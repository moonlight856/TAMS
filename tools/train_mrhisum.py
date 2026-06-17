"""
Train TAMS on MrHiSum (TripleSumm-style HDF5: V+T+A per-second, gt_score regression).

Checkpoint selection: maximize val (Kendall τ + Spearman ρ), aligned with TripleSumm-style
summary correlation metrics (see gravit/utils/mrhisum_metrics.py).

Optimizations for large scale: CPU dataset tensors + DataLoader workers, CUDA AMP, optional
gradient checkpointing, cosine LR over full training, auxiliary rank loss, modality dropout.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import setup_paths  # noqa: F401

from gravit.utils.cfg_defaults import merge_defaults
from gravit.utils.logger import get_logger
from gravit.utils.mrhisum_metrics import evaluate_highlight, evaluate_summary, pack_variable_length_scores
from gravit.utils.parser import get_args, get_cfg
from gravit.utils.seed import seed_everything
from gravit.models import get_loss_func
from model import build_model
from mrhisum_dataset import MrHiSumDataset
from loss_ext import (
    gate_entropy_loss,
    pairwise_rank_loss,
    list_mle_loss,
    topology_regularization_loss,
    margin_separation_loss,
)
from tqdm import tqdm


def _compute_gumbel_temperature(epoch: int, cfg: dict) -> float:
    """Linear annealing of Gumbel temperature from init to final over specified epochs."""
    t_init = float(cfg.get("matl_temperature_init", 1.0))
    t_final = float(cfg.get("matl_temperature_final", 0.1))
    t_epochs = int(cfg.get("matl_temperature_epochs", 15))
    if t_epochs <= 0:
        return t_final
    progress = min(epoch / t_epochs, 1.0)
    return t_init + (t_final - t_init) * progress


def _collate_mrhisum(batch):
    """Batch size must be 1 (variable-length graphs)."""
    return batch[0]


def _apply_mrhisum_modality_dropout(x: torch.Tensor, cfg: dict, training: bool) -> torch.Tensor:
    """Randomly zero one modality slice during training (robustness; TripleSumm-style trimodal)."""
    if not training:
        return x
    p = float(cfg.get("mrhisum_modality_dropout_p", 0.0))
    if p <= 0:
        return x
    nm = int(cfg.get("num_modality", 1))
    if nm < 2:
        return x
    if torch.rand(()) >= p:
        return x
    which = cfg.get("mrhisum_modality_dropout_which", "aux")
    if which == "aux":
        candidates = list(range(1, nm))
    else:
        candidates = list(range(nm))
    m = int(candidates[torch.randint(len(candidates), (1,)).item()])
    out = x.clone()
    modality_dims = cfg.get("modality_dims")
    if modality_dims and len(modality_dims) >= nm:
        start = sum(modality_dims[:m])
        out[:, start : start + modality_dims[m]] = 0
    else:
        fd = x.shape[1]
        if fd % nm != 0:
            return x
        chunk = fd // nm
        out[:, m * chunk : (m + 1) * chunk] = 0
    return out


def _forward_mrhisum(model, cfg, x, e, e_attr, device, amp_enabled: bool, video_id: str | None = None,
                     amp_dtype=torch.float16):
    x = _apply_mrhisum_modality_dropout(x, cfg, model.training)
    nc = None
    ms = None
    if cfg.get("use_step0"):
        from modules_step0 import default_visual_confidence, build_modality_scale_videoxum

        nc = default_visual_confidence(x.shape[0], device, x.dtype)
        if cfg.get("step0_multimodal_json") and video_id is not None:
            nm = int(cfg.get("num_modality", 1))
            ms = build_modality_scale_videoxum(cfg, str(video_id), x.shape[0], nm, device, x.dtype)
    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
        logits = model(x, e, e_attr, None, node_conf=nc, modality_scale=ms)
    return logits


@torch.no_grad()
def _val_metrics(loader, cfg, model, device, loss_func, amp_enabled: bool,
                 amp_dtype=torch.float16) -> tuple[float, float, float, float, float]:
    model.eval()
    preds_list: list[list[float]] = []
    gts_list: list[list[float]] = []
    loss_sum = 0.0
    n = 0
    for sample in tqdm(loader, desc="val", leave=False):
        x, y, e, e_attr, vid = sample
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True)
        e_attr = e_attr.to(device, non_blocking=True)
        logits = _forward_mrhisum(model, cfg, x, e, e_attr, device, amp_enabled, video_id=vid, amp_dtype=amp_dtype)
        pred = logits.squeeze(-1).float()
        batch_loss = loss_func(pred, y)
        if torch.isfinite(batch_loss):
            loss_sum += float(batch_loss.item())
            n += 1
        preds_list.append(pred.detach().cpu().numpy().tolist())
        gts_list.append(y.detach().cpu().numpy().tolist())
    pp, gg, mask_arr = pack_variable_length_scores(preds_list, gts_list)
    kt, sr = evaluate_summary(pp, gg, mask_arr)
    fps = float(cfg.get("mrhisum_eval_fps", 1.0))
    shot_sec = float(cfg.get("mrhisum_hl_shot_seconds", 5.0))
    try:
        m50, m15 = evaluate_highlight(pp, gg, mask_arr, fps=fps, shot_duration_seconds=shot_sec)
    except ImportError as e:
        print(e, file=sys.stderr)
        m50, m15 = float("nan"), float("nan")
    loss_m = loss_sum / max(n, 1) if n > 0 else float("nan")
    return loss_m, kt, sr, m50, m15


def train(cfg: dict) -> None:
    seed_everything(int(cfg.get("seed", 42)))
    merge_defaults(cfg)

    path_result = os.path.join(cfg["root_result"], cfg["exp_name"])
    if cfg.get("split") is not None:
        path_result = os.path.join(path_result, f"split{cfg['split']}")
    os.makedirs(path_result, exist_ok=True)

    logger = get_logger(path_result, file_name="train")
    logger.info(cfg["exp_name"])

    mr = Path(cfg.get("mrhisum_root") or "")
    if not mr.is_dir():
        raise FileNotFoundError(f"mrhisum_root is not a directory: {mr.resolve()}")

    device_str = cfg.get("device", "cuda:0")
    if not torch.cuda.is_available() and "cuda" in device_str:
        device = torch.device("cpu")
        logger.info("CUDA not available; using CPU")
    else:
        device = torch.device(device_str)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_ds = MrHiSumDataset.from_cfg("train", cfg, device=torch.device("cpu"))
    val_ds = MrHiSumDataset.from_cfg("val", cfg, device=torch.device("cpu"))

    cfg["num_modality"] = train_ds.num_modalities
    cfg["videoxum_modality_dim"] = train_ds.modality_dim
    cfg["modality_dims"] = train_ds.modality_dims_list

    with open(os.path.join(path_result, "cfg.yaml"), "w", encoding="utf-8") as f:
        yaml.dump({k: v for k, v in cfg.items() if v is not None}, f, default_flow_style=False, sort_keys=False)

    model = build_model(cfg).to(device)
    bs = int(cfg.get("batch_size", 1))
    if bs != 1:
        logger.warning("MrHiSum graphs are variable-length; forcing batch_size=1")
        bs = 1

    nw = int(cfg.get("mrhisum_dataloader_num_workers", 0))
    pm = bool(cfg.get("mrhisum_dataloader_pin_memory", True)) and device.type == "cuda"
    pw = bool(cfg.get("mrhisum_persistent_workers", False)) and nw > 0
    dl_kw = dict(
        batch_size=bs,
        collate_fn=_collate_mrhisum,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=pw,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kw)

    loss_func = get_loss_func(cfg)
    loss_func_val = get_loss_func(cfg, "val")
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("wd", 0.001)))
    tmax = cfg.get("mrhisum_cosine_t_max")
    if tmax is None:
        tmax = int(cfg["num_epoch"])
    else:
        tmax = int(tmax)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(tmax, 1))

    amp_enabled = device.type == "cuda" and bool(cfg.get("use_amp", True))
    amp_dtype_name = str(cfg.get("amp_dtype", "float16")).lower()
    if amp_dtype_name == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    if amp_enabled and amp_dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            logger.warning("BF16 not supported on this GPU; disabling AMP")
            amp_enabled = False
        else:
            logger.info("Using CUDA AMP (bfloat16) — no overflow risk; set use_amp: false to disable")
    elif amp_enabled:
        logger.info("Using CUDA AMP (float16); set use_amp: false to disable")

    use_grad_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler) if device.type == "cuda" else None

    if cfg.get("gradient_checkpointing"):
        logger.info("gradient_checkpointing enabled (lower VRAM, more compute)")

    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    rank_warmup_epochs = int(cfg.get("rank_warmup_epochs", 2))
    nan_abort_ratio = float(cfg.get("nan_abort_ratio", 0.8))
    nan_consecutive_limit = int(cfg.get("nan_consecutive_abort", 3))
    nan_lr_decay = float(cfg.get("nan_lr_decay", 0.5))

    best_score = -1e9
    epoch_best = 0
    patience = int(cfg.get("patience", 0))
    patience_cnt = 0
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    total_batches = len(train_loader)
    consecutive_high_nan = 0
    rank_suppressed_until = 0
    save_topk = int(cfg.get("mrhisum_save_topk", 1))
    save_topk = max(save_topk, 1)
    topk_records: list[tuple[float, int, str]] = []  # (score, epoch, path)

    # ---- Self-distillation teacher predictions (optional) ----
    # When enabled via cfg["teacher_preds_path"] + cfg["lambda_distill"]>0, the
    # student additionally regresses against pre-cached teacher scores. Teacher
    # predictions are stored in a single .npz keyed by video_id (variable-length).
    # This adds zero per-step compute beyond a small CPU dict lookup; predictions
    # are pinned in CPU memory once and copied to GPU per-step.
    teacher_preds: dict[str, torch.Tensor] = {}
    teacher_path = cfg.get("teacher_preds_path")
    lambda_distill = float(cfg.get("lambda_distill", 0.0))
    if lambda_distill > 0.0 and teacher_path:
        if not os.path.isfile(teacher_path):
            logger.warning(f"teacher_preds_path not found: {teacher_path} -- distillation disabled")
            lambda_distill = 0.0
        else:
            with open(teacher_path, "rb") as _tf:
                raw = pickle.load(_tf)
            for k, v in raw.items():
                arr = np.asarray(v, dtype=np.float32)
                teacher_preds[k] = torch.from_numpy(arr)
            logger.info(
                f"Loaded {len(teacher_preds)} teacher predictions from {teacher_path} "
                f"(lambda_distill={lambda_distill})"
            )

    # ---- Resume support (optional) ----
    # When cfg["resume_ckpt"] points to an existing ckpt and cfg["start_epoch"]>1,
    # we load the model weights, fast-forward the cosine scheduler, and skip the
    # already-completed epochs. Optimizer state is reset (acceptable: AdamW only
    # loses 1-step momentum, which a fresh epoch quickly rebuilds). Best-score
    # tracker is also restored from cfg["resume_best_score"] / ["resume_best_epoch"]
    # so patience/early-stopping continues correctly.
    start_epoch = int(cfg.get("start_epoch", 1))
    resume_ckpt = cfg.get("resume_ckpt")
    if resume_ckpt and start_epoch > 1:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"resume_ckpt not found: {resume_ckpt}")
        sd = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=False)
        # Fast-forward the cosine scheduler by (start_epoch - 1) steps so LR aligns.
        for _ in range(start_epoch - 1):
            scheduler.step()
        if "resume_best_score" in cfg:
            best_score = float(cfg["resume_best_score"])
        if "resume_best_epoch" in cfg:
            epoch_best = int(cfg["resume_best_epoch"])
        # Re-populate topk_records from previously-saved ckpts so save_topk works.
        # Each existing ckpt gets a synthetic score so that:
        #   - The known best epoch ranks highest (= resume_best_score).
        #   - Other existing ckpts get monotonically-decreasing scores by recency
        #     (closer-to-best epochs preserved over far-from-best).
        #   - New epochs that beat the worst existing displace it (correct top-K
        #     semantics); new epochs that beat the best update best_score.
        # Optional cfg["resume_topk_scores"] (dict {epoch:score}) overrides this.
        explicit_scores = cfg.get("resume_topk_scores") or {}
        existing_topk = sorted(
            [p for p in Path(path_result).glob("ckpt_top*.pt")],
            key=lambda p: int(p.stem.replace("ckpt_top", "")),
        )
        for p in existing_topk:
            ep_n = int(p.stem.replace("ckpt_top", ""))
            if ep_n in explicit_scores:
                score_p = float(explicit_scores[ep_n])
            elif ep_n == epoch_best:
                score_p = float(best_score)
            else:
                # Synthetic: best minus 0.001 per epoch distance from best.
                score_p = float(best_score) - 0.001 * abs(epoch_best - ep_n)
            topk_records.append((score_p, ep_n, str(p)))
        logger.info(
            f"RESUMED from {resume_ckpt} -> start_epoch={start_epoch}, "
            f"best_score={best_score:.4f}, best_epoch={epoch_best}, "
            f"recovered_topk={len(topk_records)} ckpts, lr={optimizer.param_groups[0]['lr']:.6f}"
        )

    for epoch in range(start_epoch, int(cfg["num_epoch"]) + 1):
        model.train()
        loss_tr = 0.0
        n_tr = 0
        skipped_nonfinite = 0
        scaler_overflows = 0

        cur_lr = optimizer.param_groups[0]["lr"]
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = float(cfg["lr"]) * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            cur_lr = warmup_lr

        rank_active = cfg.get("use_rank_loss") and epoch > rank_warmup_epochs and epoch > rank_suppressed_until

        if cfg.get("use_matl") and hasattr(model, "matl") and model.matl is not None:
            tau_temp = _compute_gumbel_temperature(epoch, cfg)
            model.matl.set_temperature(tau_temp)
            freeze_ep = int(cfg.get("matl_freeze_epochs", 0))
            if freeze_ep > 0:
                model.matl.requires_grad_(epoch > freeze_ep)

        for sample in tqdm(train_loader, desc=f"epoch {epoch}"):
            x, y, e, e_attr, vid = sample
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)
            e_attr = e_attr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_mrhisum(model, cfg, x, e, e_attr, device, amp_enabled, video_id=vid, amp_dtype=amp_dtype)
            pred = logits.squeeze(-1).float()
            loss = loss_func(pred, y)
            if rank_active:
                rank_type = cfg.get("rank_loss_type", "list_mle")
                lam_rank = float(cfg.get("lambda_rank", 0.05))
                if rank_type == "list_mle":
                    loss = loss + lam_rank * list_mle_loss(
                        pred, y, max_items=int(cfg.get("rank_list_max_items", 256)),
                    )
                else:
                    loss = loss + lam_rank * pairwise_rank_loss(
                        pred, y,
                        max_pairs=int(cfg.get("rank_pairs_per_video", 48)),
                        margin=float(cfg.get("rank_margin", 0.05)),
                    )
            # Opt-in margin separation loss: encourages a clear gap between
            # top-k and bottom-k frames (directly improves mAP@K & τ/ρ on
            # heavily-imbalanced highlight regression).  Bounded in [0, margin]
            # so it never blows up training. Only active when explicitly
            # enabled in cfg, preserving behavior for other datasets.
            margin_warmup_done = epoch > int(cfg.get("margin_warmup_epochs", 0))
            margin_lam = float(cfg.get("lambda_margin", 0.0))
            if margin_lam > 0.0 and margin_warmup_done:
                loss = loss + margin_lam * margin_separation_loss(
                    pred, y,
                    margin=float(cfg.get("margin_value", 0.3)),
                    top_frac=float(cfg.get("margin_top_frac", 0.15)),
                )
            # Multi-scale margin: extra top-5% (mAP15 lever) and top-50%
            # (mAP50 lever) constraints. Each is independent and bounded,
            # so adding them cannot destabilise training.
            margin_lam_top5 = float(cfg.get("lambda_margin_top5", 0.0))
            if margin_lam_top5 > 0.0 and margin_warmup_done:
                loss = loss + margin_lam_top5 * margin_separation_loss(
                    pred, y,
                    margin=float(cfg.get("margin_value_top5", 0.4)),
                    top_frac=float(cfg.get("margin_top_frac_top5", 0.05)),
                )
            margin_lam_top50 = float(cfg.get("lambda_margin_top50", 0.0))
            if margin_lam_top50 > 0.0 and margin_warmup_done:
                loss = loss + margin_lam_top50 * margin_separation_loss(
                    pred, y,
                    margin=float(cfg.get("margin_value_top50", 0.2)),
                    top_frac=float(cfg.get("margin_top_frac_top50", 0.5)),
                )
            # Self-distillation: regress pred toward teacher_pred (cached, .npz).
            # Skipped silently if teacher_preds is empty or vid not present, so this
            # branch is fully no-op for any other dataset / when disabled.
            if lambda_distill > 0.0 and teacher_preds:
                t_arr = teacher_preds.get(vid)
                if t_arr is not None and t_arr.shape[0] == pred.shape[0]:
                    teacher_y = t_arr.to(device, non_blocking=True)
                    loss = loss + lambda_distill * torch.nn.functional.mse_loss(pred, teacher_y)
            lam_gate = float(cfg.get("lambda_gate", 0.0))
            if lam_gate > 0 and getattr(model, "_last_gate_alpha", None) is not None:
                loss = loss + lam_gate * gate_entropy_loss(model._last_gate_alpha.float())
            if cfg.get("use_matl") and hasattr(model, "_last_all_weights") and model._last_all_weights:
                lam_ts = float(cfg.get("lambda_topo_sparse", 0.01))
                lam_td = float(cfg.get("lambda_topo_div", 0.005))
                rho_min = float(cfg.get("topo_rho_min", 0.3))
                if lam_ts > 0 or lam_td > 0:
                    loss = loss + topology_regularization_loss(
                        model._last_all_weights,
                        lambda_sparse=lam_ts,
                        lambda_div=lam_td,
                        rho_min=rho_min,
                    )
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                continue
            if use_grad_scaler and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < old_scale:
                    scaler_overflows += 1
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            loss_tr += float(loss.detach().item())
            n_tr += 1

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            pass
        else:
            scheduler.step()

        skip_ratio = skipped_nonfinite / max(total_batches, 1)
        loss_v, kt, sr, m50, m15 = _val_metrics(val_loader, cfg, model, device, loss_func_val, amp_enabled, amp_dtype=amp_dtype)
        combo = kt + sr
        train_loss_str = f"{loss_tr / max(n_tr, 1):.4f}" if n_tr > 0 else "nan"
        skip_msg = f" skipped={skipped_nonfinite}({skip_ratio:.0%})" if skipped_nonfinite else ""
        overflow_msg = f" scaler_overflows={scaler_overflows}" if scaler_overflows else ""
        logger.info(
            f"Epoch [{epoch:03d}|{cfg['num_epoch']:03d}] lr={cur_lr:.6f} "
            f"train_loss={train_loss_str}{skip_msg}{overflow_msg} "
            f"val_loss={loss_v:.4f} tau={kt:.4f} rho={sr:.4f} mAP50={m50:.2f} mAP15={m15:.2f} combo={combo:.4f}"
        )

        if skip_ratio > nan_abort_ratio and epoch > warmup_epochs:
            consecutive_high_nan += 1
            if consecutive_high_nan >= nan_consecutive_limit:
                logger.warning(
                    f"NaN abort: {skip_ratio:.0%} non-finite for {consecutive_high_nan} consecutive "
                    f"epochs (limit {nan_consecutive_limit}). Stopping."
                )
                break
            for pg in optimizer.param_groups:
                pg["lr"] = max(pg["lr"] * nan_lr_decay, 1e-6)
            rank_suppressed_until = epoch + 3
            logger.warning(
                f"High NaN rate ({skip_ratio:.0%}), lr decayed to {optimizer.param_groups[0]['lr']:.6f}, "
                f"rank loss suppressed until epoch {rank_suppressed_until + 1} "
                f"({consecutive_high_nan}/{nan_consecutive_limit} before abort)"
            )
        elif skip_ratio < nan_abort_ratio * 0.5:
            consecutive_high_nan = max(0, consecutive_high_nan - 1)

        # Composite monitor: τ+ρ dominates, but small mAP weight discourages
        # selecting an epoch where mAP regresses while τ/ρ are similar.
        if cfg.get("mrhisum_monitor", "combo") == "combo_with_map":
            mon = combo + 0.005 * (m50 + m15)
        else:
            mon = combo

        if mon > best_score:
            best_score = mon
            epoch_best = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(path_result, "ckpt_best.pt"))
            logger.info(f"  new best (tau+rho) epoch {epoch_best}")
        else:
            patience_cnt += 1
            if patience > 0 and patience_cnt >= patience:
                logger.info(f"Early stop: no improvement for {patience} epochs")
                break

        # ---- Top-K rolling checkpoint window for SWA-style averaging ----
        # Always-on (k=1 by default) – maintains existing behavior. Set
        # mrhisum_save_topk: 3 (or 5) in cfg to keep the K best checkpoints
        # so they can be averaged at eval time via --avg_ckpts.
        if save_topk > 1:
            ckpt_path_k = os.path.join(path_result, f"ckpt_top{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path_k)
            topk_records.append((float(mon), int(epoch), ckpt_path_k))
            topk_records.sort(key=lambda r: r[0], reverse=True)
            while len(topk_records) > save_topk:
                _, _, drop_path = topk_records.pop()
                try:
                    if os.path.isfile(drop_path):
                        os.remove(drop_path)
                except OSError:
                    pass

    if save_topk > 1 and topk_records:
        kept = sorted(topk_records, key=lambda r: r[0], reverse=True)
        kept_str = ", ".join(f"epoch{e}({s:.4f})" for s, e, _ in kept)
        logger.info(f"Top-{save_topk} checkpoints kept: {kept_str}")

    logger.info(f"Training finished. Best epoch {epoch_best}, best tau+rho={best_score:.4f}")
    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    args = get_args()
    cfg = get_cfg(args)
    train(cfg)
