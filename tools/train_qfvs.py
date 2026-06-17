"""
QFVS training entry: expects graphs under data/graphs/{graph_name}/split{k}/{train,val}/*.pt
with fields x, edge_index, edge_attr, y, user_summary, picks, g, and query_emb [D].
"""

import setup_paths  # noqa: F401

import argparse
import json
import os
from pathlib import Path

import yaml
import torch
import torch.optim as optim
from gravit.utils.parser import get_args, get_cfg
from gravit.utils.logger import get_logger
from gravit.utils.seed import seed_everything
from gravit.models import get_loss_func
from model import build_model, build_dataloaders
from loss_ext import pairwise_rank_loss, query_contrastive_loss, gate_entropy_loss, topology_regularization_loss
from training_common import vs_rank_target


@torch.no_grad()
def get_label(pred, usr_sum, loss_func, num=5, eps=0.5, y=None):
    usr_sum = usr_sum.to(pred.device)
    losses = [loss_func(pred[:, 0], yy).item() for yy in usr_sum]
    idxs = sorted(range(usr_sum.shape[0]), key=lambda x: losses[x])
    idxs = idxs[: min([len(idxs), num])]
    if y is None:
        y = (1 - eps) * torch.mean(usr_sum[idxs], dim=0, keepdim=False) + eps * torch.mean(usr_sum, dim=0, keepdim=False)
    else:
        y = (1 - eps) * torch.mean(usr_sum[idxs], dim=0, keepdim=False) + eps * y[:, 0]
    return y[:, None].detach()


def _forward(model, cfg, data, device):
    nb = device.type == "cuda"
    x = data.x.to(device, non_blocking=nb)
    edge_index = data.edge_index.to(device, non_blocking=nb)
    edge_attr = data.edge_attr.to(device, non_blocking=nb)
    c = data.c.to(device, non_blocking=nb) if cfg["use_spf"] else None
    if cfg.get("model_name") == "TAMS":
        q = getattr(data, "query_emb", None)
        qe = q.to(device, non_blocking=nb) if q is not None else None
        from modules_step0 import resolve_node_confidence

        nc = resolve_node_confidence(cfg, data, x.shape[0], device, x.dtype)
        return model(x, edge_index, edge_attr, c, query_emb=qe, node_conf=nc)
    return model(x, edge_index, edge_attr, c)


def _sharghi_jsonl_path(path_result: str) -> str:
    return os.path.join(path_result, "sharghi_epochs.jsonl")


def _maybe_clear_sharghi_jsonl(path_result: str, cfg: dict, epoch: int) -> None:
    """Start a fresh jsonl each run unless ``log_sharghi_append: true``."""
    if not cfg.get("log_sharghi_epoch") or epoch != 1:
        return
    if cfg.get("log_sharghi_append", False):
        return
    open(_sharghi_jsonl_path(path_result), "w", encoding="utf-8").close()


@torch.no_grad()
def _eval_sharghi_epoch_metrics(cfg: dict, model, device: torch.device) -> tuple[float, float, float]:
    """Held-out participant Sharghi P/R/F1 (0–1) for paper Fig.4 curves."""
    from eval_qfvs_sharghi import eval_sharghi_tams_model, scan_groups

    origin = Path(cfg["qfvs_origin_root"]) if cfg.get("qfvs_origin_root") else (
        Path(cfg["root_data"]) / "annotations" / "QFVS" / "origin_data"
    )
    if not origin.is_dir():
        origin = Path(__file__).resolve().parent.parent / "data" / "annotations" / "QFVS" / "origin_data"
    groups = scan_groups(origin)
    shot_stride = int(cfg.get("qfvs_shot_stride", 1))
    graph_name = cfg.get("graph_name", "QFVS_ut")
    root_data = Path(cfg["root_data"])
    split_i = int(cfg["split"])
    return eval_sharghi_tams_model(
        model,
        origin_root=origin,
        groups=groups,
        split=split_i,
        top_frac=None,
        shot_stride=shot_stride,
        root_data=root_data,
        graph_name=graph_name,
        cfg=cfg,
        device=device,
    )


def _log_sharghi_for_epoch(
    cfg: dict,
    model,
    device: torch.device,
    path_result: str,
    epoch: int,
    val_loss: float,
    logger,
) -> str | None:
    """
    Optional per-epoch Sharghi log for Fig.4 (``log_sharghi_epoch: true`` in QFVS yaml only).
    Writes ``sharghi_epochs.jsonl`` and returns a short suffix for ``train.log``.
    """
    if not cfg.get("log_sharghi_epoch"):
        return None
    every = max(1, int(cfg.get("log_sharghi_every", 1)))
    if epoch % every != 0:
        return None

    p_s, r_s, f1_s = _eval_sharghi_epoch_metrics(cfg, model, device)
    rec = {"epoch": epoch, "P": p_s, "R": r_s, "F1": f1_s, "val_loss": val_loss}
    sharghi_path = _sharghi_jsonl_path(path_result)
    with open(sharghi_path, "a", encoding="utf-8") as sf:
        sf.write(json.dumps(rec) + "\n")

    from eval_qfvs_sharghi import PARTICIPANTS

    split_i = int(cfg["split"])
    holdout = PARTICIPANTS[split_i - 1] if 1 <= split_i <= len(PARTICIPANTS) else f"split{split_i}"
    logger.info(
        f"  Sharghi ep {epoch:03d} P={p_s*100:.2f} R={r_s*100:.2f} F1={f1_s*100:.2f} (held-out {holdout})"
    )
    return f" sharghi_P={p_s:.4f} sharghi_R={r_s:.4f} sharghi_F1={f1_s:.4f}"


def val(val_loader, cfg, model, device, loss_func, amp_enabled: bool):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for data in val_loader:
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = _forward(model, cfg, data, device)
            y = data.y.to(device)
            y_t = get_label(
                logits,
                data.user_summary[:, data.picks],
                loss_func,
                cfg["k"],
                cfg["eps"],
                None if cfg["use_mean_sum"] else y,
            )
            loss_sum += loss_func(logits, y_t).item()
    return loss_sum / len(val_loader)


def train(cfg):
    seed_everything(int(cfg.get("seed", 42)))
    cfg["dataset"] = "QFVS"

    path_graphs = os.path.join(cfg["root_data"], f'graphs/{cfg["graph_name"]}')
    path_result = os.path.join(cfg["root_result"], f'{cfg["exp_name"]}')
    if cfg.get("split") is not None:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
        path_result = os.path.join(path_result, f'split{cfg["split"]}')
    os.makedirs(path_result, exist_ok=True)

    logger = get_logger(path_result, file_name="train")
    logger.info(cfg["exp_name"])
    with open(os.path.join(path_result, "cfg.yaml"), "w") as f:
        yaml.dump({k: v for k, v in cfg.items() if v is not None}, f, default_flow_style=False, sort_keys=False)

    device_str = cfg.get("device", "cuda:0")
    if "cuda" in device_str and not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("CUDA not available; using CPU")
    else:
        device = torch.device(device_str)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    train_loader, val_loader = build_dataloaders(cfg, path_graphs)

    loss_func = get_loss_func(cfg)
    loss_func_val = get_loss_func(cfg, "val")
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tmax = cfg.get("qfvs_cosine_t_max")
    if tmax is None:
        tmax = int(cfg["num_epoch"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(tmax), 1))

    amp_enabled = device.type == "cuda" and bool(cfg.get("use_amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    if amp_enabled:
        logger.info("QFVS: CUDA AMP enabled (use_amp: false in cfg to disable)")

    best = float("inf")
    epoch_best = 0
    for epoch in range(1, cfg["num_epoch"] + 1):
        _maybe_clear_sharghi_jsonl(path_result, cfg, epoch)
        model.train()
        if cfg.get("use_matl") and hasattr(model, "matl") and model.matl is not None:
            t_init = float(cfg.get("matl_temperature_init", 1.0))
            t_final = float(cfg.get("matl_temperature_final", 0.1))
            t_epochs = int(cfg.get("matl_temperature_epochs", 15))
            progress = min(epoch / max(t_epochs, 1), 1.0)
            model.matl.set_temperature(t_init + (t_final - t_init) * progress)
            freeze_ep = int(cfg.get("matl_freeze_epochs", 0))
            if freeze_ep > 0:
                model.matl.requires_grad_(epoch > freeze_ep)
        total = 0.0
        for data in train_loader:
            optimizer.zero_grad(set_to_none=True)
            y = data.y.to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = _forward(model, cfg, data, device)
                y_t = get_label(
                    logits,
                    data.user_summary[:, data.picks],
                    loss_func,
                    cfg["k"],
                    cfg["eps"],
                    None if cfg["use_mean_sum"] else y,
                )
                loss = loss_func(logits, y_t)
                if cfg.get("use_rank_loss"):
                    rt = vs_rank_target(cfg, data, device, y_t)
                    loss = loss + float(cfg.get("lambda_rank", 0.1)) * pairwise_rank_loss(
                        logits[:, 0],
                        rt,
                        max_pairs=cfg.get("rank_pairs_per_video", 32),
                    )
                if cfg.get("use_query_loss") and getattr(data, "query_emb_neg", None) is not None:
                    nb = device.type == "cuda"
                    qn = data.query_emb_neg.to(device, non_blocking=nb)
                    from modules_step0 import resolve_node_confidence

                    xn = data.x.to(device, non_blocking=nb)
                    nc_n = resolve_node_confidence(cfg, data, xn.shape[0], device, xn.dtype)
                    logits_neg = model(
                        xn,
                        data.edge_index.to(device, non_blocking=nb),
                        data.edge_attr.to(device, non_blocking=nb),
                        data.c.to(device, non_blocking=nb) if cfg["use_spf"] else None,
                        query_emb=qn,
                        node_conf=nc_n,
                    )
                    sp = torch.sigmoid(logits).mean()
                    sn = torch.sigmoid(logits_neg).mean()
                    loss = loss + cfg.get("lambda_q", 0.1) * query_contrastive_loss(sp, sn)
                lam_gate = float(cfg.get("lambda_gate", 0.0))
                if lam_gate > 0 and getattr(model, "_last_gate_alpha", None) is not None:
                    loss = loss + lam_gate * gate_entropy_loss(model._last_gate_alpha)
                if cfg.get("use_matl") and hasattr(model, "_last_all_weights") and model._last_all_weights:
                    lam_ts = float(cfg.get("lambda_topo_sparse", 0.01))
                    lam_td = float(cfg.get("lambda_topo_div", 0.005))
                    rho_min = float(cfg.get("topo_rho_min", 0.3))
                    if lam_ts > 0 or lam_td > 0:
                        loss = loss + topology_regularization_loss(
                            model._last_all_weights, lambda_sparse=lam_ts,
                            lambda_div=lam_td, rho_min=rho_min,
                        )
            if scaler is not None:
                scaler.scale(loss).backward()
                max_gn = float(cfg.get("max_grad_norm", 0))
                if max_gn > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                max_gn = float(cfg.get("max_grad_norm", 0))
                if max_gn > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
                optimizer.step()
            total += float(loss.detach().item())
        scheduler.step()
        lv = val(val_loader, cfg, model, device, loss_func_val, amp_enabled)
        if lv < best:
            best = lv
            epoch_best = epoch
            torch.save(model.state_dict(), os.path.join(path_result, "ckpt_best.pt"))

        save_interval = int(cfg.get("save_checkpoint_interval", 0))
        if save_interval > 0 and epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(path_result, f"ckpt_ep{epoch:03d}.pt"))

        sharghi_suffix = _log_sharghi_for_epoch(cfg, model, device, path_result, epoch, lv, logger) or ""
        logger.info(
            f"Epoch [{epoch:03d}|{cfg['num_epoch']:03d}] train_loss {total/len(train_loader):.4f} "
            f"val {lv:.4f} best {epoch_best}{sharghi_suffix}"
        )
    logger.info("Training finished")


if __name__ == "__main__":
    args = get_args()
    if getattr(args, "all_splits", False):
        snap = {k: v for k, v in vars(args).items() if k not in ("all_splits",)}
        for s in range(1, 5):
            ns = argparse.Namespace(**snap)
            ns.split = s
            train(get_cfg(ns))
    else:
        train(get_cfg(args))
