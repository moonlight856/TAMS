import setup_paths  # noqa: F401

import argparse
import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from gravit.utils.parser import get_args, get_cfg
from gravit.utils.logger import get_logger
from gravit.utils.seed import seed_everything
from gravit.utils.cfg_defaults import merge_defaults
from gravit.utils.val_metrics import f1_topk_budget, spearman_kendall
from gravit.models import get_loss_func
from videoxum_dataset import VideoXumDataset, videoxum_collate_fn
from model import build_model
from loss_ext import pairwise_rank_loss, gate_entropy_loss, topology_regularization_loss
from training_common import videoxum_rank_target
from tqdm import tqdm


def _apply_videoxum_modality_dropout(x: torch.Tensor, cfg: dict, training: bool) -> torch.Tensor:
    """Randomly zero one modality slice during training for robustness."""
    if not training:
        return x
    p = float(cfg.get("videoxum_modality_dropout_p", 0.0))
    if p <= 0:
        return x
    nm = int(cfg.get("num_modality", 1))
    if nm < 2:
        return x
    if torch.rand(()) >= p:
        return x
    fd = x.shape[1]
    if fd % nm != 0:
        return x
    chunk = fd // nm
    which = cfg.get("videoxum_modality_dropout_which", "aux")
    if which == "aux":
        candidates = list(range(1, nm))
    else:
        candidates = list(range(nm))
    m = int(candidates[torch.randint(len(candidates), (1,)).item()])
    out = x.clone()
    out[:, m * chunk : (m + 1) * chunk] = 0
    return out


def _videoxum_modality_scale(cfg: dict, vid: str, num_nodes: int, device, dtype):
    from modules_step0 import build_modality_scale_videoxum

    nm = int(cfg.get("num_modality", 1))
    return build_modality_scale_videoxum(cfg, str(vid), num_nodes, nm, device, dtype)


@torch.no_grad()
def get_label(pred, usr_sum, loss_func, num=5, eps=0.5, y=None):
    usr_sum = usr_sum.to(pred.device)
    losses = [loss_func(pred[:, 0], y).item() for y in usr_sum]
    idxs = sorted(range(usr_sum.shape[0]), key=lambda x: losses[x])
    idxs = idxs[: min([len(idxs), num])]
    if y is None:
        y = (1 - eps) * torch.mean(usr_sum[idxs], dim=0, keepdim=False) + eps * torch.mean(usr_sum, dim=0, keepdim=False)
    else:
        y = (1 - eps) * torch.mean(usr_sum[idxs], dim=0, keepdim=False) + eps * y[:, 0]
    return y[:, None].detach()


def val(val_loader, cfg, model, device, loss_func):
    model.eval()
    loss_sum = 0.0
    rank_sum = 0.0
    n_rank = 0
    spears: list[float] = []
    kendalls: list[float] = []
    f1_proxies: list[float] = []
    budget = float(cfg.get("joint_val_budget", 0.15))
    with torch.no_grad():
        for data in tqdm(val_loader, desc="val", leave=False):
            x, y, e, e_attr, vid = data
            x = x.to(device)
            y = y
            e = e.to(device)
            e_attr = e_attr.to(device)
            c = None
            ms = _videoxum_modality_scale(cfg, vid, x.shape[0], device, x.dtype)
            logits = model(x, e, e_attr, c, modality_scale=ms)
            y_t = get_label(logits, y, loss_func, cfg["k"], cfg["eps"], None)
            loss_sum += loss_func(logits, y_t).item()
            if cfg.get("use_rank_loss"):
                rt = videoxum_rank_target(cfg, y, y_t, logits.device)
                r = pairwise_rank_loss(
                    logits[:, 0],
                    rt,
                    max_pairs=cfg.get("rank_pairs_per_video", 32),
                )
                rank_sum += float(r.item())
                n_rank += 1
            pred = logits[:, 0].detach().cpu().numpy().ravel()
            gt = torch.mean(y, dim=0).detach().cpu().numpy().ravel()
            rho, tau = spearman_kendall(pred, gt)
            if rho is not None:
                spears.append(rho)
            if tau is not None:
                kendalls.append(tau)
            if cfg.get("joint_use_f1_proxy", True) and pred.size > 0:
                f1_proxies.append(f1_topk_budget(pred, gt, budget=budget))
    loss_mean = loss_sum / len(val_loader)
    rank_mean = rank_sum / max(n_rank, 1) if cfg.get("use_rank_loss") else 0.0
    spearman_mean = float(np.mean(spears)) if spears else 0.0
    kendall_mean = float(np.mean(kendalls)) if kendalls else 0.0
    f1_proxy_mean = float(np.mean(f1_proxies)) if f1_proxies else 0.0
    return loss_mean, rank_mean, spearman_mean, kendall_mean, f1_proxy_mean


def train(cfg):
    seed_everything(int(cfg.get("seed", 42)))

    path_result = os.path.join(cfg["root_result"], f'{cfg["exp_name"]}')
    split = cfg.get("split")
    if split is not None:
        path_result = os.path.join(path_result, f"split{split}")
    os.makedirs(path_result, exist_ok=True)

    logger = get_logger(path_result, file_name="train")
    logger.info(cfg["exp_name"])
    logger.info("Saving the configuration file")
    with open(os.path.join(path_result, "cfg.yaml"), "w") as f:
        yaml.dump({k: v for k, v in cfg.items() if v is not None}, f, default_flow_style=False, sort_keys=False)

    logger.info("Preparing a model and data loaders")
    device_str = cfg.get("device", "cuda:0")
    if not torch.cuda.is_available() and "cuda" in device_str:
        device = torch.device("cpu")
        logger.info("CUDA not available; using CPU")
    else:
        device = torch.device(device_str)

    merge_defaults(cfg)
    cpu = torch.device("cpu")
    train_ds = VideoXumDataset.from_cfg("train", cfg, device=cpu)
    # NOTE: training-time validation uses "test" split (VideoXum convention).
    # eval_videoxum.py uses "val" split for final reporting — metrics differ.
    val_ds = VideoXumDataset.from_cfg("test", cfg, device=cpu)
    x0, _, _, _, _ = train_ds[0]
    nmod = train_ds.num_modalities
    prev_nm = cfg.get("num_modality")
    if prev_nm is not None and int(prev_nm) != nmod:
        logger.info("num_modality: cfg had %s, using dataset value %s", prev_nm, nmod)
    cfg["num_modality"] = nmod
    cfg["videoxum_modality_dim"] = int(x0.shape[1]) // nmod

    model = build_model(cfg).to(device)
    nw = int(cfg.get("videoxum_dataloader_num_workers", 0))
    pm = bool(cfg.get("videoxum_dataloader_pin_memory", True)) and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=videoxum_collate_fn,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        collate_fn=videoxum_collate_fn,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=nw > 0,
    )

    loss_func = get_loss_func(cfg)
    loss_func_val = get_loss_func(cfg, "val")
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch_t_max = int(cfg.get("sch_param", cfg["num_epoch"]))
    if sch_t_max < cfg["num_epoch"]:
        sch_t_max = cfg["num_epoch"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch_t_max)

    logger.info("Training process started")
    epoch_best = 0
    best_metric = float("inf")

    for epoch in range(1, cfg["num_epoch"] + 1):
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
        loss_sum = 0.0
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            x, y, e, e_attr, vid = data
            x = x.to(device)
            y = y
            e = e.to(device)
            e_attr = e_attr.to(device)
            c = None
            x = _apply_videoxum_modality_dropout(x, cfg, True)
            ms = _videoxum_modality_scale(cfg, vid, x.shape[0], device, x.dtype)
            logits = model(x, e, e_attr, c, modality_scale=ms)
            y_t = get_label(logits, y, loss_func, cfg["k"], cfg["eps"], None)
            loss = loss_func(logits, y_t)
            if cfg.get("use_rank_loss"):
                rt = videoxum_rank_target(cfg, y, y_t, logits.device)
                loss = loss + float(cfg.get("lambda_rank", 0.1)) * pairwise_rank_loss(
                    logits[:, 0],
                    rt,
                    max_pairs=cfg.get("rank_pairs_per_video", 32),
                )
            lam_gate = float(cfg.get("lambda_gate", 0.0))
            if lam_gate > 0 and hasattr(model, "_last_gate_alpha") and model._last_gate_alpha is not None:
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
            loss.backward()
            max_gn = float(cfg.get("max_grad_norm", 0))
            if max_gn > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
            loss_sum += loss.item()
            optimizer.step()

        scheduler.step()
        loss_train = loss_sum / len(train_loader)
        loss_val, rank_val, spearman_val, kendall_val, f1_proxy_val = val(
            val_loader, cfg, model, device, loss_func_val
        )

        monitor = cfg.get("monitor_metric", "loss")
        wf = float(cfg.get("monitor_joint_w_f1", 1.0))
        wt = float(cfg.get("monitor_joint_w_tau", 0.2))
        wr = float(cfg.get("monitor_joint_w_rho", 0.2))
        if monitor == "val_spearman":
            metric = -spearman_val
        elif monitor == "joint":
            jf = f1_proxy_val if cfg.get("joint_use_f1_proxy", True) else 0.0
            joint_score = wf * jf + wt * kendall_val + wr * spearman_val
            metric = -joint_score
        elif monitor == "f1":
            metric = -f1_proxy_val
        elif monitor == "composite":
            metric = loss_val + cfg.get("lambda_rank", 0.1) * rank_val
        else:
            metric = loss_val

        if metric < best_metric:
            best_metric = metric
            epoch_best = epoch
            torch.save(model.state_dict(), os.path.join(path_result, "ckpt_best.pt"))

        logger.info(
            f'Epoch [{epoch:03d}|{cfg["num_epoch"]:03d}] loss_train: {loss_train:.4f}, '
            f"loss_val: {loss_val:.4f}, val_spearman: {spearman_val:.4f}, val_kendall: {kendall_val:.4f}, "
            f"val_f1_proxy: {f1_proxy_val:.4f}, monitor: {metric:.4f}, best: epoch {epoch_best:03d}"
        )

    logger.info("Training finished")


if __name__ == "__main__":
    args = get_args()
    cfg_path = args.cfg
    if getattr(args, "all_splits", False):
        snap = {k: v for k, v in vars(args).items() if k not in ("all_splits",)}
        for s in range(1, 6):
            ns = argparse.Namespace(**snap)
            ns.cfg = cfg_path
            ns.split = s
            cfg = get_cfg(ns)
            train(cfg)
    else:
        cfg = get_cfg(args)
        train(cfg)
