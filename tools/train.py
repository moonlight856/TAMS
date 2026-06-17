import setup_paths  # noqa: F401 — repo root on sys.path for `python tools/train.py`

import argparse
import os

import yaml
import numpy as np
import torch
import torch.optim as optim
from gravit.utils.parser import get_args, get_cfg
from gravit.utils.logger import get_logger
from gravit.utils.seed import seed_everything
from gravit.utils.val_metrics import f1_topk_budget, spearman_kendall, knapsack_f1_from_graph
from gravit.models import get_loss_func
from model import build_model, build_dataloaders
from loss_ext import pairwise_rank_loss, gate_entropy_loss, topology_regularization_loss, margin_separation_loss, segment_rank_loss, list_mle_loss
from training_common import vs_rank_target


@torch.no_grad()
def get_label(pred, usr_sum, loss_func, num=5, eps=0.5, y=None):
    usr_sum = usr_sum.to(pred.device)
    losses = [loss_func(pred[:, 0], u).item() for u in usr_sum]
    idxs = sorted(range(usr_sum.shape[0]), key=lambda x: losses[x])
    idxs = idxs[: min([len(idxs), num])]
    mean_topk = torch.mean(usr_sum[idxs], dim=0, keepdim=False)
    if y is None:
        label = (1 - eps) * mean_topk + eps * torch.mean(usr_sum, dim=0, keepdim=False)
    else:
        gt = y[:, 0]
        if gt.shape[0] != mean_topk.shape[0]:
            gt = gt[: mean_topk.shape[0]]
        label = (1 - eps) * mean_topk + eps * gt
    return label[:, None].detach()


def _forward(model, cfg, data, device):
    x, y = data.x.to(device), data.y.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    c = None
    if cfg["use_spf"]:
        c = data.c.to(device)
    if cfg.get("model_name") == "TAMS":
        q = getattr(data, "query_emb", None)
        qe = q.to(device) if q is not None else None
        from modules_step0 import resolve_node_confidence

        nc = resolve_node_confidence(cfg, data, x.shape[0], device, x.dtype)
        ms = None
        if cfg.get("apply_modality_scale") and cfg.get("step0_multimodal_json") and cfg.get("use_step0"):
            from modules_step0 import build_modality_scale_videoxum, _graph_id_str
            vid = _graph_id_str(data)
            if vid is not None:
                ms = build_modality_scale_videoxum(
                    cfg, vid, x.shape[0], cfg.get("num_modality", 1), device, x.dtype,
                )
        return model(
            x, edge_index, edge_attr, c,
            query_emb=qe, node_conf=nc, modality_scale=ms,
        )
    return model(x, edge_index, edge_attr, c)


def _has_change_points(data) -> bool:
    return (hasattr(data, 'change_points')
            and data.change_points is not None
            and data.change_points.numel() > 0)


def val(val_loader, cfg, model, device, loss_func, amp_enabled: bool):
    model.eval()
    loss_sum = 0.0
    rank_sum = 0.0
    n_rank = 0
    spears: list[float] = []
    kendalls: list[float] = []
    f1_proxies: list[float] = []
    knapsack_f1s: list[float] = []
    budget = float(cfg.get("joint_val_budget", 0.15))
    eval_type = cfg.get("eval_type", "VS_max")
    with torch.no_grad():
        for data in val_loader:
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = _forward(model, cfg, data, device)
            y = data.y.to(device)
            y_t = get_label(logits, data.user_summary[:, data.picks], loss_func, cfg["k"], cfg["eps"], None if cfg["use_mean_sum"] else y)
            loss_sum += loss_func(logits, y_t).item()
            if cfg.get("use_rank_loss"):
                rt = vs_rank_target(cfg, data, device, y_t)
                r = pairwise_rank_loss(
                    logits[:, 0],
                    rt,
                    max_pairs=cfg.get("rank_pairs_per_video", 32),
                )
                rank_sum += float(r.item())
                n_rank += 1
            pred = logits[:, 0].detach().cpu().numpy().ravel()
            gt = data.y.detach().cpu().numpy().ravel()
            rho, tau = spearman_kendall(pred, gt)
            if rho is not None:
                spears.append(rho)
            if tau is not None:
                kendalls.append(tau)
            if cfg.get("joint_use_f1_proxy", True) and pred.size > 0:
                f1_proxies.append(f1_topk_budget(pred, gt, budget=budget))

            if _has_change_points(data):
                eval_temp = float(cfg.get("eval_temperature", 1.0))
                centered = logits[:, 0].detach()
                if cfg.get("eval_center_logits", False):
                    centered = centered - centered.mean()
                pred_sig = torch.sigmoid(centered * eval_temp).cpu().numpy().ravel()
                picks_np = data.picks.cpu().numpy().ravel()
                cp_np = data.change_points.cpu().numpy()
                nf = int(data.n_frames_full.view(-1)[0].item())
                usm_np = data.user_summary.cpu().numpy().astype(np.int8)
                kf1 = knapsack_f1_from_graph(pred_sig, picks_np, cp_np, nf, usm_np, eval_type)
                knapsack_f1s.append(kf1)

    loss_mean = loss_sum / len(val_loader)
    rank_mean = rank_sum / max(n_rank, 1) if cfg.get("use_rank_loss") else 0.0
    spearman_mean = float(np.mean(spears)) if spears else 0.0
    kendall_mean = float(np.mean(kendalls)) if kendalls else 0.0
    f1_proxy_mean = float(np.mean(f1_proxies)) if f1_proxies else 0.0
    knapsack_f1_mean = float(np.mean(knapsack_f1s)) if knapsack_f1s else 0.0
    return loss_mean, rank_mean, spearman_mean, kendall_mean, f1_proxy_mean, knapsack_f1_mean


def train(cfg):
    seed_everything(int(cfg.get("seed", 42)))

    path_graphs = os.path.join(cfg["root_data"], f'graphs/{cfg["graph_name"]}')
    path_result = os.path.join(cfg["root_result"], f'{cfg["exp_name"]}')
    split = cfg.get("split")
    if split is not None:
        path_graphs = os.path.join(path_graphs, f"split{split}")
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

    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    train_loader, val_loader = build_dataloaders(cfg, path_graphs)

    loss_func = get_loss_func(cfg)
    loss_func_val = get_loss_func(cfg, "val")
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    warmup_epochs = int(cfg.get("lr_warmup_epochs", 0))
    cosine_t_max = int(cfg.get("cosine_t_max", 0))
    if cosine_t_max <= 0:
        cosine_t_max = max(cfg["num_epoch"] - warmup_epochs, 1)

    if warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max)

    amp_enabled = device.type == "cuda" and bool(cfg.get("use_amp", True))
    if amp_enabled:
        logger.info("Using CUDA AMP (fp16) for forward; set use_amp: false in cfg to disable")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    logger.info("Training process started")
    epoch_best = 0
    best_metric = float("inf")

    eps_start = float(cfg.get("eps_start", cfg["eps"]))
    eps_end = float(cfg.get("eps_end", cfg["eps"]))
    eps_warmup = int(cfg.get("eps_warmup_epochs", 0))

    for epoch in range(1, cfg["num_epoch"] + 1):
        if eps_warmup > 0 and eps_start != eps_end:
            t = min(epoch / max(eps_warmup, 1), 1.0)
            current_eps = eps_start + (eps_end - eps_start) * t
        else:
            current_eps = cfg["eps"]

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
        for data in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = _forward(model, cfg, data, device)
                y = data.y.to(device)
                y_t = get_label(
                    logits,
                    data.user_summary[:, data.picks],
                    loss_func,
                    cfg["k"],
                    current_eps,
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
                lam_margin = float(cfg.get("lambda_margin", 0.0))
                if lam_margin > 0:
                    rt_margin = vs_rank_target(cfg, data, device, y_t)
                    loss = loss + lam_margin * margin_separation_loss(logits[:, 0], rt_margin)
                lam_seg = float(cfg.get("lambda_seg_rank", 0.0))
                if lam_seg > 0 and _has_change_points(data):
                    rt_seg = vs_rank_target(cfg, data, device, y_t)
                    loss = loss + lam_seg * segment_rank_loss(
                        logits[:, 0], rt_seg,
                        data.picks, data.change_points,
                    )
                lam_lmle = float(cfg.get("lambda_list_mle", 0.0))
                if lam_lmle > 0:
                    rt_lmle = vs_rank_target(cfg, data, device, y_t)
                    loss = loss + lam_lmle * list_mle_loss(
                        logits[:, 0], rt_lmle,
                        max_items=int(cfg.get("list_mle_max_items", 256)),
                    )
            scaler.scale(loss).backward()
            max_gn = float(cfg.get("max_grad_norm", 0))
            if max_gn > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().item())

        scheduler.step()

        loss_train = loss_sum / len(train_loader)

        cfg["eps"] = current_eps
        loss_val, rank_val, spearman_val, kendall_val, f1_proxy_val, knapsack_f1_val = val(
            val_loader, cfg, model, device, loss_func_val, amp_enabled
        )

        monitor = cfg.get("monitor_metric", "loss")
        wf = float(cfg.get("monitor_joint_w_f1", 1.0))
        wt = float(cfg.get("monitor_joint_w_tau", 0.2))
        wr = float(cfg.get("monitor_joint_w_rho", 0.2))
        if monitor == "val_spearman":
            metric = -spearman_val
        elif monitor == "joint":
            if knapsack_f1_val > 0:
                jf = knapsack_f1_val / 100.0
            elif cfg.get("joint_use_f1_proxy", True):
                jf = f1_proxy_val
            else:
                jf = 0.0
            joint_score = wf * jf + wt * kendall_val + wr * spearman_val
            metric = -joint_score
        elif monitor == "f1":
            metric = -(knapsack_f1_val / 100.0) if knapsack_f1_val > 0 else -f1_proxy_val
        elif monitor == "composite":
            metric = loss_val + cfg.get("lambda_rank", 0.1) * rank_val
        else:
            metric = loss_val

        warmup = min(int(cfg.get("monitor_warmup_epochs", 0)), cfg["num_epoch"] - 1)
        if epoch > warmup and metric < best_metric:
            best_metric = metric
            epoch_best = epoch
            torch.save(model.state_dict(), os.path.join(path_result, "ckpt_best.pt"))

        save_interval = int(cfg.get("save_checkpoint_interval", 0))
        if save_interval > 0 and epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(path_result, f"ckpt_ep{epoch:03d}.pt"))

        logger.info(
            f'Epoch [{epoch:03d}|{cfg["num_epoch"]:03d}] eps: {current_eps:.3f}, loss_train: {loss_train:.4f}, '
            f"loss_val: {loss_val:.4f}, val_spearman: {spearman_val:.4f}, val_kendall: {kendall_val:.4f}, "
            f"val_f1_proxy: {f1_proxy_val:.4f}, val_f1_knap: {knapsack_f1_val:.2f}, "
            f"monitor: {metric:.4f}, best: epoch {epoch_best:03d}"
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
