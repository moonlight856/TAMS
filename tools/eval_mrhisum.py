"""
Evaluate TAMS on MrHiSum test split: τ, ρ, mAP@50, mAP@15 (TripleSumm convention).

Usage:
  python tools/eval_mrhisum.py --cfg <path/to/saved/cfg.yaml>
  python tools/eval_mrhisum.py --exp_name TAMS_Net_MrHiSum_v2_final --split 4
  # Optional eval-time post-processing (works on any saved ckpt):
  python tools/eval_mrhisum.py --exp_name TAMS_Net_MrHiSum_v2_final --smoothing_sigma 2.0
  python tools/eval_mrhisum.py --exp_name TAMS_Net_MrHiSum_v2_final --avg_ckpts ckpt_best.pt ckpt_top2.pt ckpt_top3.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

import setup_paths  # noqa: F401

from gravit.utils.cfg_defaults import merge_defaults
from gravit.utils.logger import get_logger
from gravit.utils.mrhisum_metrics import evaluate_highlight, evaluate_summary, pack_variable_length_scores
from gravit.models import get_loss_func
from model import build_model
from mrhisum_dataset import MrHiSumDataset
from tqdm import tqdm


def _collate_mrhisum(batch):
    return batch[0]


def _gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian temporal smoothing on a 1-D score sequence.

    Smoothing is a standard post-processing trick for video summarization
    metrics (τ/ρ/mAP) that often gains 1-3 absolute points without retraining.
    """
    if sigma is None or sigma <= 0.0 or x.size <= 1:
        return x
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(t ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    pad = np.pad(x.astype(np.float64), (radius, radius), mode="edge")
    return np.convolve(pad, kernel, mode="valid").astype(np.float32)


def _average_state_dicts(paths: list[str]) -> dict:
    """Average parameters across multiple checkpoints (poor-man's SWA)."""
    sd_avg: dict | None = None
    n = 0
    for p in paths:
        if not os.path.isfile(p):
            print(f"[WARN] missing ckpt: {p}", file=sys.stderr)
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if sd_avg is None:
            sd_avg = {k: v.detach().clone().float() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if k in sd_avg and v.shape == sd_avg[k].shape:
                    sd_avg[k] = sd_avg[k] + v.detach().float()
        n += 1
    if sd_avg is None or n == 0:
        raise FileNotFoundError(f"No valid checkpoints to average: {paths}")
    sd_avg = {k: (v / float(n)).to(dtype=torch.float32) for k, v in sd_avg.items()}
    return sd_avg


def evaluate_run(cfg: dict) -> None:
    merge_defaults(cfg)
    path_result = os.path.join(cfg["root_result"], cfg["exp_name"])
    if cfg.get("split") is not None:
        path_result = os.path.join(path_result, f"split{cfg['split']}")

    logger = get_logger(path_result, file_name="eval")
    logger.info(cfg["exp_name"])
    logger.info(path_result)

    device_str = cfg.get("device", "cuda:0")
    if "cuda" in device_str and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    test_ds = MrHiSumDataset.from_cfg("test", cfg, device=torch.device("cpu"))
    cfg["num_modality"] = test_ds.num_modalities
    cfg["videoxum_modality_dim"] = test_ds.modality_dim
    cfg["modality_dims"] = test_ds.modality_dims_list

    model = build_model(cfg).to(device)
    avg_list = cfg.get("eval_avg_ckpts")
    if avg_list:
        avg_paths = [
            p if os.path.isabs(p) else os.path.join(path_result, p)
            for p in avg_list
        ]
        logger.info(f"Averaging {len(avg_paths)} checkpoints: {avg_list}")
        sd = _average_state_dicts(avg_paths)
        model.load_state_dict(sd, strict=False)
    else:
        ckpt = os.path.join(path_result, "ckpt_best.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    model.eval()

    # Match training-time AMP precision to avoid systematic ckpt vs eval drift.
    amp_enabled = device.type == "cuda" and bool(cfg.get("use_amp", True))
    amp_dtype_name = str(cfg.get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    if amp_enabled and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_enabled = False

    nw = int(cfg.get("mrhisum_dataloader_num_workers", 0))
    pm = bool(cfg.get("mrhisum_dataloader_pin_memory", True)) and device.type == "cuda"
    pw = bool(cfg.get("mrhisum_persistent_workers", False)) and nw > 0
    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=_collate_mrhisum,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=pw,
    )
    loss_func = get_loss_func(cfg, "val")

    preds_list: list[list[float]] = []
    gts_list: list[list[float]] = []
    loss_sum = 0.0
    n_loss = 0

    with torch.no_grad():
        for sample in tqdm(loader, desc="test"):
            x, y, e, e_attr, _vid = sample
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)
            e_attr = e_attr.to(device, non_blocking=True)
            nc = None
            if cfg.get("use_step0"):
                from modules_step0 import default_visual_confidence
                nc = default_visual_confidence(x.shape[0], device, x.dtype)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                logits = model(x, e, e_attr, None, node_conf=nc)
            pred = logits.squeeze(-1).float()
            batch_loss = loss_func(pred, y)
            if torch.isfinite(batch_loss):
                loss_sum += batch_loss.item()
                n_loss += 1
            pred_np = pred.detach().cpu().numpy()
            sigma = float(cfg.get("eval_smoothing_sigma", 0.0))
            sigmas_multi = cfg.get("eval_smoothing_sigmas", None)
            if sigmas_multi:
                acc = np.zeros_like(pred_np, dtype=np.float64)
                for s in sigmas_multi:
                    s = float(s)
                    if s <= 0.0:
                        acc += pred_np.astype(np.float64)
                    else:
                        acc += _gaussian_smooth_1d(pred_np, s).astype(np.float64)
                pred_np = (acc / float(len(sigmas_multi))).astype(np.float32)
            elif sigma > 0.0:
                pred_np = _gaussian_smooth_1d(pred_np, sigma)
            preds_list.append(pred_np.tolist())
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

    loss_m = loss_sum / max(n_loss, 1) if n_loss > 0 else float("nan")
    logger.info("Computing MrHiSum metrics (TripleSumm-style)")
    print(f"test_loss: {loss_m:.6f}, tau: {kt:.6f}, rho: {sr:.6f}, mAP50: {m50:.4f}, mAP15: {m15:.4f}")
    test_ds.close()


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    merge_defaults(cfg)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MrHiSum evaluation for TAMS")
    ap.add_argument("--cfg", type=str, default=None, help="Path to cfg yaml (e.g. saved under results/)")
    ap.add_argument("--root_result", type=str, default="./results")
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--split", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--smoothing_sigma", type=float, default=None,
        help="Gaussian temporal smoothing on predictions (sigma in seconds, e.g. 1.5/2.0/3.0; 0 disables).",
    )
    ap.add_argument(
        "--smoothing_sigmas", type=str, default=None,
        help="Comma-separated list of sigmas for multi-scale TTA (e.g. '2,3,4'). Overrides --smoothing_sigma.",
    )
    ap.add_argument(
        "--avg_ckpts", type=str, nargs="+", default=None,
        help="Average multiple checkpoint files (relative to results dir) before eval.",
    )
    args = ap.parse_args()

    if args.cfg:
        cfg = _load_cfg(args.cfg)
    elif args.exp_name:
        path_exp = os.path.join(args.root_result, args.exp_name)
        if args.split is not None:
            path_exp = os.path.join(path_exp, f"split{args.split}")
        cfg_path = os.path.join(path_exp, "cfg.yaml")
        if not os.path.isfile(cfg_path):
            raise SystemExit(f"Missing {cfg_path}; train first or pass --cfg")
        cfg = _load_cfg(cfg_path)
    else:
        ap.error("Provide --cfg or --exp_name")

    if args.device is not None:
        cfg["device"] = args.device
    if args.smoothing_sigma is not None:
        cfg["eval_smoothing_sigma"] = float(args.smoothing_sigma)
    if args.smoothing_sigmas is not None:
        cfg["eval_smoothing_sigmas"] = [float(s) for s in args.smoothing_sigmas.split(",") if s.strip()]
    if args.avg_ckpts is not None:
        cfg["eval_avg_ckpts"] = list(args.avg_ckpts)

    evaluate_run(cfg)
