"""
QFVS evaluation (Sharghi et al. 2017; set-based P/R/F1 metrics).

Default prediction rule: ``--qfvs_pred topk_gt`` — mark the top-|GT+| nodes by
sigmoid score (GT+ = nodes where mean user label > 0.5). This avoids degenerate
all-zero predictions when logits are negative everywhere (common with BCE on
sparse positives). Use ``--qfvs_pred sigmoid_thresh --qfvs_thresh 0.5`` for
fixed-threshold eval.
Precision / recall / F1 follow the set form:
  P = |S ∩ Ŝ| / |Ŝ|,  R = |S ∩ Ŝ| / |S|,  F1 = 2PR/(P+R).

Optional ``--qfvs_agg max_user``: F1 = max over the 3 users (closer to VS_max-style).

For Sharghi *shot-level* metrics (dense-tag IoU + bipartite matching, CVPR'17 style),
use ``tools/eval_qfvs_sharghi.py`` (in-repo random baseline + TAMS export). This
script remains the **frame/node-subset** pipeline for ``train_qfvs.py``.
"""

import setup_paths  # noqa: F401

import os
import argparse
import yaml
import torch
import numpy as np
from gravit.utils.cfg_defaults import merge_defaults
from gravit.utils.logger import get_logger
from model import build_model, build_dataloaders


def _forward(model, cfg, data, device):
    nb = device.type == "cuda"
    x = data.x.to(device, non_blocking=nb)
    edge_index = data.edge_index.to(device, non_blocking=nb)
    edge_attr = data.edge_attr.to(device, non_blocking=nb)
    c = data.c.to(device, non_blocking=nb) if cfg["use_spf"] else None
    q = getattr(data, "query_emb", None)
    qe = q.to(device, non_blocking=nb) if q is not None else None
    if cfg.get("model_name") == "TAMS":
        from modules_step0 import resolve_node_confidence

        nc = resolve_node_confidence(cfg, data, x.shape[0], device, x.dtype)
        return model(x, edge_index, edge_attr, c, query_emb=qe, node_conf=nc)
    return model(x, edge_index, edge_attr, c)


def _prf1(pred_bin: np.ndarray, gt_bin: np.ndarray):
    tp = np.logical_and(pred_bin, gt_bin).sum()
    p = tp / max(pred_bin.sum(), 1)
    r = tp / max(gt_bin.sum(), 1)
    f1 = 2 * p * r / max(p + r, 1e-8) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f1 * 100)


def _prf1_max_user(pred_bin: np.ndarray, user_at_picks: np.ndarray):
    """user_at_picks: [3, n_nodes] binary."""
    best = (0.0, 0.0, 0.0)
    for u in range(user_at_picks.shape[0]):
        gt = user_at_picks[u].astype(np.uint8)
        p, r, f1 = _prf1(pred_bin, gt)
        if f1 > best[2]:
            best = (p, r, f1)
    return best


def _pred_from_scores(
    scores: np.ndarray,
    mode: str,
    gt_bin: np.ndarray,
    thresh: float,
) -> np.ndarray:
    """
    Build binary prediction on graph nodes.
    - sigmoid_thresh: scores are already sigmoid probabilities; mark > thresh.
    - topk_gt: predict exactly k=|GT+| highest scores (standard when labels are
      ultra-sparse and a fixed 0.5 threshold yields all-negative logits).
    """
    n = int(scores.shape[0])
    if mode == "sigmoid_thresh":
        return (scores > thresh).astype(np.uint8)
    if mode == "topk_gt":
        g = int(gt_bin.sum())
        if g <= 0:
            return np.zeros(n, dtype=np.uint8)
        k = min(g, n)
        out = np.zeros(n, dtype=np.uint8)
        out[np.argsort(-scores)[:k]] = 1
        return out
    raise ValueError(f"Unknown qfvs_pred mode: {mode}")


def evaluate(
    cfg,
    *,
    agg: str = "mean",
    qfvs_pred: str = "topk_gt",
    qfvs_thresh: float = 0.5,
) -> tuple[float, float, float]:
    path_graphs = os.path.join(cfg["root_data"], f'graphs/{cfg["graph_name"]}')
    path_result = os.path.join(cfg["root_result"], f'{cfg["exp_name"]}')
    if cfg.get("split") is not None:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
        path_result = os.path.join(path_result, f'split{cfg["split"]}')
    cfg["dataset"] = "QFVS"

    logger = get_logger(path_result, file_name="eval")
    device_str = cfg.get("device", "cuda:0")
    device = torch.device("cpu" if ("cuda" in device_str and not torch.cuda.is_available()) else device_str)

    _, val_loader = build_dataloaders(cfg, path_graphs)
    model = build_model(cfg).to(device)
    state = torch.load(
        os.path.join(path_result, "ckpt_best.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state)
    model.eval()
    amp_enabled = device.type == "cuda" and bool(cfg.get("use_amp", True))

    ps, rs, f1s = [], [], []
    with torch.no_grad():
        for data in val_loader:
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = _forward(model, cfg, data, device)
            prob = torch.sigmoid(logits.squeeze()).detach().cpu().numpy().astype(np.float64)
            up = data.user_summary[:, data.picks].float().numpy()
            if agg == "max_user":
                best = (0.0, 0.0, 0.0)
                for u in range(up.shape[0]):
                    gt_u = up[u]
                    gt_bin = (gt_u > 0.5).astype(np.uint8)
                    pred = _pred_from_scores(prob, qfvs_pred, gt_bin, qfvs_thresh)
                    p, r, f1 = _prf1(pred, gt_bin)
                    if f1 > best[2]:
                        best = (p, r, f1)
                p, r, f1 = best
            else:
                gt = up.mean(0)
                gt_bin = (gt > 0.5).astype(np.uint8)
                pred = _pred_from_scores(prob, qfvs_pred, gt_bin, qfvs_thresh)
                p, r, f1 = _prf1(pred, gt_bin)
            ps.append(p)
            rs.append(r)
            f1s.append(f1)

    mp, mr, mf = float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))
    logger.info(
        f"QFVS P={mp * 100:.2f} R={mr * 100:.2f} F1={mf:.2f} "
        f"(agg={agg}, pred={qfvs_pred}{'' if qfvs_pred != 'sigmoid_thresh' else f',thresh={qfvs_thresh}'}; "
        f"P/R scaled 0-100)"
    )
    return mp, mr, mf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--root_data", type=str, default=None)
    parser.add_argument("--root_result", type=str, default=None)
    parser.add_argument("--split", type=int, default=None, help="Fold 1..4 (overrides cfg if set)")
    parser.add_argument(
        "--qfvs_agg",
        type=str,
        default="mean",
        choices=("mean", "max_user"),
        help="mean: GT = majority mean of 3 users at picks; max_user: max F1 over users",
    )
    parser.add_argument(
        "--qfvs_pred",
        type=str,
        default="topk_gt",
        choices=("topk_gt", "sigmoid_thresh"),
        help=(
            "topk_gt: select top-|GT+| nodes by score (recommended; sparse QFVS labels). "
            "sigmoid_thresh: binary if sigmoid > --qfvs_thresh (often all-zero with 0.5 + negative logits)."
        ),
    )
    parser.add_argument(
        "--qfvs_thresh",
        type=float,
        default=0.5,
        help="Used when --qfvs_pred sigmoid_thresh",
    )
    parser.add_argument(
        "--all_splits",
        action="store_true",
        help="Evaluate split1..split4 and print macro-average P/R/F1",
    )
    args = parser.parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    merge_defaults(base_cfg)
    if args.root_data is not None:
        base_cfg["root_data"] = args.root_data
    if args.root_result is not None:
        base_cfg["root_result"] = args.root_result
    if args.split is not None:
        base_cfg["split"] = args.split

    ev_kw = dict(agg=args.qfvs_agg, qfvs_pred=args.qfvs_pred, qfvs_thresh=args.qfvs_thresh)
    if args.all_splits:
        acc = []
        for s in range(1, 5):
            cfg = {**base_cfg, "split": s}
            merge_defaults(cfg)
            p, r, f1 = evaluate(cfg, **ev_kw)
            acc.append((p, r, f1))
        mp = float(np.mean([a[0] for a in acc]))
        mr = float(np.mean([a[1] for a in acc]))
        mf = float(np.mean([a[2] for a in acc]))
        print(f"QFVS 4-split macro-avg: P={mp * 100:.2f} R={mr * 100:.2f} F1={mf:.2f}")
    else:
        evaluate(base_cfg, **ev_kw)
