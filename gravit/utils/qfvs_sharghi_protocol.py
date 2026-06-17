"""
Sharghi et al. CVPR 2017 QFVS *official-style* shot metric (dense tag IoU + matching).

Precision / recall / F1 as in the benchmark: bipartite matching between GT summary shots
and predicted summary shots, edge weight = IoU of per-shot concept tag sets; then
P = sum(matched IoU) / |pred|, R = sum(matched IoU) / |GT|, F1 = harmonic mean.

**Index alignment (important):** In the public QFVS ``origin_data`` release, User_Summary
integers and ``Dense_per_shot_tags`` lines are effectively on the **same 5s-shot axis**
(max user index ≈ number of dense lines). Use ``shot_stride=1`` so shot_id = (frame_1based-1).
If your pipeline stores **raw video frame indices** at fps *F* with 5s shots, use
``shot_stride = round(F * 5)`` (e.g. 150 at 30fps).

This module does not depend on PyG or the training model; tools call it with shot sets + tags.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_shot_tag_sets(origin_root: Path, participant: str) -> list[frozenset[str]]:
    """One frozenset of concept names per 5s shot (line order = shot index 0..)."""
    path = origin_root / "Dense_per_shot_tags" / participant / f"{participant}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    sets: list[frozenset[str]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                sets.append(frozenset())
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            sets.append(frozenset(parts))
    return sets


def tag_set_iou(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def frames_1based_to_shots(frames: list[int], *, shot_stride: int = 1) -> set[int]:
    """Map 1-based user-summary indices to 0-based dense shot ids: floor((f1-1) / shot_stride)."""
    step = max(int(shot_stride), 1)
    out: set[int] = set()
    for f1 in frames:
        if f1 < 1:
            continue
        z = f1 - 1
        out.add(z // step)
    return out


def gt_shot_set_union_users(users: dict[int, list[int]], *, shot_stride: int = 1) -> set[int]:
    s: set[int] = set()
    for u in (1, 2, 3):
        s |= frames_1based_to_shots(users[u], shot_stride=shot_stride)
    return s


def max_weight_matching_iou_sum(
    gt_shots: set[int],
    pred_shots: set[int],
    tag_sets: list[frozenset[str]],
) -> tuple[float, int, int]:
    """
    Maximum total IoU via min-cost assignment on rectangular -IoU matrix (n_gt × n_pred).
    scipy matches min(n_gt, n_pred) pairs (standard Sharghi-style bipartite matching).
    Returns (sum_iou, n_gt, n_pred).
    """
    g_list = sorted(gt_shots)
    p_list = sorted(pred_shots)
    n_g, n_p = len(g_list), len(p_list)
    if n_g == 0 or n_p == 0:
        return 0.0, n_g, n_p
    cost = np.zeros((n_g, n_p), dtype=np.float64)
    for i in range(n_g):
        ti = tag_sets[g_list[i]] if 0 <= g_list[i] < len(tag_sets) else frozenset()
        for j in range(n_p):
            tj = tag_sets[p_list[j]] if 0 <= p_list[j] < len(tag_sets) else frozenset()
            cost[i, j] = -tag_set_iou(ti, tj)
    ri, cj = linear_sum_assignment(cost)
    total = 0.0
    for i, j in zip(ri, cj, strict=False):
        ti = tag_sets[g_list[i]] if 0 <= g_list[i] < len(tag_sets) else frozenset()
        tj = tag_sets[p_list[j]] if 0 <= p_list[j] < len(tag_sets) else frozenset()
        total += tag_set_iou(ti, tj)
    return float(total), n_g, n_p


def sharghi_prf1_from_sets(
    gt_shots: set[int],
    pred_shots: set[int],
    tag_sets: list[frozenset[str]],
) -> tuple[float, float, float]:
    """Returns P, R, F1 each in [0, 1] (not scaled to 100)."""
    s, n_g, n_p = max_weight_matching_iou_sum(gt_shots, pred_shots, tag_sets)
    if n_p <= 0:
        p = 0.0
    else:
        p = s / n_p
    if n_g <= 0:
        r = 0.0
    else:
        r = s / n_g
    if p + r <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * p * r / (p + r)
    return p, r, f1


def top_frac_shot_mask(scores: np.ndarray, top_frac: float) -> set[int]:
    """Pick floor-like top fraction of shots by score (at least one if n>0)."""
    n = int(scores.shape[0])
    if n <= 0:
        return set()
    k = max(1, int(round(float(top_frac) * n)))
    k = min(k, n)
    idx = np.argsort(-scores)[:k]
    return set(int(x) for x in idx.tolist())


def top_k_shot_mask(scores: np.ndarray, k: int) -> set[int]:
    n = int(scores.shape[0])
    if n <= 0:
        return set()
    k = max(0, min(int(k), n))
    if k == 0:
        return set()
    idx = np.argsort(-scores)[:k]
    return set(int(x) for x in idx.tolist())
