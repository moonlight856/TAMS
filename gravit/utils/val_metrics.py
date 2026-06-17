"""
Validation metrics for checkpoint selection.
Proxy F1 at fixed top-k budget aligns with 15% summary budget at sample granularity.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


def f1_topk_budget(pred: np.ndarray, gt: np.ndarray, budget: float = 0.15) -> float:
    """
    F1 between top-k indices by pred vs by gt, k = round(n * budget), k >= 1.
    Scale-free proxy for knapsack F1 during training (no H5 / segments).
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    n = int(pred.size)
    if n < 2:
        return 0.0
    k = max(1, int(round(n * float(budget))))
    k = min(k, n)
    pred_set = set(np.argsort(-pred)[:k].tolist())
    gt_set = set(np.argsort(-gt)[:k].tolist())
    inter = len(pred_set & gt_set)
    if inter == 0:
        return 0.0
    p = inter / max(len(pred_set), 1)
    r = inter / max(len(gt_set), 1)
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def knapsack_f1_from_graph(
    pred_scores: np.ndarray,
    picks: np.ndarray,
    change_points: np.ndarray,
    n_frames: int,
    user_summary: np.ndarray,
    eval_type: str = "VS_max",
) -> float:
    """
    Compute actual knapsack F1, matching eval.py protocol exactly.
    Used during validation so the monitor selects the best checkpoint
    for the real evaluation metric, not a frame-level proxy.
    """
    from gravit.utils import protocol
    from gravit.utils.score_adapter import (
        change_points_valid,
        synthetic_shot_segment_count,
        uniform_change_points,
    )

    if not change_points_valid(change_points):
        n_seg = synthetic_shot_segment_count(n_frames, len(pred_scores))
        change_points = uniform_change_points(n_frames, n_seg)

    frame_scores = protocol.frame_scores_from_sampled_scores(
        np.asarray(pred_scores, dtype=np.float32), picks, n_frames,
    )
    s_scores, s_lengths = protocol.segment_scores_and_lengths(frame_scores, change_points)
    final_len = int(n_frames * 0.15)
    segments = protocol.select_segments("strict_main", final_len, s_scores, s_lengths)

    pred_summary = np.zeros(n_frames, dtype=np.int8)
    for seg in segments:
        pred_summary[int(change_points[seg][0]) : int(change_points[seg][1])] = 1

    n_user = user_summary.shape[0]
    f1_scores = np.empty(n_user, dtype=np.float64)
    for u in range(n_user):
        us = np.zeros(n_frames, dtype=np.int8)
        us_len = min(n_frames, user_summary.shape[1])
        us[:us_len] = user_summary[u][:us_len]
        tp = int((pred_summary & us).sum())
        p_den = max(int(pred_summary.sum()), 1)
        r_den = max(int(us.sum()), 1)
        precision = float(tp) / p_den
        recall = float(tp) / r_den
        if (precision + recall) == 0:
            f1_scores[u] = 0.0
        else:
            f1_scores[u] = 2 * precision * recall * 100 / (precision + recall)

    return float(max(f1_scores)) if eval_type == "VS_max" else float(np.mean(f1_scores))


def spearman_kendall(pred: np.ndarray, gt: np.ndarray) -> tuple[float | None, float | None]:
    """Spearman rho and Kendall tau (rank by descending importance, same as eval.py)."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    if pred.size < 2 or np.nanstd(pred) < 1e-8 or np.nanstd(gt) < 1e-8:
        return None, None
    rho, _ = spearmanr(pred, gt)
    tau, _ = kendalltau(rankdata(-pred), rankdata(-gt))
    rho_f = float(rho) if rho == rho and not np.isinf(rho) else None
    tau_f = float(tau) if tau == tau and not np.isinf(tau) else None
    return rho_f, tau_f
