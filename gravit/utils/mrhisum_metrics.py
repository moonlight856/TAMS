"""
MrHiSum / TripleSumm-style metrics: Kendall τ, Spearman ρ, mAP@50, mAP@15.

Logic adapted from TripleSumm (MIT): https://github.com/smkim37/TripleSumm
``utils/compute_metrics.py`` — shot pooling for highlight AP uses 5s windows at fps=1
(aligned with per-second feature layout in the released MrHiSum HDF5).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

try:
    from sklearn.metrics import average_precision_score
except ImportError:  # pragma: no cover
    average_precision_score = None


def pack_variable_length_scores(
    preds_list: list[list[float]],
    gts_list: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], np.ndarray]:
    """Pad batch of variable-length score sequences; mask True on valid positions."""
    if not preds_list:
        return [], [], np.zeros((0, 0), dtype=bool)
    max_t = max(len(p) for p in preds_list)
    pp, gg = [], []
    mask = np.zeros((len(preds_list), max_t), dtype=bool)
    for i, (p, g) in enumerate(zip(preds_list, gts_list)):
        L = len(p)
        pp.append(list(p) + [0.0] * (max_t - L))
        gg.append(list(g) + [0.0] * (max_t - L))
        mask[i, :L] = True
    return pp, gg, mask


def _trim_to_mask(pred_list: list, gt_list: list, mask: np.ndarray) -> tuple[list, list]:
    out_p, out_g = [], []
    for i in range(len(pred_list)):
        valid = np.where(mask[i])[0]
        end = int(valid[-1]) + 1 if len(valid) else 0
        out_p.append(np.asarray(pred_list[i][:end], dtype=np.float64))
        out_g.append(np.asarray(gt_list[i][:end], dtype=np.float64))
    return out_p, out_g


def evaluate_summary(
    pred_score: Sequence[Sequence[float]],
    gt_score: Sequence[Sequence[float]],
    mask: np.ndarray,
) -> tuple[float, float]:
    """Mean Kendall τ and Spearman ρ over videos (TripleSumm ``evaluate_summary``)."""
    pred_list = [list(map(float, p)) for p in pred_score]
    gt_list = [list(map(float, g)) for g in gt_score]
    pred_list, gt_list = _trim_to_mask(pred_list, gt_list, mask)

    ktau_list, srho_list = [], []
    for p, g in zip(pred_list, gt_list):
        if len(p) < 2 or len(g) < 2:
            continue
        kt = kendalltau(p, g)[0]
        sr = spearmanr(p, g)[0]
        ktau_list.append(0.0 if kt is None or np.isnan(kt) else float(kt))
        srho_list.append(0.0 if sr is None or np.isnan(sr) else float(sr))
    if not ktau_list:
        return 0.0, 0.0
    return float(np.mean(ktau_list)), float(np.mean(srho_list))


def _calculate_ap_for_video(
    pred_score: np.ndarray,
    gt_score: np.ndarray,
    rho: float,
    shot_duration_seconds: float = 5.0,
    fps: float = 1.0,
) -> float:
    if average_precision_score is None:
        raise ImportError("mAP metrics require scikit-learn: pip install scikit-learn")

    shot_length_frames = int(shot_duration_seconds * fps)
    num_frames = len(pred_score)
    if shot_length_frames <= 0:
        return 0.0
    num_shots = math.ceil(num_frames / shot_length_frames)
    padding_size = num_shots * shot_length_frames - num_frames

    pred_padded = np.pad(pred_score, (0, padding_size), mode="constant")
    gt_padded = np.pad(gt_score, (0, padding_size), mode="constant")
    pred_shot_scores = np.mean(pred_padded.reshape(-1, shot_length_frames), axis=1)
    gt_shot_scores = np.mean(gt_padded.reshape(-1, shot_length_frames), axis=1)

    # sklearn requires finite y_score; NaN/Inf can appear from unstable AMP or diverged weights.
    if not np.all(np.isfinite(pred_shot_scores)):
        pred_shot_scores = np.nan_to_num(pred_shot_scores, nan=0.0, posinf=0.0, neginf=0.0)

    num_shots = len(gt_shot_scores)
    top_k = int(math.ceil(num_shots * rho))
    top_k = max(0, min(top_k, num_shots))
    top_k_indices = np.argsort(gt_shot_scores)[-top_k:] if top_k > 0 else np.array([], dtype=int)
    gt_binary_labels = np.zeros(num_shots)
    gt_binary_labels[top_k_indices] = 1

    if np.sum(gt_binary_labels) == 0:
        return 0.0
    return float(average_precision_score(gt_binary_labels, pred_shot_scores))


def evaluate_highlight(
    pred_score_list: Sequence[Sequence[float]],
    gt_score_list: Sequence[Sequence[float]],
    mask: np.ndarray,
    fps: float = 1.0,
    shot_duration_seconds: float = 5.0,
) -> tuple[float, float]:
    """mAP@50 and mAP@15 in percent (TripleSumm ``evaluate_highlight``)."""
    pred_list = [list(map(float, p)) for p in pred_score_list]
    gt_list = [list(map(float, g)) for g in gt_score_list]
    pred_list, gt_list = _trim_to_mask(pred_list, gt_list, mask)

    ap50_list, ap15_list = [], []
    for pred_score, gt_score in zip(pred_list, gt_list):
        p = np.asarray(pred_score, dtype=np.float64)
        g = np.asarray(gt_score, dtype=np.float64)
        if len(p) == 0:
            continue
        ap50_list.append(_calculate_ap_for_video(p, g, rho=0.50, fps=fps, shot_duration_seconds=shot_duration_seconds))
        ap15_list.append(_calculate_ap_for_video(p, g, rho=0.15, fps=fps, shot_duration_seconds=shot_duration_seconds))

    if not ap50_list:
        return 0.0, 0.0
    return float(np.mean(ap50_list) * 100.0), float(np.mean(ap15_list) * 100.0)
