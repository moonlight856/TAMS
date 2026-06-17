"""
Track-A (strict_main): same knapsack + change_points + 15% budget as original eval.py.
Track-B (extended): optional greedy segment selection for ablation only.
"""

from __future__ import annotations

import numpy as np

from gravit.utils.vs import knapsack


def frame_scores_from_sampled_scores(scores: np.ndarray, picks: np.ndarray, n_frames: int) -> np.ndarray:
    """Map sampled-frame importance scores to full-frame vector (same as eval.py)."""
    picks = np.append(np.asarray(picks, dtype=np.int64), [n_frames - 1])
    n_samples = len(scores)
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for idx in range(n_samples):
        frame_scores[picks[idx] : picks[idx + 1]] = scores[idx]
    return frame_scores


def segment_scores_and_lengths(frame_scores: np.ndarray, gt_segments: np.ndarray):
    """
    Match legacy eval_tool / original eval.py exactly:
    s_lengths uses inclusive endpoints (b - a + 1) while per-segment mean
    uses Python slice [a:b] (b - a frames).  This intentional mismatch
    replicates the standard evaluation protocol shared by all published
    baselines (PGL-SUM, CA-SUM, etc.) and is necessary for
    fair comparison.
    """
    n_segments = len(gt_segments)
    s_scores = np.empty(n_segments, dtype=np.float32)
    s_lengths = np.empty(n_segments, dtype=np.int32)
    for idx in range(n_segments):
        a, b = int(gt_segments[idx][0]), int(gt_segments[idx][1])
        s_lengths[idx] = b - a + 1
        seg = frame_scores[a:b]
        s_scores[idx] = float(seg.mean()) if seg.size else 0.0
    return s_scores, s_lengths


def select_segments_strict_main(final_len: int, s_scores: np.ndarray, s_lengths: np.ndarray):
    """Main protocol: knapsack-based segment selection."""
    return knapsack.fill_knapsack(final_len, s_scores, s_lengths)


def select_segments_extended_greedy(final_len: int, s_scores: np.ndarray, s_lengths: np.ndarray):
    """
    Track-B: greedy by score density. Not comparable to main table.
    """
    n = len(s_scores)
    density = s_scores / np.maximum(s_lengths.astype(np.float32), 1.0)
    order = np.argsort(-density)
    picked = []
    used = 0
    for i in order:
        li = int(s_lengths[i])
        if used + li <= final_len:
            picked.append(int(i))
            used += li
    return sorted(picked)


def select_segments(
    protocol_mode: str,
    final_len: int,
    s_scores: np.ndarray,
    s_lengths: np.ndarray,
):
    if protocol_mode == "extended":
        return select_segments_extended_greedy(final_len, s_scores, s_lengths)
    return select_segments_strict_main(final_len, s_scores, s_lengths)
