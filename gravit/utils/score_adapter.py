"""
Shot-level scores -> sampled-frame scores for VS evaluation (SumMe / TVSum / VideoXum).
"""

from __future__ import annotations

import numpy as np


def synthetic_shot_segment_count(n_frames: int, n_steps: int) -> int:
    """
    Same heuristic as data/build_vs_shot_graphs fallback when H5 KTS is missing.
    Keeps eval knapsack segmentation closer to training shot graphs.
    """
    nf = max(1, int(n_frames))
    ns = max(1, int(n_steps))
    return int(min(64, max(4, min(ns, max(1, nf // 25)))))


def uniform_change_points(n_frames: int, n_segments: int) -> np.ndarray:
    """
    Non-overlapping segments covering [0, n_frames-1], count == n_segments.
    Used when H5 change_points is missing but shot-level eval must map to picks.
    """
    n_frames = max(1, int(n_frames))
    n_segments = max(1, int(n_segments))
    n_segments = min(n_segments, n_frames)
    edges = np.linspace(0, n_frames - 1, num=n_segments + 1, dtype=np.int64)
    return np.stack([edges[:-1], edges[1:]], axis=1)


def change_points_valid(cp) -> bool:
    a = np.asarray(cp) if cp is not None else None
    if a is None or a.size == 0:
        return False
    if a.dtype == object and a.ndim == 0 and (a.item() is None):
        return False
    try:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 2:
            return False
        return True
    except (TypeError, ValueError):
        return False


def frame_to_shot_index(frame_idx: int, change_points: np.ndarray) -> int:
    for s, seg in enumerate(change_points):
        a, b = int(seg[0]), int(seg[1])
        if a <= frame_idx <= b:
            return s
    return -1


def build_frame_to_shot_map(n_frames: int, change_points: np.ndarray) -> np.ndarray:
    m = np.full(n_frames, -1, dtype=np.int32)
    for s, seg in enumerate(change_points):
        a, b = int(seg[0]), int(seg[1])
        m[a : b + 1] = s
    return m


def shot_scores_to_sampled_frame_scores(
    shot_scores: np.ndarray,
    picks: np.ndarray,
    change_points: np.ndarray,
    n_frames: int,
    n_samples: int,
) -> np.ndarray:
    shot_scores = np.asarray(shot_scores, dtype=np.float32).reshape(-1)
    picks = np.asarray(picks, dtype=np.int64).reshape(-1)
    fmap = build_frame_to_shot_map(n_frames, change_points)
    out = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        fr = int(min(picks[i], n_frames - 1))
        sid = int(fmap[fr])
        if 0 <= sid < len(shot_scores):
            out[i] = shot_scores[sid]
    return out


def identity_sample_scores_if_one_shot_per_sample(shot_scores: np.ndarray, n_samples: int) -> np.ndarray:
    s = np.asarray(shot_scores, dtype=np.float32).reshape(-1)
    if s.shape[0] == n_samples:
        return s.copy()
    raise ValueError(f"Expected shot_scores len {n_samples}, got {s.shape[0]}")
