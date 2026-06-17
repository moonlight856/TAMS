"""Load per-participant QFVS frame features from ``V{n}_*.h5`` (community release)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PARTICIPANT_TO_VI = {"P01": 1, "P02": 2, "P03": 3, "P04": 4}


def participant_h5_path(h5_dir: Path, participant: str, backbone: str) -> Path:
    vi = PARTICIPANT_TO_VI[participant]
    name = f"V{vi}_{backbone}.h5"
    p = h5_dir / name
    if not p.is_file():
        raise FileNotFoundError(f"Missing H5: {p} (expected V1..V4_{backbone}.h5)")
    return p


def load_feature_matrix(h5_dir: Path, participant: str, backbone: str) -> tuple[np.ndarray, int]:
    """
    Returns (features float32 [T, D], feat_dim D).
    """
    import h5py

    path = participant_h5_path(h5_dir, participant, backbone)
    with h5py.File(path, "r") as f:
        if "feature" not in f:
            raise KeyError(f"{path}: no 'feature' dataset")
        x = np.asarray(f["feature"][:], dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{path}: feature must be 2D, got {x.shape}")
    return x, int(x.shape[1])


def aligned_features(full: np.ndarray, n_frames_ann: int) -> tuple[np.ndarray, int]:
    """
    Clip or pad annotation length to H5 length.
    Returns (feats_first_n, n_use) where n_use = min(T, n_frames_ann).
    """
    t = int(full.shape[0])
    n_use = min(t, int(n_frames_ann))
    if n_use < 1:
        n_use = 1
    return full[:n_use], n_use
