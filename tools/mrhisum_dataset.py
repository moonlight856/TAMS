"""
MrHiSum multimodal HDF5 loader (TripleSumm layout under ``data/annotations/MrHiSum``).

Per-video: visual [T,1024], text [T,768], audio [T,768], T ≈ duration in seconds.

Two modes controlled by ``mrhisum_no_pad`` config:
  - False (default, legacy): zero-pad text/audio to D_vis, concat → [T, 3*D_vis].
  - True (recommended): concat raw dims → [T, D_vis+D_txt+D_aud] = [T, 2560].
    Sets ``modality_dims`` in config so the model can split correctly.

Edges are built with ``tau`` (or ``max(multi_scale_taus)`` if configured).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _pad_feat(feat: np.ndarray, d_target: int) -> np.ndarray:
    t, d = feat.shape[0], feat.shape[1]
    if d == d_target:
        return feat.astype(np.float32)
    if d > d_target:
        return feat[:, :d_target].astype(np.float32)
    pad = np.zeros((t, d_target - d), dtype=np.float32)
    return np.concatenate([feat.astype(np.float32), pad], axis=1)


class MrHiSumDataset(Dataset):
    def __init__(
        self,
        split: str,
        root: str | Path,
        tau: int = 2,
        device: torch.device | None = None,
        *,
        cfg: dict | None = None,
    ):
        self._no_pad = False
        if cfg is not None:
            from gravit.utils.cfg_defaults import merge_defaults

            merge_defaults(cfg)
            root = cfg.get("mrhisum_root", root)
            tau = int(cfg.get("tau", tau))
            self._no_pad = bool(cfg.get("mrhisum_no_pad", False))
            sage_tau = cfg.get("sage_tau")
            if sage_tau is not None:
                tau = int(sage_tau)
            else:
                ms_taus = cfg.get("multi_scale_taus")
                if ms_taus and isinstance(ms_taus, list):
                    tau = max(max(t for t in ms_taus if t > 0), tau)
        self.root = Path(root)
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.tau = max(tau, 1)

        split_path = self.root / "mrhisum_split.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing {split_path}")
        with open(split_path, encoding="utf-8") as f:
            sp = json.load(f)
        key = f"{split}_keys"
        if key not in sp:
            raise KeyError(f"{split_path} missing '{key}'")
        self.video_ids: list[str] = list(sp[key])

        self._gt: h5py.File | None = None
        self._vis: h5py.File | None = None
        self._txt: h5py.File | None = None
        self._aud: h5py.File | None = None

        with h5py.File(self.root / "mrhisum_feat_visual_inceptionv3.h5", "r") as vf:
            self._d_vis = int(vf[self.video_ids[0]].shape[1])
        with h5py.File(self.root / "mrhisum_feat_text_roberta.h5", "r") as tf:
            self._d_txt = int(tf[self.video_ids[0]].shape[1])
        with h5py.File(self.root / "mrhisum_feat_audio_ast.h5", "r") as af:
            self._d_aud = int(af[self.video_ids[0]].shape[1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_gt"] = None
        state["_vis"] = None
        state["_txt"] = None
        state["_aud"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_h5(self) -> None:
        if self._vis is not None:
            return
        self._gt = h5py.File(self.root / "mrhisum_gt.h5", "r")
        self._vis = h5py.File(self.root / "mrhisum_feat_visual_inceptionv3.h5", "r")
        self._txt = h5py.File(self.root / "mrhisum_feat_text_roberta.h5", "r")
        self._aud = h5py.File(self.root / "mrhisum_feat_audio_ast.h5", "r")

    @property
    def num_modalities(self) -> int:
        return 3

    @property
    def modality_dim(self) -> int:
        return self._d_vis

    @property
    def modality_dims_list(self) -> list[int]:
        """Per-modality feature dimensions [D_vis, D_txt, D_aud]."""
        if self._no_pad:
            return [self._d_vis, self._d_txt, self._d_aud]
        return [self._d_vis, self._d_vis, self._d_vis]

    @classmethod
    def from_cfg(cls, split: str, cfg: dict, device: torch.device | None = None):
        from gravit.utils.cfg_defaults import merge_defaults

        merge_defaults(cfg)
        if device is None:
            dev = cfg.get("device", "cpu")
            device = torch.device(dev) if isinstance(dev, str) else torch.device("cpu")
            if device.type == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")
        root = cfg.get("mrhisum_root", "./data/annotations/MrHiSum")
        return cls(split, root, device=device, cfg=cfg)

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int):
        self._ensure_h5()
        vid = self.video_ids[idx]
        vis = np.asarray(self._vis[vid][...], dtype=np.float32)
        txt = np.asarray(self._txt[vid][...], dtype=np.float32)
        aud = np.asarray(self._aud[vid][...], dtype=np.float32)
        t = vis.shape[0]
        if txt.shape[0] != t or aud.shape[0] != t:
            raise ValueError(f"{vid}: length mismatch vis={vis.shape[0]} txt={txt.shape[0]} aud={aud.shape[0]}")

        if self._no_pad:
            x = np.concatenate([vis, txt, aud], axis=1)
        else:
            d = self._d_vis
            x = np.concatenate([_pad_feat(vis, d), _pad_feat(txt, d), _pad_feat(aud, d)], axis=1)

        gt_score = np.asarray(self._gt[vid]["gt_score"][...], dtype=np.float32).reshape(-1)
        if gt_score.shape[0] != t:
            raise ValueError(f"{vid}: gt_score len {gt_score.shape[0]} != T {t}")

        n = x.shape[0]
        ii = np.arange(n)
        offsets = np.arange(-self.tau, self.tau + 1)
        src = np.repeat(ii, len(offsets))
        dst = np.clip((ii[:, None] + offsets).ravel(), 0, n - 1)
        keep = src != dst  # remove self-loops duplicates from boundary clamping
        pair_ids = src[keep] * n + dst[keep]
        _, unique_idx = np.unique(pair_ids, return_index=True)
        src_u = src[keep][unique_idx]
        dst_u = dst[keep][unique_idx]
        self_src = ii
        self_dst = ii
        src_all = np.concatenate([src_u, self_src])
        dst_all = np.concatenate([dst_u, self_dst])
        attr = np.sign(src_all.astype(np.int64) - dst_all.astype(np.int64))

        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(gt_score, dtype=torch.float32)
        e_t = torch.from_numpy(np.stack([src_all, dst_all]).astype(np.int64))
        ea_t = torch.from_numpy(attr.astype(np.int64))
        return x_t, y_t, e_t, ea_t, vid

    def close(self) -> None:
        for f in (self._gt, self._vis, self._txt, self._aud):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self._gt = self._vis = self._txt = self._aud = None
