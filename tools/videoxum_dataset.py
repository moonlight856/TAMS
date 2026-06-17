import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _pad_to_dim(feat: np.ndarray, d_target: int) -> np.ndarray:
    d = feat.shape[1]
    if d == d_target:
        return feat
    if d > d_target:
        return feat[:, :d_target].copy()
    pad = np.zeros((feat.shape[0], d_target - d), dtype=np.float32)
    return np.concatenate([feat.astype(np.float32), pad], axis=1)


def _to_2d_float(b: np.ndarray) -> np.ndarray:
    """Vision/text/audio arrays: ensure (n_shots, dim). 1D (D,) → (1, D) for video-level embeddings."""
    b = np.asarray(b, dtype=np.float32)
    if b.ndim == 0:
        raise ValueError("modality array is scalar")
    if b.ndim == 1:
        return b.reshape(1, -1)
    if b.ndim > 2:
        b = b.reshape(b.shape[0], -1)
    return b


def _align_rows_to_n(b: np.ndarray, ref_n: int) -> np.ndarray:
    """
    Match shot count to vision (ref_n). vt_clipscore often stores text as (1, D) (one caption vector).
    """
    n = b.shape[0]
    if n == ref_n:
        return b
    if ref_n <= 0:
        raise ValueError("ref_n must be positive")
    if n == 1:
        return np.repeat(b, ref_n, axis=0)
    idx = np.linspace(0, n - 1, ref_n, dtype=np.float64)
    idx = np.clip(np.round(idx).astype(np.int64), 0, n - 1)
    return b[idx]


def _stack_modalities(blocks: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """Equalize shot lengths (to first block), feature dims (pad), concat along channel."""
    if not blocks:
        raise ValueError("no modality blocks")
    blocks = [_to_2d_float(b) for b in blocks]
    ref_n = int(blocks[0].shape[0])
    aligned = [blocks[0]] + [_align_rows_to_n(b, ref_n) for b in blocks[1:]]
    dmax = max(b.shape[1] for b in aligned)
    parts = [_pad_to_dim(b, dmax) for b in aligned]
    return np.concatenate(parts, axis=1), len(parts), dmax


class VideoXumDataset(Dataset):
    """
    Online τ-neighbor graphs for VideoXum.

    * ``fea='blip'``: npz key ``features`` (vision only).
    * ``fea='vt_clipscore'``: keys ``vision``, ``text``; audio from ``.../audio/{vid}.npz`` (keys ``audio`` / ``features`` / ``rms``)
      if present, otherwise key ``audio`` in the **same** vt npz.
      Rows are aligned to ``vision`` shot count: ``text`` / ``audio`` may be broadcast from length-1 (video-level) rows.
      If audio files are missing, set ``videoxum_audio_fallback: zeros`` in cfg (placeholder); default ``error`` requires real audio.
    """

    def __init__(
        self,
        mode="train",
        fea="blip",
        tau=2,
        device=None,
        *,
        use_text=False,
        use_audio=False,
        audio_feature_dir=None,
        cfg=None,
    ):
        if cfg is not None:
            from gravit.utils.cfg_defaults import merge_defaults

            merge_defaults(cfg)
            fea = cfg.get("videoxum_fea", fea)
            tau = int(cfg.get("tau", tau))
            use_text = bool(cfg.get("use_text", use_text))
            use_audio = bool(cfg.get("use_audio", use_audio))
            ar = cfg.get("videoxum_audio_feature_dir")
            if ar:
                audio_feature_dir = ar
        if device is None:
            device = torch.device("cpu")
        self.audio_fallback = "error"
        if cfg is not None:
            self.audio_fallback = str(cfg.get("videoxum_audio_fallback", "error")).lower()
        self.menu_pth = r"./data/annotations/videoxum/{}_videoxum.json".format(mode)
        self.feature_pth = r"./data/annotations/videoxum/{}".format(fea)
        self.fea = fea
        self.tau = tau
        self.device = device
        self.use_text = use_text
        self.use_audio = use_audio
        self.audio_feature_dir = audio_feature_dir
        with open(self.menu_pth, "r", encoding="utf-8") as f:
            self.menu = json.load(f)
        self._num_modalities = self._infer_num_modalities()

    def _infer_num_modalities(self) -> int:
        if self.fea == "blip":
            return 1
        n = 1
        if self.use_text:
            n += 1
        if self.use_audio:
            n += 1
        return max(n, 1)

    @property
    def num_modalities(self) -> int:
        return self._num_modalities

    @classmethod
    def from_cfg(cls, mode: str, cfg: dict, device: torch.device | None = None):
        from gravit.utils.cfg_defaults import merge_defaults

        merge_defaults(cfg)
        if device is None:
            dev = cfg.get("device", "cpu")
            device = torch.device(dev) if isinstance(dev, str) else torch.device("cpu")
            if device.type == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")
        return cls(mode, cfg=cfg, device=device)

    def _load_feature_matrix(self, vid: str) -> np.ndarray:
        base = Path(self.feature_pth) / f"{vid}.npz"
        if not base.is_file():
            raise FileNotFoundError(f"Missing features: {base}")
        raw = np.load(base, allow_pickle=True)
        if self.fea == "blip":
            return raw["features"].astype(np.float32)

        blocks: list[np.ndarray] = []
        if "vision" not in raw.files:
            raise KeyError(f"{base} missing 'vision' (vt_clipscore layout)")
        blocks.append(raw["vision"].astype(np.float32))
        if self.use_text:
            if "text" not in raw.files:
                raise KeyError(f"{base} missing 'text' while use_text=True")
            blocks.append(raw["text"].astype(np.float32))
        if self.use_audio:
            audio_arr = None
            adir = self.audio_feature_dir or str(Path(self.feature_pth).parent / "audio")
            ap = Path(adir) / f"{vid}.npz"
            if ap.is_file():
                ar = np.load(ap, allow_pickle=True)
                key = None
                for cand in ("audio", "features", "rms"):
                    if cand in ar.files:
                        key = cand
                        break
                if key is None:
                    raise KeyError(f"{ap} needs one of keys: audio, features, rms")
                audio_arr = ar[key]
            elif "audio" in raw.files:
                audio_arr = raw["audio"]
            elif self.audio_fallback == "zeros":
                v0 = np.asarray(blocks[0], dtype=np.float32)
                n = int(v0.shape[0])
                d = int(v0.shape[1]) if v0.ndim >= 2 else 1
                if len(blocks) > 1:
                    t0 = np.asarray(blocks[1], dtype=np.float32)
                    if t0.ndim >= 2:
                        d = max(d, int(t0.shape[1]))
                audio_arr = np.zeros((n, d), dtype=np.float32)
            else:
                raise FileNotFoundError(
                    f"use_audio=True but missing sidecar {ap} and no 'audio' in {base}"
                )
            blocks.append(np.asarray(audio_arr, dtype=np.float32))
        x, nmod, _ = _stack_modalities(blocks)
        if nmod != self._num_modalities:
            self._num_modalities = nmod
        return x

    def __getitem__(self, item):
        line = self.menu[item]
        vid = line["video_id"]
        vsum_onehot = line["vsum_onehot"]

        feature = self._load_feature_matrix(vid)

        edges = [[], []]
        edge_attr = []

        for i in range(feature.shape[0]):
            for j in range(max([0, i - self.tau]), min([feature.shape[0] - 1, i + self.tau])):
                edges[0].append(i)
                edges[1].append(j)
                if i > j:
                    edge_attr.append(1)
                elif i < j:
                    edge_attr.append(-1)
                else:
                    edge_attr.append(0)

        return (
            torch.tensor(feature, dtype=torch.float, device=self.device),
            torch.tensor(vsum_onehot, dtype=torch.float, device=self.device),
            torch.tensor(edges, dtype=torch.long, device=self.device),
            torch.tensor(edge_attr, device=self.device),
            vid,
        )

    def __len__(self):
        return len(self.menu)


def videoxum_collate_fn(batch):
    """
    VideoXum graphs are variable size; training/eval use ``batch_size: 1``.
    Avoids ``default_collate`` failing on the string ``video_id`` (Step0 multimodal JSON key).
    """
    if len(batch) != 1:
        raise ValueError(
            "VideoXum expects batch_size=1 (variable shot counts). Got batch of size "
            f"{len(batch)}. Set batch_size: 1 in cfg."
        )
    return batch[0]


if __name__ == "__main__":
    for data in VideoXumDataset():
        x, y, e, e_attr, vid = data
        print(x.shape, y.shape, e.shape, e_attr.shape, vid)
