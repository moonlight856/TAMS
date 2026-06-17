"""
QFVS query strings -> GloVe mean vector.

Uses a one-pass scan of a GloVe ``.txt`` file and writes a small ``.npz`` cache
under ``processed/`` so 5GB files are not re-read every run.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_TOKEN_SPLIT = re.compile(r"[_\s]+")


def query_key_to_tokens(query_key: str) -> list[str]:
    parts = [p for p in _TOKEN_SPLIT.split(query_key.strip()) if p]
    return [p.lower() for p in parts]


def collect_vocab_from_groups(groups: dict[tuple[str, str], dict]) -> set[str]:
    words: set[str] = set()
    for (_p, qk) in groups:
        words.update(query_key_to_tokens(qk))
    return words


def _load_cache(cache_path: Path) -> dict[str, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    z = np.load(cache_path, allow_pickle=True)
    words = z["words"]
    vecs = z["vecs"]
    return {str(w): vecs[i].astype(np.float32) for i, w in enumerate(words)}


def _save_cache(cache_path: Path, emb: dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(emb.keys())
    np.savez_compressed(
        cache_path,
        words=np.array(keys, dtype=object),
        vecs=np.stack([emb[k] for k in keys], axis=0).astype(np.float32),
    )


def build_or_load_glove_subset(
    glove_txt: Path,
    needed_words: set[str],
    cache_path: Path,
    dim: int = 300,
) -> dict[str, np.ndarray]:
    cached = _load_cache(cache_path)
    if cached is not None and needed_words.issubset(set(cached.keys())):
        return cached

    need = {w.lower() for w in needed_words}
    found: dict[str, np.ndarray] = {}
    if not glove_txt.is_file():
        raise FileNotFoundError(glove_txt)

    pending = set(need)
    with open(glove_txt, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != dim + 1:
                continue
            w = parts[0].lower()
            if w in pending:
                found[w] = np.array(parts[1:], dtype=np.float32)
                pending.discard(w)
            if not pending:
                break

    for w in need:
        if w not in found:
            found[w] = np.zeros(dim, dtype=np.float32)

    _save_cache(cache_path, found)
    return found


def mean_query_embedding(query_key: str, table: dict[str, np.ndarray], dim: int = 300) -> np.ndarray:
    toks = query_key_to_tokens(query_key)
    if not toks:
        return np.zeros(dim, dtype=np.float32)
    vecs = [table.get(t, np.zeros(dim, dtype=np.float32)) for t in toks]
    v = np.mean(np.stack(vecs, axis=0), axis=0)
    n = float(np.linalg.norm(v) + 1e-8)
    return (v / n).astype(np.float32)


class QFVSQueryGlove:
    """Lazy table: build subset from glove file on first use."""

    def __init__(
        self,
        glove_txt: Path,
        groups: dict,
        cache_path: Path | None = None,
        dim: int = 300,
    ):
        self.glove_txt = Path(glove_txt)
        vocab = collect_vocab_from_groups(groups)
        self.dim = dim
        if cache_path is None:
            cache_path = self.glove_txt.parent.parent / "processed" / "glove_qfvs_subset.npz"
        self.cache_path = Path(cache_path)
        self._table = build_or_load_glove_subset(self.glove_txt, vocab, self.cache_path, dim=dim)

    def embed(self, query_key: str) -> np.ndarray:
        return mean_query_embedding(query_key, self._table, dim=self.dim)
