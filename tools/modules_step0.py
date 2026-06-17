"""Step0: modality presence / shot-level confidence handling."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_MM: dict[str, dict[str, Any]] = {}


def clear_step0_cache() -> None:
    _CACHE.clear()
    _CACHE_MM.clear()


def default_visual_confidence(num_nodes: int, device, dtype) -> torch.Tensor:
    return torch.ones(num_nodes, device=device, dtype=dtype)


def _load_json(path: str) -> dict[str, Any]:
    ap = os.path.abspath(path)
    if ap not in _CACHE:
        with open(ap, encoding="utf-8") as f:
            _CACHE[ap] = json.load(f)
    return _CACHE[ap]


def _graph_id_str(data) -> str | None:
    if data is None:
        return None
    g = getattr(data, "g", None)
    if g is None:
        return None
    if isinstance(g, torch.Tensor):
        return str(int(g.view(-1)[0].item()))
    if isinstance(g, (list, tuple)) and len(g) > 0:
        return str(int(g[0]))
    return str(g)


def resolve_node_confidence(cfg: dict, data, num_nodes: int, device, dtype) -> torch.Tensor | None:
    """
    If ``use_step0`` is False, returns None (caller / model uses all-ones).
    If True and ``step0_cache_json`` is set, load per-video visual confidence vector.
    Else fall back to ones (canonical visual anchor).
    """
    if not cfg.get("use_step0"):
        return None
    path = cfg.get("step0_cache_json")
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        return default_visual_confidence(num_nodes, device, dtype)
    cache = _load_json(path)
    vid = _graph_id_str(data)
    raw = None
    if vid is not None and vid in cache:
        raw = cache[vid]
    elif "__default__" in cache:
        raw = cache["__default__"]
    if raw is None:
        return default_visual_confidence(num_nodes, device, dtype)
    if isinstance(raw, (int, float)):
        return torch.full((num_nodes,), float(raw), device=device, dtype=dtype)
    if isinstance(raw, list):
        vals = [float(x) for x in raw]
        if len(vals) < num_nodes:
            pad = float(vals[-1]) if vals else 1.0
            vals = vals + [pad] * (num_nodes - len(vals))
        elif len(vals) > num_nodes:
            vals = vals[:num_nodes]
        return torch.tensor(vals, device=device, dtype=dtype)
    return default_visual_confidence(num_nodes, device, dtype)


def _load_json_mm(path: str) -> dict[str, Any]:
    ap = os.path.abspath(path)
    if ap not in _CACHE_MM:
        with open(ap, encoding="utf-8") as f:
            _CACHE_MM[ap] = json.load(f)
    return _CACHE_MM[ap]


def _confidence_vec(raw: Any, num_nodes: int, device, dtype, default_fill: float) -> torch.Tensor:
    if raw is None:
        return torch.full((num_nodes,), float(default_fill), device=device, dtype=dtype)
    if isinstance(raw, (int, float)):
        v = max(0.0, min(1.0, float(raw)))
        return torch.full((num_nodes,), v, device=device, dtype=dtype)
    if isinstance(raw, list):
        vals = [max(0.0, min(1.0, float(x))) for x in raw]
        if len(vals) < num_nodes:
            pad = float(vals[-1]) if vals else default_fill
            vals = vals + [pad] * (num_nodes - len(vals))
        elif len(vals) > num_nodes:
            vals = vals[:num_nodes]
        return torch.tensor(vals, device=device, dtype=dtype)
    return torch.full((num_nodes,), float(default_fill), device=device, dtype=dtype)


def build_modality_scale_videoxum(
    cfg: dict,
    video_id: str,
    num_nodes: int,
    num_modalities: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    """
    Per-modality multipliers [N] for TAMS fusion (after per-modality linear/SAGE).
    Returns None if disabled or no JSON; else length ``num_modalities`` list.
    """
    if not cfg.get("use_step0"):
        return None
    path = cfg.get("step0_multimodal_json")
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        return None
    nm = int(num_modalities)
    if nm < 1:
        return None

    cache = _load_json_mm(path)
    keys_m = ("v", "t", "a")
    entry = None
    if video_id in cache:
        entry = cache[video_id]
    elif "__defaults__" in cache:
        entry = cache["__defaults__"]
    if entry is None or not isinstance(entry, dict):
        entry = {}

    def _get_raw(k: str) -> Any:
        if k in entry:
            return entry[k]
        return None

    defaults_top = cache.get("__defaults__") if isinstance(cache.get("__defaults__"), dict) else {}
    out: list[torch.Tensor] = []
    for i in range(nm):
        k = keys_m[i] if i < len(keys_m) else "v"
        raw = _get_raw(k)
        if raw is None and k in defaults_top:
            raw = defaults_top[k]
        d0 = float(cfg.get("step0_default_confidence", 1.0))
        d0 = max(0.0, min(1.0, d0))
        out.append(_confidence_vec(raw, num_nodes, device, dtype, d0))
    return out
