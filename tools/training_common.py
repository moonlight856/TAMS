"""Shared training helpers: rank target computation."""

from __future__ import annotations

import torch


def vs_rank_target(cfg: dict, data, device: torch.device, y_t: torch.Tensor) -> torch.Tensor:
    """
    Pairwise rank loss target for SumMe/TVSum/QFVS-style PyG batches.
    ``gtscore``: ``data.y`` first column; else pseudo target from ``get_label`` (``y_t[:,0]``).
    """
    if cfg.get("rank_supervision", "pseudo") != "gtscore":
        return y_t[:, 0]
    g = data.y.to(device)
    if g.dim() == 1:
        g = g.unsqueeze(1)
    return g[:, 0]


def videoxum_rank_target(cfg: dict, y: torch.Tensor, y_t: torch.Tensor, logits_device: torch.device) -> torch.Tensor:
    """VideoXum: ``gtscore`` = mean over annotator dim of one-hot / multi-label ``y``."""
    if cfg.get("rank_supervision", "pseudo") != "gtscore":
        return y_t[:, 0]
    return torch.mean(y.float(), dim=0).to(logits_device)
