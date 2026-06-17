"""Temporal text shot topology: SAGE on text channel before fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class TemporalShotSAGE(nn.Module):
    """One-hop GraphSAGE over the same τ-neighbor graph as the visual stream."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.conv = SAGEConv(in_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        return self.norm(h + x)
