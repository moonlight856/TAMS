"""Query-video cross-attention fusion."""

from __future__ import annotations

import torch
from torch import nn


class QueryCrossAttnFusion(nn.Module):
    """
    Each frame node attends to a single global query token (concept embedding).
    Pre-norm + residual + dropout for stability.
    """

    def __init__(self, dim: int, query_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.ln_h = nn.LayerNorm(dim)
        self.ln_q = nn.LayerNorm(query_dim)
        self.q_in = nn.Linear(query_dim, dim)
        self.mha = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if query.dim() > 1:
            query = query.view(-1)
        qh = self.ln_h(h).unsqueeze(0)
        qq = self.ln_q(query).view(1, 1, -1)
        kv = self.q_in(qq)
        out, _ = self.mha(qh, kv, kv)
        out = self.dropout(out.squeeze(0))
        return h + out
