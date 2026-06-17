"""Modality fusion modules: gated, concat-proj, cross-modal attention."""

import torch
import torch.nn as nn


class GatedModalityFusion(nn.Module):
    def __init__(self, dim: int, n_modalities: int = 2):
        super().__init__()
        self.gate = nn.Linear(dim * n_modalities, n_modalities)

    def forward(self, *hs: torch.Tensor) -> torch.Tensor:
        if len(hs) == 1:
            return hs[0]
        x = torch.cat(hs, dim=-1)
        w = torch.softmax(self.gate(x), dim=-1)
        out = sum(w[:, i : i + 1] * hs[i] for i in range(len(hs)))
        return out


class ConcatLinearFusion(nn.Module):
    """Concat modality projections then linear fuse to shared dim (alternative to softmax gating)."""

    def __init__(self, dim: int, n_modalities: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(dim * n_modalities, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, *hs: torch.Tensor) -> torch.Tensor:
        if len(hs) == 1:
            return hs[0]
        x = torch.cat(hs, dim=-1)
        h = self.drop(self.norm(self.proj(x)))
        return h


class CrossModalFusionBlock(nn.Module):
    """
    Fusion-token cross-attention: a neutral learnable token queries per-modality
    features, producing an unbiased fused representation at each time step.

    Fusion-token cross-attention designed for graph pipelines — operates
    independently at each node (time step), no sequence-level attention.
    """

    def __init__(self, dim: int, n_heads: int = 4, n_modalities: int = 3, dropout: float = 0.1):
        super().__init__()
        self.fusion_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, *hs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hs: M tensors of shape [T, D] (per-modality features).
        Returns:
            Fused features [T, D].
        """
        if len(hs) == 1:
            return hs[0]
        T = hs[0].shape[0]
        f = self.fusion_token.expand(T, -1)
        kv = torch.stack(hs, dim=1)                        # [T, M, D]
        q = self.norm1(f).unsqueeze(1)                      # [T, 1, D]
        h, _ = self.cross_attn(q, kv, kv)                  # [T, 1, D]
        h = self.drop(h.squeeze(1)) + f                    # [T, D]
        h = self.drop(self.ffn(self.norm2(h))) + h          # [T, D]
        return h

    def forward_with_weights(self, *hs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Like forward but also returns [T, M] attention weights for visualization."""
        if len(hs) == 1:
            return hs[0], torch.ones(hs[0].shape[0], 1, device=hs[0].device)
        T = hs[0].shape[0]
        f = self.fusion_token.expand(T, -1)
        kv = torch.stack(hs, dim=1)
        q = self.norm1(f).unsqueeze(1)
        h, attn_w = self.cross_attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        h = self.drop(h.squeeze(1)) + f
        h = self.drop(self.ffn(self.norm2(h))) + h
        return h, attn_w.squeeze(1)
