"""Modality alignment and missing-token fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class VisualOnlyInput(nn.Module):
    """Identity on visual stream when only V is used."""

    def forward(self, x_visual: torch.Tensor, *args, **kwargs):
        return x_visual


class ModalityInputFusion(nn.Module):
    """
    Per-modality missing-token fusion.

    Each modality gets its own LayerNorm + learnable missing token.
    Output = c · LN(h) + (1 - c) · e_miss.
    """

    def __init__(self, dim: int, n_modalities: int = 1):
        super().__init__()
        self.n_modalities = n_modalities
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_modalities)])
        self.missing_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(n_modalities)]
        )
        self._init_missing()

    def _init_missing(self):
        for p in self.missing_tokens:
            nn.init.normal_(p, std=0.02)

    def forward(
        self,
        h_parts: list[torch.Tensor],
        modality_conf: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            h_parts: list of [N, D] projected modality features (after layer011/012/linear_mod3).
            modality_conf: list of [N] confidence in [0, 1] per modality (from Step0).
                           If None, all confidences are 1 (identity pass-through).

        Returns:
            list of [N, D] fused features (same length as h_parts).
        """
        out: list[torch.Tensor] = []
        for i, h in enumerate(h_parts):
            if i >= self.n_modalities:
                out.append(h)
                continue
            h_norm = self.norms[i](h)
            if modality_conf is not None and i < len(modality_conf):
                c = modality_conf[i].to(device=h.device, dtype=h.dtype).clamp(0.0, 1.0)
                c = c.unsqueeze(-1)  # [N, 1]
                e_miss = self.missing_tokens[i].unsqueeze(0).expand_as(h)  # [N, D]
                h_fused = c * h_norm + (1.0 - c) * e_miss
            else:
                h_fused = h_norm
            out.append(h_fused)
        return out
