"""
MATL: Modality-Adaptive Topology Learner (TAMS Innovation 1).

Replaces fixed τ-neighbor shared topology with per-modality learned, differentiable
graph topologies via Gumbel-Sigmoid edge predictors.

Each modality gets its own EdgePredictor MLP that outputs edge logits within a
candidate radius R_l. Gumbel-Sigmoid provides differentiable discrete edges during
training and hard edges at inference. Final edge weights combine three factors:
  1. z_ij: Gumbel-Sigmoid edge existence (differentiable discrete)
  2. Content similarity: learned bilinear attention between node features
  3. Confidence: c_i * c_j from Step0 / TGMI
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityAdaptiveTopologyLearner(nn.Module):
    """
    Per-modality learnable graph topology with Gumbel-Sigmoid edge selection.

    For each scale layer l, candidate edges are |i-j| <= R_l. An EdgePredictor
    per modality decides which edges exist via differentiable sampling.

    Args:
        dim: hidden feature dimension (channel1).
        n_modalities: max number of modalities (v/t/a = 3).
        scales: list of candidate radii [R_1, R_2, ...] for multi-scale layers.
        max_rel_pos: maximum relative position for position embedding.
        temperature_init: initial Gumbel temperature (annealed during training).
    """

    def __init__(
        self,
        dim: int,
        n_modalities: int,
        scales: list[int],
        max_rel_pos: int | None = None,
        temperature_init: float = 1.0,
        shared_topology: bool = False,
    ):
        super().__init__()
        self.scales = scales
        self.n_mod = n_modalities
        self.shared_topology = shared_topology
        # Auto-size relative-position embedding to the largest scale so that
        # increasing multi_scale_taus actually takes effect (otherwise
        # |i-j| > max_rel_pos collapses into a single shared embedding).
        if max_rel_pos is None:
            max_rel_pos = max(int(max(scales)) if scales else 50, 50)
        self.max_rel_pos = int(max_rel_pos)
        pe_dim = dim // 4

        self.rel_pos_emb = nn.Embedding(2 * self.max_rel_pos + 1, pe_dim)

        n_predictors = 1 if shared_topology else n_modalities
        self.edge_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2 + pe_dim, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )
            for _ in range(n_predictors)
        ])
        for ep in self.edge_predictors:
            nn.init.zeros_(ep[-1].weight)
            nn.init.zeros_(ep[-1].bias)

        self.content_proj = nn.ModuleList([
            nn.Linear(dim, dim // 2) for _ in range(n_predictors)
        ])
        self.content_attn = nn.ParameterList([
            nn.Parameter(torch.randn(dim)) for _ in range(n_predictors)
        ])

        self.register_buffer("temperature", torch.tensor(temperature_init))

    def set_temperature(self, tau: float) -> None:
        self.temperature.fill_(tau)

    def forward(
        self,
        h_parts: list[torch.Tensor],
        scale_idx: int,
        node_conf: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            h_parts: list of [T, D] per-modality features.
            scale_idx: 0-indexed layer index selecting candidate radius.
            node_conf: optional list of [T] confidence per modality.

        Returns:
            adj_list: list of [2, E_m] edge indices per modality.
            weight_list: list of [E_m] edge weights per modality.
        """
        T = h_parts[0].shape[0]
        device = h_parts[0].device
        R = self.scales[min(scale_idx, len(self.scales) - 1)]

        ii = torch.arange(T, device=device)
        offsets = torch.arange(-R, R + 1, device=device)
        src = ii.repeat_interleave(2 * R + 1)
        dst = (ii.unsqueeze(1) + offsets).clamp(0, T - 1).reshape(-1)

        # Deduplicate boundary edges (clamping maps multiple offsets to same dst).
        pair_key = torch.unique(src * T + dst, sorted=True)
        src = pair_key // T
        dst = pair_key % T

        rel = (src - dst).clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        pe = self.rel_pos_emb(rel)

        adj_list: list[torch.Tensor] = []
        weight_list: list[torch.Tensor] = []

        n_active = min(self.n_mod, len(h_parts))
        for m in range(n_active):
            h = h_parts[m]
            feat = torch.cat([h[src], h[dst], pe], dim=-1)
            pred_idx = 0 if self.shared_topology else m
            logit = self.edge_predictors[pred_idx](feat).squeeze(-1)

            with torch.amp.autocast("cuda", enabled=False):
                logit_f = logit.float()
                if self.training:
                    u1 = torch.rand_like(logit_f).clamp(1e-6, 1 - 1e-6)
                    u2 = torch.rand_like(logit_f).clamp(1e-6, 1 - 1e-6)
                    g = -torch.log(-torch.log(u1)) + torch.log(-torch.log(u2))
                    z = torch.sigmoid((logit_f + g) / self.temperature.float().clamp(min=0.01))
                else:
                    z = (logit_f > 0).float()

            p = self.content_proj[pred_idx]
            s = torch.sigmoid((p(h[src]) * p(h[dst])).sum(-1))

            if node_conf is not None and m < len(node_conf):
                c = node_conf[m].to(device=device, dtype=h.dtype)
                conf = c[src] * c[dst]
            else:
                conf = 1.0

            w = z * s * conf
            adj_list.append(torch.stack([src, dst]))
            weight_list.append(w)

        return adj_list, weight_list

    def get_mean_edge_density(
        self,
        weight_list: list[torch.Tensor],
    ) -> list[float]:
        """Return mean z (edge density) per modality for monitoring."""
        densities = []
        for w in weight_list:
            densities.append(float(w.mean().item()))
        return densities
