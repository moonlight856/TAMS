"""
OT-CMA: Optimal Transport Cross-Modal Alignment.

Replaces TBCMI with a principled optimal transport framework for cross-modal
feature fusion, combined with an Adaptive Modality Relevance Gate (AMRG).

Key innovations over TBCMI:
  1. AMRG: automatically suppresses modalities with low temporal
     informativeness (e.g., video-level constant text features).
  2. Entropic OT alignment via log-domain Sinkhorn computes differentiable
     soft alignment between modality feature distributions.
  3. Topology-modulated cost: the transport cost matrix incorporates graph
     structure descriptors from MATL, coupling alignment to learned topology.

Mathematical formulation:
  Given per-modality features {h^m}_{m=1}^M and MATL adjacency/weights:

  (AMRG)    σ²_m = Var_t(h^m).mean();   g = σ(MLP([σ²_1,...,σ²_M]))
  (Cost)    C^{ij} = ||φ_i(h^i) - φ_j(h^j)||_2 + α·||d^i - d^j||_2
  (Sinkhorn) π^{ij} = argmin_{π∈U(a,b)} <C,π> - εH(π)
  (Transport) t^{i←j}_n = T · Σ_k π_{nk} h^j_k
  (Fuse)    h_fused = FFN(mean(all transports)) + residual(desc_avg)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveModalityRelevanceGate(nn.Module):
    """
    Soft-gates each modality by its temporal feature variance.

    Modalities whose features are near-constant across time (variance → 0)
    receive a gate value near the sigmoid bias.  Initialized at identity
    (zero weight + zero bias → sigmoid(0)=0.5) so early training is stable.
    """

    def __init__(self, n_modalities: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_modalities, n_modalities * 4),
            nn.GELU(),
            nn.Linear(n_modalities * 4, n_modalities),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self, h_parts: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        variances = []
        for h in h_parts:
            variances.append(h.var(dim=0).mean())
        var_vec = torch.stack(variances)
        gates = torch.sigmoid(self.gate(var_vec))
        gated = [gates[m] * h_parts[m] for m in range(len(h_parts))]
        return gated, gates


class OTCrossModalAlignment(nn.Module):
    """
    Cross-modal fusion via entropic optimal transport with topology-aware cost.

    Args:
        dim: hidden dimension (channel1).
        n_modalities: maximum number of modalities.
        sinkhorn_reg: entropic regularisation ε (larger → smoother plans).
        sinkhorn_iter: Sinkhorn iterations K.
        topo_weight: weight α for topology descriptor distance in cost.
        dropout: FFN / residual dropout.
    """

    def __init__(
        self,
        dim: int,
        n_modalities: int = 3,
        sinkhorn_reg: float = 0.1,
        sinkhorn_iter: int = 8,
        topo_weight: float = 0.1,
        dropout: float = 0.1,
        topo_stop_grad: bool = False,
    ):
        super().__init__()
        self.n_mod = n_modalities
        self.reg = sinkhorn_reg
        self.n_iter = sinkhorn_iter
        self.topo_weight = topo_weight
        self.topo_stop_grad = topo_stop_grad

        cost_dim = max(dim // 4, 32)
        self.cost_proj = nn.ModuleList([
            nn.Linear(dim, cost_dim) for _ in range(n_modalities)
        ])

        self.topo_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
            for _ in range(n_modalities)
        ])

        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _log_sinkhorn(self, C: torch.Tensor) -> torch.Tensor:
        """Log-domain Sinkhorn for uniform marginals (numerically stable).

        Always computed in FP32 to avoid FP16 overflow in logsumexp/exp.
        """
        with torch.amp.autocast("cuda", enabled=False):
            C = C.float()
            T = C.shape[0]
            reg = max(self.reg, 1e-4)
            log_K = -C / reg
            log_mu = torch.full((T,), -math.log(max(T, 1)), device=C.device, dtype=C.dtype)

            log_u = torch.zeros(T, device=C.device, dtype=C.dtype)
            log_v = torch.zeros(T, device=C.device, dtype=C.dtype)

            for _ in range(self.n_iter):
                log_v = log_mu - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
                log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

            return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp()

    # ------------------------------------------------------------------
    def _topology_descriptors(
        self,
        h_parts: list[torch.Tensor],
        adj_list: list[torch.Tensor],
        weight_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        descs: list[torch.Tensor] = []
        n_active = min(self.n_mod, len(h_parts))
        for m in range(n_active):
            h = h_parts[m]
            ei = adj_list[m]
            ew = weight_list[m]
            T = h.shape[0]
            src, dst = ei[0], ei[1]

            msg = h[dst] * ew.unsqueeze(-1)
            agg = torch.zeros_like(h).scatter_add_(
                0, src.unsqueeze(-1).expand_as(msg), msg,
            )
            ws = torch.zeros(T, device=h.device, dtype=h.dtype).scatter_add_(0, src, ew)
            with torch.amp.autocast("cuda", enabled=False):
                desc = self.topo_encoders[m](agg.float() / (ws.unsqueeze(-1).float() + 1e-8))
            descs.append(desc)
        return descs

    # ------------------------------------------------------------------
    def forward(
        self,
        h_parts: list[torch.Tensor],
        adj_list: list[torch.Tensor],
        weight_list: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(h_parts) == 1:
            return h_parts[0]

        self._last_transport_summaries = []
        descs = self._topology_descriptors(h_parts, adj_list, weight_list)
        n_active = len(descs)
        T = h_parts[0].shape[0]

        fused = torch.zeros_like(h_parts[0])
        n_pairs = 0

        for i in range(n_active):
            for j in range(n_active):
                if i == j:
                    continue
                # FP32 for cdist to avoid FP16 overflow in squared-distance sums
                with torch.amp.autocast("cuda", enabled=False):
                    ci = self.cost_proj[i](h_parts[i].float())
                    cj = self.cost_proj[j](h_parts[j].float())
                    C_feat = torch.cdist(ci, cj, p=2)

                    di = descs[i].float().detach() if self.topo_stop_grad else descs[i].float()
                    dj = descs[j].float().detach() if self.topo_stop_grad else descs[j].float()
                    C_topo = torch.cdist(di, dj, p=2)

                    C = C_feat + self.topo_weight * C_topo
                    c_max = C.max().detach().clamp(min=1e-8)
                    C = C / c_max

                P = self._log_sinkhorn(C)
                with torch.no_grad():
                    p_float = P.detach().float()
                    diag_mass = torch.diagonal(p_float).sum() if p_float.ndim == 2 else torch.tensor(0.0)
                    entropy = -(p_float * (p_float + 1e-12).log()).sum()
                    entry = {
                        "src_mod": int(i),
                        "dst_mod": int(j),
                        "T": int(T),
                        "diag_mass": float(diag_mass.cpu()),
                        "entropy": float(entropy.cpu()),
                        "row_peak_mean": float(p_float.max(dim=1).values.mean().cpu()),
                    }
                    if bool(getattr(self, "trace_transport_plan", False)):
                        max_size = int(getattr(self, "trace_transport_size", 64))
                        plan = p_float.cpu()
                        if plan.shape[0] > max_size or plan.shape[1] > max_size:
                            plan = F.interpolate(
                                plan.unsqueeze(0).unsqueeze(0),
                                size=(max_size, max_size),
                                mode="area",
                            ).squeeze(0).squeeze(0)
                        entry["plan"] = plan
                    self._last_transport_summaries.append(entry)

                transported = T * (P @ h_parts[j].float())
                fused = fused + transported
                n_pairs += 1

        fused = fused / max(n_pairs, 1)

        desc_avg = sum(descs) / len(descs)
        h = self.norm1(self.drop(self.out_proj(fused)) + desc_avg)
        h = self.norm2(self.drop(self.ffn(h)) + h)
        return h
