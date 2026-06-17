"""
TA-GNN: Topology-Aware Graph Neural Network (TAMS Innovation 2).

Replaces CA-GBT's hard dual-threshold {0, 0.5, 0.8} message weighting with
continuous differentiable edge weights from MATL. Preserves the MLP + shortcut
structure of ThresBlock for fair ablation comparison.

Key differences from CA-GBT:
  - Edge weights come from MATL (learned, continuous) instead of cosine discretization.
  - No fixed tau_low/tau_high buffers.
  - Confidence is already baked into MATL weights, no redundant multiplication.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class TopologyAwareMessagePassing(gnn.MessagePassing):
    """
    TA-GNN message passing layer with MATL-provided edge weights.

    h_i^{l+1} = MLP(h_i^l + Σ_{j∈N(i)} w_ij · h_j^l) + shortcut(h_i^l)

    Args:
        in_ch: input channel dimension.
        out_ch: output channel dimension.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(aggr="add")
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch, bias=False),
            nn.GELU(),
            nn.Linear(out_ch, out_ch, bias=False),
        )
        self.shortcut = (
            nn.Identity() if in_ch == out_ch
            else nn.Linear(in_ch, out_ch, bias=False)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [T, D] node features.
            edge_index: [2, E] edge indices.
            edge_weight: [E] differentiable edge weights from MATL.
        """
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.fc(x + agg) + self.shortcut(x)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.unsqueeze(-1)


class DirectionalTopologyGNN(nn.Module):
    """
    Direction-specific TA-GNN with learned per-node fusion gate.

    Temporal dependencies are inherently asymmetric: forward (causal) and
    backward (retrospective) information flow differ in nature.  This module
    gives each direction its own parametrisation and learns a per-node gate
    to adaptively weight the three streams.

    Streams:
        forward  (src > dst): later frames aggregate from earlier ones.
        backward (src < dst): earlier frames aggregate from later ones.
        undirected:           symmetric neighbourhood aggregation.

    Gate: α = softmax(W₂ · GELU(W₁ · [h_f ‖ h_b ‖ h_u]))  ∈ R^{T×3}
    Output: h = Σ_d α_d ⊙ h_d
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fwd_gnn = TopologyAwareMessagePassing(in_ch, out_ch)
        self.bwd_gnn = TopologyAwareMessagePassing(in_ch, out_ch)
        self.undi_gnn = TopologyAwareMessagePassing(in_ch, out_ch)
        self.gate_fc = nn.Sequential(
            nn.Linear(out_ch * 3, out_ch),
            nn.GELU(),
            nn.Linear(out_ch, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [T, D] node features.
            edge_index: [2, E] full edge indices (all directions).
            edge_weight: [E] differentiable edge weights from MATL.

        Returns:
            [T, D] direction-fused node features.
        """
        mask_f = edge_index[0] > edge_index[1]
        mask_b = edge_index[0] < edge_index[1]

        h_f = self.fwd_gnn(x, edge_index[:, mask_f], edge_weight[mask_f])
        h_b = self.bwd_gnn(x, edge_index[:, mask_b], edge_weight[mask_b])
        h_u = self.undi_gnn(x, edge_index, edge_weight)

        cat = torch.cat([h_f, h_b, h_u], dim=-1)
        gate = torch.softmax(self.gate_fc(cat), dim=-1)  # [T, 3]
        self._last_gate = gate.detach()
        if bool(getattr(self, "trace_gate_history", False)):
            if not hasattr(self, "_gate_history"):
                self._gate_history = []
            self._gate_history.append(gate.detach())
        return gate[:, 0:1] * h_f + gate[:, 1:2] * h_b + gate[:, 2:3] * h_u
