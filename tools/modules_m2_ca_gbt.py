"""
Confidence-aware dual-threshold GBT message passing (CA-GBT).
Extends ThresBlock tri-level similarity mask with per-edge confidence scaling.
"""

import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class CAThresBlock(gnn.MessagePassing):
    def __init__(self, in_channel: int, out_channel: int, tau_low: float = 0.2, tau_high: float = 0.5):
        super().__init__(aggr="add")
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.register_buffer("tau_low", torch.tensor(float(tau_low)), persistent=False)
        self.register_buffer("tau_high", torch.tensor(float(tau_high)), persistent=False)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, out_channel, bias=False),
            nn.GELU(),
            nn.Linear(out_channel, out_channel, bias=False),
        )
        self.shortcut = nn.Identity() if in_channel == out_channel else nn.Linear(in_channel, out_channel, bias=False)

    def _tri_level_sim(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        t1 = float(self.tau_low)
        t2 = float(self.tau_high)
        with torch.no_grad():
            xi_f = x_i.float()
            xj_f = x_j.float()
            ni = xi_f.norm(dim=1).clamp(min=1e-6)
            nj = xj_f.norm(dim=1).clamp(min=1e-6)
            sim = (xi_f * xj_f).sum(dim=1) / (ni * nj)
            sim = torch.where(sim > t2, torch.tensor(0.8, device=sim.device),
                   torch.where(sim >= t1, torch.tensor(0.5, device=sim.device),
                               torch.zeros_like(sim)))
        return sim.to(dtype=x_i.dtype)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_conf: torch.Tensor | None = None):
        if node_conf is None:
            node_conf = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        conf_ij = node_conf[edge_index[0]] * node_conf[edge_index[1]]
        return self.fc(x + self.propagate(edge_index, x=x, conf_ij=conf_ij)) + self.shortcut(x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, conf_ij: torch.Tensor) -> torch.Tensor:
        sim = self._tri_level_sim(x_i, x_j)
        w = sim * conf_ij * self.alpha.view(-1)
        return x_j * w.unsqueeze(-1)
