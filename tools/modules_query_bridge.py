"""Global query conditioning for QFVS (QueryBridge)."""

import torch
from torch import nn
from torch_geometric.nn import Linear


class QueryBridge(nn.Module):
    """
    Minimal attention-style bridge: h' = h + softmax((W_s h)·q_proj) * q_proj
    """

    def __init__(self, dim: int, query_dim: int | None = None):
        super().__init__()
        qd = query_dim or dim
        self.lin_s = Linear(dim, dim, bias=False)
        self.lin_q = Linear(qd, dim, bias=False)

    def forward(self, h: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if query.dim() > 1:
            query = query.view(-1)
        q_proj = self.lin_q(query)
        hs = self.lin_s(h)
        logits = (hs * q_proj.unsqueeze(0)).sum(dim=-1, keepdim=True)
        beta = torch.softmax(logits, dim=0)
        return h + beta * q_proj.unsqueeze(0)
