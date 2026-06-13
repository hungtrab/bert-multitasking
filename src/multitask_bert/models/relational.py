"""Rich relational layer for the related Para+STS task pair.

Implements the v1/report architecture closely: both paraphrase and STS pairs
flow through a shared sentence-pair layer over
``[h_u; h_v; |h_u - h_v|; normalised(h_u) * normalised(h_v)]``.
"""

from __future__ import annotations

import torch
from torch import nn


class RichRelationalLayer(nn.Module):
    """Shared v1-style sentence-pair layer."""

    def __init__(self, hidden_size: int, *, output_size: int | None = None,
                 dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        out = output_size or hidden_size
        self.eps = eps
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, out),
        )

    def hadamard(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        nu = hu.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        nv = hv.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return (hu * hv) / (nu * nv + self.eps)            # sum -> cosine

    def forward(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        features = torch.cat([hu, hv, (hu - hv).abs(), self.hadamard(hu, hv)], dim=-1)
        return self.net(features)


class RelationalParaphraseHead(nn.Module):
    """Paraphrase classifier on top of the shared relational layer."""

    def __init__(self, hidden_size: int, *, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        return self.head(shared).squeeze(-1)


class RelationalSTSHead(nn.Module):
    """STS regressor on top of the shared relational layer."""

    def __init__(self, hidden_size: int, *, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        # v1 trains STS against labels scaled to [0, 1].
        return torch.sigmoid(self.head(shared).squeeze(-1))
