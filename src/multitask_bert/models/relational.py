"""Rich relational layer for the related Para+STS task pair.

Implements the architecture described in the v1 report (Section 3.3):
both paraphrase and STS pairs flow through a *shared* hidden layer that
takes the scaled Hadamard product of the two sentence embeddings as input.

Key differences from v1:
* Cleanly separated module so the relational layer is testable in isolation.
* Uses :func:`torch.nn.functional.cosine_similarity`-style scaling so cosine
  is recoverable as the sum of features (sanity check from the report).
"""

from __future__ import annotations

import torch
from torch import nn


class RichRelationalLayer(nn.Module):
    """Shared hidden layer: Hadamard product -> hidden -> activation."""

    def __init__(self, hidden_size: int, *, output_size: int | None = None,
                 dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        out = output_size or hidden_size
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(hidden_size, out),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def hadamard(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        nu = hu.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        nv = hv.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return (hu * hv) / (nu * nv + self.eps)            # sum -> cosine

    def forward(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        return self.net(self.hadamard(hu, hv))


class RelationalParaphraseHead(nn.Module):
    """Paraphrase classifier on top of the shared relational layer."""

    def __init__(self, hidden_size: int, *, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
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
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        # Output bounded to [0, 5] via sigmoid * 5.
        return torch.sigmoid(self.head(shared).squeeze(-1)) * 5.0
