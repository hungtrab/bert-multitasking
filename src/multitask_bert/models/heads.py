"""Task-specific heads for SST, Quora paraphrase, and STS.

Default heads (used when ``model.use_relational_layer = false``):

* :class:`SentimentHead` — maps ``[CLS]`` to 5 sentiment logits.
* :class:`ParaphraseHead` — SBERT-style ``[h_u; h_v; |h_u - h_v|]``  ->  1 logit.
* :class:`STSCosineHead` — cosine similarity scaled to ``[0, 5]``.

The :class:`models.relational` module provides alternative heads that share
information between paraphrase and STS (the "rich relational" architecture).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SentimentHead(nn.Module):
    def __init__(self, hidden_size: int, *, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.net(cls)


class ParaphraseHead(nn.Module):
    """SBERT-style classifier: concat[h_u, h_v, |h_u - h_v|] -> 1 logit."""

    def __init__(self, hidden_size: int, *, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * hidden_size, 1),
        )

    def forward(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([hu, hv, (hu - hv).abs()], dim=-1)
        return self.net(feats).squeeze(-1)              # (B,)


class STSCosineHead(nn.Module):
    """Cosine similarity rescaled to the configured STS training range."""

    def __init__(self, *, scale: float = 5.0):
        super().__init__()
        self.scale = scale

    def forward(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(hu, hv, dim=-1)        # in [-1, 1]
        return (sim + 1.0) * 0.5 * self.scale
