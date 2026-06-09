"""The multitask model that ties encoder + heads together.

Two assembly modes (chosen by ``cfg.model.use_relational_layer``):

1. **Plain** (default): independent heads for each task.
2. **Rich relational**: paraphrase and STS share a single hidden layer
   (:class:`RichRelationalLayer`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .bert_encoder import build_encoder
from .heads import ParaphraseHead, SentimentHead, STSCosineHead
from .relational import (
    RelationalParaphraseHead,
    RelationalSTSHead,
    RichRelationalLayer,
)


@dataclass
class MultitaskOutputs:
    sentiment_logits: Optional[torch.Tensor] = None      # (B, 5)
    paraphrase_logit: Optional[torch.Tensor] = None      # (B,)
    sts_score: Optional[torch.Tensor] = None             # (B,)


class MultitaskBERT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = build_encoder(
            cfg.model.encoder,
            backend=str(cfg.model.get("encoder_backend", "mini")),
            freeze=cfg.model.freeze_encoder,
            load_pretrained=bool(cfg.model.get("load_pretrained", True)),
        )
        H = self.encoder.hidden_size
        dropout = float(cfg.model.hidden_dropout)

        self.sentiment_head = SentimentHead(
            H, num_classes=int(cfg.data.sst.num_classes), dropout=dropout
        )

        self.use_relational = bool(cfg.model.use_relational_layer)
        if self.use_relational:
            self.relational = RichRelationalLayer(H, dropout=dropout)
            self.paraphrase_head = RelationalParaphraseHead(H, dropout=dropout)
            self.sts_head = RelationalSTSHead(H, dropout=dropout)
        else:
            self.paraphrase_head = ParaphraseHead(H, dropout=dropout)
            self.sts_head = STSCosineHead(scale=5.0)

    # -------- single-sentence forward (SST) -------------------------------

    def predict_sentiment(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(token_ids, attention_mask)
        return self.sentiment_head(out.cls)

    def _encode_pair(
        self,
        ids1: torch.Tensor, mask1: torch.Tensor,
        ids2: torch.Tensor, mask2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.encoder(ids1, mask1).cls,
            self.encoder(ids2, mask2).cls,
        )

    def predict_paraphrase(
        self,
        ids1: torch.Tensor, mask1: torch.Tensor,
        ids2: torch.Tensor, mask2: torch.Tensor,
    ) -> torch.Tensor:
        hu, hv = self._encode_pair(ids1, mask1, ids2, mask2)
        if self.use_relational:
            shared = self.relational(hu, hv)
            return self.paraphrase_head(shared)
        return self.paraphrase_head(hu, hv)

    def predict_similarity(
        self,
        ids1: torch.Tensor, mask1: torch.Tensor,
        ids2: torch.Tensor, mask2: torch.Tensor,
    ) -> torch.Tensor:
        hu, hv = self._encode_pair(ids1, mask1, ids2, mask2)
        if self.use_relational:
            shared = self.relational(hu, hv)
            return self.sts_head(shared)
        return self.sts_head(hu, hv)
