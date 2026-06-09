"""End-to-end evaluator over the three task dev/test loaders."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .metrics import accuracy, f1_macro, normalised_pearson, pearson_r


@dataclass
class MultitaskMetrics:
    sst_acc: float = 0.0
    sst_f1: float = 0.0
    para_acc: float = 0.0
    sts_pearson: float = 0.0
    combined: float = 0.0

    def as_dict(self) -> dict:
        return {
            "sst_acc": self.sst_acc,
            "sst_f1": self.sst_f1,
            "para_acc": self.para_acc,
            "sts_pearson": self.sts_pearson,
            "combined": self.combined,
        }


class MultitaskEvaluator:
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, loaders: Mapping[str, torch.utils.data.DataLoader]) -> MultitaskMetrics:
        self.model.eval()
        sst_acc = sst_f1 = para_acc = sts_r = 0.0

        if "sst" in loaders and loaders["sst"] is not None:
            preds, labels = self._predict_singles(loaders["sst"])
            sst_acc = accuracy(preds, labels)
            sst_f1 = f1_macro(preds, labels, num_classes=5)

        if "quora" in loaders and loaders["quora"] is not None:
            preds, labels = self._predict_paraphrase(loaders["quora"])
            para_acc = accuracy(preds, labels)

        if "sts" in loaders and loaders["sts"] is not None:
            preds, labels = self._predict_sts(loaders["sts"])
            sts_r = pearson_r(preds, labels)

        combined = (sst_acc + para_acc + (sts_r + 1.0) / 2.0) / 3.0
        return MultitaskMetrics(sst_acc=sst_acc, sst_f1=sst_f1, para_acc=para_acc,
                                sts_pearson=sts_r, combined=combined)

    # ------------------------------------------------------------------ helpers

    def _predict_singles(self, loader):
        preds, labels = [], []
        for batch in loader:
            ids = batch.token_ids.to(self.device); mask = batch.attention_mask.to(self.device)
            logits = self.model.predict_sentiment(ids, mask)
            preds.extend(logits.argmax(-1).tolist())
            labels.extend(batch.labels.tolist())
        return preds, labels

    def _predict_paraphrase(self, loader):
        preds, labels = [], []
        for batch in loader:
            ids1 = batch.token_ids_1.to(self.device); m1 = batch.attention_mask_1.to(self.device)
            ids2 = batch.token_ids_2.to(self.device); m2 = batch.attention_mask_2.to(self.device)
            logit = self.model.predict_paraphrase(ids1, m1, ids2, m2)
            preds.extend((logit.sigmoid() > 0.5).long().tolist())
            labels.extend(batch.labels.tolist())
        return preds, labels

    def _predict_sts(self, loader):
        preds, labels = [], []
        for batch in loader:
            ids1 = batch.token_ids_1.to(self.device); m1 = batch.attention_mask_1.to(self.device)
            ids2 = batch.token_ids_2.to(self.device); m2 = batch.attention_mask_2.to(self.device)
            score = self.model.predict_similarity(ids1, m1, ids2, m2)
            preds.extend(score.tolist())
            labels.extend(batch.labels.tolist())
        return preds, labels

    @torch.no_grad()
    def write_predictions(
        self,
        loaders: Mapping[str, torch.utils.data.DataLoader],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Run prediction over every loader (typically *test* loaders) and
        write CSVs in the format expected by the v1 grader.
        """
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        produced: dict[str, Path] = {}
        self.model.eval()

        if "sst" in loaders and loaders["sst"] is not None:
            preds, _ = self._predict_singles(loaders["sst"])
            ids_seen = []
            for batch in loaders["sst"]:
                ids_seen.extend(batch.ids)
            out = output_dir / "sst-output.csv"
            with open(out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["id", "Predicted_Sentiment"])
                for i, p in zip(ids_seen, preds):
                    w.writerow([i, p])
            produced["sst"] = out

        if "quora" in loaders and loaders["quora"] is not None:
            preds, _ = self._predict_paraphrase(loaders["quora"])
            ids_seen = []
            for batch in loaders["quora"]:
                ids_seen.extend(batch.ids)
            out = output_dir / "para-output.csv"
            with open(out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["id", "Predicted_Is_Paraphrase"])
                for i, p in zip(ids_seen, preds):
                    w.writerow([i, p])
            produced["quora"] = out

        if "sts" in loaders and loaders["sts"] is not None:
            preds, _ = self._predict_sts(loaders["sts"])
            ids_seen = []
            for batch in loaders["sts"]:
                ids_seen.extend(batch.ids)
            out = output_dir / "sts-output.csv"
            with open(out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["id", "Predicted_Similarity"])
                for i, p in zip(ids_seen, preds):
                    w.writerow([i, p])
            produced["sts"] = out

        return produced
