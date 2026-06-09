"""Datasets for the three tasks.

CSV layouts (matching v1 of the codebase):

* **SST-5**: ``ids-sst-{train,dev}.csv``  ->  columns ``id, sentence, sentiment``
  (sentiment in {0,1,2,3,4}). Test student file: ``id, sentence``.

* **Quora paraphrase**: ``quora-{train,dev}.csv``  ->  columns
  ``id, sentence1, sentence2, is_duplicate``. Test student: no label.

* **STS Benchmark**: ``sts-{train,dev}.csv``  ->  columns
  ``id, sentence1, sentence2, similarity``. Test student: no label.

Two flavours of batches are emitted:

*   :class:`SinglesBatch` — single-sentence (SST).
*   :class:`PairsBatch` — sentence-pair (Quora, STS).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------- batches


@dataclass
class SinglesBatch:
    token_ids: torch.Tensor          # (B, L)
    attention_mask: torch.Tensor     # (B, L)
    labels: torch.Tensor             # (B,) long
    ids: List[str]
    sentences: List[str]


@dataclass
class PairsBatch:
    token_ids_1: torch.Tensor
    attention_mask_1: torch.Tensor
    token_ids_2: torch.Tensor
    attention_mask_2: torch.Tensor
    labels: torch.Tensor             # long for paraphrase, float for STS
    ids: List[str]
    sentences_1: List[str]
    sentences_2: List[str]


# ----------------------------------------------------------------------- helpers


def _read_csv(path: Path) -> List[dict[str, str]]:
    rows: List[dict[str, str]] = []
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in header else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _normalise(text: str) -> str:
    return " ".join(
        text.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


# ----------------------------------------------------------------------- datasets


class _BaseDataset(Dataset):
    """Common pad/collate logic shared by single and pair datasets."""

    name: str = "base"
    is_pair: bool = False

    def __init__(self, rows: Sequence[dict[str, str]], tokenizer, max_seq_len: int = 128):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    # subclasses override __getitem__ + collate_fn

    @classmethod
    def from_csv(cls, csv_path: str | Path, tokenizer, *, max_seq_len: int = 128) -> "_BaseDataset":
        rows = _read_csv(Path(csv_path))
        return cls(rows, tokenizer, max_seq_len=max_seq_len)


class SST5Dataset(_BaseDataset):
    """Stanford Sentiment Treebank 5-class."""

    name = "sst"
    is_pair = False
    sentence_field = "sentence"
    label_field = "sentiment"
    id_field = "id"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "sentence": _normalise(row[self.sentence_field]),
            "label": int(row.get(self.label_field, -1)),
            "id": row.get(self.id_field, str(idx)),
        }

    def collate_fn(self, batch: List[dict[str, Any]]) -> SinglesBatch:
        sentences = [b["sentence"] for b in batch]
        encoded = self.tokenizer(
            sentences, padding=True, truncation=True,
            max_length=self.max_seq_len, return_tensors="pt",
        )
        return SinglesBatch(
            token_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=torch.tensor([b["label"] for b in batch], dtype=torch.long),
            ids=[b["id"] for b in batch],
            sentences=sentences,
        )


class _PairDataset(_BaseDataset):
    sent1_field = "sentence1"
    sent2_field = "sentence2"
    label_field = "label"
    id_field = "id"
    label_dtype = torch.long
    is_pair = True

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        label_str = row.get(self.label_field, None)
        if label_str is None or label_str == "":
            label: float | int = -1
        else:
            label = float(label_str) if self.label_dtype == torch.float else int(float(label_str))
        return {
            "sentence_1": _normalise(row[self.sent1_field]),
            "sentence_2": _normalise(row[self.sent2_field]),
            "label": label,
            "id": row.get(self.id_field, str(idx)),
        }

    def collate_fn(self, batch: List[dict[str, Any]]) -> PairsBatch:
        s1 = [b["sentence_1"] for b in batch]
        s2 = [b["sentence_2"] for b in batch]
        e1 = self.tokenizer(s1, padding=True, truncation=True,
                            max_length=self.max_seq_len, return_tensors="pt")
        e2 = self.tokenizer(s2, padding=True, truncation=True,
                            max_length=self.max_seq_len, return_tensors="pt")
        labels_list = [b["label"] for b in batch]
        labels = torch.tensor(labels_list, dtype=self.label_dtype)
        return PairsBatch(
            token_ids_1=e1["input_ids"],
            attention_mask_1=e1["attention_mask"],
            token_ids_2=e2["input_ids"],
            attention_mask_2=e2["attention_mask"],
            labels=labels,
            ids=[b["id"] for b in batch],
            sentences_1=s1,
            sentences_2=s2,
        )


class QuoraParaphraseDataset(_PairDataset):
    name = "quora"
    sent1_field = "sentence1"
    sent2_field = "sentence2"
    label_field = "is_duplicate"
    label_dtype = torch.long


class STSBenchmarkDataset(_PairDataset):
    name = "sts"
    sent1_field = "sentence1"
    sent2_field = "sentence2"
    label_field = "similarity"
    label_dtype = torch.float
