from pathlib import Path

import torch

from multitask_bert.data.datasets import QuoraParaphraseDataset, SST5Dataset


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        batch = len(texts)
        return {
            "input_ids": torch.ones(batch, 4, dtype=torch.long),
            "attention_mask": torch.ones(batch, 4, dtype=torch.long),
        }


def test_reads_tab_separated_csv_suffix(tmp_path: Path):
    path = tmp_path / "ids-sst-train.csv"
    path.write_text("\tid\tsentence\tsentiment\n0\tabc\tGreat movie.\t4\n", encoding="utf-8")

    ds = SST5Dataset.from_csv(path, DummyTokenizer())

    assert len(ds) == 1
    assert ds[0]["sentence"] == "great movie ."
    assert ds[0]["label"] == 4
    assert ds[0]["id"] == "abc"


def test_pair_label_accepts_float_encoded_integer(tmp_path: Path):
    path = tmp_path / "quora-train.csv"
    path.write_text(
        "\tid\tsentence1\tsentence2\tis_duplicate\n"
        "0\tqid\tHow are you?\tHow do you do?\t1.0\n",
        encoding="utf-8",
    )

    ds = QuoraParaphraseDataset.from_csv(path, DummyTokenizer())

    assert ds[0]["label"] == 1

