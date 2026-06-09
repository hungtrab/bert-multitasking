"""Build DataLoaders for the three tasks from a config object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from torch.utils.data import DataLoader

from .datasets import (
    QuoraParaphraseDataset,
    STSBenchmarkDataset,
    SST5Dataset,
    _BaseDataset,
)


@dataclass
class MultitaskLoaders:
    """A dataclass holding train/dev/test loaders for the three tasks."""

    sst_train: DataLoader
    sst_dev: DataLoader
    sst_test: DataLoader | None
    quora_train: DataLoader
    quora_dev: DataLoader
    quora_test: DataLoader | None
    sts_train: DataLoader
    sts_dev: DataLoader
    sts_test: DataLoader | None

    def train_loaders(self) -> dict[str, DataLoader]:
        return {"sst": self.sst_train, "quora": self.quora_train, "sts": self.sts_train}

    def dev_loaders(self) -> dict[str, DataLoader]:
        return {"sst": self.sst_dev, "quora": self.quora_dev, "sts": self.sts_dev}

    def test_loaders(self) -> dict[str, DataLoader | None]:
        return {"sst": self.sst_test, "quora": self.quora_test, "sts": self.sts_test}


def _make_loader(ds: _BaseDataset, batch_size: int, *, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ds.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def _resolve(data_dir: Path, name: str | None) -> Path | None:
    if name is None:
        return None
    p = data_dir / name
    return p if p.exists() else None


def build_loaders(
    cfg,
    tokenizer,
    *,
    num_workers: int = 0,
    task_batch_sizes: Mapping[str, int] | None = None,
) -> MultitaskLoaders:
    """Construct a :class:`MultitaskLoaders` from a YAML config + tokenizer.

    If ``task_batch_sizes`` is not given, the round-robin ``training.batch_size``
    is used for all tasks.
    """
    data_dir = Path(cfg.paths.data_dir)
    max_seq_len = int(cfg.data.max_seq_len)

    if task_batch_sizes is None:
        bs = int(cfg.training.batch_size)
        task_batch_sizes = {"sst": bs, "quora": bs, "sts": bs}

    sst_train = SST5Dataset.from_csv(data_dir / cfg.data.sst.train, tokenizer, max_seq_len=max_seq_len)
    sst_dev = SST5Dataset.from_csv(data_dir / cfg.data.sst.dev, tokenizer, max_seq_len=max_seq_len)
    sst_test_path = _resolve(data_dir, cfg.data.sst.get("test"))
    sst_test_ds = SST5Dataset.from_csv(sst_test_path, tokenizer, max_seq_len=max_seq_len) if sst_test_path else None

    quora_train = QuoraParaphraseDataset.from_csv(data_dir / cfg.data.quora.train, tokenizer, max_seq_len=max_seq_len)
    quora_dev = QuoraParaphraseDataset.from_csv(data_dir / cfg.data.quora.dev, tokenizer, max_seq_len=max_seq_len)
    quora_test_path = _resolve(data_dir, cfg.data.quora.get("test"))
    quora_test_ds = QuoraParaphraseDataset.from_csv(quora_test_path, tokenizer, max_seq_len=max_seq_len) if quora_test_path else None

    sts_train = STSBenchmarkDataset.from_csv(data_dir / cfg.data.sts.train, tokenizer, max_seq_len=max_seq_len)
    sts_dev = STSBenchmarkDataset.from_csv(data_dir / cfg.data.sts.dev, tokenizer, max_seq_len=max_seq_len)
    sts_test_path = _resolve(data_dir, cfg.data.sts.get("test"))
    sts_test_ds = STSBenchmarkDataset.from_csv(sts_test_path, tokenizer, max_seq_len=max_seq_len) if sts_test_path else None

    return MultitaskLoaders(
        sst_train=_make_loader(sst_train, task_batch_sizes["sst"], shuffle=True, num_workers=num_workers),
        sst_dev=_make_loader(sst_dev, task_batch_sizes["sst"], shuffle=False, num_workers=num_workers),
        sst_test=_make_loader(sst_test_ds, task_batch_sizes["sst"], shuffle=False, num_workers=num_workers) if sst_test_ds else None,
        quora_train=_make_loader(quora_train, task_batch_sizes["quora"], shuffle=True, num_workers=num_workers),
        quora_dev=_make_loader(quora_dev, task_batch_sizes["quora"], shuffle=False, num_workers=num_workers),
        quora_test=_make_loader(quora_test_ds, task_batch_sizes["quora"], shuffle=False, num_workers=num_workers) if quora_test_ds else None,
        sts_train=_make_loader(sts_train, task_batch_sizes["sts"], shuffle=True, num_workers=num_workers),
        sts_dev=_make_loader(sts_dev, task_batch_sizes["sts"], shuffle=False, num_workers=num_workers),
        sts_test=_make_loader(sts_test_ds, task_batch_sizes["sts"], shuffle=False, num_workers=num_workers) if sts_test_ds else None,
    )
