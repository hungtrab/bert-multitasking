"""Report-style step-count helpers for multitask training."""

from __future__ import annotations

from torch.utils.data import DataLoader


def full_batches(loader: DataLoader) -> int:
    """Return the number of complete batches, matching the v1 source loops."""
    batch_size = getattr(loader, "batch_size", None) or 1
    return max(1, len(loader.dataset) // int(batch_size))


def round_robin_steps(loaders, mode: str) -> int:
    """Step count for equal-batch round-robin training.

    The report specifies floor(len(sts_train_data) / batch_size) for the
    equally weighted setting. Other modes remain available for ablations.
    """
    if mode == "sst":
        return full_batches(loaders.sst_train)
    if mode == "quora":
        return full_batches(loaders.quora_train)
    if mode == "max":
        return max(
            full_batches(loaders.sst_train),
            full_batches(loaders.quora_train),
            full_batches(loaders.sts_train),
        )
    if mode == "sts":
        return full_batches(loaders.sts_train)
    raise ValueError(f"Unknown training.round_robin_steps: {mode}")


def interleaved_steps(loaders) -> int:
    """Step count for report rrobin-full training."""
    return max(1, min(len(loaders.sts_train.dataset), full_batches(loaders.quora_train)))
