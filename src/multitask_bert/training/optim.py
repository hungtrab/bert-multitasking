"""Optimiser + LR-scheduler factories driven by config."""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


_NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight")


def _decay_groups(parameters: Iterable[tuple[str, torch.nn.Parameter]], wd: float):
    decay, no_decay = [], []
    for name, p in parameters:
        if not p.requires_grad:
            continue
        if any(nd in name for nd in _NO_DECAY):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model: torch.nn.Module, cfg) -> AdamW:
    groups = _decay_groups(model.named_parameters(), float(cfg.training.weight_decay))
    return AdamW(groups, lr=float(cfg.training.learning_rate))


def build_scheduler(optimizer, cfg, num_training_steps: int) -> LambdaLR:
    """Linear warmup then linear decay to zero (BERT-style)."""
    warmup = int(cfg.training.warmup_steps)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, num_training_steps - warmup)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)
