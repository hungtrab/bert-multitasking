"""Round-robin multitask trainer.

Algorithm: at each step, draw one batch from each of the three task
loaders (recycling shorter loaders), run a forward+backward+step *for
each*. The order is fixed: SST -> Quora -> STS. Equal batch size for all
three tasks (set by ``training.batch_size``).
"""

from __future__ import annotations

from itertools import cycle

import torch
from tqdm import tqdm

from .trainer import MultitaskTrainer


class RoundRobinTrainer(MultitaskTrainer):
    def train_one_epoch(self) -> None:
        self.model.train()
        sst_iter = iter(self.loaders.sst_train)
        quora_iter = cycle(self.loaders.quora_train)   # often longer
        sts_iter = cycle(self.loaders.sts_train)        # often shorter
        # We iterate as many times as the SST loader has batches.
        n_steps = len(self.loaders.sst_train)
        pbar = tqdm(range(n_steps), desc=f"epoch {self.state.epoch + 1}")
        loss_sum = 0.0; loss_count = 0

        for _ in pbar:
            sst_batch = next(sst_iter)
            quora_batch = next(quora_iter)
            sts_batch = next(sts_iter)

            # IMPORTANT: forward → backward → step must complete for one task
            # before the next task's forward pass. The three tasks share the
            # BERT encoder; if we pre-computed all three losses first, the
            # optimizer step on task 1 would mutate encoder weights in-place
            # and invalidate the autograd graphs of tasks 2 and 3
            # ("variable modified by an inplace operation"). Using lazy
            # callables keeps each task's forward right before its step.
            for loss_fn, batch in (
                (self.sst_loss, sst_batch),
                (self.quora_loss, quora_batch),
                (self.sts_loss, sts_batch),
            ):
                batch_loss = loss_fn(batch)
                self.optimizer_step(batch_loss)
                loss_sum += batch_loss.item(); loss_count += 1

            if self.state.global_step % int(self.cfg.training.log_interval) == 0:
                pbar.set_postfix(avg_loss=loss_sum / max(1, loss_count))
