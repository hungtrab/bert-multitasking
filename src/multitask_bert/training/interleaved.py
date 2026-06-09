"""Interleaved trainer (variable per-task batch sizes).

Each loader is built with a per-task batch size from
``cfg.training.task_batch_sizes``. We iterate as many times as the
*largest* loader (Quora) — short loaders are recycled. This implements
"interleaved fine-tuning" from the v1 report: more Quora data per step
without starving SST/STS.
"""

from __future__ import annotations

from itertools import cycle

from tqdm import tqdm

from .trainer import MultitaskTrainer


class InterleavedTrainer(MultitaskTrainer):
    def train_one_epoch(self) -> None:
        self.model.train()
        sst_iter = cycle(self.loaders.sst_train)
        quora_iter = iter(self.loaders.quora_train)     # the largest
        sts_iter = cycle(self.loaders.sts_train)
        n_steps = len(self.loaders.quora_train)
        pbar = tqdm(range(n_steps), desc=f"epoch {self.state.epoch + 1}")
        loss_sum = 0.0; loss_count = 0

        for _ in pbar:
            quora_batch = next(quora_iter)
            sst_batch = next(sst_iter)
            sts_batch = next(sts_iter)

            # forward → backward → step per task before next task's forward
            # (shared encoder; see note in round_robin.py).
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
