"""Base trainer — common bookkeeping: logging, checkpointing, eval hooks.

Subclasses implement :meth:`MultitaskTrainer.train_one_epoch` to define how
batches from the three tasks are sequenced (round-robin, interleaved, ...).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..data import MultitaskLoaders
from ..evaluation.evaluator import MultitaskEvaluator, MultitaskMetrics
from ..losses import SmartRegulariser
from ..utils import get_logger
from .optim import build_optimizer, build_scheduler
from .steps import interleaved_steps, round_robin_steps


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_score: float = -float("inf")
    history: list[dict] = field(default_factory=list)


class MultitaskTrainer:
    """Glue between model / data / optimiser / evaluator."""

    def __init__(
        self,
        cfg,
        *,
        model: torch.nn.Module,
        loaders: MultitaskLoaders,
        device: torch.device,
        output_dir: Path,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.loaders = loaders
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("mtbert.trainer", log_file=self.output_dir / "train.log")

        if str(cfg.training.strategy) == "interleaved":
            steps_per_epoch = interleaved_steps(loaders)
        else:
            step_mode = str(cfg.training.get("round_robin_steps", "sts"))
            steps_per_epoch = round_robin_steps(loaders, step_mode)
        self.num_training_steps = int(cfg.training.num_epochs) * 3 * steps_per_epoch

        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, self.num_training_steps)
        self.evaluator = MultitaskEvaluator(self.model, self.device)

        self.scaler = (
            torch.amp.GradScaler("cuda")
            if (cfg.training.amp and device.type == "cuda")
            else None
        )

        self.smart: Optional[SmartRegulariser] = None
        if bool(cfg.losses.smart.enabled):
            self.smart = SmartRegulariser(
                epsilon=float(cfg.losses.smart.epsilon),
                lambda_s=float(cfg.losses.smart.lambda_s),
            )

        self.state = TrainState()

    def task_loss_weight(self, task: str) -> float:
        weights = self.cfg.training.get("task_loss_weights", {})
        return float(weights.get(task, 1.0))

    # ------------------------------------------------------------------ public

    def fit(self) -> TrainState:
        for epoch in range(int(self.cfg.training.num_epochs)):
            self.state.epoch = epoch
            self.log.info(f"=== Epoch {epoch + 1}/{self.cfg.training.num_epochs} ===")
            t0 = time.time()
            self.train_one_epoch()
            t1 = time.time()
            metrics = self.evaluate()
            self.log.info(
                f"  epoch {epoch + 1} done in {t1 - t0:.1f}s | "
                f"val SST acc={metrics.sst_acc:.4f}  Para acc={metrics.para_acc:.4f}  "
                f"STS r={metrics.sts_pearson:.4f}  S={metrics.combined:.4f}"
            )
            self.state.history.append({"epoch": epoch + 1, **metrics.as_dict()})

            if metrics.combined > self.state.best_score:
                self.state.best_score = metrics.combined
                self.save_checkpoint(self.output_dir / "best.pt", metrics=metrics)

            self.save_checkpoint(self.output_dir / "last.pt", metrics=metrics)
            with open(self.output_dir / "history.json", "w") as f:
                json.dump(self.state.history, f, indent=2)

        return self.state

    # ------------------------------------------------------------------ extension hooks

    def train_one_epoch(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ losses

    def sst_loss(self, batch) -> torch.Tensor:
        ids = batch.token_ids.to(self.device)
        mask = batch.attention_mask.to(self.device)
        y = batch.labels.to(self.device)
        logits = self.model.predict_sentiment(ids, mask)
        loss = F.cross_entropy(logits, y)
        if self.smart is not None:
            head = self.model.sentiment_head

            def fwd(emb):
                return head(self.model.encoder.forward_from_embeddings(emb, mask).cls)

            loss = loss + self.smart(
                self.model, encoder=self.model.encoder,
                input_ids=ids, attention_mask=mask,
                forward_fn=fwd, clean_output=logits, divergence="kl",
            )
        return loss

    def quora_loss(self, batch) -> torch.Tensor:
        ids1 = batch.token_ids_1.to(self.device); m1 = batch.attention_mask_1.to(self.device)
        ids2 = batch.token_ids_2.to(self.device); m2 = batch.attention_mask_2.to(self.device)
        y = batch.labels.to(self.device).float()
        logit = self.model.predict_paraphrase(ids1, m1, ids2, m2)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        if self.smart is not None:
            loss = loss + self._smart_pair(
                ids1, m1, ids2, m2,
                clean_output=logit,
                task="para",
                divergence="mse",
            )
        return loss

    def sts_loss(self, batch) -> torch.Tensor:
        ids1 = batch.token_ids_1.to(self.device); m1 = batch.attention_mask_1.to(self.device)
        ids2 = batch.token_ids_2.to(self.device); m2 = batch.attention_mask_2.to(self.device)
        y = batch.labels.to(self.device).float() / float(self.cfg.losses.get("sts_label_scale", 5.0))
        score = self.model.predict_similarity(ids1, m1, ids2, m2)
        loss = F.mse_loss(score, y)
        if self.smart is not None:
            loss = loss + self._smart_pair(
                ids1, m1, ids2, m2,
                clean_output=score,
                task="sts",
                divergence="mse",
            )
        return loss

    def _pair_output_from_cls(self, hu: torch.Tensor, hv: torch.Tensor, *, task: str) -> torch.Tensor:
        if bool(self.model.use_relational):
            shared = self.model.relational(hu, hv)
            if task == "para":
                return self.model.paraphrase_head(shared)
            return self.model.sts_head(shared)
        if task == "para":
            return self.model.paraphrase_head(hu, hv)
        return self.model.sts_head(hu, hv)

    def _smart_pair(
        self,
        ids1: torch.Tensor,
        m1: torch.Tensor,
        ids2: torch.Tensor,
        m2: torch.Tensor,
        *,
        clean_output: torch.Tensor,
        task: str,
        divergence: str,
    ) -> torch.Tensor:
        """SMART perturbation for both sides of a sentence-pair task."""
        if self.smart is None:  # pragma: no cover - defensive
            return torch.zeros((), device=self.device)

        clean_h2 = self.model.encoder(ids2, m2).cls.detach()

        def fwd_left(emb):
            h1 = self.model.encoder.forward_from_embeddings(emb, m1).cls
            return self._pair_output_from_cls(h1, clean_h2, task=task)

        left = self.smart(
            self.model, encoder=self.model.encoder,
            input_ids=ids1, attention_mask=m1,
            forward_fn=fwd_left, clean_output=clean_output,
            divergence=divergence,
        )

        clean_h1 = self.model.encoder(ids1, m1).cls.detach()

        def fwd_right(emb):
            h2 = self.model.encoder.forward_from_embeddings(emb, m2).cls
            return self._pair_output_from_cls(clean_h1, h2, task=task)

        right = self.smart(
            self.model, encoder=self.model.encoder,
            input_ids=ids2, attention_mask=m2,
            forward_fn=fwd_right, clean_output=clean_output,
            divergence=divergence,
        )
        return 0.5 * (left + right)

    # ------------------------------------------------------------------ optim step

    def optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.training.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.training.grad_clip))
            self.optimizer.step()
        self.scheduler.step()
        self.state.global_step += 1

    # ------------------------------------------------------------------ eval / ckpt

    def evaluate(self) -> MultitaskMetrics:
        self.model.eval()
        with torch.no_grad():
            metrics = self.evaluator.evaluate(self.loaders.dev_loaders())
        self.model.train()
        return metrics

    def save_checkpoint(self, path: Path, *, metrics: MultitaskMetrics | None = None) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "cfg": dict(self.cfg),
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "metrics": metrics.as_dict() if metrics else None,
            },
            path,
        )
        self.log.info(f"  saved checkpoint -> {path}")
