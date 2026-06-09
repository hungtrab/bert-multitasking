"""Train the multitask BERT model.

Usage::

    python -m scripts.train --config configs/round_robin.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from multitask_bert.data import build_loaders
from multitask_bert.models import MultitaskBERT, build_tokenizer
from multitask_bert.training import InterleavedTrainer, RoundRobinTrainer
from multitask_bert.utils import get_logger, load_config, seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--device", default=None,
                   help="cpu | cuda | cuda:0 (default: auto)")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("mtbert.train", log_file=output_dir / "stdout.log")
    log.info(f"experiment={cfg.experiment_name}  device={device}  out={output_dir}")

    tokenizer = build_tokenizer(cfg.model.encoder)

    strategy = cfg.training.strategy
    if strategy == "interleaved":
        loaders = build_loaders(
            cfg, tokenizer,
            num_workers=args.num_workers,
            task_batch_sizes=dict(cfg.training.task_batch_sizes),
        )
    else:
        loaders = build_loaders(cfg, tokenizer, num_workers=args.num_workers)

    model = MultitaskBERT(cfg)

    trainer_cls = {"round_robin": RoundRobinTrainer,
                   "interleaved": InterleavedTrainer}.get(strategy)
    if trainer_cls is None:
        raise ValueError(f"Unknown training.strategy: {strategy}")

    trainer = trainer_cls(cfg, model=model, loaders=loaders, device=device,
                          output_dir=output_dir)
    state = trainer.fit()
    log.info(f"training done. best combined score = {state.best_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
