"""Evaluate a checkpoint on the dev set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from multitask_bert.data import build_loaders
from multitask_bert.evaluation import MultitaskEvaluator
from multitask_bert.models import MultitaskBERT, build_tokenizer
from multitask_bert.utils import load_config


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = build_tokenizer(cfg.model.encoder)
    loaders = build_loaders(cfg, tokenizer)

    model = MultitaskBERT(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    evaluator = MultitaskEvaluator(model, device)
    metrics = evaluator.evaluate(loaders.dev_loaders())
    print(json.dumps(metrics.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
