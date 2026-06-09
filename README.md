# Multitask BERT v2

A clean, modular refactor of the Multitask BERT codebase used for our NLP
course project. v1 (Stanford CS224N original) lives at
`../Multitask BERT/source_code/`. **This v2 keeps the same modelling ideas
but reorganises the codebase for a 5-person team** so that tasks can be
parallelised and reviewed independently.

## What we model

A single BERT encoder fine-tuned jointly on three tasks:

| Task | Type | Dataset | Metric |
|---|---|---|---|
| **Sentiment Analysis (SST)** | 5-class classification | SST-5 (~8.5k) | accuracy |
| **Paraphrase Detection** | binary classification | Quora QP (~140k) | accuracy |
| **Semantic Textual Similarity (STS)** | regression `[0, 5]` | STSBenchmark (~5.7k) | Pearson r |

Aggregated score: `S = (acc_sst + acc_para + (r + 1) / 2) / 3`.

Methods supported:

* **Round-robin multitask fine-tuning** (equal batch size across tasks).
* **Interleaved fine-tuning** (variable batch sizes — bigger slice of Quora per step).
* **Rich relational layer** for the related Para/STS pair.
* **SMART** regularisation — symmetric KL adversarial smoothing + Bregman proximal step.

All four are toggleable via YAML config — no code changes needed for ablations.

## Quick start

```bash
make setup                                            # venv + deps
make data                                             # symlink data from v1
make train CONFIG=configs/round_robin.yaml            # main experiment
make eval  CHECKPOINT=runs/round_robin/best.pt        # evaluate
make test                                             # unit tests
```

## Layout

```
multitask-bert-v2/
├── README.md
├── pyproject.toml
├── Makefile
├── configs/                  # everything an experiment needs
│   ├── default.yaml
│   ├── round_robin.yaml
│   ├── interleaved.yaml
│   ├── rich_relational.yaml
│   └── smart.yaml
├── src/multitask_bert/
│   ├── data/                 # SST / Quora / STS dataset + loader factory
│   ├── models/               # encoder, task heads, relational layer
│   ├── losses/               # SMART regularisation
│   ├── training/             # round-robin & interleaved trainers, optim
│   ├── evaluation/           # metrics + evaluator
│   └── utils/                # config, logging, paths
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/                    # pytest
└── docs/
    ├── ARCHITECTURE.md
    ├── DATA.md
    └── TASKS.md              # 5-person split
```

## Differences from v1

| Concern | v1 | v2 |
|---|---|---|
| BERT impl | custom from-scratch (`model/bert.py`) | custom self-implemented encoder in `models/bert_encoder.py`, with optional loading from `bert-base-uncased` weight files |
| Config | scattered argparse + globals | one YAML per experiment |
| Multitask loops | inline in `multitask_classifier.py` | `training/round_robin.py`, `training/interleaved.py` |
| SMART | `model/proximateGD.py` ad-hoc | `losses/smart.py` with unit tests |
| Evaluation | `evaluation.py` 350+ lines | `evaluation/{metrics,evaluator}.py` with single contract |
| Tests | none | `tests/` with pytest |

The encoder forward pass no longer calls `transformers.AutoModel`. We still use
the pretrained tokenizer/config/weight-file format so token IDs and pretrained
parameters remain compatible with `bert-base-uncased`.

## Team

5 members: see [`docs/TASKS.md`](docs/TASKS.md) for the agreed split and
ownership.

| # | Member | Owns |
|---|---|---|
| 1 | Vu Thuong Tin | data pipeline + dataset registry |
| 2 | Chu Anh Duc | encoder + tokenizer |
| 3 | Tran Quang Hung | training loops + SMART |
| 4 | Nguyen Xuan Khai | task heads + relational layer + losses |
| 5 | Do Dang Vu | evaluation + scripts + CI + report |
