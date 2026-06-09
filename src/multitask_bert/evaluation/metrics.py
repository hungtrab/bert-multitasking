"""Standalone metric functions."""

from __future__ import annotations

import numpy as np


def accuracy(preds, labels) -> float:
    preds = np.asarray(preds); labels = np.asarray(labels)
    return float((preds == labels).mean())


def f1_macro(preds, labels, *, num_classes: int | None = None) -> float:
    preds = np.asarray(preds); labels = np.asarray(labels)
    classes = np.unique(np.concatenate([preds, labels])) if num_classes is None else np.arange(num_classes)
    f1s = []
    for c in classes:
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def pearson_r(preds, labels) -> float:
    preds = np.asarray(preds, dtype=float); labels = np.asarray(labels, dtype=float)
    if preds.size < 2:
        return 0.0
    cov = ((preds - preds.mean()) * (labels - labels.mean())).mean()
    sp = preds.std(); sl = labels.std()
    return float(cov / (sp * sl + 1e-12))


def normalised_pearson(preds, labels) -> float:
    """Map Pearson r in [-1, 1] to [0, 1] (used for the combined score)."""
    return (pearson_r(preds, labels) + 1.0) / 2.0
