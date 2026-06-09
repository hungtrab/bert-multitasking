"""Reproducibility helper."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 11711) -> None:
    """Set Python, NumPy and PyTorch RNGs (best-effort)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:  # pragma: no cover
        pass
