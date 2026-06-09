"""Tiny stdlib logging helper — single entry point so all modules format alike."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "[%(asctime)s] %(name)s %(levelname)s — %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "mtbert", *, log_file: Path | str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
