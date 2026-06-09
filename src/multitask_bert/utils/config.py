"""YAML config loader with simple ``_base_`` inheritance and ``${var}``
interpolation.

Example::

    cfg = load_config("configs/round_robin.yaml")
    cfg.training.num_epochs    # attribute access
    cfg["training"]["num_epochs"]  # dict access also works
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_VAR_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_.]*)\}")


class Config(dict):
    """A dict that exposes keys as attributes, recursively."""

    def __getattr__(self, item: str) -> Any:
        if item in self:
            value = self[item]
            return Config(value) if isinstance(value, dict) else value
        raise AttributeError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __dir__(self):
        return list(super().__dir__()) + list(self.keys())


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _interpolate(obj: Any, root: dict) -> Any:
    """Replace ``${a.b}`` placeholders with values looked up under ``root``."""
    if isinstance(obj, dict):
        return {k: _interpolate(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(x, root) for x in obj]
    if isinstance(obj, str):
        def repl(match: re.Match[str]) -> str:
            path = match.group(1).split(".")
            cur: Any = root
            for p in path:
                cur = cur[p]
            return str(cur)

        return _VAR_RE.sub(repl, obj)
    return obj


def _load_raw(path: Path) -> dict:
    """Load YAML chain without interpolation, recursively resolving ``_base_``."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base_name = raw.pop("_base_", None)
    if base_name:
        base_path = (path.parent / base_name).resolve()
        base = _load_raw(base_path)
        return _deep_merge(base, raw)
    return raw


def load_config(path: str | os.PathLike) -> Config:
    """Load YAML, resolving ``_base_`` inheritance and ``${var}`` placeholders.

    ``${var}`` substitution is applied **once** to the final merged config so
    placeholders see the most-derived values.
    """
    merged = _load_raw(Path(path).resolve())
    merged = _interpolate(merged, merged)
    return Config(merged)
