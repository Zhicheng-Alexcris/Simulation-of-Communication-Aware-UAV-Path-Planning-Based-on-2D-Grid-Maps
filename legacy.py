"""Narrow compatibility layer for the existing city and spline modules."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def load_legacy_modules(legacy_root: str | Path):
    legacy_root = Path(legacy_root).resolve()
    if not (legacy_root / "environment.py").exists():
        raise FileNotFoundError(f"environment.py not found under {legacy_root}")
    root_text = str(legacy_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    environment = importlib.import_module("environment")
    path_planner = importlib.import_module("path_planner")
    return environment, path_planner

