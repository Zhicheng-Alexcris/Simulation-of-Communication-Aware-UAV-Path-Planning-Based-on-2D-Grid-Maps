"""Integration shim selecting the final replay-feedback optimizer."""

from __future__ import annotations

from . import integration as _base
from .optimizer_v9_dynamic4 import FinalReplayFeedbackCADiffusionES


_ORIGINAL_PLAN = _base.plan_drone_ca_diffusion_es


def plan_drone_ca_diffusion_es(*args, **kwargs):
    kwargs.setdefault("patience", None)
    previous = _base.CADiffusionES
    _base.CADiffusionES = FinalReplayFeedbackCADiffusionES
    try:
        return _ORIGINAL_PLAN(*args, **kwargs)
    finally:
        _base.CADiffusionES = previous
