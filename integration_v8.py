"""Integration shim selecting the replay-aware v8 optimizer."""

from __future__ import annotations

from . import integration as _base
from .optimizer_v8_fixed import ReplayAwareAdaptiveSkeletonCADiffusionES


_ORIGINAL_PLAN = _base.plan_drone_ca_diffusion_es


def plan_drone_ca_diffusion_es(*args, **kwargs):
    kwargs.setdefault("patience", None)
    previous = _base.CADiffusionES
    _base.CADiffusionES = ReplayAwareAdaptiveSkeletonCADiffusionES
    try:
        return _ORIGINAL_PLAN(*args, **kwargs)
    finally:
        _base.CADiffusionES = previous
