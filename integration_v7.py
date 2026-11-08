"""Integration shim selecting the v7 skeleton-conditioned optimizer."""

from __future__ import annotations

from . import integration as _base
from .optimizer_v7 import SkeletonConditionedCADiffusionES


_ORIGINAL_PLAN = _base.plan_drone_ca_diffusion_es


def plan_drone_ca_diffusion_es(*args, **kwargs):
    previous = _base.CADiffusionES
    _base.CADiffusionES = SkeletonConditionedCADiffusionES
    try:
        return _ORIGINAL_PLAN(*args, **kwargs)
    finally:
        _base.CADiffusionES = previous
