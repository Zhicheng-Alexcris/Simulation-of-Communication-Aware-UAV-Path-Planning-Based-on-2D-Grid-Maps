"""Read-only objective summaries: no evaluation, selection, or random sampling."""

from __future__ import annotations

import numpy as np


def objective_summary(batch, prefix, reference, communication_weight):
    """Summarize a CandidateBatch using fixed, recorded diagnostic scales.

    *_std is sample SD across candidates (ddof=1), not across repeated runs.
    Empty/all-nonfinite subsets have NaN statistics; a singleton has SD 0.
    Fixed score is diagnostic only, without constraint penalties or clipping.
    """
    reference = np.asarray(reference, dtype=float)
    weight = float(communication_weight)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)) or np.any(reference <= 0):
        raise ValueError("Objective logging requires two finite positive scales")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Objective logging weight must lie in [0, 1]")
    if batch is None:
        length = communication = violation = np.empty(0, dtype=float)
    else:
        length = np.asarray(batch.length, dtype=float)
        communication = np.asarray(batch.communication, dtype=float)
        violation = np.asarray(batch.constraint_violation, dtype=float)
    score = (1 - weight) * length / reference[0] + weight * communication / reference[1]
    metrics = {"length_m": length, "communication": communication,
               "fixed_score": score, "constraint_violation": violation}
    result = {}
    for subset, mask in (("all", np.ones(len(length), dtype=bool)),
                         ("feasible", np.isfinite(violation) & (violation <= 1e-9))):
        base = f"{prefix}_{subset}"
        result[f"{base}_count"] = float(np.count_nonzero(mask))
        for metric, values in metrics.items():
            selected = values[mask]
            finite = selected[np.isfinite(selected)]
            key = f"{base}_{metric}"
            result[f"{key}_finite_count"] = float(len(finite))
            statistics = {name: float("nan") for name in ("mean", "std", "min", "median", "max")}
            if len(finite):
                statistics = {"mean": float(finite.mean()),
                              "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                              "min": float(finite.min()), "median": float(np.median(finite)),
                              "max": float(finite.max())}
            result.update({f"{key}_{name}": value for name, value in statistics.items()})
    return result
