"""Per-evaluation diagnostics. Never evaluates paths or samples random values."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


CANDIDATE_FIELDS = [
    "evaluation_id", "generation", "stage", "batch_id", "index_in_batch", "batch_size",
    "length_m", "communication_cost", "fixed_cost", "constraint_violation",
    "is_feasible", "best_feasible_fixed_cost_so_far", "min_clearance_m",
    "collision_count", "mean_pathloss_db", "worst_pathloss_db", "separation_violation_m",
    "topology_origin", "length_scale_m", "communication_scale", "communication_weight",
]


class CandidateEvaluationLog:
    def __init__(self, reference, communication_weight):
        self.reference = np.asarray(reference, dtype=float).copy()
        self.weight = float(communication_weight)
        if self.reference.shape != (2,) or not np.all(np.isfinite(self.reference)) or np.any(self.reference <= 0):
            raise ValueError("Candidate logging requires two positive finite reference scales")
        if not np.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("Candidate logging weight must lie within [0, 1]")
        self.records = []
        self.batch_id = 0
        self.best = float("inf")

    def record_batch(self, batch, *, evaluation_end, generation, stage):
        count = len(batch.length)
        first = int(evaluation_end) - count + 1
        if first != len(self.records) + 1:
            raise RuntimeError("Per-candidate log does not match the objective evaluation counter")
        self.batch_id += 1
        scores = ((1 - self.weight) * batch.length / self.reference[0]
                  + self.weight * batch.communication / self.reference[1])
        for i in range(count):
            violation = float(batch.constraint_violation[i])
            feasible = bool(np.isfinite(violation) and violation <= 1e-9)
            cost = float(scores[i])
            if feasible and np.isfinite(cost):
                self.best = min(self.best, cost)
            self.records.append({
                "evaluation_id": first + i, "generation": int(generation), "stage": str(stage),
                "batch_id": self.batch_id, "index_in_batch": i, "batch_size": count,
                "length_m": float(batch.length[i]), "communication_cost": float(batch.communication[i]),
                "fixed_cost": cost, "constraint_violation": violation, "is_feasible": int(feasible),
                "best_feasible_fixed_cost_so_far": self.best if np.isfinite(self.best) else float("nan"),
                "min_clearance_m": float(batch.min_clearance[i]),
                "collision_count": int(batch.collision_count[i]),
                "mean_pathloss_db": float(batch.mean_pathloss_db[i]),
                "worst_pathloss_db": float(batch.worst_pathloss_db[i]),
                "separation_violation_m": float(batch.separation_violation[i]),
                "topology_origin": int(batch.topology_origin[i]),
                "length_scale_m": float(self.reference[0]),
                "communication_scale": float(self.reference[1]), "communication_weight": self.weight,
            })

    def validate_total(self, evaluation_count):
        if len(self.records) != int(evaluation_count):
            raise RuntimeError(f"Candidate rows={len(self.records)} but evaluations={evaluation_count}")


def export_candidate_csv(result, path, *, context=None):
    """Write once per completed UAV; no plotting/interpolation or synthetic points."""
    records = result.candidate_history
    if len(records) != result.evaluation_count:
        raise ValueError("Refusing to export incomplete per-evaluation records")
    if [row["evaluation_id"] for row in records] != list(range(1, result.evaluation_count + 1)):
        raise ValueError("Candidate evaluation IDs must be contiguous and unique")
    context = dict(context or {})
    if set(context).intersection(CANDIDATE_FIELDS):
        raise ValueError("Context fields may not overwrite candidate fields")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(context) + CANDIDATE_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({**context, **row})
    temporary.replace(path)
    return path
