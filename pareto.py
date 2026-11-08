"""Constraint-aware Pareto ranking utilities."""

from __future__ import annotations

import numpy as np


def constrained_non_dominated_sort(
    objectives: np.ndarray, constraint_violation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objectives = np.asarray(objectives, dtype=float)
    violation = np.asarray(constraint_violation, dtype=float).reshape(-1)
    if objectives.ndim != 2 or len(objectives) != len(violation):
        raise ValueError("Objectives and constraint violations have incompatible shapes")
    count = len(objectives)
    if count == 0:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=float)
    objectives = np.where(np.isfinite(objectives), objectives, np.inf)
    violation = np.where(np.isfinite(violation), np.maximum(violation, 0.0), np.inf)
    feasible = violation <= 1e-9

    def dominates(left: int, right: int) -> bool:
        if feasible[left] and not feasible[right]:
            return True
        if not feasible[left] and feasible[right]:
            return False
        if not feasible[left] and not feasible[right]:
            if violation[left] < violation[right] - 1e-12:
                return True
            if violation[left] > violation[right] + 1e-12:
                return False
        return bool(
            np.all(objectives[left] <= objectives[right])
            and np.any(objectives[left] < objectives[right])
        )

    dominated_sets = [[] for _ in range(count)]
    domination_count = np.zeros(count, dtype=int)
    for left in range(count):
        for right in range(left + 1, count):
            if dominates(left, right):
                dominated_sets[left].append(right)
                domination_count[right] += 1
            elif dominates(right, left):
                dominated_sets[right].append(left)
                domination_count[left] += 1

    fronts: list[list[int]] = []
    current = np.flatnonzero(domination_count == 0).tolist()
    while current:
        fronts.append(current)
        following: list[int] = []
        for left in current:
            for right in dominated_sets[left]:
                domination_count[right] -= 1
                if domination_count[right] == 0:
                    following.append(right)
        current = following

    rank = np.full(count, count, dtype=int)
    crowding = np.zeros(count, dtype=float)
    for front_rank, front in enumerate(fronts):
        rank[front] = front_rank
        if len(front) <= 2:
            crowding[front] = np.inf
            continue
        for objective_index in range(objectives.shape[1]):
            values = objectives[front, objective_index]
            sorted_local = np.argsort(values)
            sorted_global = np.asarray(front, dtype=int)[sorted_local]
            crowding[sorted_global[0]] = np.inf
            crowding[sorted_global[-1]] = np.inf
            value_range = values[sorted_local[-1]] - values[sorted_local[0]]
            if value_range <= 1e-12:
                continue
            for local_index in range(1, len(front) - 1):
                global_index = sorted_global[local_index]
                if np.isinf(crowding[global_index]):
                    continue
                crowding[global_index] += (
                    values[sorted_local[local_index + 1]]
                    - values[sorted_local[local_index - 1]]
                ) / value_range

    order = np.asarray(
        sorted(range(count), key=lambda index: (rank[index], -crowding[index])),
        dtype=int,
    )
    return order, rank, crowding


def feasible_pareto_mask(
    objectives: np.ndarray, constraint_violation: np.ndarray
) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)
    violation = np.asarray(constraint_violation, dtype=float).reshape(-1)
    mask = np.zeros(len(objectives), dtype=bool)
    feasible_indices = np.flatnonzero(violation <= 1e-9)
    if len(feasible_indices) == 0:
        return mask
    feasible_objectives = objectives[feasible_indices]
    keep = np.ones(len(feasible_indices), dtype=bool)
    for index in range(len(feasible_indices)):
        dominated = np.all(feasible_objectives <= feasible_objectives[index], axis=1) & np.any(
            feasible_objectives < feasible_objectives[index], axis=1
        )
        dominated[index] = False
        if np.any(dominated):
            keep[index] = False
    mask[feasible_indices[keep]] = True
    return mask


def normalize_objectives(objectives: np.ndarray) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2 or len(objectives) == 0:
        raise ValueError("At least one objective vector is required")
    lower = objectives.min(axis=0)
    upper = objectives.max(axis=0)
    return (objectives - lower) / np.maximum(upper - lower, 1e-12)


def compromise_index(
    objectives: np.ndarray,
    weights: tuple[float, ...] | np.ndarray | None = None,
) -> int:
    objectives = np.asarray(objectives, dtype=float)
    normalized = normalize_objectives(objectives)
    if weights is None:
        objective_weights = np.ones(objectives.shape[1], dtype=float)
    else:
        objective_weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(objective_weights) != objectives.shape[1]:
            raise ValueError("Compromise weights must match the objective dimension")
        if np.any(objective_weights < 0.0) or not np.any(objective_weights > 0.0):
            raise ValueError("Compromise weights must be non-negative and non-zero")
    objective_weights /= objective_weights.sum()
    distance = np.sqrt(
        np.sum(objective_weights[None, :] * normalized ** 2, axis=1)
    )
    return int(np.argmin(distance))


def reference_direction_indices(
    objectives: np.ndarray,
    constraint_violation: np.ndarray,
    count: int,
    *,
    communication_weight_min: float = 0.25,
    communication_weight_max: float = 0.95,
) -> np.ndarray:
    """Select parents across length/communication trade-off directions."""
    objectives = np.asarray(objectives, dtype=float)
    violation = np.asarray(constraint_violation, dtype=float).reshape(-1)
    count = int(count)
    if objectives.ndim != 2 or objectives.shape[1] != 2:
        raise ValueError("Reference-direction selection requires two objectives")
    if len(objectives) != len(violation) or len(objectives) == 0:
        raise ValueError("Objectives and constraint violations have incompatible shapes")
    if count < 1:
        return np.array([], dtype=int)
    if not 0.0 <= communication_weight_min <= communication_weight_max <= 1.0:
        raise ValueError("Communication weight range must lie within [0, 1]")

    feasible = np.flatnonzero(violation <= 1e-9)
    pool = feasible if len(feasible) else np.arange(len(objectives), dtype=int)
    normalized = normalize_objectives(objectives[pool])
    if not len(feasible):
        scaled_violation = violation[pool] / max(
            float(np.max(violation[pool])), 1e-12
        )
        normalized = np.column_stack([normalized, scaled_violation])

    weights = np.linspace(
        float(communication_weight_min),
        float(communication_weight_max),
        count,
    )
    selected = []
    for communication_weight in weights:
        objective_weights = np.asarray(
            [1.0 - communication_weight, communication_weight], dtype=float
        )
        weighted = normalized[:, :2] * (objective_weights[None, :] + 1e-3)
        score = np.max(weighted, axis=1) + 0.05 * np.sum(weighted, axis=1)
        if normalized.shape[1] > 2:
            score += 1000.0 * normalized[:, 2]
        selected.append(int(pool[int(np.argmin(score))]))
    return np.asarray(selected, dtype=int)

def diverse_front_indices(objectives: np.ndarray, count: int) -> np.ndarray:
    """Farthest-point truncation that preserves every Pareto extreme."""
    objectives = np.asarray(objectives, dtype=float)
    count = int(count)
    if objectives.ndim != 2 or len(objectives) == 0:
        raise ValueError("At least one objective vector is required")
    if count < 1:
        return np.array([], dtype=int)
    if len(objectives) <= count:
        return np.arange(len(objectives), dtype=int)

    normalized = normalize_objectives(objectives)
    boundary = []
    for objective_index in range(objectives.shape[1]):
        boundary.extend(
            [
                int(np.argmin(objectives[:, objective_index])),
                int(np.argmax(objectives[:, objective_index])),
            ]
        )
    selected = list(dict.fromkeys(boundary))[:count]
    available = np.ones(len(objectives), dtype=bool)
    available[selected] = False
    while len(selected) < count:
        candidates = np.flatnonzero(available)
        distance = np.linalg.norm(
            normalized[candidates, None, :] - normalized[np.asarray(selected)][None, :, :],
            axis=2,
        )
        chosen = int(candidates[int(np.argmax(np.min(distance, axis=1)))])
        selected.append(chosen)
        available[chosen] = False
    return np.asarray(selected, dtype=int)


def stratified_reference_direction_indices(
    objectives: np.ndarray,
    constraint_violation: np.ndarray,
    count: int,
    *,
    rng=None,
    direction_count: int = 11,
    communication_weight_min: float = 0.25,
    communication_weight_max: float = 0.95,
) -> np.ndarray:
    """Allocate parents across objective extremes and interior trade-off strata."""
    objectives = np.asarray(objectives, dtype=float)
    violation = np.asarray(constraint_violation, dtype=float).reshape(-1)
    count = int(count)
    direction_count = max(3, int(direction_count))
    if objectives.ndim != 2 or objectives.shape[1] != 2:
        raise ValueError("Stratified selection requires two objectives")
    if len(objectives) != len(violation) or len(objectives) == 0:
        raise ValueError("Objectives and constraint violations have incompatible shapes")
    if count < 1:
        return np.array([], dtype=int)
    if not 0.0 <= communication_weight_min <= communication_weight_max <= 1.0:
        raise ValueError("Communication weight range must lie within [0, 1]")
    if rng is None:
        rng = np.random.default_rng(0)

    feasible = np.flatnonzero(violation <= 1e-9)
    if len(feasible):
        pool = feasible
    else:
        minimum = float(np.min(violation))
        tolerance = max(1e-9, 0.05 * max(minimum, 1.0))
        pool = np.flatnonzero(violation <= minimum + tolerance)
    normalized = normalize_objectives(objectives[pool])

    interior_count = max(1, direction_count - 2)
    directions = np.concatenate(
        [
            [0.0],
            np.linspace(
                float(communication_weight_min),
                float(communication_weight_max),
                interior_count,
            ),
            [1.0],
        ]
    )
    directions = np.unique(directions)
    rankings = []
    for communication_weight in directions:
        weights = np.asarray(
            [1.0 - communication_weight, communication_weight], dtype=float
        )
        if communication_weight <= 1e-12:
            score = normalized[:, 0]
        elif communication_weight >= 1.0 - 1e-12:
            score = normalized[:, 1]
        else:
            weighted = normalized * (weights[None, :] + 1e-3)
            score = np.max(weighted, axis=1) + 0.05 * np.sum(weighted, axis=1)
        rankings.append(np.argsort(score, kind="stable"))

    selected_local: list[int] = []
    usage = np.zeros(len(pool), dtype=int)
    for ranking in rankings:
        unused = [int(index) for index in ranking if usage[int(index)] == 0]
        chosen = unused[0] if unused else int(ranking[0])
        selected_local.append(chosen)
        usage[chosen] += 1
        if len(selected_local) >= count:
            return pool[np.asarray(selected_local, dtype=int)]

    slot = 0
    while len(selected_local) < count:
        ranking = rankings[slot % len(rankings)]
        top = ranking[: min(8, len(ranking))]
        adjusted = np.arange(len(top), dtype=float) + 0.75 * usage[top]
        adjusted += rng.uniform(0.0, 1e-6, size=len(top))
        chosen = int(top[int(np.argmin(adjusted))])
        selected_local.append(chosen)
        usage[chosen] += 1
        slot += 1
    return pool[np.asarray(selected_local, dtype=int)]

def hypervolume_2d(objectives: np.ndarray, reference: tuple[float, float]) -> float:
    """Exact dominated hypervolume for a two-objective minimization front."""
    points = np.asarray(objectives, dtype=float)
    reference_point = np.asarray(reference, dtype=float).reshape(2)
    if points.size == 0:
        return 0.0
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("hypervolume_2d expects an array with shape (N, 2)")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    points = points[np.all(points < reference_point, axis=1)]
    if len(points) == 0:
        return 0.0

    keep = feasible_pareto_mask(points, np.zeros(len(points), dtype=float))
    front = points[keep]
    front = front[np.argsort(front[:, 0])]
    volume = 0.0
    current_y = float(reference_point[1])
    for x_value, y_value in front:
        if y_value >= current_y:
            continue
        volume += max(0.0, float(reference_point[0] - x_value)) * (
            current_y - float(y_value)
        )
        current_y = float(y_value)
    return float(volume)
