"""Common black-box evaluation shared by all formal experiment methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryMetrics:
    path_length_m: float
    communication_cost: float
    mean_pathloss_db: float
    worst_pathloss_db: float
    pathloss_p95_db: float
    outage_ratio: float
    collision_count: int
    min_clearance_m: float
    avg_clearance_m: float
    turn_mean_deg: float
    turn_p95_deg: float
    turn_max_deg: float
    straightness_ratio: float
    constraint_violation: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def path_length(path: np.ndarray) -> float:
    points = np.asarray(path, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def resample_path(path: np.ndarray, sample_count: int = 120) -> np.ndarray:
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"Expected a trajectory with shape (N, 3), got {points.shape}")
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    if cumulative[-1] <= 1e-12:
        return np.repeat(points[:1], int(sample_count), axis=0)
    targets = np.linspace(0.0, cumulative[-1], int(sample_count))
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(3)]
    )


def _segment_samples(path: np.ndarray, spacing_m: float = 4.0) -> np.ndarray:
    points = np.asarray(path, dtype=float)
    segments = []
    for start, end in zip(points[:-1], points[1:]):
        count = max(8, int(np.ceil(np.linalg.norm(end - start) / spacing_m)) + 1)
        segments.append(
            start[None, :] + np.linspace(0.0, 1.0, count)[:, None] * (end - start)[None, :]
        )
    return np.concatenate(segments, axis=0) if segments else points.copy()


def _building_bounds(buildings) -> np.ndarray:
    if not buildings:
        return np.empty((0, 6), dtype=float)
    return np.asarray(
        [
            [
                building.x_bounds[0], building.x_bounds[1],
                building.y_bounds[0], building.y_bounds[1],
                building.z_bounds[0], building.z_bounds[1],
            ]
            for building in buildings
        ],
        dtype=float,
    )


def _building_clearance(
    points: np.ndarray,
    buildings=None,
    *,
    bounds: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    bounds = _building_bounds(buildings) if bounds is None else np.asarray(bounds, dtype=float)
    if len(bounds) == 0:
        return np.full(len(points), np.inf, dtype=float)
    x = points[:, 0:1]
    y = points[:, 1:2]
    z = points[:, 2:3]
    dx = np.maximum(np.maximum(bounds[None, :, 0] - x, 0.0), x - bounds[None, :, 1])
    dy = np.maximum(np.maximum(bounds[None, :, 2] - y, 0.0), y - bounds[None, :, 3])
    dz = np.maximum(np.maximum(bounds[None, :, 4] - z, 0.0), z - bounds[None, :, 5])
    return np.sqrt(dx * dx + dy * dy + dz * dz).min(axis=1)


def _collision_count(
    points: np.ndarray,
    buildings,
    safety_margin: float,
    *,
    bounds: np.ndarray | None = None,
) -> int:
    """Count samples inside any margin-expanded axis-aligned building."""
    points = np.asarray(points, dtype=float)
    bounds = _building_bounds(buildings) if bounds is None else np.asarray(bounds, dtype=float)
    if len(points) == 0 or len(bounds) == 0:
        return 0
    margin = float(safety_margin)
    lower = bounds[:, [0, 2, 4]] - margin
    upper = bounds[:, [1, 3, 5]] + margin
    inside = np.all(
        (points[:, None, :] >= lower[None, :, :])
        & (points[:, None, :] <= upper[None, :, :]),
        axis=2,
    )
    return int(np.count_nonzero(np.any(inside, axis=1)))


def is_trajectory_geometrically_feasible(
    path: np.ndarray,
    buildings,
    *,
    safety_margin: float,
    sample_count: int = 120,
) -> bool:
    """Fast equivalent of the collision/clearance feasibility part of evaluation."""
    sampled = resample_path(np.asarray(path, dtype=float), sample_count=sample_count)
    bounds = _building_bounds(buildings)
    collision_samples = _segment_samples(sampled)
    # An expanded-AABB collision test is at least as strict as the Euclidean
    # point-clearance constraint, so zero collisions also guarantees clearance.
    return _collision_count(
        collision_samples,
        buildings,
        safety_margin,
        bounds=bounds,
    ) == 0

def turn_angle_statistics(path: np.ndarray) -> tuple[float, float, float]:
    points = np.asarray(path, dtype=float)
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    deltas = np.diff(points[:, :2], axis=0)
    first = deltas[:-1]
    second = deltas[1:]
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    valid = (first_norm > 1e-9) & (second_norm > 1e-9)
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    cosine = np.sum(first[valid] * second[valid], axis=1) / (
        first_norm[valid] * second_norm[valid]
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return (
        float(np.mean(angles)),
        float(np.percentile(angles, 95)),
        float(np.max(angles)),
    )


def evaluate_trajectories(
    paths: list[np.ndarray] | np.ndarray,
    env,
    radio_map,
    *,
    pipe_radius: float,
    outage_threshold_db: float,
    sample_count: int = 120,
) -> list[TrajectoryMetrics]:
    """Evaluate a trajectory batch with exactly the same geometry and radio rules."""
    sampled = [
        resample_path(np.asarray(path, dtype=float), sample_count=sample_count)
        for path in paths
    ]
    if not sampled:
        return []

    lengths = np.asarray([path_length(path) for path in sampled], dtype=float)
    building_bounds = _building_bounds(env.buildings)
    collision_counts = np.asarray(
        [
            _collision_count(
                _segment_samples(path),
                env.buildings,
                pipe_radius,
                bounds=building_bounds,
            )
            for path in sampled
        ],
        dtype=int,
    )
    clearance_arrays = [
        _building_clearance(path, bounds=building_bounds) for path in sampled
    ]
    minimum_clearance = np.asarray(
        [float(np.min(values)) for values in clearance_arrays], dtype=float
    )
    average_clearance = np.asarray(
        [float(np.mean(values)) for values in clearance_arrays], dtype=float
    )

    count = len(sampled)
    if radio_map is None:
        communication_cost = np.zeros(count, dtype=float)
        mean_pathloss = np.full(count, np.nan, dtype=float)
        worst_pathloss = np.full(count, np.nan, dtype=float)
        pathloss_p95 = np.full(count, np.nan, dtype=float)
        outage_ratio = np.full(count, np.nan, dtype=float)
    else:
        flat = np.concatenate(sampled, axis=0)
        if hasattr(radio_map, "pathloss_and_normalized_cost"):
            pathloss_flat, communication_flat = (
                radio_map.pathloss_and_normalized_cost(flat)
            )
        else:
            pathloss_flat = radio_map.pathloss(flat)
            communication_flat = radio_map.normalized_cost(flat)
        pathloss_values = np.asarray(pathloss_flat, dtype=float).reshape(
            count, sample_count
        )
        communication_values = np.asarray(
            communication_flat, dtype=float
        ).reshape(count, sample_count)
        communication_cost = communication_values.mean(axis=1)
        mean_pathloss = pathloss_values.mean(axis=1)
        worst_pathloss = pathloss_values.max(axis=1)
        pathloss_p95 = np.percentile(pathloss_values, 95, axis=1)
        outage_ratio = np.mean(
            pathloss_values >= float(outage_threshold_db), axis=1
        )

    results = []
    for index, path in enumerate(sampled):
        turn_mean, turn_p95, turn_max = turn_angle_statistics(path)
        direct = float(np.linalg.norm(path[-1] - path[0]))
        straightness = float(lengths[index] / max(direct, 1e-12))
        clearance_violation = max(
            0.0, float(pipe_radius) - minimum_clearance[index]
        )
        constraint_violation = float(
            collision_counts[index] + clearance_violation
        )
        results.append(
            TrajectoryMetrics(
                path_length_m=float(lengths[index]),
                communication_cost=float(communication_cost[index]),
                mean_pathloss_db=float(mean_pathloss[index]),
                worst_pathloss_db=float(worst_pathloss[index]),
                pathloss_p95_db=float(pathloss_p95[index]),
                outage_ratio=float(outage_ratio[index]),
                collision_count=int(collision_counts[index]),
                min_clearance_m=float(minimum_clearance[index]),
                avg_clearance_m=float(average_clearance[index]),
                turn_mean_deg=turn_mean,
                turn_p95_deg=turn_p95,
                turn_max_deg=turn_max,
                straightness_ratio=straightness,
                constraint_violation=constraint_violation,
            )
        )
    return results


def evaluate_trajectory(
    path: np.ndarray,
    env,
    radio_map,
    *,
    pipe_radius: float,
    outage_threshold_db: float,
    sample_count: int = 120,
) -> TrajectoryMetrics:
    return evaluate_trajectories(
        [np.asarray(path, dtype=float)],
        env,
        radio_map,
        pipe_radius=pipe_radius,
        outage_threshold_db=outage_threshold_db,
        sample_count=sample_count,
    )[0]