"""Sampling-based feasibility projection for diffusion trajectories.

This module never evaluates the radio map and never selects the final
trajectory. It only supplies a few collision-free control-point seeds when a
fully denoised diffusion population contains no feasible member.
"""

from __future__ import annotations

import numpy as np


def _point_free(env, point: np.ndarray, margin: float) -> bool:
    x, y, z = point
    if not (env.xlim[0] <= x <= env.xlim[1] and env.ylim[0] <= y <= env.ylim[1]):
        return False
    return not any(
        building.contains_with_margin((x, y, z), margin)
        for building in env.buildings
    )


def _segment_free(env, start: np.ndarray, end: np.ndarray, margin: float) -> bool:
    distance = float(np.linalg.norm(end - start))
    sample_count = max(2, int(np.ceil(distance / 2.0)) + 1)
    for ratio in np.linspace(0.0, 1.0, sample_count):
        point = (1.0 - ratio) * start + ratio * end
        if not _point_free(env, point, margin):
            return False
    return True


class _Tree:
    def __init__(self, root: np.ndarray) -> None:
        self.points = [np.asarray(root, dtype=float)]
        self.parents = [-1]

    def nearest(self, target: np.ndarray) -> int:
        points = np.asarray(self.points)
        return int(np.argmin(np.linalg.norm(points[:, :2] - target[:2], axis=1)))

    def add(self, point: np.ndarray, parent: int) -> int:
        self.points.append(np.asarray(point, dtype=float))
        self.parents.append(parent)
        return len(self.points) - 1

    def trace(self, index: int) -> list[np.ndarray]:
        path = []
        while index >= 0:
            path.append(self.points[index])
            index = self.parents[index]
        return path[::-1]


def _extend(env, tree: _Tree, target: np.ndarray, margin: float, step: float):
    parent = tree.nearest(target)
    source = tree.points[parent]
    delta = np.asarray(target, dtype=float) - source
    distance = float(np.linalg.norm(delta[:2]))
    if distance <= 1e-9:
        return "reached", parent
    point = source + delta * min(1.0, step / distance)
    if not _segment_free(env, source, point, margin):
        return "trapped", parent
    index = tree.add(point, parent)
    return ("reached" if distance <= step else "advanced"), index


def _connect(env, tree: _Tree, target: np.ndarray, margin: float, step: float):
    while True:
        status, index = _extend(env, tree, target, margin, step)
        if status != "advanced":
            return status, index


def _rrt_connect(
    env,
    start: np.ndarray,
    goal: np.ndarray,
    rng,
    margin: float,
    *,
    iterations: int = 12000,
    step: float = 8.0,
) -> np.ndarray | None:
    left = _Tree(start)
    right = _Tree(goal)
    left_is_start = True
    for _ in range(iterations):
        if rng.random() < 0.15:
            sample = right.points[0]
        else:
            sample = np.asarray(
                [rng.uniform(*env.xlim), rng.uniform(*env.ylim), start[2]],
                dtype=float,
            )
        status, left_index = _extend(env, left, sample, margin, step)
        if status != "trapped":
            target = left.points[left_index]
            other_status, right_index = _connect(env, right, target, margin, step)
            if other_status == "reached":
                left_path = left.trace(left_index)
                right_path = right.trace(right_index)
                if left_is_start:
                    return np.asarray(left_path + right_path[-2::-1], dtype=float)
                return np.asarray(right_path + left_path[-2::-1], dtype=float)
        left, right = right, left
        left_is_start = not left_is_start
    return None


def _shortcut(env, path: np.ndarray, rng, margin: float) -> np.ndarray:
    points = [np.asarray(point, dtype=float) for point in path]
    for _ in range(200):
        if len(points) <= 2:
            break
        left, right = sorted(rng.integers(0, len(points), size=2))
        if right <= left + 1:
            continue
        if _segment_free(env, points[left], points[right], margin):
            points = points[: left + 1] + points[right:]
    return np.asarray(points, dtype=float)


def _resample(path: np.ndarray, count: int) -> np.ndarray:
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    )
    query = np.linspace(0.0, cumulative[-1], int(count))
    return np.column_stack(
        [np.interp(query, cumulative, path[:, dimension]) for dimension in range(3)]
    )


def topology_seed_candidate(
    env,
    start: np.ndarray,
    goal: np.ndarray,
    rng,
    *,
    control_point_count: int,
    safety_radius: float,
    path_sample_count: int = 120,
    search_buffer_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return latent controls and the verified piecewise-linear seed path."""
    # Keep a 1 m search buffer above the enforced pipe radius. Larger fixed
    # margins incorrectly close the narrow but valid route used by Drone 9.
    margin = float(safety_radius) + max(0.0, float(search_buffer_m))
    polyline = _rrt_connect(env, start, goal, rng, margin)
    if polyline is None:
        return None
    shortened = _shortcut(env, polyline, rng, margin)
    controls = _resample(shortened, control_point_count)
    safe_path = _resample(shortened, path_sample_count)
    controls[0] = start
    controls[-1] = goal
    safe_path[0] = start
    safe_path[-1] = goal
    if not all(
        _segment_free(env, left, right, float(safety_radius))
        for left, right in zip(safe_path[:-1], safe_path[1:])
    ):
        return None
    return controls.astype(np.float32), safe_path.astype(float)


def topology_seed_control_points(
    env,
    start: np.ndarray,
    goal: np.ndarray,
    rng,
    *,
    control_point_count: int,
    safety_radius: float,
) -> np.ndarray | None:
    candidate = topology_seed_candidate(
        env,
        start,
        goal,
        rng,
        control_point_count=control_point_count,
        safety_radius=safety_radius,
        path_sample_count=max(120, int(control_point_count)),
    )
    return None if candidate is None else candidate[0]