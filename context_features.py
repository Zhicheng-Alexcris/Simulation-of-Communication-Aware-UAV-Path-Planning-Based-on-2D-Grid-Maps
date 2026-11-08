"""Rotation-aligned obstacle and radio context for trajectory conditioning."""

from __future__ import annotations

import numpy as np

from .trajectory_frame import LocalTrajectoryFrame


LONGITUDINAL_SAMPLES = 9
LATERAL_OFFSETS = np.asarray([-0.35, -0.175, 0.0, 0.175, 0.35], dtype=float)


def _context_points(frame: LocalTrajectoryFrame) -> np.ndarray:
    longitudinal = np.linspace(0.1, 0.9, LONGITUDINAL_SAMPLES)
    local_end = frame.world_to_local(frame.end)
    points = []
    for progress in longitudinal:
        for lateral in LATERAL_OFFSETS:
            points.append(
                [
                    progress * local_end[0],
                    lateral,
                    progress * local_end[2],
                ]
            )
    return frame.local_to_world(np.asarray(points, dtype=float))


def _signed_distance_to_buildings(points: np.ndarray, buildings) -> np.ndarray:
    if not buildings:
        return np.full(len(points), np.inf, dtype=float)
    bounds = np.asarray(
        [
            [
                building.x_bounds[0],
                building.x_bounds[1],
                building.y_bounds[0],
                building.y_bounds[1],
                building.z_bounds[0],
                building.z_bounds[1],
            ]
            for building in buildings
        ],
        dtype=float,
    )
    lower = bounds[:, [0, 2, 4]]
    upper = bounds[:, [1, 3, 5]]
    point_matrix = points[:, None, :]
    outside_delta = np.maximum(
        np.maximum(lower[None, :, :] - point_matrix, 0.0),
        point_matrix - upper[None, :, :],
    )
    outside_distance = np.linalg.norm(outside_delta, axis=2)
    inside = np.all(
        (point_matrix >= lower[None, :, :])
        & (point_matrix <= upper[None, :, :]),
        axis=2,
    )
    distance_to_face = np.min(
        np.minimum(
            point_matrix - lower[None, :, :],
            upper[None, :, :] - point_matrix,
        ),
        axis=2,
    )
    signed = np.where(inside, -distance_to_face, outside_distance)
    return np.min(signed, axis=1)


def _signed_environment_clearance(
    points: np.ndarray,
    env,
    layer_index: int,
    safety_radius: float,
) -> np.ndarray:
    building_clearance = _signed_distance_to_buildings(points, env.buildings)
    layer = env.safe_layers[int(layer_index)]
    boundary_clearance = np.min(
        np.column_stack(
            [
                points[:, 0] - env.xlim[0],
                env.xlim[1] - points[:, 0],
                points[:, 1] - env.ylim[0],
                env.ylim[1] - points[:, 1],
                points[:, 2] - layer.z_min,
                layer.z_max - points[:, 2],
            ]
        ),
        axis=1,
    )
    obstacle_margin = building_clearance - float(safety_radius)
    return np.minimum(obstacle_margin, boundary_clearance)


def local_context_condition(
    frame: LocalTrajectoryFrame,
    env,
    layer_index: int,
    radio_map,
    *,
    safety_radius: float,
) -> np.ndarray:
    """Return endpoint geometry plus local SDF and radio-cost grid features."""
    points = _context_points(frame)
    clearance_scale = max(10.0, 0.2 * frame.scale)
    clearance = np.clip(
        _signed_environment_clearance(
            points, env, layer_index, float(safety_radius)
        )
        / clearance_scale,
        -1.0,
        1.0,
    )

    if radio_map is None:
        radio_cost = np.zeros(len(points), dtype=float)
    else:
        layer = env.safe_layers[int(layer_index)]
        radio_points = points.copy()
        radio_points[:, 0] = np.clip(radio_points[:, 0], *env.xlim)
        radio_points[:, 1] = np.clip(radio_points[:, 1], *env.ylim)
        radio_points[:, 2] = np.clip(
            radio_points[:, 2], layer.z_min, layer.z_max
        )
        radio_cost = np.asarray(
            radio_map.normalized_cost(radio_points), dtype=float
        )

    return np.concatenate(
        [frame.condition.astype(float), clearance, radio_cost]
    ).astype(np.float32)
