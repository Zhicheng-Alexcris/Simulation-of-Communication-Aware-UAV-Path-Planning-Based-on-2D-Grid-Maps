"""A*-free generation and validation of legal trajectory-prior data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .context_features import local_context_condition
from .evaluation import evaluate_trajectory
from .legacy import load_legacy_modules
from .topology_repair import topology_seed_candidate
from .trajectory_frame import LocalTrajectoryFrame


GENERATOR_VERSION = "local_context_perturbed_replay_v7_30cp"
SUPPORTED_GENERATOR_VERSIONS = {
    "topology_rejection_v2_30cp",
    "endpoint_residual_v3_30cp",
    "local_frame_v4_30cp",
    "local_context_v5_30cp",
    "local_context_replay_v6_30cp",
    GENERATOR_VERSION,
}


def _point_clearance(point: np.ndarray, buildings) -> float:
    distances = []
    for building in buildings:
        dx = max(building.x_bounds[0] - point[0], 0.0, point[0] - building.x_bounds[1])
        dy = max(building.y_bounds[0] - point[1], 0.0, point[1] - building.y_bounds[1])
        dz = max(building.z_bounds[0] - point[2], 0.0, point[2] - building.z_bounds[1])
        distances.append(float(np.sqrt(dx * dx + dy * dy + dz * dz)))
    return min(distances) if distances else float("inf")


def _sample_free_endpoint(env, layer_index: int, safety_radius: float, rng) -> np.ndarray:
    layer = env.safe_layers[layer_index]
    altitude = float(layer.mid)
    for _ in range(1000):
        point = np.array(
            [
                rng.uniform(env.xlim[0] + 2.0, env.xlim[1] - 2.0),
                rng.uniform(env.ylim[0] + 2.0, env.ylim[1] - 2.0),
                altitude,
            ],
            dtype=float,
        )
        if _point_clearance(point, env.buildings) >= safety_radius:
            return point
    raise RuntimeError("Could not sample a collision-free endpoint")


def _sample_control_points(
    start: np.ndarray,
    end: np.ndarray,
    env,
    layer_index: int,
    control_point_count: int,
    rng,
) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, control_point_count)
    controls = start[None, :] + progress[:, None] * (end - start)[None, :]
    horizontal = end[:2] - start[:2]
    horizontal_norm = np.linalg.norm(horizontal) + 1e-12
    normal = np.array([-horizontal[1], horizontal[0]]) / horizontal_norm

    amplitude_primary = rng.uniform(-85.0, 85.0)
    amplitude_secondary = rng.uniform(-25.0, 25.0)
    lateral = np.sin(np.pi * progress) * (
        amplitude_primary + amplitude_secondary * np.sin(2.0 * np.pi * progress)
    )
    controls[:, :2] += lateral[:, None] * normal[None, :]
    controls[1:-1, :2] += rng.normal(0.0, 3.0, size=(control_point_count - 2, 2))

    layer = env.safe_layers[layer_index]
    vertical_amplitude = min(0.4 * (layer.z_max - layer.z_min), 1.5)
    controls[:, 2] = layer.mid + rng.uniform(
        -vertical_amplitude, vertical_amplitude, size=control_point_count
    ) * np.sin(np.pi * progress)
    controls[:, 0] = np.clip(controls[:, 0], env.xlim[0], env.xlim[1])
    controls[:, 1] = np.clip(controls[:, 1], env.ylim[0], env.ylim[1])
    controls[:, 2] = np.clip(controls[:, 2], layer.z_min, layer.z_max)
    controls[0] = start
    controls[-1] = end
    return controls


def _turn_p95(path: np.ndarray) -> float:
    segment = np.diff(path, axis=0)
    norm = np.linalg.norm(segment, axis=1)
    valid = norm > 1e-6
    segment = segment[valid]
    if len(segment) < 2:
        return 0.0
    unit = segment / np.linalg.norm(segment, axis=1, keepdims=True)
    cosine = np.sum(unit[:-1] * unit[1:], axis=1)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.percentile(angles, 95))


def _hard_replay_cases(env, hard_drone_ids: tuple[int, ...]):
    cases = []
    layer_midpoints = [float(layer.mid) for layer in env.safe_layers]
    for drone_id in hard_drone_ids:
        if drone_id < 0 or drone_id >= len(env.starts):
            continue
        start_raw, end_raw = env.get_start_end(int(drone_id))
        validation_layer = (
            int(drone_id)
            if int(drone_id) < len(env.safe_layers)
            else int(drone_id) % len(env.safe_layers)
        )
        nearest_layer = int(
            np.argmin(
                [abs(float(start_raw[2]) - midpoint) for midpoint in layer_midpoints]
            )
        )
        for layer_index in sorted({validation_layer, nearest_layer}):
            start = np.asarray(
                [
                    float(start_raw[0]),
                    float(start_raw[1]),
                    env.clamp_to_layer(
                        max(float(start_raw[2]), env.min_height), layer_index
                    ),
                ],
                dtype=float,
            )
            end = np.asarray(
                [
                    float(end_raw[0]),
                    float(end_raw[1]),
                    env.clamp_to_layer(
                        max(float(end_raw[2]), env.min_height), layer_index
                    ),
                ],
                dtype=float,
            )
            cases.append((int(drone_id), layer_index, start, end))
    return cases


def _perturb_replay_endpoint(
    base: np.ndarray,
    env,
    layer_index: int,
    safety_radius: float,
    rng,
    *,
    minimum_offset_m: float,
    maximum_offset_m: float,
) -> np.ndarray | None:
    layer = env.safe_layers[int(layer_index)]
    for _ in range(300):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(float(minimum_offset_m), float(maximum_offset_m))
        candidate = np.asarray(base, dtype=float).copy()
        candidate[:2] += radius * np.asarray([np.cos(angle), np.sin(angle)])
        candidate[2] = float(np.clip(candidate[2], layer.z_min, layer.z_max))
        if (
            candidate[0] < env.xlim[0] + 2.0
            or candidate[0] > env.xlim[1] - 2.0
            or candidate[1] < env.ylim[0] + 2.0
            or candidate[1] > env.ylim[1] - 2.0
        ):
            continue
        if np.linalg.norm(candidate[:2] - np.asarray(base)[:2]) < minimum_offset_m:
            continue
        if _point_clearance(candidate, env.buildings) < float(safety_radius) + 0.5:
            continue
        return candidate
    return None

def _generate_hard_replays(
    env,
    planner_module,
    radio_map,
    rng,
    *,
    target_count: int,
    hard_drone_ids: tuple[int, ...],
    control_point_count: int,
    safety_radius: float,
    replay_minimum_offset_m: float,
    replay_maximum_offset_m: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], int]:
    cases = _hard_replay_cases(env, hard_drone_ids)
    if target_count <= 0:
        return [], [], [], 0
    if not cases:
        raise ValueError("No valid hard-drone replay cases were configured")

    trajectories = []
    conditions = []
    layer_indices = []
    attempts = 0
    maximum_attempts = max(120, int(target_count) * 80)
    search_buffers = (4.0, 2.0, 1.0)
    while len(trajectories) < target_count and attempts < maximum_attempts:
        _, layer_index, base_start, base_end = cases[attempts % len(cases)]
        search_buffer = search_buffers[(attempts // len(cases)) % len(search_buffers)]
        attempts += 1
        start = _perturb_replay_endpoint(
            base_start,
            env,
            layer_index,
            safety_radius,
            rng,
            minimum_offset_m=float(replay_minimum_offset_m),
            maximum_offset_m=float(replay_maximum_offset_m),
        )
        end = _perturb_replay_endpoint(
            base_end,
            env,
            layer_index,
            safety_radius,
            rng,
            minimum_offset_m=float(replay_minimum_offset_m),
            maximum_offset_m=float(replay_maximum_offset_m),
        )
        if start is None or end is None:
            continue
        if np.linalg.norm(end[:2] - start[:2]) < 70.0:
            continue
        candidate = topology_seed_candidate(
            env,
            start,
            end,
            rng,
            control_point_count=control_point_count,
            safety_radius=float(safety_radius),
            path_sample_count=180,
            search_buffer_m=search_buffer,
        )
        if candidate is None:
            continue
        controls, _ = candidate
        planner = planner_module.SplinePathPlanner(
            env,
            n_control_points=control_point_count,
            pipe_radius=float(safety_radius),
        )
        planner.set_cruise_layer(layer_index)
        path = np.asarray(
            planner.create_b_spline(controls, n_samples=180), dtype=float
        )
        layer = env.safe_layers[layer_index]
        tolerance = 1e-7
        if (
            not np.isfinite(path).all()
            or np.any(path[:, 0] < env.xlim[0] - tolerance)
            or np.any(path[:, 0] > env.xlim[1] + tolerance)
            or np.any(path[:, 1] < env.ylim[0] - tolerance)
            or np.any(path[:, 1] > env.ylim[1] + tolerance)
            or np.any(path[:, 2] < layer.z_min - tolerance)
            or np.any(path[:, 2] > layer.z_max + tolerance)
        ):
            continue
        metrics = evaluate_trajectory(
            path,
            env,
            radio_map,
            pipe_radius=float(safety_radius),
            outage_threshold_db=110.0,
            sample_count=180,
        )
        if (
            metrics.constraint_violation > 1e-9
            or metrics.turn_p95_deg > 35.0
        ):
            continue
        channel_acceptance = float(
            np.clip(1.1 - 0.7 * metrics.communication_cost, 0.35, 1.0)
        )
        if rng.random() > channel_acceptance:
            continue
        trajectories.append(np.asarray(controls, dtype=np.float32).reshape(-1))
        conditions.append(np.concatenate([start, end]).astype(np.float32))
        layer_indices.append(int(layer_index))

    if len(trajectories) < target_count:
        raise RuntimeError(
            f"Generated {len(trajectories)}/{target_count} hard-topology replays "
            f"after {attempts} attempts"
        )
    return trajectories, conditions, layer_indices, attempts

def generate_legal_dataset(
    legacy_root: str | Path,
    output_path: str | Path,
    *,
    sample_count: int = 1200,
    control_point_count: int = 30,
    safety_radius: float = 3.5,
    radio_map=None,
    seed: int = 0,
    hard_replay_fraction: float = 0.15,
    hard_drone_ids: tuple[int, ...] = (0, 1, 5, 6, 8),
    replay_minimum_offset_m: float = 12.0,
    replay_maximum_offset_m: float = 35.0,
    minimum_endpoint_distance: float = 70.0,
    max_attempt_factor: int = 300,
) -> Path:
    """Generate legal trajectories with local obstacle and radio context."""
    environment_module, planner_module = load_legacy_modules(legacy_root)
    env = environment_module.Environment()
    if radio_map is None:
        raise ValueError("radio_map is required for the local-context prior")
    if not 0.0 <= float(hard_replay_fraction) < 1.0:
        raise ValueError("hard_replay_fraction must lie within [0, 1)")
    if not 0.0 < float(replay_minimum_offset_m) <= float(replay_maximum_offset_m):
        raise ValueError("Replay offsets must satisfy 0 < minimum <= maximum")
    rng = np.random.default_rng(seed)
    replay_target = int(round(int(sample_count) * float(hard_replay_fraction)))
    if hard_replay_fraction > 0.0 and sample_count > 0:
        replay_target = max(1, replay_target)
    random_target = int(sample_count) - replay_target
    trajectories: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    layer_indices: list[int] = []
    replay_flags: list[bool] = []
    attempts = 0
    max_attempts = max(1, random_target) * int(max_attempt_factor)

    while len(trajectories) < random_target and attempts < max_attempts:
        attempts += 1
        layer_index = int(rng.integers(0, len(env.safe_layers)))
        start = _sample_free_endpoint(env, layer_index, safety_radius, rng)
        end = _sample_free_endpoint(env, layer_index, safety_radius, rng)
        if np.linalg.norm(end[:2] - start[:2]) < minimum_endpoint_distance:
            continue
        controls = _sample_control_points(
            start, end, env, layer_index, control_point_count, rng
        )
        planner = planner_module.SplinePathPlanner(
            env, n_control_points=control_point_count, pipe_radius=safety_radius
        )
        planner.set_cruise_layer(layer_index)
        path = np.asarray(planner.create_b_spline(controls, n_samples=180), dtype=float)
        if not np.isfinite(path).all():
            continue
        if np.any(path[:, 0] < env.xlim[0]) or np.any(path[:, 0] > env.xlim[1]):
            continue
        if np.any(path[:, 1] < env.ylim[0]) or np.any(path[:, 1] > env.ylim[1]):
            continue
        layer = env.safe_layers[layer_index]
        if np.any(path[:, 2] < layer.z_min) or np.any(path[:, 2] > layer.z_max):
            continue
        metrics = evaluate_trajectory(
            path,
            env,
            None,
            pipe_radius=safety_radius,
            outage_threshold_db=110.0,
            sample_count=180,
        )
        if (
            metrics.collision_count > 0
            or metrics.min_clearance_m < safety_radius
            or metrics.turn_p95_deg > 35.0
        ):
            continue
        trajectories.append(controls.astype(np.float32).reshape(-1))
        conditions.append(np.concatenate([start, end]).astype(np.float32))
        layer_indices.append(layer_index)
        replay_flags.append(False)

    if len(trajectories) < random_target:
        raise RuntimeError(
            f"Generated {len(trajectories)}/{random_target} random legal trajectories "
            f"after {attempts} attempts"
        )
    replay_trajectories, replay_conditions, replay_layers, replay_attempts = (
        _generate_hard_replays(
            env,
            planner_module,
            radio_map,
            rng,
            target_count=replay_target,
            hard_drone_ids=tuple(int(value) for value in hard_drone_ids),
            control_point_count=control_point_count,
            safety_radius=float(safety_radius),
            replay_minimum_offset_m=float(replay_minimum_offset_m),
            replay_maximum_offset_m=float(replay_maximum_offset_m),
        )
    )
    trajectories.extend(replay_trajectories)
    conditions.extend(replay_conditions)
    layer_indices.extend(replay_layers)
    replay_flags.extend([True] * len(replay_trajectories))

    raw_trajectories = np.asarray(trajectories, dtype=np.float32)
    raw_conditions = np.asarray(conditions, dtype=np.float32)
    controls = raw_trajectories.reshape(-1, control_point_count, 3)
    frames = [
        LocalTrajectoryFrame.from_endpoints(condition[:3], condition[3:])
        for condition in raw_conditions
    ]
    model_trajectories = np.asarray(
        [frame.encode_residual(item) for frame, item in zip(frames, controls)],
        dtype=np.float32,
    )
    model_conditions = np.asarray(
        [
            local_context_condition(
                frame,
                env,
                int(layer_index),
                radio_map,
                safety_radius=float(safety_radius),
            )
            for frame, layer_index in zip(frames, layer_indices)
        ],
        dtype=np.float32,
    )
    trajectory_mean = model_trajectories.mean(axis=0)
    trajectory_std = model_trajectories.std(axis=0) + 1e-6
    condition_mean = model_conditions.mean(axis=0)
    condition_std = model_conditions.std(axis=0) + 1e-6

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=(model_trajectories - trajectory_mean) / trajectory_std,
        C=(model_conditions - condition_mean) / condition_std,
        Xraw=raw_trajectories,
        Xmodelraw=model_trajectories,
        Craw=model_conditions,
        endpoint_world=raw_conditions,
        Xmean=trajectory_mean,
        Xstd=trajectory_std,
        Cmean=condition_mean,
        Cstd=condition_std,
        layer_index=np.asarray(layer_indices, dtype=np.int16),
        is_hard_replay=np.asarray(replay_flags, dtype=bool),
        hard_replay_count=np.asarray(len(replay_trajectories), dtype=np.int32),
        hard_replay_fraction=np.asarray(float(hard_replay_fraction)),
        hard_drone_ids=np.asarray(hard_drone_ids, dtype=np.int16),
        replay_minimum_offset_m=np.asarray(float(replay_minimum_offset_m)),
        replay_maximum_offset_m=np.asarray(float(replay_maximum_offset_m)),
        replay_excludes_exact_endpoints=np.asarray(True),
        generator_version=np.asarray(GENERATOR_VERSION),
        trajectory_representation=np.asarray("local_frame_residual_v1"),
        condition_representation=np.asarray("local_sdf_radio_grid_v1"),
        context_longitudinal_samples=np.asarray(9, dtype=np.int16),
        context_lateral_samples=np.asarray(5, dtype=np.int16),
        uses_astar=np.asarray(False),
        safety_radius=np.asarray(float(safety_radius)),
        attempts=np.asarray(attempts + replay_attempts),
        random_attempts=np.asarray(attempts),
        replay_attempts=np.asarray(replay_attempts),
    )
    return output_path


def load_prior_dataset(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    required = {"X", "C", "Xmean", "Xstd", "Cmean", "Cstd", "generator_version"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Prior dataset is missing fields: {sorted(missing)}")
    version = str(data["generator_version"].item())
    if version not in SUPPORTED_GENERATOR_VERSIONS or bool(
        data.get("uses_astar", True)
    ):
        raise ValueError("Refusing an A*-generated or unsupported trajectory prior")
    return {key: data[key] for key in data.files}
