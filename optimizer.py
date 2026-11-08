"""A*-free Channel-Aware Multi-Objective Diffusion-ES."""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .candidate_logging import CandidateEvaluationLog
from .objective_logging import objective_summary
from .context_features import local_context_condition
from .diffusion_model import seed_everything
from .evaluation import evaluate_trajectories, is_trajectory_geometrically_feasible
from .pareto import (
    compromise_index,
    constrained_non_dominated_sort,
    diverse_front_indices,
    feasible_pareto_mask,
    hypervolume_2d,
    stratified_reference_direction_indices,
)
from .topology_repair import topology_seed_candidate
from .trajectory_frame import LocalTrajectoryFrame


@dataclass
class CandidateBatch:
    latent: np.ndarray
    paths: list[np.ndarray]
    length: np.ndarray
    communication: np.ndarray
    mean_pathloss_db: np.ndarray
    worst_pathloss_db: np.ndarray
    constraint_violation: np.ndarray
    collision_count: np.ndarray
    min_clearance: np.ndarray
    separation_violation: np.ndarray
    topology_origin: np.ndarray

    @property
    def objectives(self) -> np.ndarray:
        return np.column_stack([self.length, self.communication])

    def take(self, indices: np.ndarray) -> "CandidateBatch":
        indices = np.asarray(indices, dtype=int)
        return CandidateBatch(
            latent=self.latent[indices],
            paths=[self.paths[index] for index in indices],
            length=self.length[indices],
            communication=self.communication[indices],
            mean_pathloss_db=self.mean_pathloss_db[indices],
            worst_pathloss_db=self.worst_pathloss_db[indices],
            constraint_violation=self.constraint_violation[indices],
            collision_count=self.collision_count[indices],
            min_clearance=self.min_clearance[indices],
            separation_violation=self.separation_violation[indices],
            topology_origin=self.topology_origin[indices],
        )

    @classmethod
    def concatenate(cls, left: "CandidateBatch", right: "CandidateBatch"):
        return cls(
            latent=np.concatenate([left.latent, right.latent], axis=0),
            paths=left.paths + right.paths,
            length=np.concatenate([left.length, right.length]),
            communication=np.concatenate([left.communication, right.communication]),
            mean_pathloss_db=np.concatenate([left.mean_pathloss_db, right.mean_pathloss_db]),
            worst_pathloss_db=np.concatenate([left.worst_pathloss_db, right.worst_pathloss_db]),
            constraint_violation=np.concatenate(
                [left.constraint_violation, right.constraint_violation]
            ),
            collision_count=np.concatenate([left.collision_count, right.collision_count]),
            min_clearance=np.concatenate([left.min_clearance, right.min_clearance]),
            separation_violation=np.concatenate(
                [left.separation_violation, right.separation_violation]
            ),
            topology_origin=np.concatenate(
                [left.topology_origin, right.topology_origin]
            ),
        )


@dataclass
class OptimizationResult:
    path: np.ndarray
    length: float
    communication_cost: float
    mean_pathloss_db: float
    worst_pathloss_db: float
    constraint_violation: float
    min_clearance: float
    collision_count: int
    history: list[dict[str, float]]
    pareto: CandidateBatch
    evaluation_count: int = 0
    topology_seed_count: int = 0
    offspring_resample_count: int = 0
    projection_count: int = 0
    candidate_history: list[dict[str, Any]] = field(default_factory=list)


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.min()) / max(float(values.max() - values.min()), 1e-12)


def _resample_by_distance(path: np.ndarray, distances: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    )
    if cumulative[-1] <= 1e-12:
        return np.repeat(path[:1], len(distances), axis=0)
    clipped = np.clip(distances, 0.0, cumulative[-1])
    return np.column_stack(
        [np.interp(clipped, cumulative, path[:, dimension]) for dimension in range(3)]
    )


def _adapt_truncation_step(
    current: int,
    minimum: int,
    maximum: int,
    *,
    projection_ratio: float,
    raw_feasible_ratio: float,
    stale_generations: int,
    projection_threshold: float,
) -> int:
    """Use feasibility feedback to keep diffusion mutations local."""
    current = int(np.clip(current, minimum, maximum))
    if projection_ratio > projection_threshold:
        return max(int(minimum), current - 1)
    if (
        projection_ratio < 0.5 * projection_threshold
        and raw_feasible_ratio >= 0.6
        and stale_generations >= 2
    ):
        return min(int(maximum), current + 1)
    return current

def _adapt_topology_noise_scale(
    current: float,
    minimum: float,
    *,
    projected_count: float,
    parent_count: float,
    projection_threshold: float,
    decay: float,
) -> float:
    """Shrink topology mutation noise when descendants require projection."""
    if parent_count <= 0.0:
        return float(current)
    projection_ratio = float(projected_count) / float(parent_count)
    if projection_ratio > float(projection_threshold):
        return max(float(minimum), float(current) * float(decay))
    return float(current)

def _communication_elite_parent_indices(
    objectives: np.ndarray,
    constraint_violation: np.ndarray,
    count: int,
    *,
    rng,
    elite_fraction: float = 0.1,
) -> np.ndarray:
    """Sample a fixed quota from the best feasible communication candidates."""
    objectives = np.asarray(objectives, dtype=float)
    violation = np.asarray(constraint_violation, dtype=float)
    feasible = np.flatnonzero(violation <= 1e-9)
    pool = feasible if len(feasible) else np.arange(len(objectives), dtype=int)
    order = pool[np.argsort(objectives[pool, 1], kind="stable")]
    elite_count = min(
        len(order), max(2, int(np.ceil(len(order) * float(elite_fraction))))
    )
    elite = order[:elite_count]
    selected = rng.choice(elite, size=int(count), replace=True)
    if count > 0:
        selected[0] = order[0]
    return np.asarray(selected, dtype=int)

class FeasibleParetoArchive:
    def __init__(self, maximum_size: int = 400) -> None:
        self.maximum_size = int(maximum_size)
        self.batch: CandidateBatch | None = None

    def update(self, candidates: CandidateBatch) -> None:
        merged = candidates if self.batch is None else CandidateBatch.concatenate(self.batch, candidates)
        mask = feasible_pareto_mask(merged.objectives, merged.constraint_violation)
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return
        selected = merged.take(indices)
        rounded = np.round(selected.objectives, decimals=9)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        selected = selected.take(np.sort(unique_indices))
        if len(selected.length) > self.maximum_size:
            selected = selected.take(
                diverse_front_indices(selected.objectives, self.maximum_size)
            )
        self.batch = selected


class CADiffusionES:
    """Constrained bi-objective Diffusion-ES with a feasible external archive."""

    def __init__(
        self,
        planner,
        start: np.ndarray,
        end: np.ndarray,
        prior,
        dataset: dict[str, np.ndarray],
        *,
        radio_map=None,
        other_paths: list[np.ndarray] | None = None,
        seed: int = 0,
        sample_points: int = 120,
        speed_mps: float = 12.0,
        minimum_separation: float | None = None,
        outage_threshold_db: float = 110.0,
    ) -> None:
        self.planner = planner
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.prior = prior
        self.radio_map = radio_map
        self.other_paths = [np.asarray(path, dtype=float) for path in (other_paths or [])]
        self.rng = np.random.default_rng(seed)
        self.sample_points = int(sample_points)
        self.speed_mps = float(speed_mps)
        self.minimum_separation = float(
            minimum_separation
            if minimum_separation is not None
            else 2.0 * float(planner.pipe_radius)
        )
        self.outage_threshold_db = float(outage_threshold_db)
        self.trajectory_mean = np.asarray(dataset["Xmean"], dtype=np.float32)
        self.trajectory_std = np.asarray(dataset["Xstd"], dtype=np.float32)
        self.condition_mean = np.asarray(dataset["Cmean"], dtype=np.float32)
        self.condition_std = np.asarray(dataset["Cstd"], dtype=np.float32)
        self.prior_trajectories = np.asarray(
            dataset.get("X", np.empty((0, self.trajectory_mean.size))),
            dtype=np.float32,
        )
        self.prior_conditions = np.asarray(
            dataset.get("C", np.empty((0, self.condition_mean.size))),
            dtype=np.float32,
        )
        self.prior_topology_origin = np.asarray(
            dataset.get(
                "is_hard_replay",
                np.zeros(len(self.prior_trajectories), dtype=bool),
            ),
            dtype=bool,
        ).reshape(-1)
        representation = dataset.get(
            "trajectory_representation", np.asarray("absolute_control_points_v1")
        )
        self.trajectory_representation = str(np.asarray(representation).item())
        condition_representation = dataset.get(
            "condition_representation", np.asarray("endpoint_world_v1")
        )
        self.condition_representation = str(
            np.asarray(condition_representation).item()
        )
        if len(self.prior_trajectories) != len(self.prior_conditions):
            raise ValueError("Prior trajectories and conditions must have equal length")
        if len(self.prior_topology_origin) != len(self.prior_trajectories):
            raise ValueError("Hard-replay flags must match prior trajectories")
        if self.trajectory_mean.size % 3 != 0:
            raise ValueError("Trajectory dimension must contain 3D control points")
        self.control_point_count = int(self.trajectory_mean.size // 3)
        self.endpoint_baseline = np.linspace(
            self.start,
            self.end,
            self.control_point_count,
        )
        self.local_frame = LocalTrajectoryFrame.from_endpoints(self.start, self.end)
        if self.condition_representation == "local_sdf_radio_grid_v1":
            condition = local_context_condition(
                self.local_frame,
                self.planner.env,
                self.planner.cruise_layer_idx,
                self.radio_map,
                safety_radius=float(self.planner.pipe_radius),
            )
        elif self.trajectory_representation == "local_frame_residual_v1":
            condition = self.local_frame.condition
        else:
            condition = np.concatenate([self.start, self.end]).astype(np.float32)
        if condition.size != self.condition_mean.size:
            raise ValueError(
                "Trajectory representation and prior condition dimensions do not match"
            )
        self.normalized_condition = (
            (condition - self.condition_mean) / self.condition_std
        ).astype(np.float32)
        seed_everything(seed)
        self.evaluation_count = 0
        self.topology_seed_count = 0
        self.offspring_resample_count = 0
        self.projection_count = 0
        self._feasibility_cache: OrderedDict[bytes, bool] = OrderedDict()
        self._feasibility_cache_size = 8192

    def _model_vector_to_controls(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1, 3)
        if self.trajectory_representation == "endpoint_residual_v1":
            controls = values + self.endpoint_baseline
        elif self.trajectory_representation == "local_frame_residual_v1":
            controls = self.local_frame.decode_residual(values)
        else:
            controls = values
        return self._clip_control_points(controls)

    def _controls_to_model_vector(self, controls: np.ndarray) -> np.ndarray:
        values = np.asarray(controls, dtype=float)
        if self.trajectory_representation == "endpoint_residual_v1":
            values = values - self.endpoint_baseline
        elif self.trajectory_representation == "local_frame_residual_v1":
            return self.local_frame.encode_residual(values)
        return values.reshape(-1)

    def _sample_initial_latent_population(
        self,
        population_size: int,
        *,
        anchor_fraction: float,
        anchor_truncation_step: int,
        return_topology_origin: bool = False,
    ):
        """Mix DDPM samples with nearest anchors and preserve replay provenance."""
        population_size = int(population_size)
        anchor_fraction = float(np.clip(anchor_fraction, 0.0, 1.0))
        anchor_count = min(
            population_size,
            int(round(population_size * anchor_fraction)),
            len(self.prior_trajectories),
        )
        diffusion_count = population_size - anchor_count
        parts = []
        origin_parts = []
        if diffusion_count:
            parts.append(
                np.asarray(
                    self.prior.sample(self.normalized_condition, diffusion_count),
                    dtype=np.float32,
                )
            )
            origin_parts.append(np.zeros(diffusion_count, dtype=bool))
        if anchor_count:
            distance = np.linalg.norm(
                self.prior_conditions - self.normalized_condition[None, :],
                axis=1,
            )
            pool_count = min(
                len(distance),
                max(anchor_count, anchor_count * 4),
            )
            nearest = np.argsort(distance, kind="stable")[:pool_count]
            rank = np.arange(pool_count, dtype=float)
            probabilities = np.exp(-rank / max(1.0, 0.5 * anchor_count))
            probabilities /= probabilities.sum()
            chosen = self.rng.choice(
                nearest,
                size=anchor_count,
                replace=pool_count < anchor_count,
                p=probabilities,
            )
            anchors = self.prior_trajectories[chosen].copy()
            anchor_origin = self.prior_topology_origin[chosen].copy()
            regular = np.flatnonzero(~anchor_origin)
            if anchor_truncation_step > 0 and len(regular):
                anchors[regular] = self.prior.truncated_denoise(
                    anchors[regular],
                    self.normalized_condition,
                    int(anchor_truncation_step),
                )
            parts.append(np.asarray(anchors, dtype=np.float32))
            origin_parts.append(anchor_origin)
        initial = np.concatenate(parts, axis=0)
        topology_origin = np.concatenate(origin_parts, axis=0)
        permutation = self.rng.permutation(len(initial))
        initial = initial[permutation]
        topology_origin = topology_origin[permutation]
        if return_topology_origin:
            return initial, topology_origin
        return initial
    def _clip_control_points(self, controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls, dtype=float).copy()
        environment = self.planner.env
        controls[:, 0] = np.clip(controls[:, 0], environment.xlim[0], environment.xlim[1])
        controls[:, 1] = np.clip(controls[:, 1], environment.ylim[0], environment.ylim[1])
        layer = environment.safe_layers[self.planner.cruise_layer_idx]
        controls[:, 2] = np.clip(controls[:, 2], layer.z_min, layer.z_max)
        controls[0] = self.start
        controls[-1] = self.end
        return controls

    def _latent_to_path(self, latent: np.ndarray) -> np.ndarray:
        vector = np.asarray(latent, dtype=np.float32) * self.trajectory_std
        vector = vector + self.trajectory_mean
        controls = self._model_vector_to_controls(vector)
        return np.asarray(
            self.planner.create_b_spline(controls, n_samples=self.sample_points),
            dtype=float,
        )

    def _path_is_geometrically_feasible(self, path: np.ndarray) -> bool:
        path = np.ascontiguousarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
            return False
        if not np.all(np.isfinite(path)):
            return False
        environment = self.planner.env
        layer = environment.safe_layers[self.planner.cruise_layer_idx]
        boundary_tolerance = 1e-7
        if (
            np.any(path[:, 0] < environment.xlim[0] - boundary_tolerance)
            or np.any(path[:, 0] > environment.xlim[1] + boundary_tolerance)
            or np.any(path[:, 1] < environment.ylim[0] - boundary_tolerance)
            or np.any(path[:, 1] > environment.ylim[1] + boundary_tolerance)
            or np.any(path[:, 2] < layer.z_min - boundary_tolerance)
            or np.any(path[:, 2] > layer.z_max + boundary_tolerance)
        ):
            return False

        cache_key = path.tobytes()
        cached = self._feasibility_cache.get(cache_key)
        if cached is not None:
            self._feasibility_cache.move_to_end(cache_key)
            return cached

        feasible = bool(
            is_trajectory_geometrically_feasible(
                path,
                environment.buildings,
                safety_margin=float(self.planner.pipe_radius),
                sample_count=self.sample_points,
            )
            and self._separation_violation(path) <= 1e-9
        )
        self._feasibility_cache[cache_key] = feasible
        if len(self._feasibility_cache) > self._feasibility_cache_size:
            self._feasibility_cache.popitem(last=False)
        return feasible

    def _path_to_latent(self, path: np.ndarray) -> np.ndarray:
        path = np.asarray(path, dtype=float)
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        controls = _resample_by_distance(
            path, np.linspace(0.0, length, self.control_point_count)
        )
        controls = self._clip_control_points(controls)
        raw = self._controls_to_model_vector(controls)
        return ((raw - self.trajectory_mean) / self.trajectory_std).astype(np.float32)

    def _project_to_feasible_parent(
        self,
        child_latent: np.ndarray,
        parent_latent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        child_latent = np.asarray(child_latent, dtype=np.float32)
        parent_latent = np.asarray(parent_latent, dtype=np.float32)
        parent_path = self._latent_to_path(parent_latent)
        if not self._path_is_geometrically_feasible(parent_path):
            raise RuntimeError("A feasible parent latent decoded to an infeasible path")
        for child_fraction in (0.5, 0.25, 0.125, 0.0625, 0.03125):
            projected_latent = parent_latent + child_fraction * (
                child_latent - parent_latent
            )
            projected_path = self._latent_to_path(projected_latent)
            if self._path_is_geometrically_feasible(projected_path):
                return projected_latent.astype(np.float32), projected_path
        return parent_latent.copy(), parent_path

    def _separation_violation(self, path: np.ndarray) -> float:
        if not self.other_paths:
            return 0.0
        path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        duration = path_length / max(self.speed_mps, 1e-6)
        maximum_violation = 0.0
        for other in self.other_paths:
            other_length = float(np.sum(np.linalg.norm(np.diff(other, axis=0), axis=1)))
            other_duration = other_length / max(self.speed_mps, 1e-6)
            overlap = min(duration, other_duration)
            if overlap <= 0.0:
                continue
            times = np.linspace(0.0, overlap, self.sample_points)
            position = _resample_by_distance(path, times * self.speed_mps)
            other_position = _resample_by_distance(other, times * self.speed_mps)
            minimum_distance = float(np.min(np.linalg.norm(position - other_position, axis=1)))
            maximum_violation = max(
                maximum_violation, self.minimum_separation - minimum_distance
            )
        return max(0.0, maximum_violation)

    def _evaluate_candidate_paths(
        self,
        latent: np.ndarray,
        paths: list[np.ndarray],
        *,
        topology_origin: np.ndarray | None = None,
    ) -> CandidateBatch:
        latent = np.asarray(latent, dtype=np.float32)
        if topology_origin is None:
            topology_origin = np.zeros(len(latent), dtype=bool)
        topology_origin = np.asarray(topology_origin, dtype=bool).reshape(-1)
        if len(topology_origin) != len(latent):
            raise ValueError("topology_origin must match the candidate count")
        self.evaluation_count += int(len(latent))
        metrics = evaluate_trajectories(
            paths,
            self.planner.env,
            self.radio_map,
            pipe_radius=float(self.planner.pipe_radius),
            outage_threshold_db=self.outage_threshold_db,
            sample_count=self.sample_points,
        )
        separation = np.asarray(
            [self._separation_violation(path) for path in paths], dtype=float
        )
        constraint_violation = np.asarray(
            [item.constraint_violation for item in metrics], dtype=float
        ) + separation
        batch = CandidateBatch(
            latent=latent,
            paths=[np.asarray(path, dtype=float) for path in paths],
            length=np.asarray([item.path_length_m for item in metrics], dtype=float),
            communication=np.asarray(
                [item.communication_cost for item in metrics], dtype=float
            ),
            mean_pathloss_db=np.asarray(
                [item.mean_pathloss_db for item in metrics], dtype=float
            ),
            worst_pathloss_db=np.asarray(
                [item.worst_pathloss_db for item in metrics], dtype=float
            ),
            constraint_violation=constraint_violation,
            collision_count=np.asarray(
                [item.collision_count for item in metrics], dtype=float
            ),
            min_clearance=np.asarray(
                [item.min_clearance_m for item in metrics], dtype=float
            ),
            separation_violation=separation,
            topology_origin=topology_origin,
        )
        logger = getattr(self, "_candidate_log", None)
        if logger is not None:
            logger.record_batch(
                batch, evaluation_end=self.evaluation_count,
                generation=self._candidate_generation, stage=self._candidate_stage,
            )
        return batch

    def _decode(
        self,
        latent: np.ndarray,
        *,
        topology_origin: np.ndarray | None = None,
    ) -> CandidateBatch:
        latent = np.asarray(latent, dtype=np.float32)
        paths = [self._latent_to_path(vector) for vector in latent]
        return self._evaluate_candidate_paths(
            latent, paths, topology_origin=topology_origin
        )

    def _generate_feasible_offspring(
        self,
        parent_pool: CandidateBatch,
        parent_indices: np.ndarray,
        truncation_step: int,
        *,
        resample_attempts: int,
        topology_noise_scale: float,
    ) -> tuple[CandidateBatch, dict[str, float]]:
        parent_indices = np.asarray(parent_indices, dtype=int)
        count = len(parent_indices)
        accepted_latent = np.empty((count, parent_pool.latent.shape[1]), dtype=np.float32)
        accepted_paths: list[np.ndarray | None] = [None] * count
        pending = np.arange(count, dtype=int)
        last_latent: dict[int, np.ndarray] = {}
        initially_feasible = 0
        resampled = 0
        offspring_topology_origin = parent_pool.topology_origin[parent_indices].copy()

        for attempt in range(1 + max(0, int(resample_attempts))):
            if len(pending) == 0:
                break
            proposed = np.empty(
                (len(pending), parent_pool.latent.shape[1]), dtype=np.float32
            )
            pending_topology = offspring_topology_origin[pending]
            for topology_flag in np.unique(pending_topology):
                local_positions = np.flatnonzero(
                    pending_topology == topology_flag
                )
                proposed[local_positions] = self.prior.truncated_denoise(
                    parent_pool.latent[
                        parent_indices[pending[local_positions]]
                    ],
                    self.normalized_condition,
                    1 if topology_flag else int(truncation_step),
                    noise_scale=(
                        float(topology_noise_scale) if topology_flag else 1.0
                    ),
                )
            still_pending = []
            for local_index, output_index in enumerate(pending):
                latent = np.asarray(proposed[local_index], dtype=np.float32)
                path = self._latent_to_path(latent)
                last_latent[int(output_index)] = latent
                if self._path_is_geometrically_feasible(path):
                    accepted_latent[output_index] = latent
                    accepted_paths[output_index] = path
                    if attempt == 0:
                        initially_feasible += 1
                    else:
                        resampled += 1
                else:
                    still_pending.append(int(output_index))
            pending = np.asarray(still_pending, dtype=int)

        projected = 0
        for output_index in pending:
            parent_index = int(parent_indices[output_index])
            latent, path = self._project_to_feasible_parent(
                last_latent[int(output_index)],
                parent_pool.latent[parent_index],
            )
            accepted_latent[output_index] = latent
            accepted_paths[output_index] = path
            projected += 1

        paths = [np.asarray(path, dtype=float) for path in accepted_paths]
        offspring = self._evaluate_candidate_paths(
            accepted_latent,
            paths,
            topology_origin=offspring_topology_origin,
        )
        self.offspring_resample_count += resampled
        self.projection_count += projected
        return offspring, {
            "raw_offspring_feasible_ratio": float(initially_feasible / max(1, count)),
            "resampled_offspring_count": float(resampled),
            "projected_offspring_count": float(projected),
            "topology_parent_count": float(np.sum(offspring_topology_origin)),
            "topology_projected_count": float(
                np.sum(offspring_topology_origin[pending])
            ),
            "topology_noise_scale": float(topology_noise_scale),
        }
    def _generate_channel_macro_offspring(
        self,
        parent_pool: CandidateBatch,
        parent_indices: np.ndarray,
        truncation_step: int,
        *,
        resample_attempts: int,
    ) -> tuple[CandidateBatch, dict[str, float]]:
        """Apply a large diffusion restart around communication-elite parents."""
        parent_indices = np.asarray(parent_indices, dtype=int)
        count = len(parent_indices)
        accepted_latent = np.empty(
            (count, parent_pool.latent.shape[1]), dtype=np.float32
        )
        accepted_paths: list[np.ndarray | None] = [None] * count
        pending = np.arange(count, dtype=int)
        last_latent: dict[int, np.ndarray] = {}
        initially_feasible = 0
        resampled = 0

        for attempt in range(1 + max(0, int(resample_attempts))):
            if len(pending) == 0:
                break
            proposed = self.prior.truncated_denoise(
                parent_pool.latent[parent_indices[pending]],
                self.normalized_condition,
                int(truncation_step),
                noise_scale=1.0,
            )
            still_pending = []
            for local_index, output_index in enumerate(pending):
                latent = np.asarray(proposed[local_index], dtype=np.float32)
                path = self._latent_to_path(latent)
                last_latent[int(output_index)] = latent
                if self._path_is_geometrically_feasible(path):
                    accepted_latent[output_index] = latent
                    accepted_paths[output_index] = path
                    if attempt == 0:
                        initially_feasible += 1
                    else:
                        resampled += 1
                else:
                    still_pending.append(int(output_index))
            pending = np.asarray(still_pending, dtype=int)

        projected = 0
        for output_index in pending:
            parent_index = int(parent_indices[output_index])
            latent, path = self._project_to_feasible_parent(
                last_latent[int(output_index)],
                parent_pool.latent[parent_index],
            )
            accepted_latent[output_index] = latent
            accepted_paths[output_index] = path
            projected += 1

        offspring = self._evaluate_candidate_paths(
            accepted_latent,
            [np.asarray(path, dtype=float) for path in accepted_paths],
            topology_origin=np.zeros(count, dtype=bool),
        )
        self.offspring_resample_count += resampled
        self.projection_count += projected
        return offspring, {
            "raw_offspring_feasible_ratio": float(
                initially_feasible / max(1, count)
            ),
            "resampled_offspring_count": float(resampled),
            "projected_offspring_count": float(projected),
            "channel_macro_parent_count": float(count),
            "channel_macro_feasible_ratio": float(
                initially_feasible / max(1, count)
            ),
            "channel_macro_projected_count": float(projected),
        }

    def _generate_mixed_feasible_offspring(
        self,
        parent_pool: CandidateBatch,
        parent_indices: np.ndarray,
        truncation_step: int,
        *,
        resample_attempts: int,
        topology_noise_scale: float,
        channel_macro_fraction: float,
        channel_macro_truncation_step: int,
    ) -> tuple[CandidateBatch, dict[str, float]]:
        parent_indices = np.asarray(parent_indices, dtype=int)
        count = len(parent_indices)
        macro_count = int(np.ceil(count * float(channel_macro_fraction)))
        if macro_count <= 0:
            offspring, statistics = self._generate_feasible_offspring(
                parent_pool,
                parent_indices,
                truncation_step,
                resample_attempts=resample_attempts,
                topology_noise_scale=topology_noise_scale,
            )
            statistics.setdefault("channel_macro_parent_count", 0.0)
            statistics.setdefault("channel_macro_feasible_ratio", 0.0)
            statistics.setdefault("channel_macro_projected_count", 0.0)
            return offspring, statistics

        macro_count = min(count, macro_count)
        communication_order = np.argsort(
            parent_pool.communication[parent_indices], kind="stable"
        )
        macro_positions = communication_order[:macro_count]
        local_mask = np.ones(count, dtype=bool)
        local_mask[macro_positions] = False
        local_parent_indices = parent_indices[local_mask]
        macro_parent_indices = parent_indices[macro_positions]

        local_offspring = None
        local_statistics: dict[str, float] = {}
        if len(local_parent_indices):
            local_offspring, local_statistics = self._generate_feasible_offspring(
                parent_pool,
                local_parent_indices,
                truncation_step,
                resample_attempts=resample_attempts,
                topology_noise_scale=topology_noise_scale,
            )
        macro_offspring, macro_statistics = self._generate_channel_macro_offspring(
            parent_pool,
            macro_parent_indices,
            int(channel_macro_truncation_step),
            resample_attempts=resample_attempts,
        )
        offspring = (
            macro_offspring
            if local_offspring is None
            else CandidateBatch.concatenate(local_offspring, macro_offspring)
        )

        local_count = len(local_parent_indices)
        statistics = dict(local_statistics)
        statistics["raw_offspring_feasible_ratio"] = float(
            (
                local_count
                * float(local_statistics.get("raw_offspring_feasible_ratio", 0.0))
                + macro_count
                * float(macro_statistics["raw_offspring_feasible_ratio"])
            )
            / max(1, count)
        )
        for key in ("resampled_offspring_count", "projected_offspring_count"):
            statistics[key] = float(local_statistics.get(key, 0.0)) + float(
                macro_statistics[key]
            )
        statistics["channel_macro_parent_count"] = float(macro_count)
        statistics["channel_macro_feasible_ratio"] = float(
            macro_statistics["channel_macro_feasible_ratio"]
        )
        statistics["channel_macro_projected_count"] = float(
            macro_statistics["channel_macro_projected_count"]
        )
        return offspring, statistics
    def _topology_seed_batch(
        self,
        desired_count: int,
        *,
        evaluation_budget: int | None = None,
    ) -> CandidateBatch | None:
        """Create feasible seeds only when the diffusion population is infeasible."""
        control_point_count = self.control_point_count
        accepted: list[CandidateBatch] = []
        attempts = 0
        maximum_attempts = max(12, int(desired_count) * 10)
        while len(accepted) < desired_count and attempts < maximum_attempts:
            if evaluation_budget is not None and self.evaluation_count >= evaluation_budget:
                break
            attempts += 1
            search_buffers = (4.0, 2.0, 1.0)
            candidate = topology_seed_candidate(
                self.planner.env,
                self.start,
                self.end,
                self.rng,
                control_point_count=control_point_count,
                safety_radius=float(self.planner.pipe_radius),
                path_sample_count=self.sample_points,
                search_buffer_m=search_buffers[(attempts - 1) % len(search_buffers)],
            )
            if candidate is None:
                continue
            controls, _ = candidate
            raw = self._controls_to_model_vector(controls)
            if raw.size != self.trajectory_mean.size:
                continue
            latent = ((raw - self.trajectory_mean) / self.trajectory_std).astype(
                np.float32
            )
            decoded_path = self._latent_to_path(latent)
            if not self._path_is_geometrically_feasible(decoded_path):
                continue
            decoded = self._evaluate_candidate_paths(
                latent.reshape(1, -1),
                [decoded_path],
                topology_origin=np.ones(1, dtype=bool),
            )
            accepted.append(decoded)
        self.topology_seed_count += len(accepted)
        if not accepted:
            return None
        batch = accepted[0]
        for candidate in accepted[1:]:
            batch = CandidateBatch.concatenate(batch, candidate)
        return batch
    @staticmethod
    def _scalar_score(
        batch: CandidateBatch,
        length_weight: float,
        communication_weight: float,
        constraint_weight: float,
    ) -> np.ndarray:
        return (
            length_weight * _normalized(batch.length)
            + communication_weight * batch.communication
            + constraint_weight * batch.constraint_violation
        )

    def _archive_progress_indicator(
        self,
        objectives: np.ndarray,
        hypervolume: float,
        reference: tuple[float, float],
        communication_weight: float,
    ) -> float:
        """Legacy progress score (lower is better); final selection is separate."""
        selected = compromise_index(
            objectives, weights=(1.0 - communication_weight, communication_weight)
        )
        return float(np.sum(
            objectives[selected] / np.maximum(objectives.max(axis=0), 1e-12)
        ))

    def run(
        self,
        *,
        population_size: int = 100,
        generations: int = 40,
        elite_fraction: float = 0.2,
        truncation_high: int = 6,
        truncation_low: int = 1,
        truncation_initial: int = 3,
        projection_feedback_threshold: float = 0.35,
        multi_objective: bool = True,
        length_weight: float = 0.4,
        communication_weight: float = 0.6,
        constraint_weight: float = 1000.0,
        patience: int | None = 8,
        tolerance: float = 1e-4,
        evaluation_budget: int | None = None,
        hypervolume_reference: tuple[float, float] = (800.0, 1.05),
        final_communication_weight: float = 0.65,
        communication_weight_min: float = 0.25,
        communication_weight_max: float = 0.95,
        prior_anchor_fraction: float = 0.35,
        prior_anchor_truncation_step: int = 3,
        pareto_archive_size: int = 400,
        parent_direction_count: int = 11,
        offspring_resample_attempts: int = 1,
        communication_parent_fraction: float = 0.25,
        communication_elite_fraction: float = 0.1,
        channel_macro_fraction: float = 0.0,
        channel_macro_truncation_step: int = 12,
        topology_noise_scale: float = 0.25,
        topology_noise_scale_min: float = 0.05,
        topology_noise_decay: float = 0.4,
        topology_projection_threshold: float = 0.35,
        record_candidates: bool = False,
    ) -> OptimizationResult:
        population_size = int(population_size)
        generations = int(generations)
        if population_size < 2 or generations < 1:
            raise ValueError("population_size must be >=2 and generations must be >=1")
        budget = int(
            population_size * generations
            if evaluation_budget is None
            else evaluation_budget
        )
        if budget < population_size:
            raise ValueError("evaluation_budget must cover the initial population")
        if not 0.0 <= final_communication_weight <= 1.0:
            raise ValueError("final_communication_weight must lie within [0, 1]")
        if not 0.0 <= prior_anchor_fraction <= 1.0:
            raise ValueError("prior_anchor_fraction must lie within [0, 1]")
        if int(pareto_archive_size) < 2:
            raise ValueError("pareto_archive_size must be >=2")
        if int(parent_direction_count) < 3:
            raise ValueError("parent_direction_count must be >=3")
        if int(offspring_resample_attempts) < 0:
            raise ValueError("offspring_resample_attempts must be non-negative")
        if not 0.0 <= float(communication_parent_fraction) <= 1.0:
            raise ValueError("communication_parent_fraction must lie within [0, 1]")
        if not 0.0 < float(communication_elite_fraction) <= 1.0:
            raise ValueError("communication_elite_fraction must lie within (0, 1]")
        if not 0.0 <= float(channel_macro_fraction) <= 0.5:
            raise ValueError("channel_macro_fraction must lie within [0, 0.5]")
        if not 0.0 <= float(topology_noise_scale) <= 1.0:
            raise ValueError("topology_noise_scale must lie within [0, 1]")
        if not 0.0 <= float(topology_noise_scale_min) <= float(topology_noise_scale):
            raise ValueError(
                "topology_noise_scale_min must lie between 0 and the initial scale"
            )
        if not 0.0 < float(topology_noise_decay) < 1.0:
            raise ValueError("topology_noise_decay must lie within (0, 1)")
        if not 0.0 < float(topology_projection_threshold) < 1.0:
            raise ValueError("topology_projection_threshold must lie within (0, 1)")
        if not 1 <= int(truncation_low) <= int(truncation_initial) <= int(truncation_high):
            raise ValueError(
                "truncation steps must satisfy 1 <= low <= initial <= high"
            )
        prior_steps = int(getattr(self.prior, "diffusion_steps", 25))
        if int(truncation_high) >= prior_steps:
            raise ValueError("truncation_high must be below the DDPM step count")
        if not int(truncation_high) <= int(channel_macro_truncation_step) < prior_steps:
            raise ValueError(
                "channel_macro_truncation_step must be at least truncation_high "
                "and below the DDPM step count"
            )
        if not 0.0 < float(projection_feedback_threshold) < 1.0:
            raise ValueError("projection_feedback_threshold must lie within (0, 1)")

        self.evaluation_count = 0
        self.topology_seed_count = 0
        self.offspring_resample_count = 0
        self.projection_count = 0
        self._candidate_log = (
            CandidateEvaluationLog(hypervolume_reference, final_communication_weight)
            if record_candidates else None
        )
        self._candidate_generation = 0
        self._candidate_stage = "initial_population"
        initial, initial_topology_origin = self._sample_initial_latent_population(
            population_size,
            anchor_fraction=prior_anchor_fraction,
            anchor_truncation_step=prior_anchor_truncation_step,
            return_topology_origin=True,
        )
        population = self._decode(
            initial, topology_origin=initial_topology_origin
        )
        if not np.any(population.constraint_violation <= 1e-9):
            self._candidate_stage = "initial_topology_seed"
            repaired = self._topology_seed_batch(
                max(4, min(12, population_size // 5)),
                evaluation_budget=budget,
            )
            if repaired is not None:
                merged_initial = CandidateBatch.concatenate(population, repaired)
                if multi_objective:
                    initial_order, _, _ = constrained_non_dominated_sort(
                        merged_initial.objectives,
                        merged_initial.constraint_violation,
                    )
                else:
                    initial_order = np.argsort(
                        self._scalar_score(
                            merged_initial,
                            length_weight,
                            communication_weight,
                            constraint_weight,
                        )
                    )
                population = merged_initial.take(initial_order[:population_size])
        if not np.any(population.constraint_violation <= 1e-9):
            raise RuntimeError(
                "CA-Diffusion-ES could not construct a consistent feasible initial population"
            )

        archive = FeasibleParetoArchive(maximum_size=int(pareto_archive_size))
        archive.update(population)
        history: list[dict[str, float]] = []

        def record_history(
            generation: int,
            truncation_step: int,
            reproduction: dict[str, float] | None = None,
            offspring: CandidateBatch | None = None,
        ) -> float:
            feasible_count = int(np.sum(population.constraint_violation <= 1e-9))
            if archive.batch is not None:
                archive_objectives = archive.batch.objectives
                hypervolume = hypervolume_2d(
                    archive_objectives, hypervolume_reference
                )
                indicator = self._archive_progress_indicator(
                    archive_objectives, hypervolume, hypervolume_reference,
                    final_communication_weight,
                )
                pareto_size = len(archive.batch.length)
            else:
                indicator = float(np.min(population.constraint_violation))
                hypervolume = 0.0
                pareto_size = 0
            reproduction = reproduction or {
                "raw_offspring_feasible_ratio": float(
                    feasible_count / max(1, len(population.length))
                ),
                "resampled_offspring_count": 0.0,
                "projected_offspring_count": 0.0,
                "topology_parent_count": 0.0,
                "topology_projected_count": 0.0,
                "topology_noise_scale": float(topology_noise_scale),
                "channel_macro_parent_count": 0.0,
                "channel_macro_feasible_ratio": 0.0,
                "channel_macro_projected_count": 0.0,
            }
            history.append(
                {
                    "generation": float(generation),
                    "evaluations": float(self.evaluation_count),
                    "truncation_step": float(truncation_step),
                    "feasible_count": float(feasible_count),
                    "feasible_ratio": float(
                        feasible_count / max(1, len(population.length))
                    ),
                    "pareto_size": float(pareto_size),
                    "hypervolume": float(hypervolume),
                    "indicator": indicator,
                    **reproduction,
                    "objective_logging_version": 1.0,
                    "objective_length_scale_m": float(hypervolume_reference[0]),
                    "objective_communication_scale": float(hypervolume_reference[1]),
                    "objective_score_communication_weight": float(final_communication_weight),
                    **objective_summary(
                        offspring, "offspring_postrepair", hypervolume_reference,
                        final_communication_weight,
                    ),
                    **objective_summary(
                        population, "population_selected", hypervolume_reference,
                        final_communication_weight,
                    ),
                }
            )
            return indicator

        best_indicator = record_history(0, 0)
        stale = 0
        update_count = max(0, generations - 1)
        truncation_step = int(truncation_initial)
        current_topology_noise_scale = float(topology_noise_scale)

        for generation in range(1, update_count + 1):
            remaining_budget = budget - self.evaluation_count
            if remaining_budget <= 0:
                break
            offspring_count = min(population_size, remaining_budget)
            if multi_objective:
                parent_pool = (
                    population
                    if archive.batch is None
                    else CandidateBatch.concatenate(population, archive.batch)
                )
                parent_indices = stratified_reference_direction_indices(
                    parent_pool.objectives,
                    parent_pool.constraint_violation,
                    offspring_count,
                    rng=self.rng,
                    direction_count=int(parent_direction_count),
                    communication_weight_min=communication_weight_min,
                    communication_weight_max=communication_weight_max,
                )
                communication_quota = min(
                    offspring_count,
                    int(np.ceil(offspring_count * communication_parent_fraction)),
                )
                if communication_quota > 0:
                    parent_indices[-communication_quota:] = (
                        _communication_elite_parent_indices(
                            parent_pool.objectives,
                            parent_pool.constraint_violation,
                            communication_quota,
                            rng=self.rng,
                            elite_fraction=float(communication_elite_fraction),
                        )
                    )
                parent_indices = parent_indices[
                    self.rng.permutation(len(parent_indices))
                ]
            else:
                parent_pool = population
                selection_score = self._scalar_score(
                    population,
                    length_weight,
                    communication_weight,
                    constraint_weight,
                )
                order = np.argsort(selection_score)
                elite_count = max(
                    2, int(np.ceil(elite_fraction * population_size))
                )
                elite_indices = order[:elite_count]
                elite_score = selection_score[elite_indices]
                probabilities = np.exp(-(elite_score - elite_score.min()))
                probabilities /= probabilities.sum()
                parent_indices = self.rng.choice(
                    elite_indices,
                    size=offspring_count,
                    replace=True,
                    p=probabilities,
                )
            self._candidate_generation = generation
            self._candidate_stage = "offspring_postrepair_preselection"
            offspring, reproduction = self._generate_mixed_feasible_offspring(
                parent_pool,
                parent_indices,
                int(truncation_step),
                resample_attempts=int(offspring_resample_attempts),
                topology_noise_scale=current_topology_noise_scale,
                channel_macro_fraction=float(channel_macro_fraction),
                channel_macro_truncation_step=int(
                    channel_macro_truncation_step
                ),
            )
            merged = CandidateBatch.concatenate(population, offspring)
            archive.update(offspring)

            if multi_objective:
                merged_order, _, _ = constrained_non_dominated_sort(
                    merged.objectives, merged.constraint_violation
                )
            else:
                merged_order = np.argsort(
                    self._scalar_score(
                        merged,
                        length_weight,
                        communication_weight,
                        constraint_weight,
                    )
                )
            population = merged.take(merged_order[:population_size])
            indicator = record_history(
                generation, int(truncation_step), reproduction, offspring
            )
            if indicator < best_indicator - tolerance:
                best_indicator = indicator
                stale = 0
            else:
                stale += 1
            projection_ratio = float(
                reproduction["projected_offspring_count"] / max(1, offspring_count)
            )
            truncation_step = _adapt_truncation_step(
                truncation_step,
                int(truncation_low),
                int(truncation_high),
                projection_ratio=projection_ratio,
                raw_feasible_ratio=float(
                    reproduction["raw_offspring_feasible_ratio"]
                ),
                stale_generations=stale,
                projection_threshold=float(projection_feedback_threshold),
            )
            current_topology_noise_scale = _adapt_topology_noise_scale(
                current_topology_noise_scale,
                float(topology_noise_scale_min),
                projected_count=float(
                    reproduction["topology_projected_count"]
                ),
                parent_count=float(reproduction["topology_parent_count"]),
                projection_threshold=float(topology_projection_threshold),
                decay=float(topology_noise_decay),
            )
            if patience is not None and archive.batch is not None and stale >= patience:
                break

        if archive.batch is not None:
            pareto = archive.batch
            best_index = compromise_index(
                pareto.objectives,
                weights=(1.0 - final_communication_weight, final_communication_weight),
            )
            best = pareto
        else:
            scalar = self._scalar_score(
                population,
                length_weight,
                communication_weight,
                constraint_weight,
            )
            best_index = int(
                min(
                    range(len(population.length)),
                    key=lambda index: (
                        population.constraint_violation[index], scalar[index]
                    ),
                )
            )
            best = population
            pareto = population.take(np.asarray([best_index]))

        if self._candidate_log is not None:
            self._candidate_log.validate_total(self.evaluation_count)
        return OptimizationResult(
            path=best.paths[best_index],
            length=float(best.length[best_index]),
            communication_cost=float(best.communication[best_index]),
            mean_pathloss_db=float(best.mean_pathloss_db[best_index]),
            worst_pathloss_db=float(best.worst_pathloss_db[best_index]),
            constraint_violation=float(best.constraint_violation[best_index]),
            min_clearance=float(best.min_clearance[best_index]),
            collision_count=int(best.collision_count[best_index]),
            history=history,
            pareto=pareto,
            evaluation_count=int(self.evaluation_count),
            topology_seed_count=int(self.topology_seed_count),
            offspring_resample_count=int(self.offspring_resample_count),
            projection_count=int(self.projection_count),
            candidate_history=(self._candidate_log.records if self._candidate_log is not None else []),
        )

def export_pareto_csv(result: OptimizationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "length_m",
                "communication_cost",
                "mean_predicted_pathloss_db",
                "worst_predicted_pathloss_db",
                "constraint_violation",
                "collision_count",
                "min_clearance_m",
                "separation_violation_m",
            ]
        )
        for index in range(len(result.pareto.length)):
            writer.writerow(
                [
                    result.pareto.length[index],
                    result.pareto.communication[index],
                    result.pareto.mean_pathloss_db[index],
                    result.pareto.worst_pathloss_db[index],
                    result.pareto.constraint_violation[index],
                    result.pareto.collision_count[index],
                    result.pareto.min_clearance[index],
                    result.pareto.separation_violation[index],
                ]
            )
    return path
