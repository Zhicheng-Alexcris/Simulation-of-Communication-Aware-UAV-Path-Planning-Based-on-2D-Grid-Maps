"""Topology-skeleton-conditioned residual search for CA-Diffusion-ES v7."""

from __future__ import annotations

import numpy as np

from .evaluation import _building_clearance
from .optimizer import CADiffusionES, CandidateBatch


TOPOLOGY_RESIDUAL_MAX_RADIUS_M = 3.0
TOPOLOGY_CLEARANCE_RESERVE_M = 0.5


def configure_topology_residual(
    *, max_radius_m: float, clearance_reserve_m: float
) -> None:
    global TOPOLOGY_RESIDUAL_MAX_RADIUS_M, TOPOLOGY_CLEARANCE_RESERVE_M
    if float(max_radius_m) <= 0.0:
        raise ValueError("topology_residual_max_radius_m must be positive")
    if float(clearance_reserve_m) < 0.0:
        raise ValueError("topology_clearance_reserve_m must be non-negative")
    TOPOLOGY_RESIDUAL_MAX_RADIUS_M = float(max_radius_m)
    TOPOLOGY_CLEARANCE_RESERVE_M = float(clearance_reserve_m)


def topology_residual_controls(
    skeleton: np.ndarray,
    proposal: np.ndarray,
    radius_m: np.ndarray | float,
) -> np.ndarray:
    """Express a proposal as bounded normal-plane offsets from a skeleton."""
    skeleton = np.asarray(skeleton, dtype=float).reshape(-1, 3)
    proposal = np.asarray(proposal, dtype=float).reshape(-1, 3)
    if skeleton.shape != proposal.shape:
        raise ValueError("Skeleton and proposal control points must have equal shape")
    radius = np.broadcast_to(
        np.asarray(radius_m, dtype=float).reshape(-1), (len(skeleton),)
    ).copy()
    if np.any(radius < 0.0):
        raise ValueError("Topology residual radii must be non-negative")

    tangents = np.gradient(skeleton, axis=0)
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    fallback = np.asarray([1.0, 0.0, 0.0], dtype=float)
    tangents = np.divide(
        tangents,
        np.maximum(tangent_norm, 1e-12),
        out=np.repeat(fallback[None, :], len(skeleton), axis=0),
        where=tangent_norm > 1e-12,
    )

    residual = proposal - skeleton
    residual -= np.sum(residual * tangents, axis=1, keepdims=True) * tangents
    if len(residual) > 2:
        source = residual.copy()
        residual[1:-1] = (
            0.25 * source[:-2] + 0.5 * source[1:-1] + 0.25 * source[2:]
        )
        residual -= np.sum(residual * tangents, axis=1, keepdims=True) * tangents

    residual_norm = np.linalg.norm(residual, axis=1)
    residual *= np.minimum(
        1.0, radius / np.maximum(residual_norm, 1e-12)
    )[:, None]
    residual[0] = 0.0
    residual[-1] = 0.0
    return skeleton + residual


class SkeletonConditionedCADiffusionES(CADiffusionES):
    """CA-Diffusion-ES whose hard-topology mutations stay in one safe tube."""

    search_representation = "topology_skeleton_residual_v1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._topology_reference_by_latent: dict[bytes, np.ndarray] = {}
        self._topology_radius_by_reference: dict[bytes, np.ndarray] = {}

    @staticmethod
    def _latent_key(latent: np.ndarray) -> bytes:
        return np.asarray(latent, dtype=np.float32).reshape(-1).tobytes()

    def _register_topology_reference(
        self, latent: np.ndarray, reference: np.ndarray | None = None
    ) -> None:
        latent = np.asarray(latent, dtype=np.float32).reshape(-1)
        reference = (
            latent.copy()
            if reference is None
            else np.asarray(reference, dtype=np.float32).reshape(-1).copy()
        )
        self._topology_reference_by_latent[self._latent_key(latent)] = reference

    def _topology_reference(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float32).reshape(-1)
        return self._topology_reference_by_latent.get(
            self._latent_key(latent), latent
        ).copy()

    def _latent_to_controls(self, latent: np.ndarray) -> np.ndarray:
        raw = np.asarray(latent, dtype=np.float32) * self.trajectory_std
        raw = raw + self.trajectory_mean
        return self._model_vector_to_controls(raw)

    def _controls_to_latent(self, controls: np.ndarray) -> np.ndarray:
        raw = self._controls_to_model_vector(self._clip_control_points(controls))
        return ((raw - self.trajectory_mean) / self.trajectory_std).astype(np.float32)

    def _topology_radius_profile(self, reference: np.ndarray) -> np.ndarray:
        key = self._latent_key(reference)
        cached = self._topology_radius_by_reference.get(key)
        if cached is not None:
            return cached.copy()

        path = self._latent_to_path(reference)
        clearance = _building_clearance(path, self.planner.env.buildings)
        control_positions = np.linspace(0, len(path) - 1, self.control_point_count)
        half_window = max(1, int(np.ceil(len(path) / self.control_point_count)))
        local_clearance = []
        for position in control_positions:
            center = int(round(float(position)))
            left = max(0, center - half_window)
            right = min(len(clearance), center + half_window + 1)
            local_clearance.append(float(np.min(clearance[left:right])))
        available = (
            np.asarray(local_clearance)
            - float(self.planner.pipe_radius)
            - TOPOLOGY_CLEARANCE_RESERVE_M
        )
        radius = np.clip(available, 0.0, TOPOLOGY_RESIDUAL_MAX_RADIUS_M)
        radius[0] = 0.0
        radius[-1] = 0.0
        self._topology_radius_by_reference[key] = radius.astype(np.float32)
        return radius

    def _condition_on_topology(
        self, proposal: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        skeleton_controls = self._latent_to_controls(reference)
        proposal_controls = self._latent_to_controls(proposal)
        controls = topology_residual_controls(
            skeleton_controls,
            proposal_controls,
            self._topology_radius_profile(reference),
        )
        return self._controls_to_latent(controls)

    def _sample_initial_latent_population(self, *args, **kwargs):
        result = super()._sample_initial_latent_population(*args, **kwargs)
        if kwargs.get("return_topology_origin", False):
            latent, topology_origin = result
            for item, is_topology in zip(latent, topology_origin):
                if bool(is_topology):
                    self._register_topology_reference(item)
        return result

    def _topology_seed_batch(self, *args, **kwargs):
        batch = super()._topology_seed_batch(*args, **kwargs)
        if batch is not None:
            for latent in batch.latent:
                self._register_topology_reference(latent)
        return batch

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
        accepted_latent = np.empty(
            (count, parent_pool.latent.shape[1]), dtype=np.float32
        )
        accepted_paths: list[np.ndarray | None] = [None] * count
        pending = np.arange(count, dtype=int)
        last_latent: dict[int, np.ndarray] = {}
        initially_feasible = 0
        initially_topology_feasible = 0
        resampled = 0
        offspring_topology_origin = parent_pool.topology_origin[parent_indices].copy()
        topology_references: list[np.ndarray | None] = []
        for parent_index, is_topology in zip(
            parent_indices, offspring_topology_origin
        ):
            topology_references.append(
                self._topology_reference(parent_pool.latent[parent_index])
                if bool(is_topology)
                else None
            )

        for attempt in range(1 + max(0, int(resample_attempts))):
            if len(pending) == 0:
                break
            proposed = np.empty(
                (len(pending), parent_pool.latent.shape[1]), dtype=np.float32
            )
            pending_topology = offspring_topology_origin[pending]
            for topology_flag in np.unique(pending_topology):
                local_positions = np.flatnonzero(pending_topology == topology_flag)
                proposed[local_positions] = self.prior.truncated_denoise(
                    parent_pool.latent[parent_indices[pending[local_positions]]],
                    self.normalized_condition,
                    1 if topology_flag else int(truncation_step),
                    noise_scale=(
                        float(topology_noise_scale) if topology_flag else 1.0
                    ),
                )

            still_pending = []
            for local_index, output_index in enumerate(pending):
                latent = np.asarray(proposed[local_index], dtype=np.float32)
                if bool(offspring_topology_origin[output_index]):
                    reference = topology_references[output_index]
                    if reference is None:
                        raise RuntimeError("Topology parent lost its skeleton reference")
                    latent = self._condition_on_topology(latent, reference)
                path = self._latent_to_path(latent)
                last_latent[int(output_index)] = latent
                if self._path_is_geometrically_feasible(path):
                    accepted_latent[output_index] = latent
                    accepted_paths[output_index] = path
                    if attempt == 0:
                        initially_feasible += 1
                        if bool(offspring_topology_origin[output_index]):
                            initially_topology_feasible += 1
                    else:
                        resampled += 1
                else:
                    still_pending.append(int(output_index))
            pending = np.asarray(still_pending, dtype=int)

        projected = 0
        for output_index in pending:
            parent_index = int(parent_indices[output_index])
            latent, path = self._project_to_feasible_parent(
                last_latent[int(output_index)], parent_pool.latent[parent_index]
            )
            accepted_latent[output_index] = latent
            accepted_paths[output_index] = path
            projected += 1

        for output_index, is_topology in enumerate(offspring_topology_origin):
            if bool(is_topology):
                reference = topology_references[output_index]
                if reference is None:
                    raise RuntimeError("Topology offspring lost its skeleton reference")
                self._register_topology_reference(
                    accepted_latent[output_index], reference
                )

        paths = [np.asarray(path, dtype=float) for path in accepted_paths]
        offspring = self._evaluate_candidate_paths(accepted_latent, paths)
        offspring.topology_origin = offspring_topology_origin
        self.offspring_resample_count += resampled
        self.projection_count += projected
        topology_count = int(np.sum(offspring_topology_origin))
        return offspring, {
            "raw_offspring_feasible_ratio": float(initially_feasible / max(1, count)),
            "resampled_offspring_count": float(resampled),
            "projected_offspring_count": float(projected),
            "topology_parent_count": float(topology_count),
            "topology_projected_count": float(
                np.sum(offspring_topology_origin[pending])
            ),
            "topology_noise_scale": float(topology_noise_scale),
            "topology_conditioned_count": float(topology_count),
            "topology_conditioned_feasible_ratio": float(
                initially_topology_feasible / max(1, topology_count)
            ),
            "topology_residual_max_radius_m": float(
                TOPOLOGY_RESIDUAL_MAX_RADIUS_M
            ),
        }
