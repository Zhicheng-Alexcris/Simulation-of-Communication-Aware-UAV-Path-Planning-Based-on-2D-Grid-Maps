"""Per-skeleton adaptive trust-region search for CA-Diffusion-ES v8."""

from __future__ import annotations

import numpy as np

from .optimizer import CandidateBatch
from .optimizer_v7 import (
    SkeletonConditionedCADiffusionES,
    configure_topology_residual,
)


TOPOLOGY_TRUST_SCALE_MIN = 0.125
TOPOLOGY_TRUST_DECAY = 0.5
TOPOLOGY_TRUST_RECOVERY = 1.1
TOPOLOGY_TRUST_FEASIBLE_TARGET = 0.75
TOPOLOGY_TRUST_RECOVERY_THRESHOLD = 0.9


def configure_topology_trust_region(
    *,
    max_radius_m: float,
    clearance_reserve_m: float,
    minimum_scale: float,
    decay: float,
    recovery: float,
    feasible_target: float,
) -> None:
    global TOPOLOGY_TRUST_SCALE_MIN
    global TOPOLOGY_TRUST_DECAY
    global TOPOLOGY_TRUST_RECOVERY
    global TOPOLOGY_TRUST_FEASIBLE_TARGET
    configure_topology_residual(
        max_radius_m=max_radius_m,
        clearance_reserve_m=clearance_reserve_m,
    )
    if not 0.0 < float(minimum_scale) <= 1.0:
        raise ValueError("topology_trust_scale_min must lie within (0, 1]")
    if not 0.0 < float(decay) < 1.0:
        raise ValueError("topology_trust_decay must lie within (0, 1)")
    if float(recovery) <= 1.0:
        raise ValueError("topology_trust_recovery must be greater than 1")
    if not 0.0 < float(feasible_target) < TOPOLOGY_TRUST_RECOVERY_THRESHOLD:
        raise ValueError("topology_trust_feasible_target must lie within (0, 0.9)")
    TOPOLOGY_TRUST_SCALE_MIN = float(minimum_scale)
    TOPOLOGY_TRUST_DECAY = float(decay)
    TOPOLOGY_TRUST_RECOVERY = float(recovery)
    TOPOLOGY_TRUST_FEASIBLE_TARGET = float(feasible_target)


def adapt_reference_radius_scale(
    current: float,
    feasible_ratio: float,
    *,
    minimum: float,
    decay: float,
    recovery: float,
    feasible_target: float,
    recovery_threshold: float = TOPOLOGY_TRUST_RECOVERY_THRESHOLD,
) -> float:
    current = float(np.clip(current, minimum, 1.0))
    if float(feasible_ratio) < float(feasible_target):
        return max(float(minimum), current * float(decay))
    if float(feasible_ratio) >= float(recovery_threshold):
        return min(1.0, current * float(recovery))
    return current


class AdaptiveSkeletonCADiffusionES(SkeletonConditionedCADiffusionES):
    """Apply skeleton conditioning only to online objective-free RRT repairs."""

    search_representation = "adaptive_rrt_skeleton_residual_v2"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reference_radius_scale: dict[bytes, float] = {}

    def _sample_initial_latent_population(self, *args, **kwargs):
        result = super()._sample_initial_latent_population(*args, **kwargs)
        if kwargs.get("return_topology_origin", False):
            latent, topology_origin = result
            # Hard-replay is training provenance, not an online topology skeleton.
            topology_origin = np.zeros(len(latent), dtype=bool)
            return latent, topology_origin
        return result

    def _topology_radius_profile(self, reference: np.ndarray) -> np.ndarray:
        base_radius = super()._topology_radius_profile(reference)
        scale = self._reference_radius_scale.get(
            self._latent_key(reference), 1.0
        )
        return base_radius * float(scale)

    def _reference_groups(
        self, parent_pool: CandidateBatch
    ) -> dict[bytes, list[int]]:
        feasible = parent_pool.constraint_violation <= 1e-9
        topology = np.asarray(parent_pool.topology_origin, dtype=bool)
        eligible = np.flatnonzero(feasible & topology)
        if len(eligible) == 0:
            eligible = np.flatnonzero(topology)
        groups: dict[bytes, list[int]] = {}
        for index in eligible:
            reference = self._topology_reference(parent_pool.latent[index])
            groups.setdefault(self._latent_key(reference), []).append(int(index))
        return groups

    def _rebalance_topology_parents(
        self, parent_pool: CandidateBatch, parent_indices: np.ndarray
    ) -> np.ndarray:
        parent_indices = np.asarray(parent_indices, dtype=int).copy()
        topology_slots = np.flatnonzero(
            parent_pool.topology_origin[parent_indices]
        )
        groups = self._reference_groups(parent_pool)
        if len(topology_slots) == 0 or len(groups) <= 1:
            return parent_indices

        objectives = np.asarray(parent_pool.objectives, dtype=float)
        lower = objectives.min(axis=0)
        span = np.maximum(objectives.max(axis=0) - lower, 1e-12)
        score = np.sum((objectives - lower) / span, axis=1)
        ordered_groups = sorted(
            groups,
            key=lambda key: min(score[index] for index in groups[key]),
        )
        ranked = {
            key: sorted(groups[key], key=lambda index: score[index])
            for key in ordered_groups
        }
        for order, slot in enumerate(topology_slots):
            key = ordered_groups[order % len(ordered_groups)]
            candidates = ranked[key]
            tier = (order // len(ordered_groups)) % min(4, len(candidates))
            parent_indices[slot] = candidates[tier]
        return parent_indices

    def _update_reference_scales(
        self,
        attempts: dict[bytes, int],
        feasible: dict[bytes, int],
    ) -> None:
        for key, count in attempts.items():
            ratio = float(feasible.get(key, 0)) / max(1, int(count))
            current = self._reference_radius_scale.get(key, 1.0)
            self._reference_radius_scale[key] = adapt_reference_radius_scale(
                current,
                ratio,
                minimum=TOPOLOGY_TRUST_SCALE_MIN,
                decay=TOPOLOGY_TRUST_DECAY,
                recovery=TOPOLOGY_TRUST_RECOVERY,
                feasible_target=TOPOLOGY_TRUST_FEASIBLE_TARGET,
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
        parent_indices = self._rebalance_topology_parents(
            parent_pool, parent_indices
        )
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

        reference_attempts: dict[bytes, int] = {}
        reference_feasible: dict[bytes, int] = {}
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
                reference = topology_references[output_index]
                reference_key = None
                if bool(offspring_topology_origin[output_index]):
                    if reference is None:
                        raise RuntimeError("Topology parent lost its RRT skeleton")
                    reference_key = self._latent_key(reference)
                    latent = self._condition_on_topology(latent, reference)
                    if attempt == 0:
                        reference_attempts[reference_key] = (
                            reference_attempts.get(reference_key, 0) + 1
                        )
                path = self._latent_to_path(latent)
                last_latent[int(output_index)] = latent
                if self._path_is_geometrically_feasible(path):
                    accepted_latent[output_index] = latent
                    accepted_paths[output_index] = path
                    if attempt == 0:
                        initially_feasible += 1
                        if reference_key is not None:
                            initially_topology_feasible += 1
                            reference_feasible[reference_key] = (
                                reference_feasible.get(reference_key, 0) + 1
                            )
                    else:
                        resampled += 1
                else:
                    still_pending.append(int(output_index))
            pending = np.asarray(still_pending, dtype=int)
            if attempt == 0:
                self._update_reference_scales(
                    reference_attempts, reference_feasible
                )

        projected = 0
        for output_index in pending:
            parent_index = int(parent_indices[output_index])
            latent, path = self._project_to_feasible_parent(
                last_latent[int(output_index)], parent_pool.latent[parent_index]
            )
            accepted_latent[output_index] = latent
            accepted_paths[output_index] = path
            projected += 1

        active_reference_keys = set()
        for output_index, is_topology in enumerate(offspring_topology_origin):
            if bool(is_topology):
                reference = topology_references[output_index]
                if reference is None:
                    raise RuntimeError("Topology offspring lost its RRT skeleton")
                active_reference_keys.add(self._latent_key(reference))
                self._register_topology_reference(
                    accepted_latent[output_index], reference
                )

        paths = [np.asarray(path, dtype=float) for path in accepted_paths]
        offspring = self._evaluate_candidate_paths(accepted_latent, paths)
        offspring.topology_origin = offspring_topology_origin
        self.offspring_resample_count += resampled
        self.projection_count += projected
        topology_count = int(np.sum(offspring_topology_origin))
        active_scales = [
            self._reference_radius_scale.get(key, 1.0)
            for key in active_reference_keys
        ] or [1.0]
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
            "topology_reference_count": float(len(active_reference_keys)),
            "topology_radius_scale_min": float(min(active_scales)),
            "topology_radius_scale_max": float(max(active_scales)),
        }
