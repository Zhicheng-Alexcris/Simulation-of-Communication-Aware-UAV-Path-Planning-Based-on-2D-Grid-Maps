"""Replay-aware correction for the adaptive CA-Diffusion-ES v8 search."""

from __future__ import annotations

import numpy as np

from .optimizer import CandidateBatch
from .optimizer_v7 import SkeletonConditionedCADiffusionES
from .optimizer_v8 import AdaptiveSkeletonCADiffusionES


class ReplayAwareAdaptiveSkeletonCADiffusionES(AdaptiveSkeletonCADiffusionES):
    """Use low-noise replay mutation but condition only objective-free RRT seeds."""

    search_representation = "adaptive_rrt_skeleton_residual_v2_1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._online_rrt_keys: set[bytes] = set()

    def _sample_initial_latent_population(self, *args, **kwargs):
        # Bypass v8's flag reset. Replay provenance retains low-noise mutation,
        # but is not added to _online_rrt_keys and receives no skeleton decoder.
        return SkeletonConditionedCADiffusionES._sample_initial_latent_population(
            self, *args, **kwargs
        )

    def _topology_seed_batch(self, *args, **kwargs):
        batch = SkeletonConditionedCADiffusionES._topology_seed_batch(
            self, *args, **kwargs
        )
        if batch is not None:
            for latent in batch.latent:
                self._online_rrt_keys.add(self._latent_key(latent))
        return batch

    def _is_online_reference(self, reference: np.ndarray) -> bool:
        return self._latent_key(reference) in self._online_rrt_keys

    def _reference_groups(
        self, parent_pool: CandidateBatch
    ) -> dict[bytes, list[int]]:
        feasible = parent_pool.constraint_violation <= 1e-9
        protected = np.asarray(parent_pool.topology_origin, dtype=bool)
        eligible = np.flatnonzero(feasible & protected)
        if len(eligible) == 0:
            eligible = np.flatnonzero(protected)
        groups: dict[bytes, list[int]] = {}
        for index in eligible:
            reference = self._topology_reference(parent_pool.latent[index])
            key = self._latent_key(reference)
            if key in self._online_rrt_keys:
                groups.setdefault(key, []).append(int(index))
        return groups

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
        initially_online_feasible = 0
        resampled = 0
        protected_origin = parent_pool.topology_origin[parent_indices].copy()
        references: list[np.ndarray | None] = []
        online_origin = np.zeros(count, dtype=bool)
        for output_index, (parent_index, is_protected) in enumerate(
            zip(parent_indices, protected_origin)
        ):
            if bool(is_protected):
                reference = self._topology_reference(
                    parent_pool.latent[parent_index]
                )
                references.append(reference)
                online_origin[output_index] = self._is_online_reference(reference)
            else:
                references.append(None)

        reference_attempts: dict[bytes, int] = {}
        reference_feasible: dict[bytes, int] = {}
        for attempt in range(1 + max(0, int(resample_attempts))):
            if len(pending) == 0:
                break
            proposed = np.empty(
                (len(pending), parent_pool.latent.shape[1]), dtype=np.float32
            )
            pending_protected = protected_origin[pending]
            for protected_flag in np.unique(pending_protected):
                local_positions = np.flatnonzero(
                    pending_protected == protected_flag
                )
                proposed[local_positions] = self.prior.truncated_denoise(
                    parent_pool.latent[parent_indices[pending[local_positions]]],
                    self.normalized_condition,
                    1 if protected_flag else int(truncation_step),
                    noise_scale=(
                        float(topology_noise_scale) if protected_flag else 1.0
                    ),
                )

            still_pending = []
            for local_index, output_index in enumerate(pending):
                latent = np.asarray(proposed[local_index], dtype=np.float32)
                reference = references[output_index]
                reference_key = None
                if bool(online_origin[output_index]):
                    if reference is None:
                        raise RuntimeError("Online RRT parent lost its skeleton")
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
                            initially_online_feasible += 1
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
        for output_index, is_protected in enumerate(protected_origin):
            if bool(is_protected):
                reference = references[output_index]
                if reference is None:
                    raise RuntimeError("Protected offspring lost its provenance")
                self._register_topology_reference(
                    accepted_latent[output_index], reference
                )
                if bool(online_origin[output_index]):
                    active_reference_keys.add(self._latent_key(reference))

        paths = [np.asarray(path, dtype=float) for path in accepted_paths]
        offspring = self._evaluate_candidate_paths(accepted_latent, paths)
        offspring.topology_origin = protected_origin
        self.offspring_resample_count += resampled
        self.projection_count += projected
        protected_count = int(np.sum(protected_origin))
        online_count = int(np.sum(online_origin))
        active_scales = [
            self._reference_radius_scale.get(key, 1.0)
            for key in active_reference_keys
        ] or [1.0]
        pending_online = int(np.sum(online_origin[pending]))
        return offspring, {
            "raw_offspring_feasible_ratio": float(initially_feasible / max(1, count)),
            "resampled_offspring_count": float(resampled),
            "projected_offspring_count": float(projected),
            "topology_parent_count": float(protected_count),
            "topology_projected_count": float(
                np.sum(protected_origin[pending])
            ),
            "topology_noise_scale": float(topology_noise_scale),
            "topology_conditioned_count": float(online_count),
            "topology_conditioned_feasible_ratio": float(
                initially_online_feasible / max(1, online_count)
            ),
            "topology_reference_count": float(len(active_reference_keys)),
            "topology_radius_scale_min": float(min(active_scales)),
            "topology_radius_scale_max": float(max(active_scales)),
            "online_skeleton_parent_count": float(online_count),
            "online_skeleton_projected_count": float(pending_online),
            "replay_parent_count": float(protected_count - online_count),
        }
