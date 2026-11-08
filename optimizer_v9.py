"""Replay-anchor trust-region refinement for CA-Diffusion-ES v8."""

from __future__ import annotations

import numpy as np

from .optimizer_v8_fixed import ReplayAwareAdaptiveSkeletonCADiffusionES


REPLAY_ANCHOR_MAX_RADIUS_M = 1.0


def configure_replay_anchor(*, max_radius_m: float) -> None:
    global REPLAY_ANCHOR_MAX_RADIUS_M
    if float(max_radius_m) <= 0.0:
        raise ValueError("replay_anchor_max_radius_m must be positive")
    REPLAY_ANCHOR_MAX_RADIUS_M = float(max_radius_m)


class ReplayAnchorTrustCADiffusionES(ReplayAwareAdaptiveSkeletonCADiffusionES):
    """Condition replay mutations in a small local tube without relabeling them."""

    search_representation = "adaptive_rrt_skeleton_replay_anchor_trust_v3"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._true_online_rrt_keys: set[bytes] = set(self._online_rrt_keys)
        self._replay_reference_keys: set[bytes] = set()

    def _topology_seed_batch(self, *args, **kwargs):
        batch = super()._topology_seed_batch(*args, **kwargs)
        self._true_online_rrt_keys = set(self._online_rrt_keys)
        return batch

    def _reference_groups(self, parent_pool):
        feasible = parent_pool.constraint_violation <= 1e-9
        protected = np.asarray(parent_pool.topology_origin, dtype=bool)
        eligible = np.flatnonzero(feasible & protected)
        if len(eligible) == 0:
            eligible = np.flatnonzero(protected)
        groups: dict[bytes, list[int]] = {}
        for index in eligible:
            reference = self._topology_reference(parent_pool.latent[index])
            key = self._latent_key(reference)
            if key in self._true_online_rrt_keys:
                groups.setdefault(key, []).append(int(index))
        return groups

    def _topology_radius_profile(self, reference: np.ndarray) -> np.ndarray:
        radius = super()._topology_radius_profile(reference)
        key = self._latent_key(reference)
        if key in self._replay_reference_keys:
            scale = self._reference_radius_scale.get(key, 1.0)
            radius = np.minimum(
                radius, REPLAY_ANCHOR_MAX_RADIUS_M * float(scale)
            )
        return radius

    def _generate_feasible_offspring(
        self,
        parent_pool,
        parent_indices,
        truncation_step,
        *,
        resample_attempts,
        topology_noise_scale,
    ):
        parent_indices = np.asarray(parent_indices, dtype=int)
        old_online_keys = set(self._online_rrt_keys)
        replay_keys = set()
        for parent_index in parent_indices:
            if not bool(parent_pool.topology_origin[parent_index]):
                continue
            reference = self._topology_reference(parent_pool.latent[parent_index])
            key = self._latent_key(reference)
            if key not in self._true_online_rrt_keys:
                replay_keys.add(key)
        # Reuse the tested v8 offspring engine by temporarily allowing replay
        # references through its topology decoder. They remain labeled as replay.
        self._replay_reference_keys = replay_keys
        self._online_rrt_keys = old_online_keys | replay_keys
        try:
            offspring, statistics = super()._generate_feasible_offspring(
                parent_pool,
                parent_indices,
                truncation_step,
                resample_attempts=resample_attempts,
                topology_noise_scale=topology_noise_scale,
            )
        finally:
            self._online_rrt_keys = old_online_keys
            self._replay_reference_keys = set()

        online_count = 0
        replay_count = 0
        for parent_index in parent_indices:
            if not bool(parent_pool.topology_origin[parent_index]):
                continue
            reference = self._topology_reference(parent_pool.latent[parent_index])
            if self._latent_key(reference) in self._true_online_rrt_keys:
                online_count += 1
            else:
                replay_count += 1
        statistics["online_skeleton_parent_count"] = float(online_count)
        statistics["replay_parent_count"] = float(replay_count)
        statistics["replay_conditioned_count"] = float(replay_count)
        statistics["replay_anchor_max_radius_m"] = float(
            REPLAY_ANCHOR_MAX_RADIUS_M
        )
        return offspring, statistics
