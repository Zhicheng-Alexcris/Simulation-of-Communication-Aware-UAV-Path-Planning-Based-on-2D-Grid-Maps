"""Replay-parent feasibility feedback for CA-Diffusion-ES v9."""

from __future__ import annotations

import numpy as np

from .optimizer_v9 import ReplayAnchorTrustCADiffusionES


REPLAY_PARENT_FRACTION_MIN = 0.5
REPLAY_PARENT_FRACTION_EXPLORATION = 0.5


def configure_replay_parent_feedback(
    *,
    minimum_fraction: float,
    exploration_fraction: float | None = None,
) -> None:
    global REPLAY_PARENT_FRACTION_MIN
    global REPLAY_PARENT_FRACTION_EXPLORATION
    minimum_fraction = float(minimum_fraction)
    exploration_fraction = (
        minimum_fraction
        if exploration_fraction is None
        else float(exploration_fraction)
    )
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("replay_parent_fraction_min must lie within [0, 1]")
    if not 0.0 <= exploration_fraction <= minimum_fraction:
        raise ValueError(
            "replay exploration fraction must lie between zero and "
            "replay_parent_fraction_min"
        )
    REPLAY_PARENT_FRACTION_MIN = minimum_fraction
    REPLAY_PARENT_FRACTION_EXPLORATION = exploration_fraction


class ReplayFeedbackCADiffusionES(ReplayAnchorTrustCADiffusionES):
    """Maintain a feasible replay-parent floor when no online RRT seed exists."""

    search_representation = "adaptive_rrt_replay_feedback_v3_1"

    def _is_replay_parent(self, parent_pool, index: int) -> bool:
        if not bool(parent_pool.topology_origin[index]):
            return False
        reference = self._topology_reference(parent_pool.latent[index])
        return self._latent_key(reference) not in self._true_online_rrt_keys

    def _replay_parent_fraction(self) -> float:
        return float(REPLAY_PARENT_FRACTION_MIN)

    def _enforce_replay_parent_floor(self, parent_pool, parent_indices):
        parent_indices = np.asarray(parent_indices, dtype=int).copy()
        protected = np.flatnonzero(parent_pool.topology_origin)
        has_online = False
        replay_candidates = []
        for index in protected:
            reference = self._topology_reference(parent_pool.latent[index])
            if self._latent_key(reference) in self._true_online_rrt_keys:
                has_online = True
            elif parent_pool.constraint_violation[index] <= 1e-9:
                replay_candidates.append(int(index))
        if has_online or not replay_candidates:
            return parent_indices, 0

        quota = min(
            len(parent_indices),
            int(np.ceil(len(parent_indices) * self._replay_parent_fraction())),
        )
        replay_positions = [
            position
            for position, index in enumerate(parent_indices)
            if self._is_replay_parent(parent_pool, int(index))
        ]
        missing = max(0, quota - len(replay_positions))
        if missing == 0:
            return parent_indices, quota

        objectives = np.asarray(parent_pool.objectives, dtype=float)
        lower = objectives.min(axis=0)
        span = np.maximum(objectives.max(axis=0) - lower, 1e-12)
        normalized = (objectives - lower) / span
        score = 0.35 * normalized[:, 0] + 0.65 * normalized[:, 1]
        ranked_replay = sorted(replay_candidates, key=lambda index: score[index])
        replaceable = [
            position
            for position, index in enumerate(parent_indices)
            if not self._is_replay_parent(parent_pool, int(index))
        ]
        replaceable.sort(
            key=lambda position: score[parent_indices[position]], reverse=True
        )
        elite_count = max(1, min(len(ranked_replay), 8))
        for order, position in enumerate(replaceable[:missing]):
            parent_indices[position] = ranked_replay[order % elite_count]
        return parent_indices, quota

    def _generate_feasible_offspring(
        self, parent_pool, parent_indices, truncation_step, **kwargs
    ):
        parent_indices, quota = self._enforce_replay_parent_floor(
            parent_pool, parent_indices
        )
        offspring, statistics = super()._generate_feasible_offspring(
            parent_pool, parent_indices, truncation_step, **kwargs
        )
        statistics["replay_parent_quota"] = float(quota)
        return offspring, statistics
