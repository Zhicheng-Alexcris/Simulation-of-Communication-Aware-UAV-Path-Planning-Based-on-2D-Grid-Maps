"""Consecutive-failure gate for replay-anchor trust."""

from __future__ import annotations

from .optimizer_v8_fixed import ReplayAwareAdaptiveSkeletonCADiffusionES
from .optimizer_v9 import ReplayAnchorTrustCADiffusionES
from .optimizer_v9_dynamic import FeedbackGatedReplayCADiffusionES


REPLAY_TRUST_ACTIVATION_THRESHOLD = 0.90
REPLAY_TRUST_ACTIVATION_STREAK = 2


class ConsecutiveFailureReplayCADiffusionES(FeedbackGatedReplayCADiffusionES):
    """Require two consecutive low-feasibility generations before activation."""

    search_representation = "adaptive_rrt_replay_feedback_streak_gated_v3_3"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._replay_low_feasible_streak = 0

    def _generate_feasible_offspring(
        self, parent_pool, parent_indices, truncation_step, **kwargs
    ):
        if self._replay_trust_active:
            offspring, statistics = ReplayAnchorTrustCADiffusionES._generate_feasible_offspring(
                self, parent_pool, parent_indices, truncation_step, **kwargs
            )
        else:
            offspring, statistics = ReplayAwareAdaptiveSkeletonCADiffusionES._generate_feasible_offspring(
                self, parent_pool, parent_indices, truncation_step, **kwargs
            )
        replay_count = float(statistics.get("replay_parent_count", 0.0))
        raw_ratio = float(statistics.get("raw_offspring_feasible_ratio", 1.0))
        if not self._replay_trust_active and replay_count > 0.0:
            if raw_ratio < REPLAY_TRUST_ACTIVATION_THRESHOLD:
                self._replay_low_feasible_streak += 1
            else:
                self._replay_low_feasible_streak = 0
            if self._replay_low_feasible_streak >= REPLAY_TRUST_ACTIVATION_STREAK:
                self._replay_trust_active = True
        statistics["replay_trust_active"] = float(self._replay_trust_active)
        statistics["replay_trust_low_feasible_streak"] = float(
            self._replay_low_feasible_streak
        )
        return offspring, statistics
