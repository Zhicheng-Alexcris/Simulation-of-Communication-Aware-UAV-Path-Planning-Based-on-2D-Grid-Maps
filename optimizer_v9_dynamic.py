"""Feedback-gated replay trust region for CA-Diffusion-ES v9."""

from __future__ import annotations

from .optimizer_v8_fixed import ReplayAwareAdaptiveSkeletonCADiffusionES
from .optimizer_v9 import ReplayAnchorTrustCADiffusionES
from .optimizer_v9_fixed import ReplayFeedbackCADiffusionES


REPLAY_TRUST_ACTIVATION_THRESHOLD = 0.90


class FeedbackGatedReplayCADiffusionES(ReplayFeedbackCADiffusionES):
    """Activate replay anchoring only after replay feasibility degrades."""

    search_representation = "adaptive_rrt_replay_feedback_gated_v3_2"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._replay_trust_active = False

    def _enforce_replay_parent_floor(self, parent_pool, parent_indices):
        if not self._replay_trust_active:
            return parent_indices, 0
        return super()._enforce_replay_parent_floor(parent_pool, parent_indices)

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
        if (
            not self._replay_trust_active
            and replay_count > 0.0
            and raw_ratio < REPLAY_TRUST_ACTIVATION_THRESHOLD
        ):
            self._replay_trust_active = True
        statistics["replay_trust_active"] = float(self._replay_trust_active)
        statistics["replay_trust_activation_threshold"] = float(
            REPLAY_TRUST_ACTIVATION_THRESHOLD
        )
        return offspring, statistics
