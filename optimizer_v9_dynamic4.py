"""Severe-failure replay trust with enforced replay-parent floor."""

from __future__ import annotations

from . import optimizer_v9_fixed as _feedback
from .optimizer_v8_fixed import ReplayAwareAdaptiveSkeletonCADiffusionES
from .optimizer_v9 import ReplayAnchorTrustCADiffusionES
from .optimizer_v9_fixed import ReplayFeedbackCADiffusionES


REPLAY_TRUST_ACTIVATION_THRESHOLD = 0.60
REPLAY_TRUST_ACTIVATION_STREAK = 2
REPLAY_STABLE_FEASIBLE_TARGET = 0.85
REPLAY_STABLE_FEASIBLE_STREAK = 2
REPLAY_PARENT_FRACTION_DECAY = 0.8
REPLAY_PARENT_FRACTION_RECOVERY = 0.1


def configure_dynamic_replay(
    *,
    feasible_target: float,
    feasible_streak: int,
    decay: float,
    recovery: float,
) -> None:
    global REPLAY_STABLE_FEASIBLE_TARGET
    global REPLAY_STABLE_FEASIBLE_STREAK
    global REPLAY_PARENT_FRACTION_DECAY
    global REPLAY_PARENT_FRACTION_RECOVERY
    if not REPLAY_TRUST_ACTIVATION_THRESHOLD < float(feasible_target) <= 1.0:
        raise ValueError("dynamic replay feasible target must lie within (0.6, 1]")
    if int(feasible_streak) < 1:
        raise ValueError("dynamic replay feasible streak must be positive")
    if not 0.0 < float(decay) < 1.0:
        raise ValueError("dynamic replay decay must lie within (0, 1)")
    if not 0.0 < float(recovery) <= 1.0:
        raise ValueError("dynamic replay recovery must lie within (0, 1]")
    REPLAY_STABLE_FEASIBLE_TARGET = float(feasible_target)
    REPLAY_STABLE_FEASIBLE_STREAK = int(feasible_streak)
    REPLAY_PARENT_FRACTION_DECAY = float(decay)
    REPLAY_PARENT_FRACTION_RECOVERY = float(recovery)


def adapt_replay_parent_fraction(
    current: float,
    raw_feasible_ratio: float,
    stable_streak: int,
) -> tuple[float, int]:
    maximum = float(_feedback.REPLAY_PARENT_FRACTION_MIN)
    minimum = float(_feedback.REPLAY_PARENT_FRACTION_EXPLORATION)
    current = min(maximum, max(minimum, float(current)))
    if float(raw_feasible_ratio) >= REPLAY_STABLE_FEASIBLE_TARGET:
        stable_streak = int(stable_streak) + 1
        if stable_streak >= REPLAY_STABLE_FEASIBLE_STREAK:
            current = max(minimum, current * REPLAY_PARENT_FRACTION_DECAY)
            stable_streak = 0
    elif float(raw_feasible_ratio) < REPLAY_TRUST_ACTIVATION_THRESHOLD:
        current = min(maximum, current + REPLAY_PARENT_FRACTION_RECOVERY)
        stable_streak = 0
    else:
        stable_streak = 0
    return float(current), int(stable_streak)


class FinalReplayFeedbackCADiffusionES(ReplayFeedbackCADiffusionES):
    """Gate replay trust, then enforce its parent floor after activation."""

    search_representation = "adaptive_rrt_replay_feedback_final_v3_5"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._replay_trust_active = False
        self._replay_low_feasible_streak = 0
        self._replay_stable_feasible_streak = 0
        self._replay_parent_fraction_current = float(
            _feedback.REPLAY_PARENT_FRACTION_MIN
        )

    def _replay_parent_fraction(self) -> float:
        return float(self._replay_parent_fraction_current)

    def _generate_feasible_offspring(
        self, parent_pool, parent_indices, truncation_step, **kwargs
    ):
        quota = 0
        fraction_used = 0.0
        if self._replay_trust_active:
            fraction_used = self._replay_parent_fraction()
            parent_indices, quota = self._enforce_replay_parent_floor(
                parent_pool, parent_indices
            )
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
        if self._replay_trust_active:
            (
                self._replay_parent_fraction_current,
                self._replay_stable_feasible_streak,
            ) = adapt_replay_parent_fraction(
                self._replay_parent_fraction_current,
                raw_ratio,
                self._replay_stable_feasible_streak,
            )
        statistics["replay_parent_quota"] = float(quota)
        statistics["replay_parent_fraction_used"] = float(fraction_used)
        statistics["replay_parent_fraction_next"] = float(
            self._replay_parent_fraction_current
        )
        statistics["replay_trust_active"] = float(self._replay_trust_active)
        statistics["replay_trust_activation_threshold"] = float(
            REPLAY_TRUST_ACTIVATION_THRESHOLD
        )
        statistics["replay_trust_low_feasible_streak"] = float(
            self._replay_low_feasible_streak
        )
        return offspring, statistics
