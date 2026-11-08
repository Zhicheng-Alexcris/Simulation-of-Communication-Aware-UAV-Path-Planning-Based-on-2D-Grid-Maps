"""Optional HV feedback for validation; the default v9 optimizer is unchanged."""

from __future__ import annotations

import numpy as np

from .optimizer_v9_dynamic4 import FinalReplayFeedbackCADiffusionES


class HVFeedbackCADiffusionES(FinalReplayFeedbackCADiffusionES):
    """Use archive HV to control stagnation, retaining v9 final selection."""

    def _archive_progress_indicator(
        self, objectives, hypervolume, reference, communication_weight
    ) -> float:
        # The inherited loop minimizes its progress score. The fixed reference
        # makes a tolerance of 1e-4 mean 1e-4 normalized-HV improvement.
        area = float(reference[0]) * float(reference[1])
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("HV feedback requires a finite positive reference area")
        if not np.isfinite(hypervolume):
            raise ValueError("HV feedback requires a finite hypervolume")
        return -float(hypervolume) / area
