"""Formal experiment entry point for replay-anchor feedback CA-Diffusion-ES v9."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from . import experiment as _experiment
from . import integration as _integration
from .integration_v9 import plan_drone_ca_diffusion_es
from .optimizer_v8 import configure_topology_trust_region
from .optimizer_v9 import configure_replay_anchor
from .optimizer_v9_fixed import configure_replay_parent_feedback


_ORIGINAL_VALIDATE = _experiment.validate_experiment


def _config_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise ValueError("--config is required") from exc


def _validate_experiment(root: Path, config_path: Path, *, write_manifest: bool):
    manifest = _ORIGINAL_VALIDATE(root, config_path, write_manifest=False)
    manifest["search_representation"] = "adaptive_rrt_skeleton_replay_anchor_trust_v3_1"
    manifest["topology_skeleton_source"] = "online_objective_free_rrt_only"
    manifest["hard_replay_policy"] = "local_anchor_trust_region_plus_parent_floor"
    manifest["validation_early_stopping"] = False
    if write_manifest:
        (root / "experiment_ready.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return manifest


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    fields = [
        "generation", "evaluations", "truncation_step", "feasible_count",
        "feasible_ratio", "pareto_size", "hypervolume", "indicator",
        "raw_offspring_feasible_ratio", "resampled_offspring_count",
        "projected_offspring_count", "topology_parent_count",
        "topology_projected_count", "topology_noise_scale",
        "topology_conditioned_count", "topology_conditioned_feasible_ratio",
        "topology_reference_count", "topology_radius_scale_min",
        "topology_radius_scale_max", "online_skeleton_parent_count",
        "online_skeleton_projected_count", "replay_parent_count",
        "replay_conditioned_count", "replay_anchor_max_radius_m",
        "replay_parent_quota",
    ]
    # Preserve diagnostic fields, including fields absent from generation zero.
    fields = list(dict.fromkeys(fields + [key for row in history for key in row]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    config = json.loads(_config_path().read_text(encoding="utf-8"))
    configure_topology_trust_region(
        max_radius_m=float(config.get("topology_residual_max_radius_m", 3.0)),
        clearance_reserve_m=float(config.get("topology_clearance_reserve_m", 0.5)),
        minimum_scale=float(config.get("topology_trust_scale_min", 0.125)),
        decay=float(config.get("topology_trust_decay", 0.5)),
        recovery=float(config.get("topology_trust_recovery", 1.1)),
        feasible_target=float(config.get("topology_trust_feasible_target", 0.75)),
    )
    configure_replay_anchor(
        max_radius_m=float(config.get("replay_anchor_max_radius_m", 1.0))
    )
    configure_replay_parent_feedback(
        minimum_fraction=float(config.get("replay_parent_fraction_min", 0.5))
    )
    _integration.plan_drone_ca_diffusion_es = plan_drone_ca_diffusion_es
    _experiment.validate_experiment = _validate_experiment
    _experiment._write_history = _write_history
    _experiment.main()


if __name__ == "__main__":
    main()
