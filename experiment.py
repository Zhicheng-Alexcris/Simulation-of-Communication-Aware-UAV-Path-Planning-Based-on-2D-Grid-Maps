"""Reproducible formal experiment runner for A*, A*+NSGA-II and CA-Diffusion-ES."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


REQUIRED_METHODS = {
    "astar",
    "astar_nsga2",
    "nsga2",
    "diffusion_only",
    "ca_diffusion_es",
}
RECORD_FIELDS = [
    "method", "uav_count", "trial", "seed", "drone_id", "layer_index",
    "success", "runtime_s", "scenario_runtime_s", "evaluation_count",
    "topology_seed_count", "offspring_resample_count", "projection_count",
    "pareto_size", "start_time_s", "path_length_m",
    "communication_cost", "mean_pathloss_db", "worst_pathloss_db",
    "pathloss_p95_db", "outage_ratio", "collision_count", "min_clearance_m",
    "avg_clearance_m", "turn_mean_deg", "turn_p95_deg", "turn_max_deg",
    "straightness_ratio", "constraint_violation",
]


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = {
        "methods", "uav_counts", "trials", "base_seed", "population_size",
        "generations", "evaluation_budget", "radio_map_csv", "prior_dataset",
        "prior_checkpoint", "output_dir",
    } - set(config)
    if missing:
        raise ValueError(f"Missing formal experiment fields: {sorted(missing)}")
    unknown_methods = set(config["methods"]) - REQUIRED_METHODS
    if unknown_methods:
        raise ValueError(f"Unsupported methods: {sorted(unknown_methods)}")
    expected_budget = int(config["population_size"]) * int(config["generations"])
    if int(config["evaluation_budget"]) != expected_budget:
        raise ValueError(
            "evaluation_budget must equal population_size * generations "
            f"({expected_budget})"
        )
    if int(config["trials"]) < 10:
        raise ValueError("Formal experiments require at least 10 independent trials")
    if not isinstance(config.get("record_candidates", False), bool):
        raise ValueError("record_candidates must be a JSON boolean")
    if config.get("experiment_purpose") == "candidate_diagnostics":
        if (config["uav_counts"] != [5] or config["methods"] != ["ca_diffusion_es"]
                or not config.get("record_candidates", False)):
            raise ValueError("Candidate diagnostics require U=5, CA-Diffusion-ES and record_candidates=true")
    elif sorted(int(value) for value in config["uav_counts"]) != [5, 7, 9]:
        raise ValueError("The main paper protocol fixes uav_counts to [5, 7, 9]")
    bounded = {
        "final_communication_weight": (0.0, 1.0, 0.65),
        "communication_weight_min": (0.0, 1.0, 0.25),
        "communication_weight_max": (0.0, 1.0, 0.95),
        "prior_anchor_fraction": (0.0, 1.0, 0.35),
        "communication_parent_fraction": (0.0, 1.0, 0.25),
        "communication_elite_fraction": (0.0, 1.0, 0.1),
        "topology_noise_scale": (0.0, 1.0, 0.25),
        "topology_noise_scale_min": (0.0, 1.0, 0.05),
        "topology_noise_decay": (0.0, 1.0, 0.4),
        "topology_projection_threshold": (0.0, 1.0, 0.35),
    }
    for field, (lower, upper, default) in bounded.items():
        value = float(config.get(field, default))
        if not lower <= value <= upper:
            raise ValueError(f"{field} must lie within [{lower}, {upper}]")
        if field == "communication_elite_fraction" and value <= 0.0:
            raise ValueError("communication_elite_fraction must be positive")
        if field in {"topology_noise_decay", "topology_projection_threshold"} and value <= 0.0:
            raise ValueError(f"{field} must be positive")
    if float(config.get("communication_weight_min", 0.25)) > float(
        config.get("communication_weight_max", 0.95)
    ):
        raise ValueError(
            "communication_weight_min cannot exceed communication_weight_max"
        )
    if int(config.get("prior_anchor_truncation_step", 3)) < 0:
        raise ValueError("prior_anchor_truncation_step must be non-negative")
    integer_bounds = {
        "pareto_archive_size": (2, 400),
        "parent_direction_count": (3, 11),
        "offspring_resample_attempts": (0, 1),
        "truncation_low": (1, 1),
        "truncation_initial": (1, 3),
        "truncation_high": (2, 6),
    }
    for field, (minimum, default) in integer_bounds.items():
        if int(config.get(field, default)) < minimum:
            raise ValueError(f"{field} must be >= {minimum}")
    truncation_low = int(config.get("truncation_low", 1))
    truncation_initial = int(config.get("truncation_initial", 3))
    truncation_high = int(config.get("truncation_high", 6))
    if not truncation_low <= truncation_initial <= truncation_high:
        raise ValueError(
            "truncation steps must satisfy low <= initial <= high"
        )
    feedback = float(config.get("projection_feedback_threshold", 0.35))
    if not 0.0 < feedback < 1.0:
        raise ValueError("projection_feedback_threshold must lie within (0, 1)")
    if float(config.get("topology_noise_scale_min", 0.05)) > float(
        config.get("topology_noise_scale", 0.25)
    ):
        raise ValueError(
            "topology_noise_scale_min cannot exceed topology_noise_scale"
        )
    return config


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restrict_environment(env, uav_count: int):
    if uav_count < 1 or uav_count > len(env.starts):
        raise ValueError(f"Invalid UAV count {uav_count}; available={len(env.starts)}")
    env.starts = list(env.starts[:uav_count])
    env.ends = list(env.ends[:uav_count])
    rebuild = getattr(env, "_build_vertiports_from_starts_ends", None)
    env.vertiports = rebuild(z_min=0.0, z_max=40.0, nd=2) if callable(rebuild) else []
    return env


def _assign_layers(env) -> list[int]:
    count = len(env.starts)
    base = min(5, count, len(env.safe_layers))
    layers = list(range(base))
    midpoints = [layer.mid for layer in env.safe_layers]
    for index in range(base, count):
        start_height = float(env.starts[index][2])
        layers.append(int(np.argmin([abs(start_height - value) for value in midpoints])))
    return layers


def _write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    fields = [
        "generation", "evaluations", "truncation_step", "feasible_count",
        "feasible_ratio", "pareto_size", "hypervolume", "indicator",
        "raw_offspring_feasible_ratio", "resampled_offspring_count",
        "projected_offspring_count", "topology_parent_count",
        "topology_projected_count", "topology_noise_scale",
    ]
    # Preserve diagnostic fields, including fields absent from generation zero.
    fields = list(dict.fromkeys(fields + [key for row in history for key in row]))
    _write_rows(path, history, fields)


def validate_experiment(root: Path, config_path: Path, *, write_manifest: bool) -> dict:
    config = load_config(config_path)
    radio_path = _resolve(root, config["radio_map_csv"])
    dataset_path = _resolve(root, config["prior_dataset"])
    checkpoint_path = _resolve(root, config["prior_checkpoint"])
    for required in (radio_path, dataset_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(required)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from environment import Environment
    from ca_diffusion_es.prior_data import load_prior_dataset
    from ca_diffusion_es.radio_map import load_hgnn_radio_map

    env = Environment()
    if max(config["uav_counts"]) > len(env.starts):
        raise ValueError("The environment does not contain enough UAV start/end pairs")
    dataset = load_prior_dataset(dataset_path)
    uses_astar = bool(np.asarray(dataset.get("uses_astar", [True])).reshape(-1)[0])
    if uses_astar:
        raise ValueError("The formal diffusion prior dataset must declare uses_astar=False")
    if int(dataset["X"].shape[1]) != int(config["control_points"]) * 3:
        raise ValueError("Prior dimension and formal control-point count do not match")
    radio_map = load_hgnn_radio_map(radio_path)
    output_dir = _resolve(root, config["output_dir"])
    existing_outputs = list(output_dir.rglob("*")) if output_dir.exists() else []
    if any(path.is_file() for path in existing_outputs):
        raise FileExistsError(
            f"Formal output directory is not empty: {output_dir}. "
            "Move or delete it before a new formal run."
        )

    manifest = {
        "status": "ready",
        "config": config,
        "config_path": str(config_path),
        "radio_map_path": str(radio_path),
        "radio_map_sha256": _sha256(radio_path),
        "radio_map_transmitter_count": len(radio_map.transmitters),
        "radio_map_altitudes_m": [float(value) for value in radio_map.altitudes],
        "prior_dataset_path": str(dataset_path),
        "prior_dataset_sha256": _sha256(dataset_path),
        "prior_checkpoint_path": str(checkpoint_path),
        "prior_checkpoint_sha256": _sha256(checkpoint_path),
        "prior_samples": int(dataset["X"].shape[0]),
        "prior_dimension": int(dataset["X"].shape[1]),
        "trajectory_representation": str(
            np.asarray(dataset.get("trajectory_representation", "unknown")).item()
        ),
        "condition_representation": str(
            np.asarray(dataset.get("condition_representation", "unknown")).item()
        ),
        "uses_astar": False,
        "expected_runs": len(config["methods"])
        * len(config["uav_counts"])
        * int(config["trials"]),
        "expected_optimized_uav_runs": (
            len(
                set(config["methods"])
                & {
                    "astar_nsga2",
                    "nsga2",
                    "diffusion_only",
                    "ca_diffusion_es",
                }
            )
            * sum(config["uav_counts"])
            * int(config["trials"])
        ),
        "black_box_budget_per_optimized_uav": int(config["evaluation_budget"]),
        "python": sys.version,
        "platform": platform.platform(),
        "warning": (
            "The current HGNN CSV contains one transmitter; the paper must describe "
            "single-link/worst-link evaluation unless a multi-transmitter map is exported."
            if len(radio_map.transmitters) == 1
            else ""
        ),
    }
    if write_manifest:
        target = root / "experiment_ready.json"
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_experiment(root: Path, config_path: Path) -> Path:
    manifest = validate_experiment(root, config_path, write_manifest=False)
    config = manifest["config"]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from environment import Environment
    from ca_diffusion_es.baselines import (
        run_astar_baseline,
        run_astar_nsga2_baseline,
        run_diffusion_only_baseline,
        run_nsga2_baseline,
    )
    from ca_diffusion_es.evaluation import evaluate_trajectory
    from ca_diffusion_es.integration import plan_drone_ca_diffusion_es
    from ca_diffusion_es.legacy import load_legacy_modules
    from ca_diffusion_es.candidate_logging import export_candidate_csv
    from ca_diffusion_es.optimizer import export_pareto_csv
    from ca_diffusion_es.radio_map import load_hgnn_radio_map
    from main import check_all_pipe_collisions_timeaware, schedule_with_time_windows

    _, planner_module = load_legacy_modules(root)
    radio_path = _resolve(root, config["radio_map_csv"])
    radio_map = load_hgnn_radio_map(radio_path)
    output_dir = _resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["status"] = "running"
    manifest["started_at_unix_s"] = time.time()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    records: list[dict] = []
    hv_reference = tuple(float(value) for value in config["hypervolume_reference"])

    for method in config["methods"]:
        for uav_count in config["uav_counts"]:
            for trial in range(int(config["trials"])):
                scenario_seed = int(config["base_seed"]) + 1000 * int(uav_count) + 50 * trial
                env = _restrict_environment(Environment(), int(uav_count))
                layers = _assign_layers(env)
                paths = []
                scenario_records = []
                scenario_dir = output_dir / method / f"N{uav_count}" / f"trial_{trial:02d}"
                scenario_dir.mkdir(parents=True, exist_ok=True)
                scenario_start = time.perf_counter()

                for drone_id, layer_index in enumerate(layers):
                    seed = scenario_seed + drone_id * 100
                    start_time = time.perf_counter()
                    if method == "astar":
                        result = run_astar_baseline(
                            env,
                            planner_module,
                            radio_map,
                            drone_id,
                            layer_index,
                            pipe_radius=float(config["pipe_radius_m"]),
                            outage_threshold_db=float(
                                config["outage_pathloss_threshold_db"]
                            ),
                            astar_resolution=float(config["astar_resolution_m"]),
                            astar_inflation=float(config["astar_inflation_m"]),
                            hypervolume_reference=hv_reference,
                        )
                    elif method == "astar_nsga2":
                        result = run_astar_nsga2_baseline(
                            env,
                            planner_module,
                            radio_map,
                            drone_id,
                            layer_index,
                            population_size=int(config["population_size"]),
                            generations=int(config["generations"]),
                            evaluation_budget=int(config["evaluation_budget"]),
                            seed=seed,
                            pipe_radius=float(config["pipe_radius_m"]),
                            outage_threshold_db=float(
                                config["outage_pathloss_threshold_db"]
                            ),
                            control_points=int(config["control_points"]),
                            path_samples=int(config["path_samples"]),
                            astar_resolution=float(config["astar_resolution_m"]),
                            astar_inflation=float(config["astar_inflation_m"]),
                            corridor_radius=float(config["corridor_radius_m"]),
                            hypervolume_reference=hv_reference,
                        )
                    elif method == "nsga2":
                        result = run_nsga2_baseline(
                            env,
                            planner_module,
                            radio_map,
                            drone_id,
                            layer_index,
                            population_size=int(config["population_size"]),
                            generations=int(config["generations"]),
                            evaluation_budget=int(config["evaluation_budget"]),
                            seed=seed,
                            pipe_radius=float(config["pipe_radius_m"]),
                            outage_threshold_db=float(
                                config["outage_pathloss_threshold_db"]
                            ),
                            control_points=int(config["control_points"]),
                            path_samples=int(config["path_samples"]),
                            hypervolume_reference=hv_reference,
                        )
                    elif method == "diffusion_only":
                        result = run_diffusion_only_baseline(
                            env,
                            planner_module,
                            radio_map,
                            drone_id,
                            layer_index,
                            legacy_root=root,
                            dataset_path=_resolve(root, config["prior_dataset"]),
                            checkpoint_path=_resolve(root, config["prior_checkpoint"]),
                            population_size=int(config["population_size"]),
                            evaluation_budget=int(config["evaluation_budget"]),
                            seed=seed,
                            pipe_radius=float(config["pipe_radius_m"]),
                            outage_threshold_db=float(
                                config["outage_pathloss_threshold_db"]
                            ),
                            path_samples=int(config["path_samples"]),
                            control_points=int(config["control_points"]),
                            hypervolume_reference=hv_reference,
                            final_communication_weight=float(
                                config.get("final_communication_weight", 0.65)
                            ),
                            pareto_archive_size=int(
                                config.get("pareto_archive_size", 400)
                            ),
                        )
                    elif method == "ca_diffusion_es":
                        path, result, _ = plan_drone_ca_diffusion_es(
                            env,
                            drone_id,
                            layer_index,
                            [],
                            legacy_root=root,
                            hgnn_csv=radio_path,
                            radio_map=radio_map,
                            channel_enabled=True,
                            pipe_radius=float(config["pipe_radius_m"]),
                            population_size=int(config["population_size"]),
                            generations=int(config["generations"]),
                            seed=seed,
                            multi_objective=True,
                            dataset_path=_resolve(root, config["prior_dataset"]),
                            checkpoint_path=_resolve(root, config["prior_checkpoint"]),
                            evaluation_budget=int(config["evaluation_budget"]),
                            patience=int(config["generations"]) + 1,
                            hypervolume_reference=hv_reference,
                            require_feasible=False,
                            outage_threshold_db=float(
                                config["outage_pathloss_threshold_db"]
                            ),
                            final_communication_weight=float(
                                config.get("final_communication_weight", 0.65)
                            ),
                            communication_weight_min=float(
                                config.get("communication_weight_min", 0.25)
                            ),
                            communication_weight_max=float(
                                config.get("communication_weight_max", 0.95)
                            ),
                            prior_anchor_fraction=float(
                                config.get("prior_anchor_fraction", 0.35)
                            ),
                            prior_anchor_truncation_step=int(
                                config.get("prior_anchor_truncation_step", 3)
                            ),
                            pareto_archive_size=int(
                                config.get("pareto_archive_size", 400)
                            ),
                            parent_direction_count=int(
                                config.get("parent_direction_count", 11)
                            ),
                            offspring_resample_attempts=int(
                                config.get("offspring_resample_attempts", 1)
                            ),
                            truncation_low=int(config.get("truncation_low", 1)),
                            truncation_initial=int(
                                config.get("truncation_initial", 3)
                            ),
                            truncation_high=int(config.get("truncation_high", 6)),
                            projection_feedback_threshold=float(
                                config.get("projection_feedback_threshold", 0.35)
                            ),
                            communication_parent_fraction=float(
                                config.get("communication_parent_fraction", 0.25)
                            ),
                            communication_elite_fraction=float(
                                config.get("communication_elite_fraction", 0.1)
                            ),
                            channel_macro_fraction=float(
                                config.get("channel_macro_fraction", 0.0)
                            ),
                            channel_macro_truncation_step=int(
                                config.get("channel_macro_truncation_step", 12)
                            ),
                            topology_noise_scale=float(
                                config.get("topology_noise_scale", 0.25)
                            ),
                            topology_noise_scale_min=float(
                                config.get("topology_noise_scale_min", 0.05)
                            ),
                            topology_noise_decay=float(
                                config.get("topology_noise_decay", 0.4)
                            ),
                            topology_projection_threshold=float(
                                config.get("topology_projection_threshold", 0.35)
                            ),
                            record_candidates=bool(config.get("record_candidates", False)),
                        )
                        result.path = np.asarray(path, dtype=float)
                    runtime = time.perf_counter() - start_time
                    path = np.asarray(result.path, dtype=float)
                    paths.append(path)
                    np.savetxt(
                        scenario_dir / f"trajectory_drone{drone_id}.csv",
                        path,
                        delimiter=",",
                        header="x_m,y_m,z_m",
                        comments="",
                    )
                    export_pareto_csv(
                        result, scenario_dir / f"pareto_drone{drone_id}.csv"
                    )
                    _write_history(
                        scenario_dir / f"history_drone{drone_id}.csv", result.history
                    )
                    if method == "ca_diffusion_es" and config.get("record_candidates", False):
                        export_candidate_csv(
                            result, scenario_dir / f"candidates_drone{drone_id}.csv",
                            context={"method": method, "uav_count": int(uav_count),
                                     "trial": trial, "seed": seed, "drone_id": drone_id},
                        )
                        print(f"[candidate log] U={uav_count} trial={trial:02d} "
                              f"drone={drone_id}: {len(result.candidate_history)} evaluations saved", flush=True)
                    metrics = evaluate_trajectory(
                        path,
                        env,
                        radio_map,
                        pipe_radius=float(config["pipe_radius_m"]),
                        outage_threshold_db=float(
                            config["outage_pathloss_threshold_db"]
                        ),
                        sample_count=int(config["path_samples"]),
                    )
                    record = {
                        "method": method,
                        "uav_count": int(uav_count),
                        "trial": trial,
                        "seed": seed,
                        "drone_id": drone_id,
                        "layer_index": layer_index,
                        "success": int(metrics.constraint_violation <= 1e-9),
                        "runtime_s": runtime,
                        "scenario_runtime_s": 0.0,
                        "evaluation_count": result.evaluation_count,
                        "topology_seed_count": result.topology_seed_count,
                        "offspring_resample_count": result.offspring_resample_count,
                        "projection_count": result.projection_count,
                        "pareto_size": len(result.pareto.length),
                        "start_time_s": 0.0,
                        **metrics.to_dict(),
                    }
                    scenario_records.append(record)

                start_times = schedule_with_time_windows(
                    paths,
                    pipe_radius=float(config["pipe_radius_m"]),
                    speed=float(config["speed_mps"]),
                    delta_t=float(config["scheduling_step_s"]),
                )
                conflicts = check_all_pipe_collisions_timeaware(
                    paths,
                    start_times,
                    speed=float(config["speed_mps"]),
                    pipe_radius=float(config["pipe_radius_m"]),
                )
                if conflicts:
                    raise RuntimeError(
                        f"Scheduling left {len(conflicts)} conflicts for {method}, "
                        f"N={uav_count}, trial={trial}"
                    )
                scenario_runtime = time.perf_counter() - scenario_start
                for record, scheduled_start in zip(scenario_records, start_times):
                    record["start_time_s"] = float(scheduled_start)
                    record["scenario_runtime_s"] = float(scenario_runtime)
                records.extend(scenario_records)
                _write_rows(output_dir / "records.csv", records, RECORD_FIELDS)

    manifest["status"] = "complete"
    manifest["finished_at_unix_s"] = time.time()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.legacy_root.resolve()
    config_path = args.config.resolve()
    if args.dry_run:
        print(json.dumps(validate_experiment(root, config_path, write_manifest=True), indent=2))
        return
    print(run_experiment(root, config_path))


if __name__ == "__main__":
    main()