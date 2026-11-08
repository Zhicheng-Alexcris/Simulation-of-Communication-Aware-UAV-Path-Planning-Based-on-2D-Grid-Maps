"""Paired Radio Map ablation for CA-Diffusion-ES v9.

The planning variants share trajectories, seeds, prior assets, and evaluation
budgets. Only HGNN-predicted Radio Map access is switched on or off. Both final
paths are evaluated against the CSV's ray-tracing True_PathLoss column.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np


VARIANT_IDS = ("without_radio_map", "with_radio_map")
RECORD_FIELDS = [
    "status", "variant", "variant_label", "radio_map_enabled", "source_kind",
    "seed", "effective_seed", "trial", "drone_id", "display_drone_id",
    "layer_index", "success", "runtime_s", "evaluation_count",
    "planning_final_hypervolume", "planning_pareto_size", "path_length_m",
    "collision_count", "min_clearance_m", "avg_clearance_m", "turn_mean_deg",
    "turn_p95_deg", "turn_max_deg", "straightness_ratio",
    "constraint_violation", "predicted_communication_cost",
    "predicted_mean_pathloss_db", "predicted_worst_pathloss_db",
    "predicted_pathloss_p95_db", "predicted_outage_ratio",
    "true_communication_cost", "true_mean_pathloss_db",
    "true_worst_pathloss_db", "true_pathloss_p95_db", "true_outage_ratio",
    "trajectory_path", "history_path", "error",
]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    fields: list[str] = []
    for row in history:
        for key in row:
            if key not in fields:
                fields.append(key)
    _write_csv(path, history, fields)


def _assign_layers(env) -> list[int]:
    count = len(env.starts)
    base = min(5, count, len(env.safe_layers))
    layers = list(range(base))
    midpoints = [layer.mid for layer in env.safe_layers]
    for index in range(base, count):
        height = float(env.starts[index][2])
        layers.append(int(np.argmin([abs(height - value) for value in midpoints])))
    return layers


def _variant_map(config: dict) -> dict[str, dict]:
    return {str(item["id"]): item for item in config["variants"]}


def iter_runs(config: dict):
    variants = _variant_map(config)
    for trial, seed in enumerate(config["seeds"]):
        for drone_id in config["drones"]:
            for variant_id in VARIANT_IDS:
                yield variants[variant_id], int(seed), int(drone_id), int(trial)


def _source_records(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            (int(row["seed"]), int(row["drone_id"])): row
            for row in rows
            if row.get("method") == "ca_diffusion_es"
            and int(row.get("uav_count", 0)) == 5
        }


def load_and_validate_config(root: Path, path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "drones", "seeds", "variants", "population_size", "generations",
        "evaluation_budget", "radio_map_csv", "planning_pathloss_column",
        "true_pathloss_column", "prior_dataset", "prior_checkpoint",
        "output_dir", "pipe_radius_m", "path_samples",
        "outage_pathloss_threshold_db", "hypervolume_reference",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing Radio Map ablation fields: {sorted(missing)}")
    drones = [int(value) for value in config["drones"]]
    seeds = [int(value) for value in config["seeds"]]
    if drones != [0, 1, 2, 3, 4]:
        raise ValueError("The paired Radio Map ablation fixes drones to [0, 1, 2, 3, 4]")
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("The paired Radio Map ablation requires 10 distinct seeds")
    expected_budget = int(config["population_size"]) * int(config["generations"])
    if int(config["evaluation_budget"]) != expected_budget:
        raise ValueError(
            "evaluation_budget must equal population_size * generations "
            f"({expected_budget})"
        )
    variants = _variant_map(config)
    if set(variants) != set(VARIANT_IDS) or len(config["variants"]) != 2:
        raise ValueError(f"variants must be exactly {list(VARIANT_IDS)}")
    if bool(variants["without_radio_map"].get("radio_map_enabled", True)):
        raise ValueError("without_radio_map must set radio_map_enabled=false")
    if not bool(variants["with_radio_map"].get("radio_map_enabled", False)):
        raise ValueError("with_radio_map must set radio_map_enabled=true")
    for field in ("radio_map_csv", "prior_dataset", "prior_checkpoint"):
        resolved = _resolve(root, str(config[field]))
        if not resolved.exists():
            raise FileNotFoundError(resolved)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from environment import Environment
    from ca_diffusion_es.radio_map import load_hgnn_radio_map

    if max(drones) >= len(Environment().starts):
        raise ValueError("The environment does not contain the requested UAVs")
    radio_path = _resolve(root, str(config["radio_map_csv"]))
    load_hgnn_radio_map(
        radio_path, predicted_column=str(config["planning_pathloss_column"])
    )
    load_hgnn_radio_map(
        radio_path, predicted_column=str(config["true_pathloss_column"])
    )
    if bool(config.get("reuse_formal_with_radio", False)):
        if "with_radio_source_dir" not in config:
            raise ValueError(
                "with_radio_source_dir is required when reuse_formal_with_radio=true"
            )
        source_dir = _resolve(root, str(config["with_radio_source_dir"]))
        source_manifest = json.loads(
            (source_dir / "manifest.json").read_text(encoding="utf-8")
        )
        source_config = source_manifest.get("config", {})
        for field in ("population_size", "generations", "evaluation_budget"):
            if int(source_config.get(field, -1)) != int(config[field]):
                raise ValueError(f"With-Radio source has incompatible {field}")
        source = _source_records(source_dir / "records.csv")
        expected = {
            (seed + 100 * drone_id, drone_id)
            for seed in seeds for drone_id in drones
        }
        if not expected.issubset(source):
            raise ValueError("With-Radio source does not contain all 50 paired runs")
    return config


def _run_directory(output_dir: Path, variant: str, seed: int, drone_id: int) -> Path:
    return output_dir / variant / f"seed_{seed:04d}" / f"drone_{drone_id}"


def _collect_records(output_dir: Path) -> list[dict]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output_dir.glob("*/seed_*/drone_*/run_result.json")
    ]
    return sorted(
        rows,
        key=lambda row: (
            int(row["seed"]), int(row["drone_id"]), str(row["variant"])
        ),
    )


def _radio_fields(prefix: str, metrics) -> dict[str, float]:
    return {
        f"{prefix}_communication_cost": float(metrics.communication_cost),
        f"{prefix}_mean_pathloss_db": float(metrics.mean_pathloss_db),
        f"{prefix}_worst_pathloss_db": float(metrics.worst_pathloss_db),
        f"{prefix}_pathloss_p95_db": float(metrics.pathloss_p95_db),
        f"{prefix}_outage_ratio": float(metrics.outage_ratio),
    }


def _geometry_fields(metrics) -> dict[str, float | int]:
    return {
        "path_length_m": float(metrics.path_length_m),
        "collision_count": int(metrics.collision_count),
        "min_clearance_m": float(metrics.min_clearance_m),
        "avg_clearance_m": float(metrics.avg_clearance_m),
        "turn_mean_deg": float(metrics.turn_mean_deg),
        "turn_p95_deg": float(metrics.turn_p95_deg),
        "turn_max_deg": float(metrics.turn_max_deg),
        "straightness_ratio": float(metrics.straightness_ratio),
        "constraint_violation": float(metrics.constraint_violation),
    }


def _copy_formal_result(
    source_dir: Path, run_dir: Path, trial: int, drone_id: int
) -> tuple[np.ndarray, Path, Path]:
    scenario = source_dir / "ca_diffusion_es" / "N5" / f"trial_{trial:02d}"
    source_trajectory = scenario / f"trajectory_drone{drone_id}.csv"
    source_history = scenario / f"history_drone{drone_id}.csv"
    source_pareto = scenario / f"pareto_drone{drone_id}.csv"
    for source in (source_trajectory, source_history, source_pareto):
        if not source.exists():
            raise FileNotFoundError(source)
    trajectory_path = run_dir / "trajectory.csv"
    history_path = run_dir / "history.csv"
    shutil.copy2(source_trajectory, trajectory_path)
    shutil.copy2(source_history, history_path)
    shutil.copy2(source_pareto, run_dir / "pareto.csv")
    path = np.loadtxt(trajectory_path, delimiter=",", skiprows=1)
    return np.asarray(path, dtype=float), trajectory_path, history_path


def run_experiment(
    root: Path,
    config_path: Path,
    *,
    smoke: bool = False,
    rerun_failed: bool = False,
) -> Path:
    config = load_and_validate_config(root, config_path)
    output_dir = _resolve(root, str(config["output_dir"]))
    if smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    from environment import Environment
    from ca_diffusion_es.evaluation import evaluate_trajectory
    from ca_diffusion_es.integration_v9 import plan_drone_ca_diffusion_es
    from ca_diffusion_es.optimizer import export_pareto_csv
    from ca_diffusion_es.optimizer_v8 import configure_topology_trust_region
    from ca_diffusion_es.optimizer_v9 import configure_replay_anchor
    from ca_diffusion_es.optimizer_v9_fixed import configure_replay_parent_feedback
    from ca_diffusion_es.radio_map import load_hgnn_radio_map

    configure_topology_trust_region(
        max_radius_m=float(config["topology_residual_max_radius_m"]),
        clearance_reserve_m=float(config["topology_clearance_reserve_m"]),
        minimum_scale=float(config["topology_trust_scale_min"]),
        decay=float(config["topology_trust_decay"]),
        recovery=float(config["topology_trust_recovery"]),
        feasible_target=float(config["topology_trust_feasible_target"]),
    )
    configure_replay_anchor(max_radius_m=float(config["replay_anchor_max_radius_m"]))
    configure_replay_parent_feedback(
        minimum_fraction=float(config["replay_parent_fraction_min"])
    )

    radio_path = _resolve(root, str(config["radio_map_csv"]))
    predicted_map = load_hgnn_radio_map(
        radio_path, predicted_column=str(config["planning_pathloss_column"])
    )
    true_map = load_hgnn_radio_map(
        radio_path, predicted_column=str(config["true_pathloss_column"])
    )
    dataset_path = _resolve(root, str(config["prior_dataset"]))
    checkpoint_path = _resolve(root, str(config["prior_checkpoint"]))
    source_dir = _resolve(root, str(config.get("with_radio_source_dir", ".")))
    source_records = (
        _source_records(source_dir / "records.csv")
        if bool(config.get("reuse_formal_with_radio", False)) and not smoke
        else {}
    )

    all_runs = list(iter_runs(config))
    if smoke:
        first_seed = int(config["seeds"][0])
        first_drone = int(config["drones"][0])
        all_runs = [
            item for item in all_runs
            if item[1] == first_seed and item[2] == first_drone
        ]
    population = 20 if smoke else int(config["population_size"])
    generations = 2 if smoke else int(config["generations"])
    budget = population * generations
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "config": config,
        "config_path": str(config_path),
        "expected_runs": len(all_runs),
        "expected_pairs": len(all_runs) // 2,
        "planning_column": str(config["planning_pathloss_column"]),
        "independent_evaluation_column": str(config["true_pathloss_column"]),
        "paired_seed_policy": "identical seed, prior, UAV, and evaluation budget",
        "smoke": bool(smoke),
        "started_at_unix_s": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_start = time.perf_counter()
    executed = 0
    for index, (variant, seed, drone_id, trial) in enumerate(all_runs, start=1):
        variant_id = str(variant["id"])
        enabled = bool(variant["radio_map_enabled"])
        effective_seed = int(seed) + 100 * int(drone_id)
        run_dir = _run_directory(output_dir, variant_id, seed, drone_id)
        result_path = run_dir / "run_result.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete" or not rerun_failed:
                print(
                    f"[{index}/{len(all_runs)}] skip {variant_id} "
                    f"seed={seed} drone={drone_id}",
                    flush=True,
                )
                continue
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(all_runs)}] run {variant_id} seed={seed} "
            f"drone={drone_id} effective_seed={effective_seed}",
            flush=True,
        )
        started = time.perf_counter()
        record = {
            "status": "failed",
            "variant": variant_id,
            "variant_label": str(variant["label"]),
            "radio_map_enabled": int(enabled),
            "source_kind": "new_optimization",
            "seed": int(seed),
            "effective_seed": effective_seed,
            "trial": int(trial),
            "drone_id": int(drone_id),
            "display_drone_id": int(drone_id) + 1,
            "success": 0,
            "error": "",
        }
        try:
            env = Environment()
            layer_index = _assign_layers(env)[drone_id]
            record["layer_index"] = int(layer_index)
            reuse = enabled and bool(source_records)
            if reuse:
                source_record = source_records[(effective_seed, drone_id)]
                path, trajectory_path, history_path = _copy_formal_result(
                    source_dir, run_dir, trial, drone_id
                )
                record["source_kind"] = "reused_formal_ca_v9"
                record["runtime_s"] = float(source_record["runtime_s"])
                record["evaluation_count"] = int(float(source_record["evaluation_count"]))
                with (run_dir / "pareto.csv").open(
                    "r", newline="", encoding="utf-8"
                ) as handle:
                    pareto_rows = list(csv.DictReader(handle))
                record["planning_pareto_size"] = len(pareto_rows)
                with history_path.open("r", newline="", encoding="utf-8") as handle:
                    history_rows = list(csv.DictReader(handle))
                record["planning_final_hypervolume"] = float(
                    history_rows[-1]["hypervolume"]
                )
            else:
                path, result, _ = plan_drone_ca_diffusion_es(
                    env,
                    drone_id,
                    layer_index,
                    [],
                    legacy_root=root,
                    hgnn_csv=radio_path,
                    radio_map=predicted_map if enabled else None,
                    channel_enabled=enabled,
                    pipe_radius=float(config["pipe_radius_m"]),
                    population_size=population,
                    generations=generations,
                    seed=effective_seed,
                    multi_objective=True,
                    dataset_path=dataset_path,
                    checkpoint_path=checkpoint_path,
                    evaluation_budget=budget,
                    patience=None,
                    hypervolume_reference=tuple(config["hypervolume_reference"]),
                    require_feasible=False,
                    outage_threshold_db=float(config["outage_pathloss_threshold_db"]),
                    final_communication_weight=float(config["final_communication_weight"]),
                    communication_weight_min=float(config["communication_weight_min"]),
                    communication_weight_max=float(config["communication_weight_max"]),
                    prior_anchor_fraction=float(config["prior_anchor_fraction"]),
                    prior_anchor_truncation_step=int(config["prior_anchor_truncation_step"]),
                    pareto_archive_size=int(config["pareto_archive_size"]),
                    parent_direction_count=int(config["parent_direction_count"]),
                    offspring_resample_attempts=int(config["offspring_resample_attempts"]),
                    truncation_low=int(config["truncation_low"]),
                    truncation_initial=int(config["truncation_initial"]),
                    truncation_high=int(config["truncation_high"]),
                    projection_feedback_threshold=float(config["projection_feedback_threshold"]),
                    communication_parent_fraction=float(config["communication_parent_fraction"]),
                    communication_elite_fraction=float(config["communication_elite_fraction"]),
                    topology_noise_scale=float(config["topology_noise_scale"]),
                    topology_noise_scale_min=float(config["topology_noise_scale_min"]),
                    topology_noise_decay=float(config["topology_noise_decay"]),
                    topology_projection_threshold=float(config["topology_projection_threshold"]),
                )
                trajectory_path = run_dir / "trajectory.csv"
                history_path = run_dir / "history.csv"
                np.savetxt(
                    trajectory_path,
                    np.asarray(path, dtype=float),
                    delimiter=",",
                    header="x_m,y_m,z_m",
                    comments="",
                )
                _write_history(history_path, result.history)
                export_pareto_csv(result, run_dir / "pareto.csv")
                record["runtime_s"] = float(time.perf_counter() - started)
                record["evaluation_count"] = int(result.evaluation_count)
                record["planning_pareto_size"] = int(len(result.pareto.length))
                record["planning_final_hypervolume"] = float(
                    result.history[-1]["hypervolume"]
                )

            predicted_metrics = evaluate_trajectory(
                path,
                env,
                predicted_map,
                pipe_radius=float(config["pipe_radius_m"]),
                outage_threshold_db=float(config["outage_pathloss_threshold_db"]),
                sample_count=int(config["path_samples"]),
            )
            true_metrics = evaluate_trajectory(
                path,
                env,
                true_map,
                pipe_radius=float(config["pipe_radius_m"]),
                outage_threshold_db=float(config["outage_pathloss_threshold_db"]),
                sample_count=int(config["path_samples"]),
            )
            record.update(
                {
                    "status": "complete",
                    "success": int(true_metrics.constraint_violation <= 1e-9),
                    **_geometry_fields(true_metrics),
                    **_radio_fields("predicted", predicted_metrics),
                    **_radio_fields("true", true_metrics),
                    "trajectory_path": str(trajectory_path.relative_to(output_dir)),
                    "history_path": str(history_path.relative_to(output_dir)),
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            (run_dir / "traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        record.setdefault("runtime_s", float(time.perf_counter() - started))
        result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        executed += 1
        records = _collect_records(output_dir)
        _write_csv(output_dir / "records.csv", records, RECORD_FIELDS)
        elapsed = time.perf_counter() - run_start
        remaining = max(0, len(all_runs) - index)
        average = elapsed / max(1, executed)
        print(
            f"  status={record['status']} runtime={record['runtime_s']:.1f}s "
            f"estimated_remaining={average * remaining / 3600.0:.2f}h",
            flush=True,
        )

    records = _collect_records(output_dir)
    complete = sum(row.get("status") == "complete" for row in records)
    failed = sum(row.get("status") == "failed" for row in records)
    manifest.update(
        {
            "status": "complete" if complete == len(all_runs) else "partial",
            "finished_at_unix_s": time.time(),
            "completed_runs": complete,
            "failed_runs": failed,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()
    root = args.legacy_root.resolve()
    config_path = args.config.resolve()
    config = load_and_validate_config(root, config_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "paired_uavs": len(config["drones"]),
                    "paired_seeds": len(config["seeds"]),
                    "expected_pairs": len(config["drones"]) * len(config["seeds"]),
                    "expected_records": len(list(iter_runs(config))),
                    "new_optimized_runs": len(config["drones"]) * len(config["seeds"]),
                    "evaluation_budget_per_optimized_run": config["evaluation_budget"],
                    "planning_column": config["planning_pathloss_column"],
                    "independent_evaluation_column": config["true_pathloss_column"],
                    "output_dir": str(_resolve(root, config["output_dir"])),
                },
                indent=2,
            )
        )
        return
    print(
        run_experiment(
            root,
            config_path,
            smoke=bool(args.smoke),
            rerun_failed=bool(args.rerun_failed),
        )
    )


if __name__ == "__main__":
    main()
