"""Run the controlled one-factor-at-a-time experiment for CA-Diffusion-ES v9."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np


RECORD_FIELDS = [
    "status", "configuration_id", "configuration_label", "groups", "seed",
    "effective_seed", "drone_id", "display_drone_id", "layer_index", "success",
    "runtime_s", "evaluation_count", "final_hypervolume", "hypervolume_auc",
    "initial_feasible_ratio", "mean_raw_offspring_feasible_ratio",
    "mean_population_feasible_ratio", "projection_count",
    "projection_per_100_evaluations", "offspring_resample_count",
    "topology_seed_count", "pareto_size", "path_length_m", "communication_cost",
    "mean_pathloss_db", "worst_pathloss_db", "outage_ratio", "collision_count",
    "min_clearance_m", "constraint_violation", "history_path", "error",
]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_and_validate_config(root: Path, path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "drones", "seeds", "population_size", "generations",
        "evaluation_budget", "radio_map_csv", "prior_dataset",
        "prior_checkpoint", "output_dir", "configurations",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing parameter experiment fields: {sorted(missing)}")
    drones = [int(value) for value in config["drones"]]
    seeds = [int(value) for value in config["seeds"]]
    if len(drones) != 5 or len(set(drones)) != 5:
        raise ValueError("The parameter experiment requires five distinct drone IDs")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("At least three distinct random seeds are required")
    expected_budget = int(config["population_size"]) * int(config["generations"])
    if int(config["evaluation_budget"]) != expected_budget:
        raise ValueError(
            "evaluation_budget must equal population_size * generations "
            f"({expected_budget})"
        )
    identifiers = [str(item["id"]) for item in config["configurations"]]
    if len(identifiers) != 8 or len(set(identifiers)) != len(identifiers):
        raise ValueError("Exactly eight distinct one-factor configurations are required")
    baselines = [
        item for item in config["configurations"]
        if set(item.get("groups", [])) == {"truncation", "anchor"}
    ]
    if len(baselines) != 1:
        raise ValueError("Exactly one v9 configuration must belong to both groups")
    for item in config["configurations"]:
        fraction = float(item["prior_anchor_fraction"])
        low = int(item["truncation_low"])
        initial = int(item["truncation_initial"])
        high = int(item["truncation_high"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Invalid anchor fraction in {item['id']}")
        if not 1 <= low <= initial <= high < 25:
            raise ValueError(f"Invalid truncation schedule in {item['id']}")
    for field in ("radio_map_csv", "prior_dataset", "prior_checkpoint"):
        resolved = _resolve(root, str(config[field]))
        if not resolved.exists():
            raise FileNotFoundError(resolved)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from environment import Environment

    available = len(Environment().starts)
    if min(drones) < 0 or max(drones) >= available:
        raise ValueError(f"Drone IDs must lie within [0, {available - 1}]")
    return config


def iter_runs(config: dict):
    for setting in config["configurations"]:
        for seed in config["seeds"]:
            for drone_id in config["drones"]:
                yield setting, int(seed), int(drone_id)


def _assign_layers(env) -> list[int]:
    count = len(env.starts)
    base = min(5, count, len(env.safe_layers))
    layers = list(range(base))
    midpoints = [layer.mid for layer in env.safe_layers]
    for index in range(base, count):
        height = float(env.starts[index][2])
        layers.append(int(np.argmin([abs(height - value) for value in midpoints])))
    return layers


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


def _history_statistics(history: list[dict[str, float]], budget: int) -> dict[str, float]:
    if not history:
        return {
            "final_hypervolume": 0.0,
            "hypervolume_auc": 0.0,
            "initial_feasible_ratio": 0.0,
            "mean_raw_offspring_feasible_ratio": 0.0,
            "mean_population_feasible_ratio": 0.0,
        }
    evaluations = np.asarray([float(row["evaluations"]) for row in history], dtype=float)
    hypervolume = np.asarray([float(row["hypervolume"]) for row in history], dtype=float)
    evaluations = np.concatenate([[0.0], evaluations])
    hypervolume = np.concatenate([[0.0], hypervolume])
    integral = getattr(np, "trapezoid", np.trapz)(hypervolume, evaluations)
    offspring_rows = history[1:] if len(history) > 1 else history
    return {
        "final_hypervolume": float(hypervolume[-1]),
        "hypervolume_auc": float(integral / max(1, int(budget))),
        "initial_feasible_ratio": float(history[0]["feasible_ratio"]),
        "mean_raw_offspring_feasible_ratio": float(np.mean([
            float(row["raw_offspring_feasible_ratio"]) for row in offspring_rows
        ])),
        "mean_population_feasible_ratio": float(np.mean([
            float(row["feasible_ratio"]) for row in history
        ])),
    }


def _collect_records(output_dir: Path) -> list[dict]:
    records = []
    for path in output_dir.glob("config_*/seed_*/drone_*/run_result.json"):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(
        records,
        key=lambda row: (
            str(row["configuration_id"]), int(row["seed"]), int(row["drone_id"])
        ),
    )


def _run_directory(output_dir: Path, configuration_id: str, seed: int, drone_id: int) -> Path:
    return (
        output_dir / f"config_{configuration_id}" / f"seed_{seed:04d}"
        / f"drone_{drone_id}"
    )


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
    radio_map = load_hgnn_radio_map(radio_path)
    dataset_path = _resolve(root, str(config["prior_dataset"]))
    checkpoint_path = _resolve(root, str(config["prior_checkpoint"]))

    all_runs = list(iter_runs(config))
    if smoke:
        all_runs = all_runs[:1]
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "config_path": str(config_path),
        "config": config,
        "expected_runs": len(all_runs),
        "started_at_unix_s": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "smoke": bool(smoke),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    population = 20 if smoke else int(config["population_size"])
    generations = 2 if smoke else int(config["generations"])
    budget = population * generations
    run_start = time.perf_counter()
    executed = 0
    for index, (setting, seed, drone_id) in enumerate(all_runs, start=1):
        run_dir = _run_directory(output_dir, str(setting["id"]), seed, drone_id)
        result_path = run_dir / "run_result.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete" or not rerun_failed:
                print(f"[{index}/{len(all_runs)}] skip {setting['id']} seed={seed} drone={drone_id}")
                continue
        run_dir.mkdir(parents=True, exist_ok=True)
        effective_seed = int(seed) + 100 * int(drone_id)
        print(
            f"[{index}/{len(all_runs)}] run {setting['id']} seed={seed} "
            f"drone={drone_id} effective_seed={effective_seed}",
            flush=True,
        )
        started = time.perf_counter()
        record = {
            "status": "failed",
            "configuration_id": str(setting["id"]),
            "configuration_label": str(setting["label"]),
            "groups": ";".join(setting["groups"]),
            "seed": int(seed),
            "effective_seed": effective_seed,
            "drone_id": int(drone_id),
            "display_drone_id": int(drone_id) + 1,
            "success": 0,
            "final_hypervolume": 0.0,
            "hypervolume_auc": 0.0,
            "error": "",
        }
        try:
            env = Environment()
            layer_index = _assign_layers(env)[drone_id]
            record["layer_index"] = int(layer_index)
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
                prior_anchor_fraction=float(setting["prior_anchor_fraction"]),
                prior_anchor_truncation_step=int(config["prior_anchor_truncation_step"]),
                pareto_archive_size=int(config["pareto_archive_size"]),
                parent_direction_count=int(config["parent_direction_count"]),
                offspring_resample_attempts=int(config["offspring_resample_attempts"]),
                truncation_low=int(setting["truncation_low"]),
                truncation_initial=int(setting["truncation_initial"]),
                truncation_high=int(setting["truncation_high"]),
                projection_feedback_threshold=float(config["projection_feedback_threshold"]),
                communication_parent_fraction=float(config["communication_parent_fraction"]),
                communication_elite_fraction=float(config["communication_elite_fraction"]),
                topology_noise_scale=float(config["topology_noise_scale"]),
                topology_noise_scale_min=float(config["topology_noise_scale_min"]),
                topology_noise_decay=float(config["topology_noise_decay"]),
                topology_projection_threshold=float(config["topology_projection_threshold"]),
            )
            metrics = evaluate_trajectory(
                path,
                env,
                radio_map,
                pipe_radius=float(config["pipe_radius_m"]),
                outage_threshold_db=float(config["outage_pathloss_threshold_db"]),
                sample_count=int(config["path_samples"]),
            )
            history_path = run_dir / "history.csv"
            _write_history(history_path, result.history)
            np.savetxt(
                run_dir / "trajectory.csv",
                np.asarray(path, dtype=float),
                delimiter=",",
                header="x_m,y_m,z_m",
                comments="",
            )
            export_pareto_csv(result, run_dir / "pareto.csv")
            history_stats = _history_statistics(result.history, budget)
            record.update(
                {
                    "status": "complete",
                    "success": int(metrics.constraint_violation <= 1e-9),
                    "evaluation_count": int(result.evaluation_count),
                    **history_stats,
                    "projection_count": int(result.projection_count),
                    "projection_per_100_evaluations": float(
                        100.0 * result.projection_count / max(1, result.evaluation_count)
                    ),
                    "offspring_resample_count": int(result.offspring_resample_count),
                    "topology_seed_count": int(result.topology_seed_count),
                    "pareto_size": int(len(result.pareto.length)),
                    **metrics.to_dict(),
                    "history_path": str(history_path.relative_to(output_dir)),
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        record["runtime_s"] = float(time.perf_counter() - started)
        result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        executed += 1
        records = _collect_records(output_dir)
        _write_csv(output_dir / "records.csv", records, RECORD_FIELDS)
        elapsed = time.perf_counter() - run_start
        average = elapsed / max(1, executed)
        remaining = max(0, len(all_runs) - index)
        print(
            f"  status={record['status']} runtime={record['runtime_s']:.1f}s "
            f"estimated_remaining={average * remaining / 3600.0:.2f}h",
            flush=True,
        )

    records = _collect_records(output_dir)
    completed = sum(row.get("status") == "complete" for row in records)
    failed = sum(row.get("status") == "failed" for row in records)
    manifest.update(
        {
            "status": "complete" if completed == len(all_runs) else "partial",
            "finished_at_unix_s": time.time(),
            "completed_runs": completed,
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
        print(json.dumps({
            "status": "ready",
            "drones": config["drones"],
            "seeds": config["seeds"],
            "configuration_count": len(config["configurations"]),
            "expected_single_uav_runs": len(list(iter_runs(config))),
            "evaluation_budget_per_run": config["evaluation_budget"],
            "output_dir": str(_resolve(root, config["output_dir"])),
        }, indent=2))
        return
    print(run_experiment(
        root,
        config_path,
        smoke=bool(args.smoke),
        rerun_failed=bool(args.rerun_failed),
    ))


if __name__ == "__main__":
    main()
