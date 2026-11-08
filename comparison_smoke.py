"""Small end-to-end validation for the two additional formal baselines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .experiment import _assign_layers, _resolve, _restrict_environment, _write_history


def run_smoke(
    root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    population_size: int = 50,
    evaluation_budget: int = 400,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from environment import Environment
    from ca_diffusion_es.baselines import (
        run_diffusion_only_baseline,
        run_nsga2_baseline,
    )
    from ca_diffusion_es.evaluation import evaluate_trajectory
    from ca_diffusion_es.legacy import load_legacy_modules
    from ca_diffusion_es.optimizer import export_pareto_csv
    from ca_diffusion_es.radio_map import load_hgnn_radio_map

    _, planner_module = load_legacy_modules(root)
    radio_path = _resolve(root, config["radio_map_csv"])
    radio_map = load_hgnn_radio_map(radio_path)
    env = _restrict_environment(Environment(), 5)
    drone_id = 0
    layer_index = _assign_layers(env)[drone_id]
    seed = 42
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for method in ("nsga2", "diffusion_only"):
        started = time.perf_counter()
        if method == "nsga2":
            result = run_nsga2_baseline(
                env,
                planner_module,
                radio_map,
                drone_id,
                layer_index,
                population_size=population_size,
                generations=max(1, int(np.ceil(evaluation_budget / population_size))),
                evaluation_budget=evaluation_budget,
                seed=seed,
                pipe_radius=float(config["pipe_radius_m"]),
                outage_threshold_db=float(config["outage_pathloss_threshold_db"]),
                control_points=int(config["control_points"]),
                path_samples=int(config["path_samples"]),
                hypervolume_reference=tuple(config["hypervolume_reference"]),
            )
        else:
            result = run_diffusion_only_baseline(
                env,
                planner_module,
                radio_map,
                drone_id,
                layer_index,
                legacy_root=root,
                dataset_path=_resolve(root, config["prior_dataset"]),
                checkpoint_path=_resolve(root, config["prior_checkpoint"]),
                population_size=population_size,
                evaluation_budget=evaluation_budget,
                seed=seed,
                pipe_radius=float(config["pipe_radius_m"]),
                outage_threshold_db=float(config["outage_pathloss_threshold_db"]),
                control_points=int(config["control_points"]),
                path_samples=int(config["path_samples"]),
                hypervolume_reference=tuple(config["hypervolume_reference"]),
                final_communication_weight=float(config["final_communication_weight"]),
                pareto_archive_size=int(config["pareto_archive_size"]),
            )
        if result.evaluation_count != evaluation_budget:
            raise RuntimeError(
                f"{method} used {result.evaluation_count} evaluations; "
                f"expected {evaluation_budget}"
            )
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            method_dir / "trajectory_drone0.csv",
            np.asarray(result.path, dtype=float),
            delimiter=",",
            header="x_m,y_m,z_m",
            comments="",
        )
        _write_history(method_dir / "history_drone0.csv", result.history)
        export_pareto_csv(result, method_dir / "pareto_drone0.csv")
        metrics = evaluate_trajectory(
            result.path,
            env,
            radio_map,
            pipe_radius=float(config["pipe_radius_m"]),
            outage_threshold_db=float(config["outage_pathloss_threshold_db"]),
            sample_count=int(config["path_samples"]),
        )
        record = {
            "method": method,
            "evaluation_count": int(result.evaluation_count),
            "runtime_s": float(time.perf_counter() - started),
            "pareto_size": int(len(result.pareto.length)),
            "final_hypervolume": float(result.history[-1]["hypervolume"]),
            "success": int(metrics.constraint_violation <= 1e-9),
            **metrics.to_dict(),
        }
        summary.append(record)
        print(
            f"{method}: evaluations={result.evaluation_count} "
            f"success={record['success']} runtime={record['runtime_s']:.1f}s",
            flush=True,
        )

    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--budget", type=int, default=400)
    args = parser.parse_args()
    print(
        run_smoke(
            args.legacy_root.resolve(),
            args.config.resolve(),
            args.output_dir.resolve(),
            population_size=args.population,
            evaluation_budget=args.budget,
        )
    )


if __name__ == "__main__":
    main()
