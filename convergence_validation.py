"""Paired validation of v9, HV feedback, and HV feedback with replay decay.

Uses existing radio-map/prior assets, a fixed evaluation budget, and separate
validation seeds. This is a per-UAV planning validation, not a new formal result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .parameter_experiment_v9 import _assign_layers, _write_csv, _write_history


def _resolve(root: Path, value: str) -> Path:
    return (root / value).resolve()


def load_config(root: Path, config_path: Path) -> dict:
    overrides = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base = json.loads(_resolve(root, overrides["base_config"]).read_text(encoding="utf-8-sig"))
    config = {**base, **overrides}
    if config["drones"] != [0, 1, 2, 3, 4]:
        raise ValueError("This validation uses all five N=5 UAVs: [0,1,2,3,4]")
    seeds = config["seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(int(s) != s or s < 0 for s in seeds):
        raise ValueError("Use three distinct non-negative integer validation seeds")
    if config["population_size"] != 100 or config["generations"] != 40 or config["evaluation_budget"] != 4000:
        raise ValueError("Keep the formal budget: population=100, generations=40, evaluations=4000")
    if config["path_samples"] != 120:
        raise ValueError("The current integration uses 120 trajectory samples")
    reference = np.asarray(config["hypervolume_reference"], dtype=float)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)) or np.any(reference <= 0):
        raise ValueError("A finite positive two-objective HV reference is required")
    settings = config["configurations"]
    expected = ["v9_original", "hv_feedback", "hv_replay_decay"]
    if [s["id"] for s in settings] != expected:
        raise ValueError(f"Expected configurations in this order: {expected}")
    upper = float(config["replay_parent_fraction_min"])
    for setting, rule in zip(settings, ["compromise", "hypervolume", "hypervolume"]):
        lower = float(setting["replay_exploration_fraction"])
        if setting["progress_rule"] != rule or not 0 <= lower <= upper <= 1:
            raise ValueError(f"Invalid feedback configuration: {setting['id']}")
    if any(float(s["replay_exploration_fraction"]) != upper for s in settings[:2]):
        raise ValueError("Original v9 and HV-only must retain the same fixed replay floor")
    if not float(settings[2]["replay_exploration_fraction"]) < upper:
        raise ValueError("The replay-decay variant needs a lower exploration floor")
    # Existence checks prevent prepare_prior from generating or training assets.
    for field in ("radio_map_csv", "prior_dataset", "prior_checkpoint"):
        if not _resolve(root, config[field]).is_file():
            raise FileNotFoundError(_resolve(root, config[field]))
    if float(config.get("channel_macro_fraction", 0.0)) != 0.0:
        raise ValueError("Keep macro mutation disabled in this first controlled validation")
    return config


def iter_runs(config: dict, smoke: bool = False):
    # Interleave versions at identical endpoints/seeds instead of running all
    # baseline jobs first. Completed jobs can be resumed individually.
    for seed in config["seeds"][:1] if smoke else config["seeds"]:
        for drone in config["drones"][:1] if smoke else config["drones"]:
            for setting in config["configurations"]:
                yield setting, int(seed), int(drone)


def _run_dir(output: Path, setting: dict, seed: int, drone: int) -> Path:
    return output / setting["id"] / f"seed_{seed:04d}" / f"drone_{drone}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(root: Path, config: dict, smoke: bool) -> dict:
    package = Path(__file__).resolve().parent
    files = {f"ca_diffusion_es/{p.name}": p for p in package.glob("*.py")}
    files.update({p.name: p for p in root.glob("*.py")})
    files.update({key: _resolve(root, config[key]) for key in
                  ("radio_map_csv", "prior_dataset", "prior_checkpoint")})
    return {"config": config, "smoke": smoke,
            "sha256": {name: _sha256(p) for name, p in sorted(files.items())}}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def history_curve(history: list[dict], grid: np.ndarray, area: float) -> np.ndarray:
    evaluations = np.asarray([float(r["evaluations"]) for r in history])
    hv = np.asarray([float(r["hypervolume"]) for r in history])
    if not len(hv) or not np.all(np.isfinite(hv)) or not np.all(np.isfinite(evaluations)):
        raise ValueError("History must contain finite evaluation counts and HV values")
    order = np.argsort(evaluations)
    evaluations = evaluations[order]
    if np.any(np.diff(evaluations) <= 0):
        raise ValueError("History evaluation counts must be distinct and increasing")
    # Match the existing publication report, including the first-evaluation edge.
    values = np.maximum.accumulate(np.clip(hv[order] / area, 0.0, 1.0))
    return np.interp(grid, evaluations, values, left=0.0, right=values[-1])


def _records(output: Path, jobs: list) -> list[dict]:
    rows = []
    for setting, seed, drone in jobs:
        path = _run_dir(output, setting, seed, drone) / "run_result.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _save_records(output: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if fields:
        _write_csv(output / "records.csv", rows, fields)


def run_validation(root: Path, config: dict, smoke: bool = False) -> Path:
    from environment import Environment
    from . import integration
    from .diffusion_model import DEVICE
    from .evaluation import evaluate_trajectory
    from .experiment import _restrict_environment
    from .optimizer import export_pareto_csv
    from .optimizer_convergence_validation import HVFeedbackCADiffusionES
    from .optimizer_v8 import configure_topology_trust_region
    from .optimizer_v9 import configure_replay_anchor
    from .optimizer_v9_dynamic4 import FinalReplayFeedbackCADiffusionES, configure_dynamic_replay
    from .optimizer_v9_fixed import configure_replay_parent_feedback
    from .radio_map import load_hgnn_radio_map
    import torch

    output = _resolve(root, config["output_dir"])
    if smoke:
        output = output.with_name(output.name + "_smoke")
    output.mkdir(parents=True, exist_ok=True)
    identity = _identity(root, config, smoke)
    identity["runtime"] = {"python": sys.version, "torch": torch.__version__,
                           "numpy": np.__version__, "device": str(DEVICE)}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise ValueError("Code, inputs, runtime, or settings changed. Use a NEW output_dir; existing runs cannot be mixed.")
    elif any(output.iterdir()):
        raise ValueError("Output directory is non-empty without a manifest. Use a new output_dir.")
    jobs = list(iter_runs(config, smoke))
    manifest = {"identity": identity, "status": "running", "expected_runs": len(jobs),
                "platform": platform.platform(), "resumed_at_unix_s": time.time()}
    _write_json(manifest_path, manifest)
    population = 20 if smoke else int(config["population_size"])
    generations = 2 if smoke else int(config["generations"])
    budget = population * generations
    area = float(np.prod(config["hypervolume_reference"]))

    configure_topology_trust_region(
        max_radius_m=float(config["topology_residual_max_radius_m"]),
        clearance_reserve_m=float(config["topology_clearance_reserve_m"]),
        minimum_scale=float(config["topology_trust_scale_min"]),
        decay=float(config["topology_trust_decay"]),
        recovery=float(config["topology_trust_recovery"]),
        feasible_target=float(config["topology_trust_feasible_target"]),
    )
    configure_replay_anchor(max_radius_m=float(config["replay_anchor_max_radius_m"]))
    configure_dynamic_replay(feasible_target=0.85, feasible_streak=2, decay=0.8, recovery=0.1)
    radio_path = _resolve(root, config["radio_map_csv"])
    dataset = _resolve(root, config["prior_dataset"])
    checkpoint = _resolve(root, config["prior_checkpoint"])
    radio_map = load_hgnn_radio_map(radio_path)
    # Load immutable assets once before timing any variant. No training occurs.
    integration.prepare_prior(root, dataset_path=dataset, checkpoint_path=checkpoint)
    float_fields = ("final_communication_weight", "communication_weight_min",
                    "communication_weight_max", "prior_anchor_fraction",
                    "projection_feedback_threshold", "communication_parent_fraction",
                    "communication_elite_fraction", "topology_noise_scale",
                    "topology_noise_scale_min", "topology_noise_decay", "topology_projection_threshold")
    int_fields = ("prior_anchor_truncation_step", "pareto_archive_size", "parent_direction_count",
                  "offspring_resample_attempts", "truncation_low", "truncation_initial", "truncation_high")
    options = {name: float(config[name]) for name in float_fields}
    options.update({name: int(config[name]) for name in int_fields})

    for index, (setting, seed, drone) in enumerate(jobs, 1):
        run_dir = _run_dir(output, setting, seed, drone)
        result_path = run_dir / "run_result.json"
        if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
            print(f"[{index}/{len(jobs)}] skip completed {setting['id']} seed={seed} UAV={drone + 1}", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {"configuration_id": setting["id"], "configuration_label": setting["label"],
                  "seed": seed, "drone_id": drone, "effective_seed": seed + 100 * drone,
                  "progress_rule": setting["progress_rule"],
                  "replay_exploration_fraction": setting["replay_exploration_fraction"],
                  "status": "failed", "success": 0, "error": ""}
        print(f"[{index}/{len(jobs)}] {setting['id']} seed={seed} UAV={drone + 1}", flush=True)
        original_class = integration.CADiffusionES
        try:
            configure_replay_parent_feedback(
                minimum_fraction=float(config["replay_parent_fraction_min"]),
                exploration_fraction=float(setting["replay_exploration_fraction"]),
            )
            integration.CADiffusionES = (FinalReplayFeedbackCADiffusionES
                if setting["progress_rule"] == "compromise" else HVFeedbackCADiffusionES)
            env = _restrict_environment(Environment(), 5)
            layer = _assign_layers(env)[drone]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            path, result, _ = integration.plan_drone_ca_diffusion_es(
                env, drone, layer, [], legacy_root=root, hgnn_csv=radio_path,
                radio_map=radio_map, channel_enabled=True, pipe_radius=float(config["pipe_radius_m"]),
                population_size=population, generations=generations, seed=record["effective_seed"],
                multi_objective=True, dataset_path=dataset, checkpoint_path=checkpoint,
                evaluation_budget=budget, patience=None, require_feasible=False,
                hypervolume_reference=tuple(config["hypervolume_reference"]),
                outage_threshold_db=float(config["outage_pathloss_threshold_db"]), **options,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            record["runtime_s"] = time.perf_counter() - started
            if result.evaluation_count != budget:
                raise ValueError(f"Expected {budget} evaluations, got {result.evaluation_count}")
            metrics = evaluate_trajectory(path, env, radio_map,
                pipe_radius=float(config["pipe_radius_m"]),
                outage_threshold_db=float(config["outage_pathloss_threshold_db"]),
                sample_count=int(config["path_samples"]))
            _write_history(run_dir / "history.csv", result.history)
            np.savetxt(run_dir / "trajectory.csv", np.asarray(path), delimiter=",", header="x_m,y_m,z_m", comments="")
            export_pareto_csv(result, run_dir / "pareto.csv")
            record.update(metrics.to_dict())
            record.update(status="complete", success=int(metrics.constraint_violation <= 1e-9),
                layer_index=layer, evaluation_count=result.evaluation_count,
                final_normalized_hv=float(result.history[-1]["hypervolume"]) / area,
                pareto_size=len(result.pareto.length), projection_count=result.projection_count,
                offspring_resample_count=result.offspring_resample_count,
                replay_active_generations=sum(float(r.get("replay_trust_active", 0)) > 0 for r in result.history),
                minimum_replay_floor=min(float(r.get("replay_parent_fraction_next", 0.5)) for r in result.history),
                history_path=str((run_dir / "history.csv").relative_to(output)))
            if not smoke:
                grid = np.arange(0, budget + 1, 100, dtype=float)
                curve = history_curve(result.history, grid, area)
                integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                record["normalized_hv_auc"] = float(integrate(curve, grid) / budget)
                for evaluations in (1000, 2000, 4000):
                    record[f"normalized_hv_at_{evaluations}"] = float(curve[evaluations // 100])
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"  FAILED: {record['error']}", flush=True)
        finally:
            integration.CADiffusionES = original_class
            configure_replay_parent_feedback(minimum_fraction=float(config["replay_parent_fraction_min"]))
        _write_json(result_path, record)
        _save_records(output, _records(output, jobs))
        print(f"  {record['status']}; HV={record.get('final_normalized_hv', 'unavailable')}; seconds={record.get('runtime_s', 'unavailable')}", flush=True)
    rows = _records(output, jobs)
    _save_records(output, rows)
    completed = sum(r["status"] == "complete" for r in rows)
    manifest.update(status="complete" if completed == len(jobs) else "partial",
                    completed_runs=completed, failed_runs=len(rows) - completed,
                    finished_at_unix_s=time.time())
    _write_json(manifest_path, manifest)
    if completed != len(jobs):
        raise RuntimeError(f"{len(jobs) - completed} runs failed. Rerun the SAME command to retry them; completed runs are retained.")
    return output


def generate_report(output: Path, make_plot: bool = True) -> Path:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["identity"]["config"]
    if manifest["identity"]["smoke"]:
        raise ValueError("Smoke runs are functional checks, not convergence experiments")
    jobs = list(iter_runs(config))
    records = _records(output, jobs)
    if len(records) != len(jobs) or any(r["status"] != "complete" for r in records):
        raise ValueError("Report requires all 45 runs; failed or missing runs must not be silently excluded")
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    grid = np.arange(0, config["evaluation_budget"] + 1, 100, dtype=float)
    area = float(np.prod(config["hypervolume_reference"]))
    metrics = ["normalized_hv_at_1000", "normalized_hv_at_2000", "normalized_hv_at_4000",
               "normalized_hv_auc", "runtime_s", "path_length_m", "communication_cost",
               "success", "turn_p95_deg", "outage_ratio", "replay_active_generations",
               "minimum_replay_floor"]
    summaries, seed_rows, curves = [], [], []
    plot_series = []
    for setting in config["configurations"]:
        subset = [r for r in records if r["configuration_id"] == setting["id"]]
        seed_curves = []
        local_seed_rows = []
        for seed in config["seeds"]:
            group = [r for r in subset if r["seed"] == seed]
            if sorted(r["drone_id"] for r in group) != config["drones"]:
                raise ValueError("Every seed must contain the same five UAVs")
            row = {"configuration_id": setting["id"], "seed": seed}
            row.update({metric: float(np.mean([r[metric] for r in group])) for metric in metrics})
            local_seed_rows.append(row)
            drone_curves = []
            for record in group:
                with (output / record["history_path"]).open(newline="", encoding="utf-8") as handle:
                    drone_curves.append(history_curve(list(csv.DictReader(handle)), grid, area))
            seed_curves.append(np.mean(drone_curves, axis=0))
        seed_rows.extend(local_seed_rows)
        summary = {"configuration_id": setting["id"], "completed_runs": len(subset), "seed_count": len(local_seed_rows)}
        for metric in metrics:
            values = [r[metric] for r in local_seed_rows]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std_across_seeds"] = float(np.std(values, ddof=1))
        summaries.append(summary)
        mean = np.mean(seed_curves, axis=0)
        std = np.std(seed_curves, axis=0, ddof=1)
        plot_series.append((setting, mean))
        for e, m, s in zip(grid, mean, std):
            curves.append({"configuration_id": setting["id"], "evaluations": int(e),
                           "mean_normalized_feasible_hypervolume": float(m),
                           "std_across_seeds": float(s), "seed_count": len(seed_curves)})
    paired = []
    for row in seed_rows:
        if row["configuration_id"] == "v9_original":
            continue
        baseline = next(r for r in seed_rows if r["configuration_id"] == "v9_original" and r["seed"] == row["seed"])
        paired.append({"configuration_id": row["configuration_id"], "seed": row["seed"],
                       **{f"{m}_minus_original": row[m] - baseline[m] for m in metrics}})
    for name, rows in [("summary.csv", summaries), ("seed_summary.csv", seed_rows),
                       ("paired_differences.csv", paired), ("convergence_data.csv", curves)]:
        _write_csv(report / name, rows, list(rows[0]))
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 5))
        for (setting, mean), color in zip(plot_series, ["#75b6b1", "#d55e00", "#3479ad"]):
            axis.plot(grid, mean, label=setting["label"], color=color, linewidth=2)
        axis.set(xlabel="Evaluations", ylabel="Normalized feasible hypervolume",
                 xlim=(0, config["evaluation_budget"]), ylim=(0, 1))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(report / "convergence.png", dpi=300)
        figure.savefig(report / "convergence.svg")
        plt.close(figure)
    print("\nConfiguration             HV@1000    HV@4000    mean seconds")
    for row in summaries:
        print(f"{row['configuration_id']:25s} {row['normalized_hv_at_1000_mean']:.6f}   {row['normalized_hv_at_4000_mean']:.6f}   {row['runtime_s_mean']:.2f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "smoke", "run", "report", "all"])
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.legacy_root.resolve()
    if args.action == "report":
        # Report from the saved manifest rather than silently using new settings.
        config = load_config(root, args.config.resolve())
        print(generate_report(_resolve(root, config["output_dir"])))
        return
    config = load_config(root, args.config.resolve())
    if args.action == "check":
        from .optimizer_convergence_validation import HVFeedbackCADiffusionES
        from .optimizer import CADiffusionES
        assert hasattr(CADiffusionES, "_archive_progress_indicator")
        print(json.dumps({"status": "ready", "drones": config["drones"], "seeds": config["seeds"],
            "runs": len(list(iter_runs(config))), "evaluations_per_run": config["evaluation_budget"],
            "hypervolume_reference": config["hypervolume_reference"],
            "output_dir": str(_resolve(root, config["output_dir"]))}, indent=2))
        return
    output = run_validation(root, config, smoke=args.action == "smoke")
    if args.action == "all":
        print(generate_report(output))
    else:
        print(output)


if __name__ == "__main__":
    main()
