"""Command-line entry points for prior preparation and multi-UAV planning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .integration import build_radio_map, plan_drone_ca_diffusion_es, prepare_prior
from .legacy import load_legacy_modules


DEFAULT_LEGACY_ROOT = Path(r"D:\desktop\code")
DEFAULT_HGNN_CSV = DEFAULT_LEGACY_ROOT / "ALL_predicted_vs_true_pathloss_hgnn_split_5.3.csv"


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).parent / "cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-samples", type=int, default=1200)
    parser.add_argument("--training-epochs", type=int, default=300)
    parser.add_argument("--diffusion-steps", type=int, default=25)


def _cache_paths(args) -> tuple[Path, Path]:
    return (
        args.cache_dir / "legal_prior_local_context_perturbed_replay_v7_30cp.npz",
        args.cache_dir
        / f"trajectory_prior_local_context_perturbed_replay_v7_30cp_T{args.diffusion_steps}.pt",
    )


def prepare_command(args) -> None:
    dataset_path, checkpoint_path = _cache_paths(args)
    _, dataset, final_dataset, final_checkpoint = prepare_prior(
        args.legacy_root,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        sample_count=args.dataset_samples,
        epochs=args.training_epochs,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
        hgnn_csv=args.hgnn_csv,
        hard_replay_fraction=args.hard_replay_fraction,
        force_dataset=args.force_dataset,
        force_checkpoint=args.force_checkpoint,
    )
    print(
        json.dumps(
            {
                "dataset": str(final_dataset),
                "checkpoint": str(final_checkpoint),
                "samples": int(dataset["X"].shape[0]),
                "uses_astar": False,
            },
            indent=2,
        )
    )


def plan_command(args) -> None:
    environment_module, _ = load_legacy_modules(args.legacy_root)
    env = environment_module.Environment()
    drone_ids = [int(value) for value in args.drones.split(",") if value.strip()]
    if any(drone_id < 0 or drone_id >= len(env.starts) for drone_id in drone_ids):
        raise ValueError(f"Drone ids must be between 0 and {len(env.starts) - 1}")
    dataset_path, checkpoint_path = _cache_paths(args)
    radio_map = build_radio_map(
        args.hgnn_csv, enabled=not args.no_channel
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[np.ndarray] = []
    summary = []
    for order, drone_id in enumerate(drone_ids):
        layer_index = drone_id if drone_id < len(env.safe_layers) else drone_id % len(env.safe_layers)
        pareto_path = args.output_dir / f"pareto_drone{drone_id}.csv"
        history_path = args.output_dir / f"history_drone{drone_id}.csv"
        path, result, _ = plan_drone_ca_diffusion_es(
            env,
            drone_id,
            layer_index,
            paths,
            legacy_root=args.legacy_root,
            hgnn_csv=args.hgnn_csv,
            radio_map=radio_map,
            channel_enabled=not args.no_channel,
            pipe_radius=args.pipe_radius,
            population_size=args.population,
            generations=args.generations,
            seed=args.seed + order * 100,
            multi_objective=not args.single_objective,
            pareto_csv=pareto_path,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            dataset_samples=args.dataset_samples,
            training_epochs=args.training_epochs,
            diffusion_steps=args.diffusion_steps,
        )
        paths.append(path)
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            fields = list(result.history[0]) if result.history else []
            writer = csv.DictWriter(handle, fieldnames=fields)
            if fields:
                writer.writeheader()
                writer.writerows(result.history)
        trajectory_path = args.output_dir / f"trajectory_drone{drone_id}.csv"
        np.savetxt(
            trajectory_path,
            path,
            delimiter=",",
            header="x_m,y_m,z_m",
            comments="",
        )
        summary.append(
            {
                "drone_id": drone_id,
                "layer_index": layer_index,
                "trajectory": str(trajectory_path),
                "pareto": str(pareto_path),
                "history": str(history_path),
                "length_m": result.length,
                "communication_cost": result.communication_cost,
                "mean_predicted_pathloss_db": result.mean_pathloss_db,
                "worst_predicted_pathloss_db": result.worst_pathloss_db,
                "min_clearance_m": result.min_clearance,
                "collision_count": result.collision_count,
                "constraint_violation": result.constraint_violation,
                "topology_seed_count": result.topology_seed_count,
                "offspring_resample_count": result.offspring_resample_count,
                "projection_count": result.projection_count,
                "pareto_size": len(result.pareto.length),
            }
        )
    from main import check_all_pipe_collisions_timeaware, schedule_with_time_windows

    start_times = schedule_with_time_windows(
        paths, pipe_radius=args.pipe_radius, speed=12.0, delta_t=2.0
    )
    conflicts = check_all_pipe_collisions_timeaware(
        paths, start_times, speed=12.0, pipe_radius=args.pipe_radius
    )
    if conflicts:
        raise RuntimeError(
            f"Time-window scheduling left {len(conflicts)} unresolved UAV conflicts"
        )
    for record, start_time in zip(summary, start_times):
        record["start_time_s"] = float(start_time)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Generate legal prior data and train DDPM")
    _common_parser(prepare_parser)
    prepare_parser.add_argument("--hgnn-csv", type=Path, default=DEFAULT_HGNN_CSV)
    prepare_parser.add_argument("--hard-replay-fraction", type=float, default=0.15)
    prepare_parser.add_argument("--force-dataset", action="store_true")
    prepare_parser.add_argument("--force-checkpoint", action="store_true")
    prepare_parser.set_defaults(handler=prepare_command)

    plan_parser = subparsers.add_parser("plan", help="Run A*-free CA-Diffusion-ES")
    _common_parser(plan_parser)
    plan_parser.add_argument("--hgnn-csv", type=Path, default=DEFAULT_HGNN_CSV)
    plan_parser.add_argument("--drones", default="0,1,2,3,4")
    plan_parser.add_argument("--population", type=int, default=100)
    plan_parser.add_argument("--generations", type=int, default=40)
    plan_parser.add_argument("--pipe-radius", type=float, default=3.5)
    plan_parser.add_argument("--single-objective", action="store_true")
    plan_parser.add_argument("--no-channel", action="store_true")
    plan_parser.add_argument("--output-dir", type=Path, default=Path("runs") / "ca_diffusion_es")
    plan_parser.set_defaults(handler=plan_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
