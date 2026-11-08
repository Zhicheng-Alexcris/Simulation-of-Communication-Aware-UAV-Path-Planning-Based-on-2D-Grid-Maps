"""Drop-in integration with the existing UAV corridor environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .diffusion_model import (
    ConditionalTrajectoryDDPM,
    train_prior,
)
from .legacy import load_legacy_modules
from .optimizer import CADiffusionES, OptimizationResult, export_pareto_csv
from .prior_data import generate_legal_dataset, load_prior_dataset
from .radio_map import HGNNRadioMap3D, load_hgnn_radio_map


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = PACKAGE_ROOT / "cache"


@lru_cache(maxsize=4)
def _load_prior_assets(
    dataset_path_text: str,
    dataset_mtime_ns: int,
    checkpoint_path_text: str,
    checkpoint_mtime_ns: int,
):
    """Load immutable inference assets once per process and file version."""
    del dataset_mtime_ns, checkpoint_mtime_ns
    dataset = load_prior_dataset(Path(dataset_path_text))
    prior = ConditionalTrajectoryDDPM.load(Path(checkpoint_path_text))
    return prior, dataset

def prepare_prior(
    legacy_root: str | Path,
    *,
    dataset_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    sample_count: int = 1200,
    epochs: int = 300,
    diffusion_steps: int = 25,
    seed: int = 0,
    hgnn_csv: str | Path | None = None,
    radio_map: HGNNRadioMap3D | None = None,
    hard_replay_fraction: float = 0.15,
    force_dataset: bool = False,
    force_checkpoint: bool = False,
):
    dataset_path = Path(dataset_path or DEFAULT_CACHE / "legal_prior_local_context_perturbed_replay_v7_30cp.npz")
    checkpoint_path = Path(
        checkpoint_path or DEFAULT_CACHE / f"trajectory_prior_local_context_perturbed_replay_v7_30cp_T{diffusion_steps}.pt"
    )
    if force_dataset or not dataset_path.exists():
        if radio_map is None:
            radio_path = Path(
                hgnn_csv
                or Path(legacy_root)
                / "ALL_predicted_vs_true_pathloss_hgnn_split_5.3.csv"
            )
            radio_map = load_hgnn_radio_map(radio_path)
        generate_legal_dataset(
            legacy_root,
            dataset_path,
            sample_count=sample_count,
            radio_map=radio_map,
            hard_replay_fraction=float(hard_replay_fraction),
            seed=seed,
        )
        force_checkpoint = True
    if checkpoint_path.exists() and not force_checkpoint:
        prior, dataset = _load_prior_assets(
            str(dataset_path.resolve()),
            int(dataset_path.stat().st_mtime_ns),
            str(checkpoint_path.resolve()),
            int(checkpoint_path.stat().st_mtime_ns),
        )
        expected_dimension = int(dataset["X"].shape[1])
        if prior.dimension_x != expected_dimension:
            raise ValueError("Prior checkpoint and legal trajectory dataset are incompatible")
        return prior, dataset, dataset_path, checkpoint_path

    dataset = load_prior_dataset(dataset_path)
    prior = ConditionalTrajectoryDDPM(
        dimension_x=int(dataset["X"].shape[1]),
        dimension_condition=int(dataset["C"].shape[1]),
        diffusion_steps=int(diffusion_steps),
    )
    train_prior(
        prior,
        dataset["X"],
        dataset["C"],
        epochs=epochs,
        seed=seed,
    )
    prior.save(checkpoint_path)
    return prior, dataset, dataset_path, checkpoint_path


def build_radio_map(
    hgnn_csv: str | Path | None,
    *,
    enabled: bool = True,
    predicted_column: str = "Predicted_PathLoss",
) -> HGNNRadioMap3D | None:
    if not enabled:
        return None
    if hgnn_csv is None:
        raise ValueError("hgnn_csv is required when channel awareness is enabled")
    return load_hgnn_radio_map(hgnn_csv, predicted_column=predicted_column)


def plan_drone_ca_diffusion_es(
    env,
    drone_id: int,
    layer_index: int,
    other_paths,
    *,
    legacy_root: str | Path,
    hgnn_csv: str | Path | None,
    radio_map: HGNNRadioMap3D | None = None,
    channel_enabled: bool = True,
    pipe_radius: float = 3.5,
    population_size: int = 100,
    generations: int = 40,
    seed: int = 0,
    multi_objective: bool = True,
    pareto_csv: str | Path | None = None,
    dataset_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    dataset_samples: int = 1200,
    training_epochs: int = 300,
    diffusion_steps: int = 25,
    evaluation_budget: int | None = None,
    patience: int | None = 8,
    hypervolume_reference: tuple[float, float] = (800.0, 1.05),
    require_feasible: bool = True,
    outage_threshold_db: float = 110.0,
    final_communication_weight: float = 0.65,
    communication_weight_min: float = 0.25,
    communication_weight_max: float = 0.95,
    prior_anchor_fraction: float = 0.5,
    prior_anchor_truncation_step: int = 1,
    pareto_archive_size: int = 400,
    parent_direction_count: int = 11,
    offspring_resample_attempts: int = 1,
    truncation_low: int = 1,
    truncation_initial: int = 3,
    truncation_high: int = 6,
    projection_feedback_threshold: float = 0.35,
    communication_parent_fraction: float = 0.25,
    communication_elite_fraction: float = 0.1,
    channel_macro_fraction: float = 0.0,
    channel_macro_truncation_step: int = 12,
    topology_noise_scale: float = 0.25,
    topology_noise_scale_min: float = 0.05,
    topology_noise_decay: float = 0.4,
    topology_projection_threshold: float = 0.35,
    record_candidates: bool = False,
) -> tuple[np.ndarray, OptimizationResult, object]:
    """Plan one UAV without calling A*, A* corridors, or A* fallback."""
    _, planner_module = load_legacy_modules(legacy_root)
    start_raw, end_raw = env.get_start_end(drone_id)
    start = np.asarray(
        [
            float(start_raw[0]),
            float(start_raw[1]),
            env.clamp_to_layer(max(float(start_raw[2]), env.min_height), layer_index),
        ],
        dtype=float,
    )
    end = np.asarray(
        [
            float(end_raw[0]),
            float(end_raw[1]),
            env.clamp_to_layer(max(float(end_raw[2]), env.min_height), layer_index),
        ],
        dtype=float,
    )
    planner = planner_module.SplinePathPlanner(
        env, n_control_points=30, pipe_radius=float(pipe_radius)
    )
    planner.set_cruise_layer(layer_index)
    # No corridor is set. Diffusion samples the complete admissible airspace.
    planner.set_other_drone_paths([])

    if radio_map is None:
        radio_map = build_radio_map(hgnn_csv, enabled=channel_enabled)
    prior, dataset, _, _ = prepare_prior(
        legacy_root,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        sample_count=dataset_samples,
        epochs=training_epochs,
        diffusion_steps=diffusion_steps,
        seed=seed,
        hgnn_csv=hgnn_csv,
        radio_map=radio_map,
    )
    optimizer = CADiffusionES(
        planner,
        start,
        end,
        prior,
        dataset,
        radio_map=radio_map,
        # Pairwise safety is enforced by the post-planning time-window scheduler.
        other_paths=[],
        seed=seed,
        minimum_separation=2.0 * float(pipe_radius),
        outage_threshold_db=float(outage_threshold_db),
    )
    result = optimizer.run(
        population_size=population_size,
        generations=generations,
        multi_objective=multi_objective,
        evaluation_budget=evaluation_budget,
        patience=patience,
        hypervolume_reference=hypervolume_reference,
        final_communication_weight=final_communication_weight,
        communication_weight_min=communication_weight_min,
        communication_weight_max=communication_weight_max,
        prior_anchor_fraction=prior_anchor_fraction,
        prior_anchor_truncation_step=prior_anchor_truncation_step,
        pareto_archive_size=pareto_archive_size,
        parent_direction_count=parent_direction_count,
        offspring_resample_attempts=offspring_resample_attempts,
        truncation_low=truncation_low,
        truncation_initial=truncation_initial,
        truncation_high=truncation_high,
        projection_feedback_threshold=projection_feedback_threshold,
        communication_parent_fraction=communication_parent_fraction,
        communication_elite_fraction=communication_elite_fraction,
        channel_macro_fraction=channel_macro_fraction,
        channel_macro_truncation_step=channel_macro_truncation_step,
        topology_noise_scale=topology_noise_scale,
        topology_noise_scale_min=topology_noise_scale_min,
        topology_noise_decay=topology_noise_decay,
        topology_projection_threshold=topology_projection_threshold,
        record_candidates=record_candidates,
    )
    if require_feasible and result.constraint_violation > 1e-9:
        raise RuntimeError(
            "CA-Diffusion-ES did not find a feasible trajectory; increase the "
            "population/generation budget instead of falling back to A*"
        )
    if pareto_csv is not None:
        export_pareto_csv(result, pareto_csv)
    return result.path, result, planner

