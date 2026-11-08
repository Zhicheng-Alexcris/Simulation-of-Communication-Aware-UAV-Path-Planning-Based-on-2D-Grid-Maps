"""Fair A* and A*+NSGA-II baselines for the formal comparison."""

from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .evaluation import evaluate_trajectory, resample_path
from .optimizer import (
    CADiffusionES,
    CandidateBatch,
    FeasibleParetoArchive,
    OptimizationResult,
)
from .pareto import compromise_index, feasible_pareto_mask, hypervolume_2d


def _aligned_endpoints(env, drone_id: int, layer_index: int) -> tuple[np.ndarray, np.ndarray]:
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
    return start, end


def _astar_skeleton(
    env,
    planner_module,
    start: np.ndarray,
    end: np.ndarray,
    layer_index: int,
    *,
    resolution: float,
    inflation: float,
) -> np.ndarray:
    candidates = [float(inflation), max(6.0, float(inflation) - 2.0), 6.0]
    skeleton = None
    for current_inflation in dict.fromkeys(candidates):
        skeleton = planner_module.astar_2d(
            env,
            tuple(start[:2]),
            tuple(end[:2]),
            res=float(resolution),
            inflation=float(current_inflation),
            cruise_layer=int(layer_index),
            bounds=None,
            beta_channel=0.0,
            channel_cost_fn=None,
        )
        if skeleton is not None and len(skeleton) > 2:
            break
    if skeleton is None:
        skeleton = [start, end]
    path = np.asarray(skeleton, dtype=float)
    path[:, 2] = env.layer_mid_z(layer_index)
    path[0] = start
    path[-1] = end
    return path


def _batch_from_paths(paths, latents, env, radio_map, pipe_radius, outage_threshold_db):
    metrics = [
        evaluate_trajectory(
            path,
            env,
            radio_map,
            pipe_radius=pipe_radius,
            outage_threshold_db=outage_threshold_db,
        )
        for path in paths
    ]
    return CandidateBatch(
        latent=np.asarray(latents, dtype=float),
        paths=[np.asarray(path, dtype=float) for path in paths],
        length=np.asarray([item.path_length_m for item in metrics], dtype=float),
        communication=np.asarray([item.communication_cost for item in metrics], dtype=float),
        mean_pathloss_db=np.asarray([item.mean_pathloss_db for item in metrics], dtype=float),
        worst_pathloss_db=np.asarray([item.worst_pathloss_db for item in metrics], dtype=float),
        constraint_violation=np.asarray(
            [item.constraint_violation for item in metrics], dtype=float
        ),
        collision_count=np.asarray([item.collision_count for item in metrics], dtype=float),
        min_clearance=np.asarray([item.min_clearance_m for item in metrics], dtype=float),
        separation_violation=np.zeros(len(metrics), dtype=float),
        topology_origin=np.zeros(len(metrics), dtype=bool),
    )


def _result_from_batch(
    batch: CandidateBatch,
    *,
    history: list[dict[str, float]],
    evaluation_count: int,
    communication_weight: float | None = None,
) -> OptimizationResult:
    mask = feasible_pareto_mask(batch.objectives, batch.constraint_violation)
    if np.any(mask):
        pareto = batch.take(np.flatnonzero(mask))
        weights = None
        if communication_weight is not None:
            communication_weight = float(communication_weight)
            if not 0.0 <= communication_weight <= 1.0:
                raise ValueError("communication_weight must lie within [0, 1]")
            weights = (1.0 - communication_weight, communication_weight)
        best_index = compromise_index(pareto.objectives, weights=weights)
        selected = pareto
    else:
        best_index = int(np.argmin(batch.constraint_violation))
        selected = batch
        pareto = batch.take(np.asarray([best_index]))
    return OptimizationResult(
        path=selected.paths[best_index],
        length=float(selected.length[best_index]),
        communication_cost=float(selected.communication[best_index]),
        mean_pathloss_db=float(selected.mean_pathloss_db[best_index]),
        worst_pathloss_db=float(selected.worst_pathloss_db[best_index]),
        constraint_violation=float(selected.constraint_violation[best_index]),
        min_clearance=float(selected.min_clearance[best_index]),
        collision_count=int(selected.collision_count[best_index]),
        history=history,
        pareto=pareto,
        evaluation_count=int(evaluation_count),
        topology_seed_count=0,
    )


def run_astar_baseline(
    env,
    planner_module,
    radio_map,
    drone_id: int,
    layer_index: int,
    *,
    pipe_radius: float,
    outage_threshold_db: float,
    astar_resolution: float = 2.0,
    astar_inflation: float = 12.0,
    hypervolume_reference: tuple[float, float] = (800.0, 1.05),
) -> OptimizationResult:
    start, end = _aligned_endpoints(env, drone_id, layer_index)
    skeleton = _astar_skeleton(
        env,
        planner_module,
        start,
        end,
        layer_index,
        resolution=astar_resolution,
        inflation=astar_inflation,
    )
    path = resample_path(skeleton, sample_count=120)
    batch = _batch_from_paths(
        [path], np.empty((1, 0)), env, radio_map, pipe_radius, outage_threshold_db
    )
    feasible = int(batch.constraint_violation[0] <= 1e-9)
    history = [
        {
            "generation": 0.0,
            "evaluations": 0.0,
            "truncation_step": 0.0,
            "feasible_count": float(feasible),
            "feasible_ratio": float(feasible),
            "pareto_size": float(feasible),
            "hypervolume": hypervolume_2d(
                batch.objectives if feasible else np.empty((0, 2)),
                hypervolume_reference,
            ),
            "indicator": float(batch.constraint_violation[0]),
        }
    ]
    return _result_from_batch(batch, history=history, evaluation_count=0)


class _HistoryCallback(Callback):
    def __init__(self, reference: tuple[float, float]):
        super().__init__()
        self.reference = reference
        self.records: list[dict[str, float]] = []

    def notify(self, algorithm):
        objectives = np.asarray(algorithm.pop.get("F"), dtype=float)
        constraints = np.asarray(algorithm.pop.get("G"), dtype=float)
        violation = np.maximum(constraints, 0.0).sum(axis=1)
        mask = feasible_pareto_mask(objectives, violation)
        feasible_count = int(np.sum(violation <= 1e-9))
        front = objectives[mask]
        self.records.append(
            {
                "generation": float(algorithm.n_gen - 1),
                "evaluations": float(algorithm.evaluator.n_eval),
                "truncation_step": 0.0,
                "feasible_count": float(feasible_count),
                "feasible_ratio": float(feasible_count / max(1, len(objectives))),
                "pareto_size": float(len(front)),
                "hypervolume": hypervolume_2d(front, self.reference),
                "indicator": float(np.min(violation)),
            }
        )


def run_astar_nsga2_baseline(
    env,
    planner_module,
    radio_map,
    drone_id: int,
    layer_index: int,
    *,
    population_size: int,
    generations: int,
    evaluation_budget: int,
    seed: int,
    pipe_radius: float,
    outage_threshold_db: float,
    control_points: int = 30,
    path_samples: int = 120,
    astar_resolution: float = 2.0,
    astar_inflation: float = 12.0,
    corridor_radius: float = 8.0,
    hypervolume_reference: tuple[float, float] = (800.0, 1.05),
) -> OptimizationResult:
    start, end = _aligned_endpoints(env, drone_id, layer_index)
    skeleton = _astar_skeleton(
        env,
        planner_module,
        start,
        end,
        layer_index,
        resolution=astar_resolution,
        inflation=astar_inflation,
    )
    planner = planner_module.SplinePathPlanner(
        env, n_control_points=int(control_points), pipe_radius=float(pipe_radius)
    )
    planner.set_cruise_layer(layer_index)
    planner.set_corridor(skeleton.tolist(), corridor_radius=float(corridor_radius))
    planner.set_other_drone_paths([])

    bounds = planner.generate_control_points_bounds(start, end)
    lower = []
    upper = []
    for index in range(int(control_points)):
        lower.extend(
            [bounds[f"x{index}"][0], bounds[f"y{index}"][0], bounds[f"z{index}"][0]]
        )
        upper.extend(
            [bounds[f"x{index}"][1], bounds[f"y{index}"][1], bounds[f"z{index}"][1]]
        )
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rng = np.random.default_rng(seed)
    initial = rng.uniform(lower, upper, size=(int(population_size), len(lower)))
    seed_controls = resample_path(skeleton, sample_count=int(control_points)).reshape(-1)
    initial[0] = np.clip(seed_controls, lower, upper)
    informed_count = max(1, int(population_size) // 2)
    if informed_count > 1:
        noise = rng.normal(
            0.0,
            0.08 * np.maximum(upper - lower, 1e-6),
            size=(informed_count - 1, len(lower)),
        )
        initial[1:informed_count] = np.clip(seed_controls + noise, lower, upper)

    def decode(vector: np.ndarray) -> np.ndarray:
        controls = np.asarray(vector, dtype=float).reshape(int(control_points), 3)
        controls[0] = start
        controls[-1] = end
        controls = np.asarray(
            planner._enforce_corridor_on_control_points(controls.tolist()), dtype=float
        )
        path = np.asarray(
            planner.create_b_spline(controls, n_samples=int(path_samples)), dtype=float
        )
        path = np.asarray(planner.project_path_into_corridor(path.tolist()), dtype=float)
        path[0] = start
        path[-1] = end
        return path

    class FormalNSGAProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=len(lower),
                n_obj=2,
                n_ieq_constr=2,
                xl=lower,
                xu=upper,
            )

        def _evaluate(self, variables, out, *args, **kwargs):
            objectives = []
            constraints = []
            for vector in variables:
                path = decode(vector)
                metrics = evaluate_trajectory(
                    path,
                    env,
                    radio_map,
                    pipe_radius=pipe_radius,
                    outage_threshold_db=outage_threshold_db,
                    sample_count=path_samples,
                )
                objectives.append([metrics.path_length_m, metrics.communication_cost])
                constraints.append(
                    [
                        float(metrics.collision_count),
                        float(pipe_radius) - metrics.min_clearance_m,
                    ]
                )
            out["F"] = np.asarray(objectives, dtype=float)
            out["G"] = np.asarray(constraints, dtype=float)

    callback = _HistoryCallback(hypervolume_reference)
    algorithm = NSGA2(
        pop_size=int(population_size),
        sampling=initial,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    result = minimize(
        FormalNSGAProblem(),
        algorithm,
        get_termination("n_eval", int(evaluation_budget)),
        seed=int(seed),
        callback=callback,
        verbose=False,
    )
    final_variables = np.asarray(result.pop.get("X"), dtype=float)
    final_paths = [decode(vector) for vector in final_variables]
    batch = _batch_from_paths(
        final_paths,
        final_variables,
        env,
        radio_map,
        pipe_radius,
        outage_threshold_db,
    )
    evaluations = int(callback.records[-1]["evaluations"]) if callback.records else 0
    return _result_from_batch(
        batch, history=callback.records, evaluation_count=evaluations
    )

def run_nsga2_baseline(
    env,
    planner_module,
    radio_map,
    drone_id: int,
    layer_index: int,
    *,
    population_size: int,
    generations: int,
    evaluation_budget: int,
    seed: int,
    pipe_radius: float,
    outage_threshold_db: float,
    control_points: int = 30,
    path_samples: int = 120,
    hypervolume_reference: tuple[float, float] = (800.0, 1.05),
) -> OptimizationResult:
    """Run standard NSGA-II without A*, corridor bounds, or topology repair."""
    del generations  # The exact black-box budget is the termination criterion.
    start, end = _aligned_endpoints(env, drone_id, layer_index)
    planner = planner_module.SplinePathPlanner(
        env, n_control_points=int(control_points), pipe_radius=float(pipe_radius)
    )
    planner.set_cruise_layer(layer_index)
    planner.set_other_drone_paths([])

    bounds = planner.generate_control_points_bounds(start, end)
    lower = []
    upper = []
    for index in range(int(control_points)):
        lower.extend(
            [bounds[f"x{index}"][0], bounds[f"y{index}"][0], bounds[f"z{index}"][0]]
        )
        upper.extend(
            [bounds[f"x{index}"][1], bounds[f"y{index}"][1], bounds[f"z{index}"][1]]
        )
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rng = np.random.default_rng(seed)
    initial = rng.uniform(lower, upper, size=(int(population_size), len(lower)))

    def decode(vector: np.ndarray) -> np.ndarray:
        controls = np.asarray(vector, dtype=float).reshape(int(control_points), 3)
        controls[0] = start
        controls[-1] = end
        path = np.asarray(
            planner.create_b_spline(controls, n_samples=int(path_samples)), dtype=float
        )
        path[0] = start
        path[-1] = end
        return path

    class NativeNSGAProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=len(lower),
                n_obj=2,
                n_ieq_constr=2,
                xl=lower,
                xu=upper,
            )

        def _evaluate(self, variables, out, *args, **kwargs):
            objectives = []
            constraints = []
            for vector in variables:
                metrics = evaluate_trajectory(
                    decode(vector),
                    env,
                    radio_map,
                    pipe_radius=pipe_radius,
                    outage_threshold_db=outage_threshold_db,
                    sample_count=path_samples,
                )
                objectives.append([metrics.path_length_m, metrics.communication_cost])
                constraints.append(
                    [
                        float(metrics.collision_count),
                        float(pipe_radius) - metrics.min_clearance_m,
                    ]
                )
            out["F"] = np.asarray(objectives, dtype=float)
            out["G"] = np.asarray(constraints, dtype=float)

    callback = _HistoryCallback(hypervolume_reference)
    algorithm = NSGA2(
        pop_size=int(population_size),
        sampling=initial,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    result = minimize(
        NativeNSGAProblem(),
        algorithm,
        get_termination("n_eval", int(evaluation_budget)),
        seed=int(seed),
        callback=callback,
        verbose=False,
    )
    final_variables = np.asarray(result.pop.get("X"), dtype=float)
    final_paths = [decode(vector) for vector in final_variables]
    batch = _batch_from_paths(
        final_paths,
        final_variables,
        env,
        radio_map,
        pipe_radius,
        outage_threshold_db,
    )
    evaluations = int(callback.records[-1]["evaluations"]) if callback.records else 0
    return _result_from_batch(
        batch, history=callback.records, evaluation_count=evaluations
    )


def _run_diffusion_sampling(
    optimizer: CADiffusionES,
    *,
    population_size: int,
    evaluation_budget: int,
    hypervolume_reference: tuple[float, float],
    final_communication_weight: float,
    pareto_archive_size: int,
) -> OptimizationResult:
    """Sample independent full DDPM trajectories with no evolutionary feedback."""
    population_size = int(population_size)
    evaluation_budget = int(evaluation_budget)
    if population_size < 1 or evaluation_budget < population_size:
        raise ValueError("Diffusion-only budget must cover at least one population")
    archive = FeasibleParetoArchive(maximum_size=int(pareto_archive_size))
    best_infeasible: CandidateBatch | None = None
    history: list[dict[str, float]] = []
    generation = 0
    while optimizer.evaluation_count < evaluation_budget:
        batch_size = min(
            population_size, evaluation_budget - optimizer.evaluation_count
        )
        latent = np.asarray(
            optimizer.prior.sample(optimizer.normalized_condition, batch_size),
            dtype=np.float32,
        )
        batch = optimizer._decode(latent)
        archive.update(batch)
        minimum_index = int(np.argmin(batch.constraint_violation))
        if (
            best_infeasible is None
            or batch.constraint_violation[minimum_index]
            < best_infeasible.constraint_violation[0]
        ):
            best_infeasible = batch.take(np.asarray([minimum_index]))
        feasible_count = int(np.sum(batch.constraint_violation <= 1e-9))
        pareto = archive.batch
        if pareto is None:
            pareto_size = 0
            hypervolume = 0.0
            indicator = float(np.min(batch.constraint_violation))
        else:
            pareto_size = len(pareto.length)
            hypervolume = hypervolume_2d(
                pareto.objectives, hypervolume_reference
            )
            selected = compromise_index(
                pareto.objectives,
                weights=(
                    1.0 - float(final_communication_weight),
                    float(final_communication_weight),
                ),
            )
            indicator = float(
                np.sum(
                    pareto.objectives[selected]
                    / np.maximum(pareto.objectives.max(axis=0), 1e-12)
                )
            )
        feasible_ratio = feasible_count / max(1, len(batch.length))
        history.append(
            {
                "generation": float(generation),
                "evaluations": float(optimizer.evaluation_count),
                "truncation_step": 0.0,
                "feasible_count": float(feasible_count),
                "feasible_ratio": float(feasible_ratio),
                "pareto_size": float(pareto_size),
                "hypervolume": float(hypervolume),
                "indicator": indicator,
                "raw_offspring_feasible_ratio": float(feasible_ratio),
                "resampled_offspring_count": 0.0,
                "projected_offspring_count": 0.0,
                "topology_parent_count": 0.0,
                "topology_projected_count": 0.0,
                "topology_noise_scale": 0.0,
            }
        )
        generation += 1
    final_batch = archive.batch if archive.batch is not None else best_infeasible
    if final_batch is None:
        raise RuntimeError("Diffusion-only produced no evaluable trajectories")
    return _result_from_batch(
        final_batch,
        history=history,
        evaluation_count=optimizer.evaluation_count,
        communication_weight=float(final_communication_weight),
    )


def run_diffusion_only_baseline(
    env,
    planner_module,
    radio_map,
    drone_id: int,
    layer_index: int,
    *,
    legacy_root,
    dataset_path,
    checkpoint_path,
    population_size: int,
    evaluation_budget: int,
    seed: int,
    pipe_radius: float,
    outage_threshold_db: float,
    path_samples: int = 120,
    control_points: int = 30,
    hypervolume_reference: tuple[float, float] = (800.0, 1.05),
    final_communication_weight: float = 0.65,
    pareto_archive_size: int = 400,
) -> OptimizationResult:
    """Run conditional DDPM sampling without ES, replay, or truncated denoising."""
    from .integration import prepare_prior

    start, end = _aligned_endpoints(env, drone_id, layer_index)
    planner = planner_module.SplinePathPlanner(
        env, n_control_points=int(control_points), pipe_radius=float(pipe_radius)
    )
    planner.set_cruise_layer(layer_index)
    planner.set_other_drone_paths([])
    prior, dataset, _, _ = prepare_prior(
        legacy_root,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        seed=int(seed),
        radio_map=radio_map,
    )
    optimizer = CADiffusionES(
        planner,
        start,
        end,
        prior,
        dataset,
        radio_map=radio_map,
        other_paths=[],
        seed=int(seed),
        sample_points=int(path_samples),
        minimum_separation=2.0 * float(pipe_radius),
        outage_threshold_db=float(outage_threshold_db),
    )
    return _run_diffusion_sampling(
        optimizer,
        population_size=int(population_size),
        evaluation_budget=int(evaluation_budget),
        hypervolume_reference=hypervolume_reference,
        final_communication_weight=float(final_communication_weight),
        pareto_archive_size=int(pareto_archive_size),
    )
