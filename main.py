# main.py
from environment import Environment, PIPE_RADIUS
from path_planner import SplinePathPlanner, astar_2d
from plot import PathVisualizer
from channel import (
    load_pathloss_curve_csv,
    estimate_pl_range,
    make_channel_cost_fn,
    make_channel_cost_fn_vec,
    avg_pathloss_to_ues_db,
)
import numpy as np
import math
import os
import argparse
import random
import time
import csv
import os
import sys

# 让 main.py 能 import code_diffusion 中的 Diffusion-ES 接入模块
_CODE_DIFFUSION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_diffusion")
if _CODE_DIFFUSION not in sys.path:
    sys.path.insert(0, _CODE_DIFFUSION)
from diffusion_es_main import plan_drone_diffusion_es  # noqa: E402


# ------------------------------
# 1) Remove local spikes (sharp hooks)
# ------------------------------

def _despike_path_xy(path, max_turn_deg=135.0, max_span=18.0):
    P = np.array(path, dtype=float).tolist()
    changed = True
    while changed:
        changed = False
        i = 1
        while i < len(P) - 1:
            a = np.array(P[i - 1])
            b = np.array(P[i])
            c = np.array(P[i + 1])
            ab = b - a
            bc = c - b
            ab[2] = 0.0
            bc[2] = 0.0
            lab = np.linalg.norm(ab)
            lbc = np.linalg.norm(bc)
            span = np.linalg.norm(np.array(c[:2]) - np.array(a[:2]))
            if lab < 1e-9 or lbc < 1e-9:
                i += 1
                continue
            cosang = np.dot(ab, bc) / (lab * lbc)
            cosang = max(-1.0, min(1.0, cosang))
            ang = math.degrees(math.acos(cosang))
            if ang > max_turn_deg and span < max_span:
                P.pop(i)
                changed = True
            else:
                i += 1
    return P


# ------------------------------
# 2) Remove backtracking loops / mid spikes
# ------------------------------
def remove_backtracking_loops_xy(path, max_back_deg=100.0, window=12, loop_close_dist=8.0):
    P = np.array(path, dtype=float).tolist()
    if len(P) < 6:
        return P

    def angle_deg(u, v):
        nu = np.linalg.norm(u) + 1e-12
        nv = np.linalg.norm(v) + 1e-12
        c = np.dot(u, v) / (nu * nv)
        c = max(-1.0, min(1.0, c))
        return math.degrees(math.acos(c))

    changed = True
    while changed:
        changed = False

        # (A) remove local reverse-turn points
        for i in range(1, len(P) - 1):
            a = np.array(P[i - 1])
            b = np.array(P[i])
            c = np.array(P[i + 1])

            v1 = b - a
            v2 = c - b
            v1[2] = 0
            v2[2] = 0

            ang = angle_deg(v1[:2], v2[:2])
            if ang > max_back_deg:
                P.pop(i)
                changed = True
                break

        if changed:
            continue

        # (B) remove small loop closure
        for i in range(len(P) - window - 1):
            p1 = np.array(P[i])[:2]
            p2 = np.array(P[i + window])[:2]
            if np.linalg.norm(p2 - p1) < loop_close_dist:
                mid = i + window // 2
                for _ in range(window - 1):
                    if mid < len(P) - 1:
                        P.pop(mid)
                changed = True
                break

    return P


# ------------------------------
# 3) Enforce tail monotonic towards end (avoid tail backtrack)
# ------------------------------
def _enforce_monotone_tail_towards_end(path, end_xy, backtrack=50.0):
    if len(path) < 3:
        return path
    P = np.array(path, dtype=float)
    s = 0.0
    cut_idx = None
    best_d = np.inf
    end_xy = np.asarray(end_xy, dtype=float)
    for i in range(len(P) - 2, -1, -1):
        s += np.linalg.norm(P[i + 1, :2] - P[i, :2])
        d = np.linalg.norm(P[i, :2] - end_xy)
        if d < best_d - 1e-6:
            best_d = d
        else:
            cut_idx = i + 1
            break
        if s >= backtrack:
            break
    if cut_idx is None or cut_idx >= len(P) - 1:
        return P.tolist()
    zc = P[cut_idx, 2]
    Q = np.vstack([P[:cut_idx + 1], np.array([end_xy[0], end_xy[1], zc])])
    return Q.tolist()


# ------------------------------
# 4) Arc-length resampling for uniform density
# ------------------------------
def resample_by_arclength(path_xyz, n=280):
    pts = np.asarray(path_xyz, dtype=float)
    if len(pts) < 2:
        return pts.tolist()

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return pts.tolist()

    new_s = np.linspace(0, s[-1], n)
    new_pts = np.zeros((n, 3))
    for dim in range(3):
        new_pts[:, dim] = np.interp(new_s, s, pts[:, dim])
    return new_pts.tolist()


# ------------------------------
# 5) Time-window scheduling (simple delay)
# ------------------------------
def schedule_with_time_windows(paths, pipe_radius=2.0, speed=12.0, delta_t=2.0):
    from path_planner import capsules_min_distance

    def segs_and_times(p, v):
        segs = []
        s = 0.0
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            L = float(np.linalg.norm(b - a))
            t0 = s / v
            s += L
            t1 = s / v
            segs.append((a, b, L, t0, t1))
        return segs

    n = len(paths)
    start_times = [0.0 for _ in range(n)]
    segs_all = [segs_and_times(np.asarray(p), speed) for p in paths]

    changed = True
    loop_guard = 0
    while changed and loop_guard < 400:
        changed = False
        loop_guard += 1
        for i in range(n):
            for j in range(i + 1, n):
                segs_i = segs_all[i]
                segs_j = segs_all[j]
                si = start_times[i]
                sj = start_times[j]
                conflict = False
                for a1, b1, _, ti0, ti1 in segs_i:
                    for a2, b2, _, tj0, tj1 in segs_j:
                        if max(ti0 + si, tj0 + sj) < min(ti1 + si, tj1 + sj):
                            if capsules_min_distance((a1, b1), pipe_radius, (a2, b2), pipe_radius) < 0:
                                conflict = True
                                break
                    if conflict:
                        break
                if conflict:
                    start_times[j] += delta_t
                    changed = True
    return start_times


def check_all_pipe_collisions_timeaware(paths, start_times, speed, pipe_radius):
    from path_planner import capsules_min_distance
    conflicts = []
    segs_all = []
    for P, t0 in zip(paths, start_times):
        P = np.asarray(P)
        segs = []
        s = 0.0
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            L = float(np.linalg.norm(b - a))
            ti0 = t0 + s / speed
            s += L
            ti1 = t0 + s / speed
            segs.append((a, b, ti0, ti1))
        segs_all.append(segs)

    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            for (a1, b1, ti0, ti1) in segs_all[i]:
                for (a2, b2, tj0, tj1) in segs_all[j]:
                    if max(ti0, tj0) < min(ti1, tj1):
                        gap = capsules_min_distance((a1, b1), pipe_radius, (a2, b2), pipe_radius)
                        if gap < 0:
                            conflicts.append({
                                "pair": (i, j),
                                "time_window": (max(ti0, tj0), min(ti1, tj1)),
                                "gap": gap
                            })
    return conflicts


# =========================================================
# Compare helpers (metrics + CSV)
# =========================================================
def path_length_xyz(path):
    P = np.asarray(path, dtype=float)
    if P.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def turn_angle_stats_xy(path):
    """
    Smoothness proxy: turning angles in XY plane.
    Returns: mean_deg, p95_deg, max_deg
    """
    P = np.asarray(path, dtype=float)
    if P.shape[0] < 3:
        return 0.0, 0.0, 0.0

    angs = []
    for i in range(1, len(P) - 1):
        a = P[i] - P[i - 1]
        b = P[i + 1] - P[i]
        a[2] = 0.0
        b[2] = 0.0
        na = np.linalg.norm(a[:2]) + 1e-12
        nb = np.linalg.norm(b[:2]) + 1e-12
        c = float(np.dot(a[:2], b[:2]) / (na * nb))
        c = max(-1.0, min(1.0, c))
        ang = float(np.degrees(np.arccos(c)))
        angs.append(ang)

    if not angs:
        return 0.0, 0.0, 0.0
    angs = np.asarray(angs, dtype=float)
    return float(np.mean(angs)), float(np.percentile(angs, 95)), float(np.max(angs))


def write_results_csv(rows, out_csv="results_algorithm_compare.csv"):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def assign_layers(env):
    """Keep your original >5 layer policy exactly."""
    n_drones = len(env.starts)
    layers = []
    base = min(5, n_drones, len(env.safe_layers))
    layers.extend(list(range(base)))

    mids = [L.mid for L in env.safe_layers]
    for i in range(base, n_drones):
        z0 = float(env.starts[i][2])
        best = int(np.argmin([abs(z0 - m) for m in mids]))
        layers.append(best)
    return layers


def restrict_environment_to_drones(env, n_drones):
    """Keep only the selected UAV endpoints and rebuild their vertiports."""
    n_drones = int(n_drones)
    if n_drones < 1 or n_drones > len(env.starts):
        raise ValueError(f"n_drones must be between 1 and {len(env.starts)}")

    env.starts = list(env.starts[:n_drones])
    env.ends = list(env.ends[:n_drones])

    rebuild = getattr(env, "_build_vertiports_from_starts_ends", None)
    if callable(rebuild):
        env.vertiports = rebuild(z_min=0.0, z_max=40.0, nd=2)
    elif hasattr(env, "vertiports"):
        # Do not plot endpoints that belong to UAVs outside this scenario.
        env.vertiports = []
    return env


# Diffusion-ES 默认超参（经验扫描得出）：在 N=5 上 8 次 trial 全零碰撞、
# 场景耗时 ~1.9s（接近 hybrid 的 ~1.8s），相比旧默认(pop<=30/gens<=25/tau_hi=12,
# ~4.7s)提速约 2.4x。更保守的替代：DE_GENS=15, DE_TAU_HI=6（配置 D, ~2.2s）。
DE_POP = 15
DE_GENS = 12
DE_TAU_HI = 5


# ------------------------------
# single-drone plan pipeline (NOW supports modes)
# ------------------------------
def _postprocess_path(planner, env, path, s, e, layer_idx, r_safe):
    """hybrid 与 diffusion_es 共用的后处理链（保证两种优化器在完全相同的后处理下对比）。"""
    path = planner.smooth_path_with_bspline(
        path, num_points=420, clamp_to_corridor=True, keep_endpoints=True
    )
    path = _despike_path_xy(path, max_turn_deg=140.0, max_span=22.0)
    path = remove_backtracking_loops_xy(path, max_back_deg=105.0, window=14, loop_close_dist=9.0)
    path = _enforce_monotone_tail_towards_end(path, end_xy=(e[0], e[1]), backtrack=60.0)
    path = planner.smooth_path(
        path, smoothing_factor=0.22, iters=3,
        keep_endpoints=True, clamp_to_corridor=True
    )
    path = resample_by_arclength(path, n=280)
    if hasattr(planner, "project_path_into_corridor"):
        path = planner.project_path_into_corridor(path)
    path = [(float(p[0]), float(p[1]),
             env.clamp_to_layer(max(float(p[2]), env.min_height), layer_idx))
            for p in path]
    if len(path) >= 2:
        path[0] = list(s)
        path[-1] = list(e)
    return path


def plan_one_drone(env, drone_id, layer_idx, other_paths,
                   pipe_radius,
                   r_safe=None,
                   astar_res=2.0,
                   astar_inflation=12.0,
                   corridor_radius=6.0,
                   pop=40, gens=30,
                   beta_channel=0.0,
                   channel_cost_fn=None,
                   channel_weight=1200.0,
                   channel_cost_fn_vec=None,
                   quick_first=True,
                   nsga_seed=None,
                   mode="hybrid",
                   tau_hi=12,
                   pareto_dir=None,
                   multi_objective=True):
    """
    mode:
      - "hybrid"    : A* corridor + NSGA refine (your current default)
      - "astar_only": only A* (XY) + smoothing + resample (no NSGA)
      - "nsga_only" : NSGA without A* corridor (use straight corridor with big radius)
      - "diffusion_es": A*-free CA-Diffusion-ES（HGNN Radio Map + Pareto archive）
    """

    if r_safe is None:
        r_safe = float(pipe_radius)
    else:
        r_safe = float(r_safe)

    # CA-Diffusion-ES: direct diffusion generation over the admissible airspace.
    if mode == "diffusion_es":
        return plan_one_drone_diffusion_es(
            env, drone_id, layer_idx, other_paths, pipe_radius, r_safe,
            astar_inflation=astar_inflation, corridor_radius=corridor_radius,
            pop=pop, gens=gens, channel_cost_fn=channel_cost_fn,
            channel_weight=channel_weight, nsga_seed=nsga_seed,
            use_warm_start=False, T=25, tau_hi=tau_hi,
            channel_cost_fn_vec=channel_cost_fn_vec,
            multi_objective=multi_objective, pareto_dir=pareto_dir)

    # 1) Align start/end z to current safe layer mid
    s0, e0 = env.get_start_end(drone_id)
    zc = env.layer_mid_z(layer_idx)

    try:
        s = (float(s0[0]), float(s0[1]), env.clamp_to_layer(max(float(s0[2]), env.min_height), layer_idx))
        e = (float(e0[0]), float(e0[1]), env.clamp_to_layer(max(float(e0[2]), env.min_height), layer_idx))
    except Exception:
        s = (float(s0[0]), float(s0[1]), float(zc))
        e = (float(e0[0]), float(e0[1]), float(zc))

    # 2) A* coarse in XY (supports channel cost) - only for hybrid / astar_only
    if mode in ("hybrid", "astar_only"):
        coarse = astar_2d(
            env, (s[0], s[1]), (e[0], e[1]),
            res=astar_res,
            inflation=astar_inflation,
            cruise_layer=layer_idx,
            bounds=None,
            beta_channel=beta_channel,
            channel_cost_fn=channel_cost_fn
        )

        # Retry with smaller inflation if A* returns trivial line
        if coarse is None or len(coarse) <= 2:
            for inf in (max(6.0, astar_inflation - 2), max(6.0, astar_inflation - 4), 6.0):
                coarse = astar_2d(
                    env, (s[0], s[1]), (e[0], e[1]),
                    res=astar_res,
                    inflation=inf,
                    cruise_layer=layer_idx,
                    bounds=None,
                    beta_channel=beta_channel,
                    channel_cost_fn=channel_cost_fn
                )
                if coarse is not None and len(coarse) > 2:
                    break

        if coarse is None:
            coarse = [s, e]
    else:
        # nsga_only: no A*
        coarse = [s, e]

    # 3) Build planner (use r_safe as conservative pipe radius)
    planner = SplinePathPlanner(env, n_control_points=10, pipe_radius=r_safe)
    planner.set_cruise_layer(layer_idx)

    if mode == "nsga_only":
        # give NSGA more freedom
        planner.set_corridor(coarse, corridor_radius=max(60.0, float(corridor_radius) * 6.0))
    else:
        planner.set_corridor(coarse, corridor_radius=corridor_radius)

    planner.set_other_drone_paths(other_paths)

    # channel penalty (NSGA cost)
    if channel_cost_fn is not None and channel_weight > 1e-12:
        planner.set_channel_cost(channel_cost_fn, weight=channel_weight,
                                 vec_fn=channel_cost_fn_vec)

    if hasattr(planner, "_endpoint_anchor_dist"):
        planner._endpoint_anchor_dist = 18.0

    # ---------------------------------------------------------
    # ASTAR ONLY: directly smooth coarse and return
    # ---------------------------------------------------------
    if mode == "astar_only":
        coarse3 = [(float(p[0]), float(p[1]),
                    env.clamp_to_layer(max(float(p[2]), env.min_height), layer_idx))
                   for p in coarse]

        path = planner.smooth_path_with_bspline(
            coarse3, num_points=420, clamp_to_corridor=True, keep_endpoints=True
        )
        path = _despike_path_xy(path, max_turn_deg=140.0, max_span=22.0)
        path = remove_backtracking_loops_xy(path, max_back_deg=105.0, window=14, loop_close_dist=9.0)
        path = _enforce_monotone_tail_towards_end(path, end_xy=(e[0], e[1]), backtrack=60.0)

        path = planner.smooth_path(
            path, smoothing_factor=0.22, iters=3,
            keep_endpoints=True, clamp_to_corridor=True
        )
        path = resample_by_arclength(path, n=280)

        path = [(float(p[0]), float(p[1]),
                 env.clamp_to_layer(max(float(p[2]), env.min_height), layer_idx))
                for p in path]

        if len(path) >= 2:
            path[0] = list(s)
            path[-1] = list(e)

        collisions = planner.check_collisions(path, env.buildings, safety_margin=r_safe)
        metrics = planner.analyze_path(path, env.buildings, safety_margin=r_safe)
        return path, collisions, metrics

    # ---------------------------------------------------------
    # HYBRID / NSGA-ONLY: quick feasible then NSGA refine
    # ---------------------------------------------------------
    if quick_first:
        coarse3 = [(float(p[0]), float(p[1]),
                    env.clamp_to_layer(max(float(p[2]), env.min_height), layer_idx))
                   for p in coarse]

        path_q = planner.smooth_path_with_bspline(
            coarse3, num_points=320, clamp_to_corridor=True, keep_endpoints=True
        )
        path_q = planner.smooth_path(path_q, smoothing_factor=0.20, iters=4,
                                     keep_endpoints=True, clamp_to_corridor=True)
        path_q = resample_by_arclength(path_q, n=260)

        if len(path_q) >= 2:
            path_q[0] = list(s)
            path_q[-1] = list(e)

        col_q = planner.check_collisions(path_q, env.buildings, safety_margin=r_safe)
        if len(col_q) == 0:
            met_q = planner.analyze_path(path_q, env.buildings, safety_margin=r_safe)
            return path_q, col_q, met_q

    # NSGA-II refine
    path = planner.optimize_path(
        drone_id=drone_id,
        population_size=pop,
        n_generations=gens,
        start=s,
        end=e,
        seed=nsga_seed,
        verbose=False,
        pareto_dir=pareto_dir
    )

    # post smoothing chain
    path = _postprocess_path(planner, env, path, s, e, layer_idx, r_safe)

    collisions = planner.check_collisions(path, env.buildings, safety_margin=r_safe)
    metrics = planner.analyze_path(path, env.buildings, safety_margin=r_safe)
    return path, collisions, metrics

    # ---------------------------------------------------------
    # DIFFUSION-ES: 复用同一 A* 走廊 + 信道 + 多机代价，运行 GPU 上的 Diffusion-ES
    # ---------------------------------------------------------
    # (此分支在 mode=="diffusion_es" 时进入，见下方调度)


def plan_one_drone_diffusion_es(env, drone_id, layer_idx, other_paths,
                                pipe_radius, r_safe, astar_inflation=12.0,
                                corridor_radius=6.0, pop=40, gens=30,
                                channel_cost_fn=None, channel_weight=1400.0,
                                nsga_seed=None, temperature=0.1,
                                tau_hi=12, tau_lo=1, use_warm_start=False,
                                train_npz=None, T=25,
                                lambda_base=1e3, mu_base=1e3, alpha=5.0,
                                elite_frac=0.2, patience=5, tol=1e-3,
                                channel_cost_fn_vec=None,
                                multi_objective=True, pareto_dir=None):
    """A*-free channel-aware multi-objective Diffusion-ES."""
    s0, e0 = env.get_start_end(drone_id)
    zc = env.layer_mid_z(layer_idx)
    try:
        s = (float(s0[0]), float(s0[1]), env.clamp_to_layer(max(float(s0[2]), env.min_height), layer_idx))
        e = (float(e0[0]), float(e0[1]), env.clamp_to_layer(max(float(e0[2]), env.min_height), layer_idx))
    except Exception:
        s = (float(s0[0]), float(s0[1]), float(zc))
        e = (float(e0[0]), float(e0[1]), float(zc))

    pop = max(2, int(pop))
    gens = max(1, int(gens))

    raw_path, history, planner, coarse, front = plan_drone_diffusion_es(
        env=env, drone_id=drone_id, layer_idx=layer_idx, other_paths=other_paths,
        pipe_radius=pipe_radius, r_safe=r_safe, astar_inflation=astar_inflation,
        corridor_radius=corridor_radius, channel_cost_fn=channel_cost_fn,
        channel_weight=channel_weight, train_npz=train_npz, T=T,
        seed=0 if nsga_seed is None else int(nsga_seed),
        n_pop=pop, n_gen=gens, use_warm_start=use_warm_start,
        temperature=temperature, tau_hi=tau_hi, tau_lo=tau_lo,
        lambda_base=lambda_base, mu_base=mu_base, alpha=alpha,
        elite_frac=elite_frac, patience=patience, tol=tol,
        channel_cost_fn_vec=channel_cost_fn_vec,
        multi_objective=multi_objective, pareto_dir=pareto_dir)

    raw_path = np.asarray(raw_path, dtype=float)

    # The new optimizer already returns a smooth feasible B-spline selected
    # from its feasible Pareto archive. Do not post-process it into another
    # objective value and never substitute an A* path.
    path = [
        (
            float(point[0]),
            float(point[1]),
            env.clamp_to_layer(max(float(point[2]), env.min_height), layer_idx),
        )
        for point in raw_path
    ]
    if len(path) >= 2:
        path[0] = list(s)
        path[-1] = list(e)
    collisions = planner.check_collisions(path, env.buildings, safety_margin=r_safe)
    metrics = planner.analyze_path(path, env.buildings, safety_margin=r_safe)
    return path, collisions, metrics


# =========================================================
# Algorithm comparison runner
# =========================================================
def run_algorithm_comparison(trials=6, Ns=(5, 7, 9),
                             modes=("astar_only", "nsga_only", "hybrid", "diffusion_es"),
                             base_seed=42, no_channel=False,
                             out_csv="artifacts/results_allmodes.csv",
                             multi_objective=False, pareto_dir=None):

    rows = []

    # CA-Diffusion-ES 多目标：dump 前沿到与 NSGA 同目录，供 Fig.9 叠加对比
    if multi_objective:
        pareto_dir = pareto_dir or "artifacts/pareto"
        os.makedirs(pareto_dir, exist_ok=True)

    CHANNEL_CSV = "processed_crossroad_path_data.csv"
    CARRIER_FREQ_HZ = 2.0e9

    for N in Ns:
        for mode in modes:
            for t in range(trials):
                SEED = None if base_seed is None else (base_seed + 1000 * N + 50 * t)

                if SEED is not None:
                    os.environ["PYTHONHASHSEED"] = str(SEED)
                    random.seed(SEED)
                    np.random.seed(SEED)

                env = Environment()

                # slice to N drones for fair compare
                restrict_environment_to_drones(env, N)

                # channel curve
                try:
                    d_arr, pl_arr = load_pathloss_curve_csv(
                        CHANNEL_CSV,
                        d_col="Straight Line Distance",
                        pl_col="Path Loss (dB)"
                    )
                except Exception:
                    d_arr, pl_arr = None, None

                layer_channel_cost = {}
                layer_channel_cost_vec = {}
                for layer_idx in range(len(env.safe_layers)):
                    pl_min, pl_max = estimate_pl_range(
                        env, layer_idx, env.ues, d_arr, pl_arr, f_hz=CARRIER_FREQ_HZ
                    )
                    layer_channel_cost[layer_idx] = make_channel_cost_fn(
                        env, layer_idx, env.ues, d_arr, pl_arr, pl_min, pl_max, f_hz=CARRIER_FREQ_HZ
                    )
                    layer_channel_cost_vec[layer_idx] = make_channel_cost_fn_vec(
                        env, layer_idx, env.ues, d_arr, pl_arr, pl_min, pl_max, f_hz=CARRIER_FREQ_HZ
                    )

                pipe_radius = float(PIPE_RADIUS)
                r_safe_default = max(2.0, float(PIPE_RADIUS) + 0.5)

                # same hyperparams as your main for fairness
                BETA_CHANNEL_ASTAR = 10.0
                CHANNEL_WEIGHT_NS = 1400.0

                n_drones = len(env.starts)
                layers = assign_layers(env)

                all_paths = []

                scenario_t0 = time.perf_counter()

                for drone_id in range(n_drones):
                    layer_idx = layers[drone_id]
                    other_paths = [np.asarray(p) for p in all_paths]
                    channel_cost_fn = layer_channel_cost[layer_idx]
                    channel_cost_fn_vec = layer_channel_cost_vec[layer_idx]

                    astar_infl = 12.0
                    corr_r = 6.0
                    pop = 45
                    gens = 35
                    beta_astar = BETA_CHANNEL_ASTAR
                    ch_w = CHANNEL_WEIGHT_NS
                    r_safe = r_safe_default

                    if mode == "diffusion_es":
                        # 轻量超参（配置 E）：大幅压缩耗时，质量已验证无损
                        pop = DE_POP
                        gens = DE_GENS
                    else:
                        if drone_id == 1:
                            astar_infl = 18.0
                            corr_r = 10.0
                            pop = 60
                            gens = 45
                            beta_astar = 6.0
                            ch_w = 900.0
                            r_safe = r_safe_default + 0.2

                    attempts = [
                        dict(
                            astar_inflation=astar_infl,
                            corridor_radius=corr_r,
                            pop=pop,
                            gens=gens,
                            beta_channel=beta_astar,
                            channel_weight=ch_w
                        ),
                        dict(
                            astar_inflation=astar_infl + 4,
                            corridor_radius=corr_r + 2,
                            pop=min(pop + (5 if mode == "diffusion_es" else 15), 80),
                            gens=min(gens + (5 if mode == "diffusion_es" else 15), 70),
                            beta_channel=max(0.0, beta_astar - 2),
                            channel_weight=ch_w
                        ),
                    ]

                    if no_channel:
                        for att in attempts:
                            att["beta_channel"] = 0.0
                            att["channel_weight"] = 0.0

                    best = None
                    drone_t0 = time.perf_counter()

                    for ai, tt in enumerate(attempts, start=1):
                        nsga_seed = None if SEED is None else (SEED + drone_id * 100 + ai)

                        path, collisions, metrics = plan_one_drone(
                            env=env,
                            drone_id=drone_id,
                            layer_idx=layer_idx,
                            other_paths=other_paths,
                            pipe_radius=pipe_radius,
                            r_safe=r_safe,
                            astar_res=2.0,
                            channel_cost_fn=channel_cost_fn,
                            channel_cost_fn_vec=channel_cost_fn_vec,
                            quick_first=True,
                            nsga_seed=nsga_seed,
                            mode=mode,
                            tau_hi=(DE_TAU_HI if mode == "diffusion_es" else 12),
                            multi_objective=multi_objective,
                            pareto_dir=pareto_dir,
                            **tt
                        )

                        if best is None or metrics.get("collision_count", 1e9) < best[2].get("collision_count", 1e9):
                            best = (path, collisions, metrics)

                        if metrics.get("collision_count", 999) == 0:
                            break

                    drone_t1 = time.perf_counter()

                    path, collisions, metrics = best
                    all_paths.append(path)

                    # additional metrics
                    L = path_length_xyz(path)
                    mean_turn, p95_turn, max_turn = turn_angle_stats_xy(path)

                    rows.append({
                        "N": N,
                        "mode": mode,
                        "trial": t,
                        "seed": SEED if SEED is not None else "None",
                        "drone_id": drone_id,
                        "layer_idx": layer_idx,
                        "runtime_s": round(drone_t1 - drone_t0, 4),
                        "path_length_m": round(L, 3),
                        "turn_mean_deg": round(mean_turn, 3),
                        "turn_p95_deg": round(p95_turn, 3),
                        "turn_max_deg": round(max_turn, 3),
                        "collision_count": int(metrics.get("collision_count", 0)),
                        "min_clearance_m": float(metrics.get("min_clearance", float("nan"))),
                        "avg_clearance_m": float(metrics.get("avg_clearance", float("nan"))),
                    })

                scenario_t1 = time.perf_counter()
                rows.append({
                    "N": N,
                    "mode": mode,
                    "trial": t,
                    "seed": SEED if SEED is not None else "None",
                    "drone_id": "ALL",
                    "layer_idx": "NA",
                    "runtime_s": round(scenario_t1 - scenario_t0, 4),
                    "path_length_m": "",
                    "turn_mean_deg": "",
                    "turn_p95_deg": "",
                    "turn_max_deg": "",
                    "collision_count": "",
                    "min_clearance_m": "",
                    "avg_clearance_m": "",
                })

    write_results_csv(rows, out_csv=out_csv)


# ------------------------------
# MAIN
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Run algorithm comparison and save CSV")
    parser.add_argument("--trials", type=int, default=6, help="Trials per (N, mode)")
    parser.add_argument("--Ns", type=str, default="5,7,9", help="Comma list like 5,7,9")
    parser.add_argument("--modes", type=str, default="astar_only,nsga_only,hybrid,diffusion_es",
                        help="Comma list of modes: astar_only,nsga_only,hybrid,diffusion_es")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed (use -1 to mean random)")
    parser.add_argument("--out_csv", type=str, default="artifacts/results_allmodes.csv",
                        help="Output CSV for --compare (use artifacts/results_hybrid_noch.csv for Geo-only)")
    parser.add_argument("--multi_objective", action="store_true",
                        help="CA-Diffusion-ES: run multi-objective (Pareto front) mode and dump front CSVs")
    parser.add_argument("--pareto_dir", type=str, default=None,
                        help="Dir to dump Pareto front CSVs (default artifacts/pareto when --multi_objective)")
    parser.add_argument("--N", type=int, default=5, help="Drones for single-scenario viz (default 5)")
    parser.add_argument("--no_channel", action="store_true",
                        help="Disable channel cost (Geo-only) for single-scenario viz")
    parser.add_argument("--viz_mode", type=str, default="hybrid",
                        help="Planning mode for single-scenario viz: astar_only,nsga_only,hybrid,diffusion_es")
    parser.add_argument("--noshow", action="store_true",
                        help="Do not call plt.show() (for headless figure generation)")
    args = parser.parse_args()

    if args.compare:
        raise RuntimeError(
            "The legacy comparison entry is disabled because it mixes channel models "
            "and optimization budgets. Run .\\run_ca_diffusion_es.ps1 formal-check "
            "followed by .\\run_ca_diffusion_es.ps1 formal."
        )

    SEED = 42

    if SEED is not None:
        os.environ["PYTHONHASHSEED"] = str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

    env = Environment()

    n_drones = args.N if args.N else len(env.starts)
    assert n_drones >= 5, "Visualization needs at least 5 drones."
    assert n_drones <= len(env.starts), f"--N exceeds available starts ({len(env.starts)})"
    restrict_environment_to_drones(env, n_drones)

    CHANNEL_CSV = "processed_crossroad_path_data.csv"
    CARRIER_FREQ_HZ = 2.0e9

    try:
        d_arr, pl_arr = load_pathloss_curve_csv(
            CHANNEL_CSV,
            d_col="Straight Line Distance",
            pl_col="Path Loss (dB)"
        )
    except Exception:
        d_arr, pl_arr = None, None

    layer_channel_cost = {}
    layer_channel_cost_vec = {}
    for layer_idx in range(len(env.safe_layers)):
        pl_min, pl_max = estimate_pl_range(
            env, layer_idx, env.ues, d_arr, pl_arr, f_hz=CARRIER_FREQ_HZ
        )
        layer_channel_cost[layer_idx] = make_channel_cost_fn(
            env, layer_idx, env.ues, d_arr, pl_arr, pl_min, pl_max, f_hz=CARRIER_FREQ_HZ
        )
        layer_channel_cost_vec[layer_idx] = make_channel_cost_fn_vec(
            env, layer_idx, env.ues, d_arr, pl_arr, pl_min, pl_max, f_hz=CARRIER_FREQ_HZ
        )

    pipe_radius = float(PIPE_RADIUS)
    r_safe_default = max(2.0, float(PIPE_RADIUS) + 0.5)

    visualizer = PathVisualizer(env, pipe_radius=pipe_radius)

    layers = assign_layers(env)

    all_paths = []
    all_metrics = []
    all_collisions = []

    BETA_CHANNEL_ASTAR = 0.0 if args.no_channel else 10.0
    CHANNEL_WEIGHT_NS = 0.0 if args.no_channel else 1400.0

    for drone_id in range(n_drones):
        layer_idx = layers[drone_id]

        other_paths = [np.asarray(p) for p in all_paths]
        channel_cost_fn = layer_channel_cost[layer_idx]
        channel_cost_fn_vec = layer_channel_cost_vec[layer_idx]

        astar_infl = 12.0
        corr_r = 6.0
        pop = 45
        gens = 35
        beta_astar = BETA_CHANNEL_ASTAR
        ch_w = CHANNEL_WEIGHT_NS
        r_safe = r_safe_default

        if drone_id == 1:
            astar_infl = 18.0
            corr_r = 10.0
            pop = 60
            gens = 45
            beta_astar = 6.0
            ch_w = 900.0
            r_safe = r_safe_default + 0.2

        attempts = [
            dict(
                astar_inflation=astar_infl,
                corridor_radius=corr_r,
                pop=pop,
                gens=gens,
                beta_channel=beta_astar,
                channel_weight=ch_w
            ),
            dict(
                astar_inflation=astar_infl + 4,
                corridor_radius=corr_r + 2,
                pop=min(pop + 15, 80),
                gens=min(gens + 15, 70),
                beta_channel=max(0.0, beta_astar - 2),
                channel_weight=ch_w
            ),
        ]

        best = None
        for ai, t in enumerate(attempts, start=1):
            nsga_seed = None if SEED is None else (SEED + drone_id * 100 + ai)

            path, collisions, metrics = plan_one_drone(
                env=env,
                drone_id=drone_id,
                layer_idx=layer_idx,
                other_paths=other_paths,
                pipe_radius=pipe_radius,
                r_safe=r_safe,
                astar_res=2.0,
                channel_cost_fn=channel_cost_fn,
                channel_cost_fn_vec=channel_cost_fn_vec,
                quick_first=True,
                nsga_seed=nsga_seed,
                mode=args.viz_mode,
                **t
            )

            if best is None or metrics["collision_count"] < best[2]["collision_count"]:
                best = (path, collisions, metrics)

            if metrics["collision_count"] == 0:
                break

        path, collisions, metrics = best

        all_paths.append(path)
        all_collisions.append(collisions)
        all_metrics.append(metrics)

    all_paths_np = [np.asarray(p) for p in all_paths]
    start_times = schedule_with_time_windows(
        all_paths_np,
        pipe_radius=pipe_radius,
        speed=12.0,
        delta_t=2.0
    )
    _ = check_all_pipe_collisions_timeaware(
        all_paths_np,
        start_times,
        speed=12.0,
        pipe_radius=pipe_radius
    )

    visualizer.plot_buildings()

    colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red',
        'tab:brown', 'tab:cyan', 'tab:olive', 'tab:pink', 'tab:gray'
    ]

    visualizer.plot_ues_3d(z=0.8, color="tab:green", marker="^", size=55)

    for i, path_np in enumerate(all_paths_np):
        visualizer.plot_trajectory_as_pipe(
            path_np.tolist(),
            color=colors[i % len(colors)]
        )

    visualizer.plot_vertiports(alpha=1.0, colors=colors, show_labels=False)

    visualizer.plot_takeoff_landing_labels(
        colors=[colors[i % len(colors)] for i in range(n_drones)],
        uav_station_color="#222222",
        top_margin=1.2
    )

    visualizer._decorate_plot()

    visualizer.set_legend_like_topdown(
        n_drones=n_drones,
        colors=[colors[i % len(colors)] for i in range(n_drones)],
        ue_color="tab:green"
    )

    os.makedirs("figs", exist_ok=True)
    visualizer.ax.view_init(elev=30, azim=-55)

    ch_tag = "noCh" if args.no_channel else "Ch"
    save_path_3d = f"figs/3d_{n_drones}uav_{ch_tag}_{args.viz_mode}.png"
    visualizer.ax.figure.savefig(save_path_3d, dpi=300, bbox_inches="tight")

    if not args.noshow:
        visualizer.show()

    visualizer.plot_topdown(
        all_paths_np,
        title=f"Top-down trajectories (XY) - {n_drones} UAVs",
        colors=[colors[i % len(colors)] for i in range(n_drones)],
        labels=[f"Drone {i + 1}" for i in range(n_drones)],
        show_buildings=True,
        show_vertiports=True,
        show_ues=True,
        save_path=f"figs/topdown_{n_drones}uav_{ch_tag}_{args.viz_mode}.png",
        dpi=300,
    )


if __name__ == "__main__":
    main()
