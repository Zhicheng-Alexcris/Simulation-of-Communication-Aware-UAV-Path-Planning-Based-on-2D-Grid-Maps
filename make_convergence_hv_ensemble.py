"""Ensemble per-evaluation feasible-HV convergence, 50 runs (U=5).

Reference-figure style (raw + moving average + +-1 std band), mapped to the
50-run x 4000-evaluation candidate logs:

  Raw data   : 50 individual per-evaluation HV curves (thin, light)
  Mean       : cross-run mean HV at each evaluation (thick red)
  +-1 std    : cross-run standard deviation band at each evaluation

HV convention identical to the paper: objectives (path length m,
communication cost), reference point (2000, 1.05), normalized by the
reference rectangle area 2100. Only feasible candidates enter the running
nondominated front.

Outputs (figs/):
  convergence_hv_ensemble50_linear.png/.pdf   primary, linear x (reference style)
  convergence_hv_ensemble50_log.png/.pdf      log-x variant
  convergence_hv_ensemble30_linear.png/.pdf   clean subset (drone 2/3/4 only)
  hv_pereval_matrix.npy                       cached 50x4000 HV matrix
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(r"D:\desktop\code")
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "ca_diffusion_es"))

from ca_diffusion_es.report import _apply_publication_axis_style  # noqa: E402

RUNS = CODE_ROOT / "runs" / "ca_v9_candidate_logs_n5_v1" / "ca_diffusion_es" / "N5"
OUT_DIR = CODE_ROOT / "figs"
CACHE = OUT_DIR / "hv_pereval_matrix.npy"
COST_CACHE = OUT_DIR / "cost_matrix.npy"

REF_POINT = (2000.0, 1.05)
REF_AREA = 2000.0 * 1.05

FS_LABEL = 11.5
FS_TICK = 10.0
FS_LEGEND = 10.0

C_RAW = "#AEB8C0"
C_MEAN = "#D62F2F"


def hv_of_front(front):
    """Staircase HV w.r.t. the reference point; front sorted by length asc."""
    if not front:
        return 0.0
    total = 0.0
    for j, (L, C) in enumerate(front):
        nxt = front[j + 1][0] if j + 1 < len(front) else REF_POINT[0]
        total += (min(nxt, REF_POINT[0]) - min(L, REF_POINT[0])) * max(
            REF_POINT[1] - C, 0.0
        )
    return total / REF_AREA


def run_hv_curve_instant(path):
    """Per-batch *instantaneous* feasible HV (not running best).

    Each generation is one batch of 100 candidates; we build the nondominated
    front from that batch's feasible candidates alone, so the HV reflects the
    current population's quality and keeps fluctuating late in the search
    (matching the cost-figure look the user prefers).
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((
                int(r["evaluation_id"]),
                float(r["length_m"]) if r["length_m"] else np.nan,
                float(r["communication_cost"]) if r["communication_cost"] else np.nan,
                float(r["is_feasible"]),
            ))
    rows.sort()
    n = len(rows)
    hv = np.full(n, np.nan)
    batch = None
    front = []
    # collect per-batch fronts, then fill each row with its batch HV
    for idx, (_, L, C, feas) in enumerate(rows):
        if feas >= 0.5 and np.isfinite(L) and np.isfinite(C):
            front.append((L, C))
        # batch boundary = every batch_size rows; batch_size is 100 here
        if (idx + 1) % 100 == 0 or idx == n - 1:
            f = []
            for L, C in front:
                if not any(qL <= L and qC <= C for qL, qC in f):
                    f.append((L, C))
            f.sort()
            hv[idx - len(front) + 1: idx + 1] = hv_of_front(f)
            front = []
    return hv


def run_hv_curve(path):
    """Per-evaluation running feasible HV for one candidates CSV."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((
                int(r["evaluation_id"]),
                float(r["length_m"]) if r["length_m"] else np.nan,
                float(r["communication_cost"]) if r["communication_cost"] else np.nan,
                float(r["is_feasible"]),
            ))
    rows.sort()
    n = len(rows)
    hv = np.empty(n)
    front = []
    last = 0.0
    for i, (_, L, C, feas) in enumerate(rows):
        if feas >= 0.5 and np.isfinite(L) and np.isfinite(C):
            dominated = False
            keep = []
            for qL, qC in front:
                if qL <= L and qC <= C:
                    dominated = True
                if not (L <= qL and C <= qC):
                    keep.append((qL, qC))
            if not dominated:
                keep.append((L, C))
                keep.sort()
                filt = []
                best_comm = np.inf
                for a, b in keep:
                    if b < best_comm:
                        filt.append((a, b))
                        best_comm = b
                front = filt
                last = hv_of_front(front)
        hv[i] = last
    return hv


def build_matrix():
    if CACHE.exists():
        return np.load(CACHE)
    curves = []
    t0 = time.perf_counter()
    trials = sorted(RUNS.glob("trial_*"))
    for ti, trial in enumerate(trials):
        for drone in range(5):
            curves.append(run_hv_curve_instant(trial / f"candidates_drone{drone}.csv"))
        print(f"[{ti + 1}/{len(trials)}] {trial.name} done "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    matrix = np.array(curves)
    np.save(CACHE, matrix)
    return matrix


def load_cost_matrix():
    """(50, 4000) per-candidate fixed_cost (raw, noisy, NOT a front statistic).

    This is the exact analog of the reference figure's per-episode makespan:
    every evaluation is one candidate with its own noisy cost.  Each run is
    normalized by the mean fixed_cost of its initial population (evals 1..100).
    """
    if COST_CACHE.exists():
        return np.load(COST_CACHE)
    curves = []
    for trial in sorted(RUNS.glob("trial_*")):
        for drone in range(5):
            ev, cost = [], []
            with open(trial / f"candidates_drone{drone}.csv",
                      newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    ev.append(int(r["evaluation_id"]))
                    cost.append(float(r["fixed_cost"])
                                if r["fixed_cost"] else np.nan)
            order = np.argsort(ev)
            cost = np.array(cost)[order]
            base = np.nanmean(cost[:100])
            curves.append(cost / base)
    matrix = np.array(curves)
    np.save(COST_CACHE, matrix)
    return matrix


def plot_ensemble(matrix, xscale, fname):
    n_runs, n_evals = matrix.shape
    x = np.arange(1, n_evals + 1)

    med = np.nanmedian(matrix, axis=0)
    p25 = np.nanpercentile(matrix, 25, axis=0)
    p75 = np.nanpercentile(matrix, 75, axis=0)

    fig = plt.figure(figsize=(7.16, 3.6), facecolor="white")
    grid = fig.add_gridspec(1, 1, left=0.10, right=0.985, top=0.96,
                            bottom=0.155)
    ax = fig.add_subplot(grid[0])

    # raw data: scatter of every run's HV at each evaluation, forming a
    # jitter cloud around the center line (reference-figure style)
    xx = np.tile(x, n_runs)
    yy = matrix.flatten()
    ok = np.isfinite(yy)
    ax.scatter(xx[ok], yy[ok], s=1.0, color=C_RAW, alpha=0.12, linewidth=0,
               zorder=2, label=f"Raw data ({n_runs} runs)")
    # interquartile band
    ax.fill_between(x, p25, p75, color=C_MEAN, alpha=0.16,
                    linewidth=0, zorder=3, label="Interquartile range")
    # ensemble median
    ax.plot(x, med, color=C_MEAN, linewidth=2.2, zorder=5,
            label=f"Median of {n_runs} runs")

    if xscale == "log":
        ax.set_xscale("log")
        ax.set_xlim(1, n_evals)
        ax.set_xticks([1, 10, 100, 1000, 4000])
        ax.set_xticklabels(["1", "10", "100", "1000", "4000"])
    else:
        ax.set_xlim(0, n_evals)
        ax.set_xticks([0, 1000, 2000, 3000, 4000])

    # zoom the y-axis close to the data so late-search fluctuations are visible
    finite = matrix[np.isfinite(matrix)]
    ylo = float(np.nanmin(finite))
    yhi = float(np.nanmax(finite))
    pad = 0.05 * (yhi - ylo)
    ax.set_ylim(max(0.0, ylo - pad), yhi + pad)

    ax.set_xlabel("Evaluations", fontsize=FS_LABEL)
    ax.set_ylabel("Reference-normalized\nfeasible hypervolume",
                  fontsize=FS_LABEL)
    _apply_publication_axis_style(ax, grid_axis="both")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(loc="upper right", fontsize=FS_LEGEND, frameon=True,
              framealpha=0.92, edgecolor="#C8CDD2", handlelength=1.8,
              borderpad=0.5).get_frame().set_linewidth(0.6)

    out = OUT_DIR / fname
    fig.savefig(out.with_suffix(".png"), dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved: {out}.png/.pdf")


def plot_cost_ensemble(matrix, fname, smooth_window=30):
    """Reference-figure look with per-candidate cost.

    Raw  : every candidate of every run, one noisy dot per evaluation
    Band : cross-run +-1 std at each evaluation (trailing-MA smoothed)
    Line : cross-run mean, moving-average smoothed (window = 30), like the
           reference figure's 'Moving Average (window=30)' center line
    """
    n_runs, n_evals = matrix.shape
    x = np.arange(1, n_evals + 1)

    center = np.nanmean(matrix, axis=0)
    spread = np.nanstd(matrix, axis=0, ddof=1)

    # trailing moving average (expanding window at the start, no edge dip)
    def trailing_ma(a, w):
        csum = np.cumsum(np.insert(a, 0, 0.0))
        idx = np.arange(len(a))
        lo = np.maximum(0, idx - w + 1)
        return (csum[idx + 1] - csum[lo]) / (idx - lo + 1)

    center_ma = trailing_ma(center, smooth_window)
    band_lo = trailing_ma(center - spread, smooth_window)
    band_hi = trailing_ma(center + spread, smooth_window)

    fig = plt.figure(figsize=(7.16, 3.6), facecolor="white")
    grid = fig.add_gridspec(1, 1, left=0.10, right=0.985, top=0.96,
                            bottom=0.155)
    ax = fig.add_subplot(grid[0])

    # raw data: every candidate of every run, one noisy dot per evaluation
    xx = np.tile(x, n_runs)
    yy = matrix.flatten()
    ok = np.isfinite(yy)
    ax.scatter(xx[ok], yy[ok], s=1.0, color=C_RAW, alpha=0.18, linewidths=0,
               zorder=2, rasterized=True, label=f"Raw data ({n_runs} runs)")
    # +-1 std band
    ax.fill_between(x, band_lo, band_hi, color=C_MEAN, alpha=0.16,
                    linewidth=0, zorder=3, label=r"$\pm 1$ std. dev.")
    # moving-average center line
    ax.plot(x, center_ma, color=C_MEAN, linewidth=2.2, zorder=5,
            label=f"Moving average (window = {smooth_window})")

    ax.set_xlim(0, n_evals)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])

    # robust y-limits: clip extreme infeasible-cost outliers
    ylo, yhi = np.nanpercentile(yy[ok], [0.5, 99.5])
    pad = 0.05 * (yhi - ylo)
    ax.set_ylim(max(0.0, ylo - pad), yhi + pad)

    ax.set_xlabel("Evaluations", fontsize=FS_LABEL)
    ax.set_ylabel("Normalized objective cost", fontsize=FS_LABEL)
    _apply_publication_axis_style(ax, grid_axis="both")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(loc="upper right", fontsize=FS_LEGEND, frameon=True,
              framealpha=0.92, edgecolor="#C8CDD2", handlelength=1.8,
              borderpad=0.5).get_frame().set_linewidth(0.6)

    out = OUT_DIR / fname
    fig.savefig(out.with_suffix(".png"), dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved: {out}.png/.pdf")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    print("matrix:", matrix.shape)

    ends = matrix[:, -1]
    print(f"50 runs  end HV: mean={ends.mean():.3f} std={ends.std(ddof=1):.3f} "
          f"range=[{ends.min():.3f}, {ends.max():.3f}]")

    # clean subset: drones 2/3/4 (indices 2,3,4 within each trial)
    drone_ids = np.tile(np.arange(5), 10)
    mask30 = np.isin(drone_ids, [2, 3, 4])
    sub = matrix[mask30]
    e30 = sub[:, -1]
    print(f"30 runs (drone 2/3/4) end HV: mean={e30.mean():.3f} "
          f"std={e30.std(ddof=1):.3f} range=[{e30.min():.3f}, {e30.max():.3f}]")

    plot_ensemble(matrix, "linear", "convergence_hv_ensemble50_linear")
    plot_ensemble(matrix, "log", "convergence_hv_ensemble50_log")
    plot_ensemble(sub, "linear", "convergence_hv_ensemble30_linear")
    plot_ensemble(sub, "log", "convergence_hv_ensemble30_log")

    # reference-figure style: raw per-candidate cost (noisy) around the median
    cost = load_cost_matrix()
    c_ends = cost[:, -1]
    print(f"50 runs  end cost: median={np.median(c_ends):.3f} "
          f"range=[{c_ends.min():.3f}, {c_ends.max():.3f}]")
    plot_cost_ensemble(cost, "convergence_cost_ensemble50")
    plot_cost_ensemble(cost[mask30], "convergence_cost_ensemble30")


if __name__ == "__main__":
    main()
