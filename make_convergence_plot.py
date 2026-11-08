"""Convergence plots for CA-Diffusion-ES from candidate logs (U=5).

Two variants:
  V1  single-run reference style: raw per-evaluation cost + moving average
      (window=30) + rolling +-1 std band  ->  convergence_single_run_*.png/pdf
  V2  ensemble: median + IQR band across all 50 runs (10 trials x 5 drones)
      ->  convergence_ensemble_median_*.png/pdf

Data: runs/ca_v9_candidate_logs_n5_v1/ca_diffusion_es/N5/trial_*/candidates_drone*.csv
Output: figs/
"""

import csv
import sys
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

FS_LABEL = 11.5
FS_TICK = 10.0
FS_LEGEND = 10.0

C_MEAN = "#D62F2F"
C_BAND = "#D62F2F"
C_RAW = "#9AA6B0"

MA_WINDOW = 30


def load_run(trial: str, drone: int):
    """Return sorted (evaluations, fixed_cost, best_so_far) for one run."""
    f = RUNS / trial / f"candidates_drone{drone}.csv"
    rows = []
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["evaluation_id"]),
                         float(r["fixed_cost"]),
                         float(r["best_feasible_fixed_cost_so_far"])))
    rows.sort()
    ev = np.array([r[0] for r in rows])
    cost = np.array([r[1] for r in rows])
    best = np.array([r[2] for r in rows])
    return ev, cost, best


def load_all_best_so_far():
    """Return (50, 4000) array of best feasible cost for every run."""
    curves = []
    for trial in sorted(RUNS.glob("trial_*")):
        for drone in range(5):
            _, _, best = load_run(trial.name, drone)
            curves.append(best)
    return np.array(curves)


def rolling(a, window):
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="valid")


def draw_base(ax):
    ax.set_xlabel("Evaluations", fontsize=FS_LABEL)
    ax.set_xlim(0, 4000)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    _apply_publication_axis_style(ax, grid_axis="both")
    ax.tick_params(labelsize=FS_TICK)


def v1_single_run(trial="trial_04", drone=2):
    """Reference style: raw cost + moving average + rolling +-1 std band."""
    ev, cost, _ = load_run(trial, drone)
    # normalize by the initial-population mean cost (evaluations 1..100)
    base = cost[:100].mean()
    y = cost / base

    ma = rolling(y, MA_WINDOW)
    ma_x = ev[MA_WINDOW - 1:]

    # rolling std of the raw cost around the moving average
    std = np.array([y[max(0, i - MA_WINDOW + 1): i + 1].std()
                    for i in range(len(y))])
    std_ma = rolling(std, MA_WINDOW)
    std_ma_x = ev[MA_WINDOW - 1:]

    fig = plt.figure(figsize=(7.16, 3.4), facecolor="white")
    grid = fig.add_gridspec(1, 1, left=0.09, right=0.985, top=0.96, bottom=0.15)
    ax = fig.add_subplot(grid[0])

    ax.plot(ev, y, color=C_RAW, linewidth=0.6, alpha=0.55, zorder=2,
            label="Raw data")
    ax.fill_between(ma_x, ma - std_ma, ma + std_ma, color=C_BAND, alpha=0.16,
                    linewidth=0, zorder=3, label=r"$\pm 1$ std. dev.")
    ax.plot(ma_x, ma, color=C_MEAN, linewidth=2.2, zorder=5,
            label=f"Moving average (window = {MA_WINDOW})")

    ax.set_ylabel("Normalized objective cost", fontsize=FS_LABEL)
    ax.set_ylim(bottom=0)
    draw_base(ax)
    ax.legend(loc="upper right", fontsize=FS_LEGEND, frameon=True,
              framealpha=0.9, edgecolor="#C8CDD2", handlelength=1.8,
              borderpad=0.5).get_frame().set_linewidth(0.6)

    out = OUT_DIR / f"convergence_single_run_{trial}_d{drone}"
    fig.savefig(out.with_suffix(".png"), dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved: {out}.png/.pdf")


def v2_ensemble():
    """Median + IQR band across all 50 runs, normalized best feasible cost."""
    arr = load_all_best_so_far()
    first_vals = np.array([c[~np.isnan(c)][0] for c in arr])
    c0 = float(np.median(first_vals))
    norm = arr / c0
    evaluations = np.arange(1, norm.shape[1] + 1)

    med = np.nanmedian(norm, axis=0)
    p25 = np.nanpercentile(norm, 25, axis=0)
    p75 = np.nanpercentile(norm, 75, axis=0)

    fig = plt.figure(figsize=(7.16, 3.4), facecolor="white")
    grid = fig.add_gridspec(1, 1, left=0.09, right=0.985, top=0.96, bottom=0.15)
    ax = fig.add_subplot(grid[0])

    for i in range(norm.shape[0]):
        ax.plot(evaluations, norm[i], color=C_RAW, linewidth=0.6, alpha=0.30,
                zorder=2, label="Raw data (individual runs)" if i == 0 else None)
    ax.fill_between(evaluations, p25, p75, color=C_BAND, alpha=0.15,
                    linewidth=0, zorder=3, label="Interquartile range")
    ax.plot(evaluations, med, color=C_MEAN, linewidth=2.2, zorder=5,
            label="Median of 50 runs")

    ax.set_ylabel("Normalized best feasible cost", fontsize=FS_LABEL)
    ax.set_ylim(bottom=0)
    draw_base(ax)
    ax.legend(loc="upper right", fontsize=FS_LEGEND, frameon=True,
              framealpha=0.9, edgecolor="#C8CDD2", handlelength=1.8,
              borderpad=0.5).get_frame().set_linewidth(0.6)

    out = OUT_DIR / "convergence_ensemble_median"
    fig.savefig(out.with_suffix(".png"), dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved: {out}.png/.pdf")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v1_single_run()
    v2_ensemble()
