"""Reference-normalized feasible-hypervolume convergence, U=5.

Plots the per-generation (not smoothed) hypervolume trajectory for the four
iterative methods, matching the paper's Fig. 4 metric (HV normalized by the
reference rectangle defined by the reference point (2000, 1.05), i.e. area
= 2100), but with the real per-generation data points kept visible so the
search dynamics (noise / plateaus / jumps) show instead of a flat curve.

Data (history_*.csv, one hypervolume row per generation):
  A* + NSGA-II      runs/formal_experiment_hv2000/astar_nsga2/N5
  NSGA-II           runs/formal_experiment_additional_baselines_hv2000/nsga2/N5
  Diffusion-only    runs/formal_experiment_additional_baselines_hv2000/diffusion_only/N5
  CA-Diffusion-ES   runs/formal_experiment_ca_v9_hv2000/ca_diffusion_es/N5
Each method: 10 trials x 5 drones = 50 runs, 40 generations, 100 evals/gen.

Output: figs/convergence_hv_u5.png/.pdf
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

RUNS = CODE_ROOT / "runs"

# (method key, label, data dir, style)
# colors follow the established high-discriminability palette used for the
# ECDF figures; marker + linestyle give triple redundancy for B/W printing.
METHODS = [
    ("astar_nsga2",    r"A$^{*}$ + NSGA-II", RUNS / "formal_experiment_hv2000" / "astar_nsga2" / "N5",
     {"color": "#0F62B5", "marker": "s", "linestyle": "--"}),
    ("nsga2",          "NSGA-II",           RUNS / "formal_experiment_additional_baselines_hv2000" / "nsga2" / "N5",
     {"color": "#14A05A", "marker": "d", "linestyle": "-."}),
    ("diffusion_only", "Diffusion-only",    RUNS / "formal_experiment_additional_baselines_hv2000" / "diffusion_only" / "N5",
     {"color": "#A55CD0", "marker": "^", "linestyle": ":"}),
    ("ca_diffusion_es", "CA-Diffusion-ES",  RUNS / "formal_experiment_ca_v9_hv2000" / "ca_diffusion_es" / "N5",
     {"color": "#D62F2F", "marker": "o", "linestyle": "-"}),
]

REF_AREA = 2100.0  # reference rectangle: 2000 (HV budget) x 1.05 (path penalty)

FS_LABEL = 11.5
FS_TICK = 10.0
FS_LEGEND = 10.0

OUT_DIR = CODE_ROOT / "figs"


def load_runs(data_dir):
    """Return (n_runs, n_gens) array of normalized per-generation HV."""
    curves = []
    for trial in sorted(data_dir.glob("trial_*")):
        for drone in range(5):
            f = trial / f"history_drone{drone}.csv"
            rows = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
            hv = np.array([float(r["hypervolume"]) for r in rows]) / REF_AREA
            curves.append(hv)
    arr = np.array(curves)
    return arr


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.16, 4.15), facecolor="white")
    grid = fig.add_gridspec(
        1, 1, left=0.085, right=0.99, top=0.96, bottom=0.30,
    )
    ax = fig.add_subplot(grid[0])

    evaluations = None
    for key, label, data_dir, st in METHODS:
        arr = load_runs(data_dir)
        gens = arr.shape[1]
        ev = np.arange(1, gens + 1) * 100.0  # 100 evals per generation
        evaluations = ev

        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)

        # inter-quartile band (real spread, not a smoothing envelope)
        ax.fill_between(
            ev, p25, p75, color=st["color"], alpha=0.13,
            linewidth=0, zorder=2,
        )
        # per-generation median line with markers on every data point so the
        # true per-generation motion (jumps / plateaus) is visible
        ax.plot(
            ev, med,
            color=st["color"], linestyle=st["linestyle"], linewidth=2.0,
            marker=st["marker"], markersize=4.5,
            markerfacecolor="white", markeredgecolor=st["color"],
            markeredgewidth=1.1, markevery=1,
            label=label, zorder=5,
        )

    ax.set_xlabel("Evaluations", fontsize=FS_LABEL)
    ax.set_ylabel("Reference-normalized\nfeasible hypervolume", fontsize=FS_LABEL)
    ax.set_xlim(0, 4000)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_ylim(0.0, 0.85)
    _apply_publication_axis_style(ax, grid_axis="both")
    ax.tick_params(labelsize=FS_TICK)

    # legend at the bottom (user preference), two rows of two columns
    legend = fig.legend(
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, fontsize=FS_LEGEND, frameon=False,
        columnspacing=1.6, handlelength=2.2, handletextpad=0.55,
        labelspacing=0.5,
    )
    legend.get_frame().set_linewidth(0.0)

    out = OUT_DIR / "convergence_hv_u5"
    fig.savefig(out.with_suffix(".png"), dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved: {out}.png/.pdf")


if __name__ == "__main__":
    main()
