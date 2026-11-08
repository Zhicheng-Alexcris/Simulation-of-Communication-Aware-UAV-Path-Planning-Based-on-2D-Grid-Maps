"""Smoothed minimum-clearance survival figure for U = 5 / 7 / 9.

Same style as make_smooth_turn_ecdf_n579.py:
  - Gaussian-kernel smoothed survival S(x) = 1 - F(x) with a reflecting
    boundary at the 3.5 m clearance requirement,
  - identical colors / markers / linestyles / bottom legend,
  - curves start exactly at S = 1 (left) and end on the x-axis (right).

Data sources (same as formal_clearance_survival_n579.png):
  astar, astar_nsga2   <- runs/formal_experiment_hv2000/records.csv
  nsga2, diffusion_only<- runs/formal_experiment_additional_baselines_hv2000/records.csv
  ca_diffusion_es      <- runs/formal_experiment_ca_v9_hv2000/records.csv

Output (in the latex photo dir):
  formal_clearance_survival_n579_smooth.{png,pdf}
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import erf

CODE_ROOT = Path(r"D:\desktop\code")
RUNS = CODE_ROOT / "runs"
OUT_DIR = Path(r"D:\desktop\UAV廊道\latex\photo1.0")

RECORD_FILES = {
    "astar": RUNS / "formal_experiment_hv2000" / "records.csv",
    "astar_nsga2": RUNS / "formal_experiment_hv2000" / "records.csv",
    "nsga2": RUNS / "formal_experiment_additional_baselines_hv2000" / "records.csv",
    "diffusion_only": RUNS / "formal_experiment_additional_baselines_hv2000" / "records.csv",
    "ca_diffusion_es": RUNS / "formal_experiment_ca_v9_hv2000" / "records.csv",
}

LABELS = {
    "astar": "A*",
    "astar_nsga2": "A* + NSGA-II",
    "nsga2": "NSGA-II",
    "diffusion_only": "Diffusion-only",
    "ca_diffusion_es": "CA-Diffusion-ES",
}

METHOD_ORDER = ["astar", "nsga2", "ca_diffusion_es", "astar_nsga2", "diffusion_only"]

STYLE = {
    "astar": {"color": "#E6A817", "marker": "o", "linestyle": "-"},           # bright golden yellow
    "astar_nsga2": {"color": "#0F62B5", "marker": "s", "linestyle": "--"},    # strong blue
    "nsga2": {"color": "#14A05A", "marker": "d", "linestyle": "-."},          # green
    "diffusion_only": {"color": "#A55CD0", "marker": "^", "linestyle": ":"},  # bright purple
    "ca_diffusion_es": {"color": "#D62F2F", "marker": "v", "linestyle": "-"}, # vivid red (proposed)
}

PROPOSED = "ca_diffusion_es"
REFERENCE_M = 3.5     # clearance requirement
LOWER_BOUND = 3.5     # reflecting boundary for the KDE (clearance >= 3.5 m)
COUNTS = (5, 7, 9)

FS_TITLE = 12.5
FS_LABEL = 11.5
FS_TICK = 10.0
FS_LEGEND = 9.5


def load_values(method: str, count: int) -> np.ndarray:
    values = []
    with RECORD_FILES[method].open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != method or int(row["uav_count"]) != count:
                continue
            try:
                success = float(row["success"])
                value = float(row["min_clearance_m"])
            except (TypeError, ValueError):
                continue
            if success >= 0.5 and np.isfinite(value):
                values.append(value)
    return np.sort(np.asarray(values, dtype=float))


def _phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def kde_cdf(values: np.ndarray, grid: np.ndarray, smooth: float = 1.0,
            lower: float = LOWER_BOUND) -> np.ndarray:
    """Gaussian-kernel CDF with a reflecting lower boundary."""
    n = values.size
    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        span = max(float(values.max()) - float(values.min()), 1.0)
        std = span / 4.0
    h = max(1.06 * std * n ** (-0.2) * smooth, 1e-3)

    base = _phi((grid[None, :] - values[:, None]) / h).mean(axis=0)
    if np.isfinite(lower):
        reflected = _phi(((2.0 * lower - grid)[None, :] - values[:, None]) / h).mean(axis=0)
        base = base - reflected
    return np.clip(base, 0.0, 1.0)


def draw_panel(axis, records_cache: dict) -> None:
    for method in METHOD_ORDER:
        values = records_cache[method]
        if values.size == 0:
            continue
        style = STYLE[method]
        # survival S(x) = 1 - F(x); left of the data the KDE decays to F≈0,
        # prepend anchors so the curve starts exactly at 1 and ends at 0
        left_pad = 6.0
        grid = np.linspace(values.min() - left_pad, values.max() + left_pad, 600)
        survival = 1.0 - kde_cdf(values, grid)
        grid = np.concatenate([[grid[0]], grid, [grid[-1]]])
        survival = np.concatenate([[1.0], survival, [0.0]])

        linewidth = 2.1 if method == PROPOSED else 1.7
        axis.plot(
            grid, survival,
            color=style["color"], linewidth=linewidth, linestyle=style["linestyle"],
            label=LABELS[method], zorder=4,
        )
        # markers evenly spaced in survival probability (curve is decreasing,
        # so read x back with reversed arrays)
        n_markers = 10 if values.size >= 20 else max(4, int(values.size * 0.2))
        probabilities = np.linspace(0.05, 0.95, n_markers)
        xs = np.interp(probabilities, survival[::-1], grid[::-1])
        axis.scatter(
            xs, probabilities,
            s=16, marker=style["marker"],
            facecolors="white", edgecolors=style["color"],
            linewidths=0.9, zorder=5,
        )
    axis.axvline(
        REFERENCE_M, color="#7A7F85", linestyle="--", linewidth=1.0, zorder=3,
    )


def build_figure() -> None:
    plt.rcParams.update({
        "font.size": FS_LABEL,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig = plt.figure(figsize=(7.16, 3.85))
    grid_spec = fig.add_gridspec(
        1, 3, left=0.055, right=0.995, top=0.94, bottom=0.26, wspace=0.30,
    )

    panel_labels = ["(a)", "(b)", "(c)"]
    x_min_all = min(
        load_values(method, count).min()
        for method in METHOD_ORDER for count in COUNTS
    )
    x_left = float(np.floor(x_min_all / 5.0) * 5.0)
    x_max_all = max(
        load_values(method, count).max()
        for method in METHOD_ORDER for count in COUNTS
    )
    x_right = float(np.ceil(x_max_all / 5.0) * 5.0)

    for col, (count, label) in enumerate(zip(COUNTS, panel_labels)):
        axis = fig.add_subplot(grid_spec[col])
        cache = {method: load_values(method, count) for method in METHOD_ORDER}
        draw_panel(axis, cache)
        axis.set_title(f"{label} U = {count}", fontsize=FS_TITLE, pad=5)
        axis.set_xlim(x_left, x_right)
        axis.set_ylim(0.0, 1.02)
        axis.set_xticks(np.arange(0, x_right + 1, 10))
        axis.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        axis.set_xlabel("Minimum clearance (m)", fontsize=FS_LABEL)
        if col == 0:
            axis.set_ylabel("Survival probability", fontsize=FS_LABEL)
        axis.grid(color="#E6EAED", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#A9B0B6")
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors="#4E545A", width=0.8, length=3.0, labelsize=FS_TICK)

    # matplotlib fills legends column-first, so this order renders as:
    # Row1: A* | NSGA-II | CA-Diffusion-ES   Row2: A* + NSGA-II | Diffusion-only | Ref.
    legend_order = ["astar", "astar_nsga2", "nsga2", "diffusion_only", "ca_diffusion_es"]
    handles = [
        Line2D(
            [0], [0],
            color=STYLE[m]["color"], linewidth=2.0, linestyle=STYLE[m]["linestyle"],
            marker=STYLE[m]["marker"], markersize=5.0,
            markerfacecolor="white", markeredgecolor=STYLE[m]["color"],
            label=LABELS[m],
        )
        for m in legend_order
    ]
    handles.append(Line2D(
        [0], [0], color="#7A7F85", linestyle="--", linewidth=1.0,
        label="Ref.: 3.5 m",
    ))
    legend = fig.legend(
        handles=handles, ncol=3, loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        fontsize=FS_LEGEND, frameon=False,
        columnspacing=1.6, handlelength=2.2, handletextpad=0.55,
    )
    legend.get_frame().set_linewidth(0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "formal_clearance_survival_n579_smooth.png"
    pdf_path = OUT_DIR / "formal_clearance_survival_n579_smooth.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


if __name__ == "__main__":
    build_figure()
