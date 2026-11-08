"""Vertical (1-column x 3-row) versions of the two smoothed distribution
figures, designed at exact print width (3.5 in = 0.49 \\textwidth) so fonts
render 1:1 in the final PDF. The two figures are placed side by side in a
cross-column figure* environment in the LaTeX source.

Outputs (in the latex photo dir):
  formal_turn_ecdf_n579_vertical.{png,pdf}
  formal_clearance_survival_n579_vertical.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

CODE_ROOT = Path(r"D:\desktop\code")
import sys

sys.path.insert(0, str(CODE_ROOT))

from make_smooth_turn_ecdf_n579 import (  # noqa: E402
    COUNTS,
    LABELS,
    METHOD_ORDER,
    PROPOSED,
    STYLE,
    kde_cdf,
    load_values as load_turn,
)
from make_smooth_clearance_survival_n579 import (  # noqa: E402
    REFERENCE_M,
    kde_cdf as clearance_kde_cdf,
    load_values as load_clearance,
)

OUT_DIR = Path(r"D:\desktop\UAV廊道\latex\photo1.0")

FS_TITLE = 9.5
FS_LABEL = 9.0
FS_TICK = 8.0
FS_LEGEND = 7.5


def draw_turn_panel(axis, cache: dict) -> None:
    for method in METHOD_ORDER:
        values = cache[method]
        if values.size == 0:
            continue
        style = STYLE[method]
        grid = np.linspace(values.min() - 6.0, values.max() + 6.0, 600)
        curve = kde_cdf(values, grid)
        grid = np.concatenate([[grid[0]], grid])
        curve = np.concatenate([[0.0], curve])
        linewidth = 1.9 if method == PROPOSED else 1.5
        axis.plot(grid, curve, color=style["color"], linewidth=linewidth,
                  linestyle=style["linestyle"], zorder=4)
        n_markers = 8 if values.size >= 20 else max(4, int(values.size * 0.2))
        probabilities = np.linspace(0.05, 0.95, n_markers)
        xs = np.interp(probabilities, curve, grid)
        axis.scatter(xs, probabilities, s=12, marker=style["marker"],
                     facecolors="white", edgecolors=style["color"],
                     linewidths=0.8, zorder=5)
    axis.axvline(35.0, color="#7A7F85", linestyle="--", linewidth=0.9, zorder=3)


def draw_clearance_panel(axis, cache: dict) -> None:
    for method in METHOD_ORDER:
        values = cache[method]
        if values.size == 0:
            continue
        style = STYLE[method]
        grid = np.linspace(values.min() - 6.0, values.max() + 6.0, 600)
        survival = 1.0 - clearance_kde_cdf(values, grid)
        grid = np.concatenate([[grid[0]], grid, [grid[-1]]])
        survival = np.concatenate([[1.0], survival, [0.0]])
        linewidth = 1.9 if method == PROPOSED else 1.5
        axis.plot(grid, survival, color=style["color"], linewidth=linewidth,
                  linestyle=style["linestyle"], zorder=4)
        n_markers = 8 if values.size >= 20 else max(4, int(values.size * 0.2))
        probabilities = np.linspace(0.05, 0.95, n_markers)
        xs = np.interp(probabilities, survival[::-1], grid[::-1])
        axis.scatter(xs, probabilities, s=12, marker=style["marker"],
                     facecolors="white", edgecolors=style["color"],
                     linewidths=0.8, zorder=5)
    axis.axvline(REFERENCE_M, color="#7A7F85", linestyle="--", linewidth=0.9, zorder=3)


def apply_axis_style(axis) -> None:
    axis.grid(color="#E6EAED", linewidth=0.5, alpha=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#A9B0B6")
        axis.spines[side].set_linewidth(0.7)
    axis.tick_params(colors="#4E545A", width=0.7, length=2.6, labelsize=FS_TICK)


def legend_handles(ref_label: str):
    # column-first filling with ncol=2 renders:
    # col1: A* / A* + NSGA-II / NSGA-II   col2: Diffusion-only / CA-Diffusion-ES / Ref.
    order = ["astar", "astar_nsga2", "nsga2", "diffusion_only", "ca_diffusion_es"]
    handles = [
        Line2D([0], [0], color=STYLE[m]["color"], linewidth=1.6,
               linestyle=STYLE[m]["linestyle"], marker=STYLE[m]["marker"],
               markersize=4.2, markerfacecolor="white",
               markeredgecolor=STYLE[m]["color"], label=LABELS[m])
        for m in order
    ]
    handles.append(Line2D([0], [0], color="#7A7F85", linestyle="--",
                          linewidth=0.9, label=ref_label))
    return handles


def build_vertical(kind: str, out_base: str) -> None:
    plt.rcParams.update({
        "font.size": FS_LABEL,
        "axes.linewidth": 0.7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig = plt.figure(figsize=(3.5, 7.55))
    grid_spec = fig.add_gridspec(
        3, 1, left=0.155, right=0.975, top=0.965, bottom=0.135, hspace=0.38,
    )

    axes = []
    for row, count in enumerate(COUNTS):
        share = axes[0] if axes else None
        axis = fig.add_subplot(grid_spec[row], sharex=share)
        axes.append(axis)

    for row, (count, label) in enumerate(zip(COUNTS, ["(a)", "(b)", "(c)"])):
        axis = axes[row]
        if kind == "turn":
            cache = {m: load_turn(m, count) for m in METHOD_ORDER}
            draw_turn_panel(axis, cache)
            axis.set_xlim(0, 175)
            axis.set_xticks([0, 50, 100, 150])
            axis.set_ylabel("Empirical CDF", fontsize=FS_LABEL)
        else:
            cache = {m: load_clearance(m, count) for m in METHOD_ORDER}
            draw_clearance_panel(axis, cache)
            x_right = float(np.ceil(max(v.max() for v in cache.values() if v.size) / 5.0) * 5.0)
            axis.set_xlim(0, x_right)
            axis.set_xticks(np.arange(0, x_right + 1, 5))
            axis.set_ylabel("Survival probability", fontsize=FS_LABEL)
        axis.set_ylim(0.0, 1.02)
        axis.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        axis.set_title(f"{label} U = {count}", fontsize=FS_TITLE, pad=3)
        if row < len(COUNTS) - 1:
            axis.tick_params(labelbottom=False)
        else:
            xlabel = "TurnP95 (deg)" if kind == "turn" else "Minimum clearance (m)"
            axis.set_xlabel(xlabel, fontsize=FS_LABEL)
        apply_axis_style(axis)

    ref_label = "Ref.: 35°" if kind == "turn" else "Ref.: 3.5 m"
    legend = fig.legend(
        handles=legend_handles(ref_label), ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        fontsize=FS_LEGEND, frameon=False,
        columnspacing=1.0, handlelength=1.9, handletextpad=0.5,
        labelspacing=0.45,
    )
    legend.get_frame().set_linewidth(0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{out_base}.png"
    pdf_path = OUT_DIR / f"{out_base}.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    build_vertical("turn", "formal_turn_ecdf_n579_vertical")
    build_vertical("clearance", "formal_clearance_survival_n579_vertical")


if __name__ == "__main__":
    main()
