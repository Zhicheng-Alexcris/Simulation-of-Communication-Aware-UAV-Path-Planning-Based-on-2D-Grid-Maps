"""Two side-by-side figures on one row: turn ECDF (3 columns) + clearance
survival (3 columns) = 6 small panels in a single row, spanning the full
text width (7.16 in). Fonts are sized at the final print size.

Left figure  : TurnP95 ECDF       (a) U=5  (b) U=7  (c) U=9
Right figure : clearance survival (d) U=5  (e) U=7  (f) U=9

Output (latex photo dir): formal_turn_clearance_sidebyside.{png,pdf}
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

# print-size fonts (figure is 7.16 in wide, no downscaling)
FS_TITLE = 9.0
FS_LABEL = 8.5
FS_TICK = 7.5
FS_LEGEND = 8.0


def draw_turn_panel(axis, cache: dict) -> None:
    for method in METHOD_ORDER:
        values = cache[method]
        if values.size == 0:
            continue
        style = STYLE[method]
        grid = np.linspace(values.min() - 6.0, values.max() + 6.0, 500)
        curve = kde_cdf(values, grid)
        grid = np.concatenate([[grid[0]], grid])
        curve = np.concatenate([[0.0], curve])
        linewidth = 1.9 if method == PROPOSED else 1.5
        axis.plot(grid, curve, color=style["color"], linewidth=linewidth,
                  linestyle=style["linestyle"], zorder=4)
        n_markers = 8 if values.size >= 20 else max(4, int(values.size * 0.2))
        probabilities = np.linspace(0.05, 0.95, n_markers)
        xs = np.interp(probabilities, curve, grid)
        axis.scatter(xs, probabilities, s=11, marker=style["marker"],
                     facecolors="white", edgecolors=style["color"],
                     linewidths=0.8, zorder=5)
    axis.axvline(35.0, color="#7A7F85", linestyle="--", linewidth=0.9, zorder=3)


def draw_clearance_panel(axis, cache: dict) -> None:
    for method in METHOD_ORDER:
        values = cache[method]
        if values.size == 0:
            continue
        style = STYLE[method]
        grid = np.linspace(values.min() - 6.0, values.max() + 6.0, 500)
        survival = 1.0 - clearance_kde_cdf(values, grid)
        grid = np.concatenate([[grid[0]], grid, [grid[-1]]])
        survival = np.concatenate([[1.0], survival, [0.0]])
        linewidth = 1.9 if method == PROPOSED else 1.5
        axis.plot(grid, survival, color=style["color"], linewidth=linewidth,
                  linestyle=style["linestyle"], zorder=4)
        n_markers = 8 if values.size >= 20 else max(4, int(values.size * 0.2))
        probabilities = np.linspace(0.05, 0.95, n_markers)
        xs = np.interp(probabilities, survival[::-1], grid[::-1])
        axis.scatter(xs, probabilities, s=11, marker=style["marker"],
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
    axis.tick_params(colors="#4E545A", width=0.7, length=2.4, labelsize=FS_TICK)


def main() -> None:
    plt.rcParams.update({
        "font.size": FS_LABEL,
        "axes.linewidth": 0.7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig = plt.figure(figsize=(7.16, 3.1))
    grid_spec = fig.add_gridspec(
        1, 6, left=0.05, right=0.995, top=0.88, bottom=0.24,
        wspace=0.46,
    )

    # left 3 panels: turn ECDF (a)-(c); right 3 panels: clearance (d)-(f)
    turn_axes = [fig.add_subplot(grid_spec[0, j]) for j in range(3)]
    clear_axes = [fig.add_subplot(grid_spec[0, j]) for j in range(3, 6)]

    for j, (axis, count, letter) in enumerate(zip(
            turn_axes, COUNTS, ["(a)", "(b)", "(c)"])):
        cache = {m: load_turn(m, count) for m in METHOD_ORDER}
        draw_turn_panel(axis, cache)
        axis.set_title(f"{letter} U = {count}", fontsize=FS_TITLE, pad=3)
        axis.set_xlim(0, 175)
        axis.set_xticks([0, 50, 100, 150])
        axis.set_ylim(0.0, 1.02)
        axis.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        if j == 1:  # group-level xlabel under the middle panel
            axis.set_xlabel("TurnP95 (deg)", fontsize=FS_LABEL)
        if j == 0:
            axis.set_ylabel("Empirical CDF", fontsize=FS_LABEL)
        apply_axis_style(axis)

    for j, (axis, count, letter) in enumerate(zip(
            clear_axes, COUNTS, ["(d)", "(e)", "(f)"])):
        cache = {m: load_clearance(m, count) for m in METHOD_ORDER}
        draw_clearance_panel(axis, cache)
        x_right = float(np.ceil(max(v.max() for v in cache.values() if v.size) / 5.0) * 5.0)
        axis.set_title(f"{letter} U = {count}", fontsize=FS_TITLE, pad=3)
        axis.set_xlim(0, x_right)
        axis.set_xticks([0, 10, 20])
        axis.set_ylim(0.0, 1.02)
        axis.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        if j == 1:  # group-level xlabel under the middle panel
            axis.set_xlabel("Minimum clearance (m)", fontsize=FS_LABEL)
        if j == 0:
            axis.set_ylabel("Survival probability", fontsize=FS_LABEL)
        apply_axis_style(axis)

    # single-row legend at the very bottom: 6 entries, left to right
    order = ["astar", "astar_nsga2", "nsga2", "diffusion_only", "ca_diffusion_es"]
    handles = [
        Line2D([0], [0], color=STYLE[m]["color"], linewidth=1.6,
               linestyle=STYLE[m]["linestyle"], marker=STYLE[m]["marker"],
               markersize=4.0, markerfacecolor="white",
               markeredgecolor=STYLE[m]["color"], label=LABELS[m])
        for m in order
    ]
    handles.append(Line2D([0], [0], color="#7A7F85", linestyle="--",
                          linewidth=0.9, label="Ref."))
    legend = fig.legend(
        handles=handles, ncol=6, loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        fontsize=FS_LEGEND, frameon=False,
        columnspacing=1.0, handlelength=1.8, handletextpad=0.45,
    )
    legend.get_frame().set_linewidth(0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "formal_turn_clearance_sidebyside.png"
    pdf_path = OUT_DIR / "formal_turn_clearance_sidebyside.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


if __name__ == "__main__":
    main()
