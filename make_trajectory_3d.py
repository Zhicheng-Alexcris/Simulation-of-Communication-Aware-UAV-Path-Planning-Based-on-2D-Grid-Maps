"""Standalone 3-D corridor trajectory figure (paper Fig. 3 part).

Fonts are specified at the FINAL print size for a double-column journal
(7.16 in text width), so nothing shrinks in LaTeX.
Output: formal_ca_trajectory_3d.{png,pdf} in the latex photo dir.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

CODE_ROOT = Path(r"D:\desktop\code")
sys.path.insert(0, str(CODE_ROOT))

from environment import Environment  # noqa: E402
from ca_diffusion_es.report import (  # noqa: E402
    UE_COLOR,
    _draw_ca_3d_panel,
    _finalize_legend,
    _load_ca_paths_by_scale,
    _uav_palette,
)

RUNS = CODE_ROOT / "runs" / "formal_experiment_ca_v9_hv2000"
OUT_DIR = Path(r"D:\desktop\UAV廊道\latex\photo1.0")
OUT_BASE = "formal_ca_trajectory_3d"
PIPE_RADIUS = 2.4

# ---- print-size font plan (double column, no downscaling) ----
FS_TITLE = 12.5  # panel titles "(d) U = 5"
FS_LABEL = 11.5  # axis labels
FS_TICK = 10.0   # tick labels
FS_LEGEND = 10.0 # shared legend


def main():
    plt.rcParams.update({
        "font.size": FS_LABEL,
        "axes.linewidth": 0.8,
    })

    method_roots = {"ca_diffusion_es": RUNS}
    counts = (5, 7, 9)
    labels = ["(d)", "(e)", "(f)"]
    paths_by_count = {c: _load_ca_paths_by_scale(method_roots, c) for c in counts}
    colors = _uav_palette(9)

    env = Environment()

    fig = plt.figure(figsize=(7.16, 3.4), facecolor="white")
    grid = fig.add_gridspec(
        1, 3, left=0.05, right=0.995, top=0.93, bottom=0.20, wspace=0.22,
    )

    base_starts = list(env.starts)
    base_ends = list(env.ends)
    rebuild = getattr(env, "_build_vertiports_from_starts_ends", None)
    for col, count in enumerate(counts):
        env.starts = base_starts[:count]
        env.ends = base_ends[:count]
        if callable(rebuild):
            env.vertiports = rebuild(z_min=0.0, z_max=40.0, nd=2)
        ax = fig.add_subplot(grid[col], projection="3d")
        _draw_ca_3d_panel(ax, env, paths_by_count[count], count, colors, PIPE_RADIUS)
        local_legend = ax.get_legend()
        if local_legend is not None:
            local_legend.remove()
        ax.set_title("")
        ax.view_init(elev=30, azim=-55)
        ax.set_xlabel("X (m)", fontsize=FS_LABEL, labelpad=8.0)
        ax.set_ylabel("Y (m)", fontsize=FS_LABEL, labelpad=8.0)
        ax.set_xticks([-100.0, 100.0])
        ax.set_yticks([-100.0, 100.0])
        # shared altitude label on the rightmost panel: mplot3d's default
        # z-label position collapses for this short-z box aspect, so the
        # label is placed explicitly just right of the z tick column
        if col == len(counts) - 1:
            ax.set_zlabel("")
            ax.text2D(
                1.23, 0.58, "Altitude (m)",
                transform=ax.transAxes, rotation=90,
                ha="left", va="center",
                fontsize=FS_LABEL, color="#20252B",
            )
        else:
            ax.set_zlabel("")
            ax.set_zticklabels([])
        try:
            ax.set_box_aspect((240.0, 240.0, 40.0), zoom=1.12)
        except Exception:
            pass
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK, pad=0.4)
        try:
            ax.zaxis.set_tick_params(labelsize=FS_TICK, pad=0.4)
        except Exception:
            pass
        ax.text2D(
            0.5, 0.96, f"{labels[col]} U = {count}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS_TITLE, color="#20252B",
        )

    # shared legend (bottom)
    handles = [
        Line2D([0], [0], color=colors[i], lw=2.2, label=f"UAV {i + 1}")
        for i in range(max(counts))
    ]
    handles.extend([
        Line2D([0], [0], linestyle="None", marker="^", markersize=6.0,
               markerfacecolor=UE_COLOR, markeredgecolor="white", label="UE"),
        Line2D([0], [0], linestyle="None", marker="o", markersize=5.6,
               markerfacecolor="white", markeredgecolor="#555555", label="Takeoff"),
        Line2D([0], [0], linestyle="None", marker="s", markersize=5.4,
               markerfacecolor="#555555", markeredgecolor="white", label="Landing"),
    ])
    legend = fig.legend(
        handles=handles, loc="lower center", ncol=6,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=FS_LEGEND, frameon=False,
        columnspacing=0.95, handlelength=1.8, handletextpad=0.5,
    )
    try:
        _finalize_legend(legend)
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{OUT_BASE}.png"
    pdf_path = OUT_DIR / f"{OUT_BASE}.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


if __name__ == "__main__":
    main()
