"""Merge paper Fig. 2 (top-down) and Fig. 3 (3-D tubes) into one figure
sized for a double-column journal (7.16 in text width).

Fonts are specified at the FINAL print size, so nothing shrinks in LaTeX.
Output: formal_ca_trajectory_combined_2d3d.{png,pdf} in the latex photo dir.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D

CODE_ROOT = Path(r"D:\desktop\code")
sys.path.insert(0, str(CODE_ROOT))

from environment import Environment  # noqa: E402
from ca_diffusion_es.report import (  # noqa: E402
    BUILDING_EDGE_2D,
    BUILDING_FACE_2D,
    UE_COLOR,
    _apply_publication_axis_style,
    _draw_ca_3d_panel,
    _finalize_legend,
    _load_ca_paths_by_scale,
    _uav_palette,
)

RUNS = CODE_ROOT / "runs" / "formal_experiment_ca_v9_hv2000"
OUT_DIR = Path(r"D:\desktop\UAV廊道\latex\photo1.0")
OUT_BASE = "formal_ca_trajectory_combined_2d3d"
PIPE_RADIUS = 2.4

# ---- print-size font plan (double column, no downscaling) ----
FS_TITLE = 12.5  # panel titles "(a) U = 5"
FS_LABEL = 11.5  # axis labels
FS_TICK = 10.0   # tick labels
FS_ANNOT = 8.0   # takeoff / landing annotations
FS_LEGEND = 10.0 # shared legend


_OUTLINE = [patheffects.withStroke(linewidth=2.0, foreground="white", alpha=0.9)]


def _deoverlap_annotations(ax, anns, iterations=300, pad_px=10.0):
    """Greedy pairwise repulsion of annotation text boxes in display space."""
    if len(anns) < 2:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    origin = np.asarray(inv.transform((0.0, 0.0)))

    def shift(ann, dx_px, dy_px):
        delta = np.asarray(inv.transform((dx_px, dy_px))) - origin
        x, y = ann.get_position()
        ann.set_position((x + delta[0], y + delta[1]))

    for _ in range(iterations):
        moved = False
        exts = [a.get_window_extent(renderer) for a in anns]
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                b1, b2 = exts[i], exts[j]
                if not b1.overlaps(b2):
                    continue
                ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if ox <= 0 or oy <= 0:
                    continue
                c1x = 0.5 * (b1.x0 + b1.x1)
                c2x = 0.5 * (b2.x0 + b2.x1)
                c1y = 0.5 * (b1.y0 + b1.y1)
                c2y = 0.5 * (b2.y0 + b2.y1)
                if ox < oy:
                    s = ox / 2.0 + pad_px
                    if c1x <= c2x:
                        shift(anns[i], -s, 0.0)
                        shift(anns[j], s, 0.0)
                    else:
                        shift(anns[i], s, 0.0)
                        shift(anns[j], -s, 0.0)
                else:
                    s = oy / 2.0 + pad_px
                    if c1y <= c2y:
                        shift(anns[i], 0.0, -s)
                        shift(anns[j], 0.0, s)
                    else:
                        shift(anns[i], 0.0, s)
                        shift(anns[j], 0.0, -s)
                moved = True
        if not moved:
            break

    # keep every label inside the axes box
    axbox = ax.get_window_extent(renderer)
    for ann in anns:
        ext = ann.get_window_extent(renderer)
        dx = dy = 0.0
        if ext.x0 < axbox.x0:
            dx = axbox.x0 - ext.x0
        elif ext.x1 > axbox.x1:
            dx = axbox.x1 - ext.x1
        if ext.y0 < axbox.y0:
            dy = axbox.y0 - ext.y0
        elif ext.y1 > axbox.y1:
            dy = axbox.y1 - ext.y1
        if dx or dy:
            shift(ann, dx, dy)


def draw_2d_panel(ax, env, users, paths, colors, count, label):
    for building in env.buildings:
        ax.add_patch(
            plt.Rectangle(
                (building.x_bounds[0], building.y_bounds[0]),
                building.x_bounds[1] - building.x_bounds[0],
                building.y_bounds[1] - building.y_bounds[0],
                facecolor=BUILDING_FACE_2D,
                edgecolor=BUILDING_EDGE_2D,
                linewidth=0.55,
                alpha=1.0,
                zorder=1,
            )
        )

    if len(users):
        ax.scatter(
            users[:, 0], users[:, 1], marker="^", s=18,
            facecolor=UE_COLOR, edgecolors="white", linewidths=0.4,
            alpha=0.92, zorder=5,
        )

    start_groups = defaultdict(list)
    for drone_id, points in paths:
        start = points[0]
        key = (round(float(start[0]), 2), round(float(start[1]), 2))
        start_groups[key].append(drone_id)
    labeled_shared = set()
    anns = []

    for drone_id, points in paths:
        color = colors[drone_id]
        ax.plot(
            points[:, 0], points[:, 1],
            color=color, linewidth=1.6, solid_capstyle="round", zorder=4,
        )
        start = np.asarray(points[0], dtype=float)
        end = np.asarray(points[-1], dtype=float)

        ax.scatter(start[0], start[1], marker="o", s=18,
                   facecolor="white", edgecolor=color, linewidth=0.9, zorder=7)
        ax.scatter(end[0], end[1], marker="s", s=17,
                   facecolor=color, edgecolor="white", linewidth=0.55, zorder=7)

        key = (round(float(start[0]), 2), round(float(start[1]), 2))
        sdx = 8.0 if start[0] <= 80.0 else -8.0
        sha = "left" if start[0] <= 80.0 else "right"
        if len(start_groups[key]) >= 2:
            if key not in labeled_shared:
                anns.append(ax.annotate(
                    "UAV Takeoff Station",
                    xy=(start[0], start[1]),
                    xytext=(start[0] + 8.0, start[1] + 9.0),
                    color="#222222", fontsize=FS_ANNOT,
                    ha="left", va="bottom", zorder=8,
                ))
                labeled_shared.add(key)
        else:
            anns.append(ax.annotate(
                f"Takeoff {drone_id + 1}",
                xy=(start[0], start[1]),
                xytext=(start[0] + sdx, start[1] + 9.0),
                color=color, fontsize=FS_ANNOT,
                ha=sha, va="bottom", zorder=8,
            ))

        edx = 8.0 if end[0] <= 80.0 else -8.0
        eha = "left" if end[0] <= 80.0 else "right"
        anns.append(ax.annotate(
            f"Landing {drone_id + 1}",
            xy=(end[0], end[1]),
            xytext=(end[0] + edx, end[1] + 9.0),
            color=color, fontsize=FS_ANNOT,
            ha=eha, va="bottom", zorder=8,
        ))

    for ann in anns:
        ann.set_path_effects(_OUTLINE)

    ax.set_title(f"{label} U = {count}", fontsize=FS_TITLE, pad=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-120.0, 120.0)
    ax.set_ylim(-128.0, 128.0)
    ax.set_xlabel("X position (m)", fontsize=FS_LABEL)
    _apply_publication_axis_style(ax, grid_axis="both")
    ax.tick_params(labelsize=FS_TICK)

    # must run AFTER limits/aspect are final, otherwise the display-space
    # text extents used for de-overlapping are stale
    _deoverlap_annotations(ax, anns)


def main():
    plt.rcParams.update({
        "font.size": FS_LABEL,
        "axes.linewidth": 0.8,
    })

    method_roots = {"ca_diffusion_es": RUNS}
    counts = (5, 7, 9)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    paths_by_count = {c: _load_ca_paths_by_scale(method_roots, c) for c in counts}
    colors = _uav_palette(9)

    env = Environment()
    users = np.asarray(getattr(env, "ues", []), dtype=float)

    fig = plt.figure(figsize=(7.16, 6.15), facecolor="white")
    outer = fig.add_gridspec(
        2, 1,
        left=0.07, right=1.0, top=0.955, bottom=0.085,
        hspace=0.13, height_ratios=[1.0, 0.85],
    )
    gs_top = outer[0].subgridspec(1, 3, wspace=0.06)
    gs_bot = outer[1].subgridspec(1, 3, wspace=0.22)

    # ---- row 1: top-down (shared Y axis) ----
    axes_2d = []
    for col, count in enumerate(counts):
        sharey = axes_2d[0] if axes_2d else None
        ax = fig.add_subplot(gs_top[col], sharey=sharey)
        draw_2d_panel(ax, env, users, paths_by_count[count], colors, count, labels[col])
        if col > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        axes_2d.append(ax)
    axes_2d[0].set_ylabel("Y position (m)", fontsize=FS_LABEL)

    # ---- row 2: 3-D tubes (Y and Altitude both on the right side) ----
    base_starts = list(env.starts)
    base_ends = list(env.ends)
    rebuild = getattr(env, "_build_vertiports_from_starts_ends", None)
    axes_3d = []
    for col, count in enumerate(counts):
        env.starts = base_starts[:count]
        env.ends = base_ends[:count]
        if callable(rebuild):
            env.vertiports = rebuild(z_min=0.0, z_max=40.0, nd=2)
        ax = fig.add_subplot(gs_bot[col], projection="3d")
        _draw_ca_3d_panel(ax, env, paths_by_count[count], count, colors, PIPE_RADIUS)
        local_legend = ax.get_legend()
        if local_legend is not None:
            local_legend.remove()
        ax.set_title("")
        # original paper view (same as report.py): pipes/buildings unchanged
        ax.view_init(elev=30, azim=-55)
        ax.set_xlabel("X (m)", fontsize=FS_LABEL, labelpad=8.0)
        ax.set_ylabel("Y (m)", fontsize=FS_LABEL, labelpad=8.0)
        # sparse ticks keep the axis midpoint free for the rotated label
        ax.set_xticks([-100.0, 100.0])
        ax.set_yticks([-100.0, 100.0])
        # altitude scale is re-drawn manually on the left of the first
        # panel (see mirror step below); hide native z labels everywhere
        ax.set_zlabel("")
        if col > 0:
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
            0.5, 0.96, f"{labels[3 + col]} U = {count}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS_TITLE, color="#20252B",
        )
        axes_3d.append(ax)

    # ---- mirror panel (d)'s altitude ticks to the LEFT of its box ----
    # the z axis renders on the right edge (overlapped by the next panel),
    # but the original paper figure shows the scale on the far left
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax0 = axes_3d[0]
    box0 = ax0.get_window_extent(renderer)
    inv = fig.transFigure.inverted()
    gap = 6.0
    for tl in ax0.zaxis.get_ticklabels():
        ext = tl.get_window_extent(renderer)
        cy = 0.5 * (ext.y0 + ext.y1)
        new_cx = box0.x0 - gap - ext.width / 2.0
        fx, fy = inv.transform((new_cx, cy))
        fig.text(fx, fy, tl.get_text(), fontsize=tl.get_fontsize(),
                 color=tl.get_color(), ha="center", va="center")
        tl.set_visible(False)
    # re-measure the mirrored ticks and put the axis label to their left
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    mirrored = [t for t in fig.texts if t.get_text().strip() in ("0", "20", "40")]
    if mirrored:
        exts = [t.get_window_extent(renderer) for t in mirrored]
        tick_left = min(e.x0 for e in exts)
        lab_cy = 0.5 * (min(e.y0 for e in exts) + max(e.y1 for e in exts))
        fx, fy = inv.transform((tick_left - 14.0, lab_cy))
        fig.text(fx, fy, "Altitude (m)", fontsize=FS_LABEL, color="#20252B",
                 rotation=90, ha="center", va="center")

    # ---- shared legend ----
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
        bbox_to_anchor=(0.5, 0.002),
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
