"""生成毕业设计可用的地图、路径、指标、损耗曲线和参数分析图。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from metrics import profile


BASELINE_COLOR = "#2463a6"
AWARE_COLOR = "#d6473e"


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "savefig.dpi": 300, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")


def _map(ax, grid, start, goal, title):
    categories = np.zeros(grid.shape, dtype=int)
    categories[grid.weak_signal] = 1
    categories[grid.obstacles] = 2
    colors = ListedColormap(["#f1f4f8", "#f3c77a", "#3d4652"])
    ax.imshow(categories, origin="lower", extent=grid.extent,
              cmap=colors, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
              interpolation="nearest", zorder=0)
    xmin, xmax, ymin, ymax = grid.extent
    # 只给中小地图绘制逐格网线，避免极细栅格拖慢出图。
    if max(grid.shape) <= 100:
        ax.set_xticks(np.arange(xmin, xmax + grid.cell_size * 0.1, grid.cell_size), minor=True)
        ax.set_yticks(np.arange(ymin, ymax + grid.cell_size * 0.1, grid.cell_size), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.4)
        ax.tick_params(which="minor", length=0)
    ax.scatter(*grid.to_world(start), marker="o", s=70, c="#23966b",
               edgecolors="white", linewidths=1, zorder=5)
    ax.scatter(*grid.to_world(goal), marker="*", s=160, c="#8e4cad",
               edgecolors="white", linewidths=0.8, zorder=5)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x (m)", ylabel="y (m)", title=title)
    ax.set_aspect("equal")


def _path(ax, grid, result, color, label, dashed=False):
    xy = np.array([grid.to_world(p) for p in result.path])
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.1,
            linestyle="--" if dashed else "-", label=label, zorder=3)


def _legend(include_paths=False):
    handles = [Patch(facecolor="#3d4652", label="Impassable obstacle"),
               Patch(facecolor="#f3c77a", label="High communication loss"),
               Patch(facecolor="#f1f4f8", label="Normal signal"),
               Line2D([], [], marker="o", color="#23966b", linestyle="None", label="Start"),
               Line2D([], [], marker="*", color="#8e4cad", linestyle="None", label="Goal")]
    if include_paths:
        handles += [Line2D([], [], color=BASELINE_COLOR, linestyle="--", label="Distance-only A*"),
                    Line2D([], [], color=AWARE_COLOR, label="Communication-aware A*")]
    return handles


def make_figures(grid, start, goal, baseline, aware, metrics, sweep, output_dir, show=False):
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 6.8), layout="constrained")
    _map(ax, grid, start, goal, "2D urban grid map")
    fig.legend(handles=_legend(), loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output_dir, "01_grid_map")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.7), layout="constrained")
    for ax, result, record, color, title in zip(
            axes, [baseline, aware], metrics, [BASELINE_COLOR, AWARE_COLOR],
            ["Distance-only A*", "Communication-aware A*"]):
        _map(ax, grid, start, goal, title)
        _path(ax, grid, result, color, title)
        ax.text(0.02, 0.98,
                f"Length: {record['path_length_m']:.1f} m\n"
                f"Weak-signal travel: {record['weak_signal_length_m']:.1f} m\n"
                f"Weight: {record['communication_weight']:g}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"}, zorder=6)
    fig.legend(handles=_legend(), loc="outside lower center", ncol=5, frameon=False)
    _save(fig, output_dir, "02_path_comparison")

    fig, ax = plt.subplots(figsize=(7.6, 7.0), layout="constrained")
    _map(ax, grid, start, goal, "Path overlay: distance and communication trade-off")
    _path(ax, grid, baseline, BASELINE_COLOR, "Distance-only A*", dashed=True)
    _path(ax, grid, aware, AWARE_COLOR, "Communication-aware A*")
    fig.legend(handles=_legend(True), loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output_dir, "03_path_overlay")

    fields = [("path_length_m", "Path length (m)"),
              ("weak_signal_length_m", "Weak-signal travel (m)"),
              ("weak_signal_ratio_percent", "Weak-signal travel (%)"),
              ("weak_signal_cells_entered", "Weak cells entered")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), layout="constrained")
    for ax, (key, title) in zip(axes.flat, fields):
        values = [r[key] for r in metrics]
        bars = ax.bar(["Distance-only A*", "Comm.-aware A*"], values,
                      color=[BASELINE_COLOR, AWARE_COLOR], width=0.55)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3)
        ax.set_title(title)
        ax.set_ylim(0, max(max(values) * 1.2, 1.0))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    _save(fig, output_dir, "04_metrics_comparison")

    fig, ax = plt.subplots(figsize=(10.2, 4.6), layout="constrained")
    for result, color, label in [(baseline, BASELINE_COLOR, "Distance-only A*"),
                                 (aware, AWARE_COLOR, "Communication-aware A*")]:
        distance, loss = profile(grid, result)
        # 节点损耗折线是剖面展示；统计量采用边两端各占一半的模型。
        ax.plot(distance, loss, color=color, linewidth=1.8, label=label)
    ax.axhline(grid.weak_threshold_db, color="#9a6700", linestyle="--",
               label="Weak-signal threshold")
    ax.set(xlabel="Distance from start (m)", ylabel="Illustrative path loss (dB)",
           title="Communication loss along each planned path")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, output_dir, "05_pathloss_profile")

    if sweep:
        weights = [r["communication_weight"] for r in sweep]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
        for ax, key, label, color in [
                (axes[0], "path_length_m", "Path length (m)", BASELINE_COLOR),
                (axes[1], "weak_signal_length_m", "Weak-signal travel (m)", AWARE_COLOR)]:
            ax.plot(weights, [r[key] for r in sweep], "o-", color=color, linewidth=1.8)
            ax.set(xlabel="Communication weight", ylabel=label)
            ax.grid(alpha=0.2)
        fig.suptitle("Sensitivity to communication weight")
        _save(fig, output_dir, "06_weight_sensitivity")

    if show:
        plt.show()
    else:
        plt.close("all")
