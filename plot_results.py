"""Plot saved CA-Diffusion-ES trajectories without rerunning optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_results(input_dir: Path) -> tuple[list[int], list[np.ndarray]]:
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing result summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    drone_ids: list[int] = []
    paths: list[np.ndarray] = []
    for record in summary:
        drone_id = int(record["drone_id"])
        trajectory_path = input_dir / f"trajectory_drone{drone_id}.csv"
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Missing trajectory: {trajectory_path}")
        path = np.loadtxt(trajectory_path, delimiter=",", skiprows=1)
        path = np.atleast_2d(path).astype(float)
        if path.shape[0] < 2 or path.shape[1] != 3:
            raise ValueError(f"Invalid trajectory shape in {trajectory_path}: {path.shape}")
        drone_ids.append(drone_id)
        paths.append(path)
    if not paths:
        raise ValueError(f"No trajectories listed in {summary_path}")
    return drone_ids, paths


def plot_saved_results(legacy_root: Path, input_dir: Path, output_dir: Path) -> list[Path]:
    sys.path.insert(0, str(legacy_root))
    from environment import Environment
    from plot import PathVisualizer

    drone_ids, paths = _load_results(input_dir)
    env = Environment()

    # Use only the endpoints in the saved run, excluding unused UAV stations.
    env.starts = [tuple(path[0]) for path in paths]
    env.ends = [tuple(path[-1]) for path in paths]
    rebuild = getattr(env, "_build_vertiports_from_starts_ends", None)
    env.vertiports = rebuild(z_min=0.0, z_max=40.0, nd=2) if callable(rebuild) else []

    colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red",
        "tab:brown", "tab:cyan", "tab:olive", "tab:pink", "tab:gray",
    ]
    selected_colors = [colors[i % len(colors)] for i in range(len(paths))]
    labels = [f"Drone {drone_id + 1}" for drone_id in drone_ids]
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = PathVisualizer(env, pipe_radius=3.5)
    visualizer.plot_buildings()
    visualizer.plot_ues_3d(z=0.8, color="tab:green", marker="^", size=55)
    for path, color in zip(paths, selected_colors):
        visualizer.plot_trajectory_as_pipe(path.tolist(), color=color)
    visualizer.plot_vertiports(alpha=1.0, colors=selected_colors, show_labels=False)
    visualizer.plot_takeoff_landing_labels(colors=selected_colors, top_margin=1.2)
    visualizer._decorate_plot()
    visualizer.set_legend_like_topdown(
        n_drones=len(paths), colors=selected_colors, ue_color="tab:green"
    )
    visualizer.ax.view_init(elev=30, azim=-55)
    path_3d = output_dir / "ca_diffusion_es_3d.png"
    visualizer.fig.savefig(path_3d, dpi=300, bbox_inches="tight")
    plt.close(visualizer.fig)

    path_topdown = output_dir / "ca_diffusion_es_topdown.png"
    visualizer.plot_topdown(
        paths,
        title=f"CA-Diffusion-ES trajectories (XY) - {len(paths)} UAVs",
        colors=selected_colors,
        labels=labels,
        show_buildings=True,
        show_vertiports=True,
        show_ues=True,
        save_path=path_topdown,
        dpi=300,
    )
    return [path_3d, path_topdown]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for output in plot_saved_results(args.legacy_root, args.input_dir, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()