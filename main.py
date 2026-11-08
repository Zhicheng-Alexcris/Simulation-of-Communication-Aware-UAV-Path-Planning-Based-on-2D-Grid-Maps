"""本科毕业设计入口；运行本文件会进行 A* 仿真并输出图表与 CSV。"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from environment import build_grid
from metrics import evaluate, profile
from path_planner import NoPathError, astar


ROOT = Path(__file__).resolve().parent


def _write_csv(path, records):
    if not records:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _save_path(output, name, grid, result):
    distance, _ = profile(grid, result)
    records = []
    for index, (col, row) in enumerate(result.path):
        x, y = grid.to_world((col, row))
        records.append({"node_index": index, "col": col, "row": row,
                        "x_m": x, "y_m": y, "distance_from_start_m": float(distance[index]),
                        "pathloss_db": float(grid.pathloss_db[row, col]),
                        "weak_signal": int(grid.weak_signal[row, col])})
    _write_csv(output / f"path_{name}.csv", records)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Communication-aware UAV path planning on a 2D grid")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json", help="地图配置 JSON")
    parser.add_argument("--weight", type=float, help="通信权重，默认使用配置值")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], help="四邻域或八邻域")
    parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), help="起点米制坐标")
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"), help="终点米制坐标")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="输出目录")
    parser.add_argument("--show", action="store_true", help="保存图片后显示窗口")
    parser.add_argument("--no-sweep", action="store_true", help="跳过通信权重敏感性实验")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    weight = args.weight if args.weight is not None else float(config["communication_weight"])
    connectivity = args.connectivity if args.connectivity is not None else config["connectivity"]
    start_xy = args.start if args.start is not None else config["start_m"]
    goal_xy = args.goal if args.goal is not None else config["goal_m"]
    grid = build_grid(config)
    start, goal = grid.to_grid(start_xy), grid.to_grid(goal_xy)

    # 两个实验使用相同的地图、起终点、邻域与启发函数，只改变通信权重。
    baseline = astar(grid, start, goal, 0.0, connectivity)
    aware = astar(grid, start, goal, weight, connectivity)
    metrics = []
    for name, result, value in [("distance_only_astar", baseline, 0.0),
                                 ("communication_aware_astar", aware, weight)]:
        metrics.append({"algorithm": name, **evaluate(grid, result, value, connectivity)})

    sweep = []
    if not args.no_sweep:
        weights = sorted(set([0.0, weight] + [float(v) for v in config["sweep_weights"]]))
        for value in weights:
            result = baseline if value == 0 else aware if value == weight else astar(grid, start, goal, value, connectivity)
            sweep.append(evaluate(grid, result, value, connectivity))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "metrics.csv", metrics)
    _write_csv(output / "weight_sensitivity.csv", sweep)
    _save_path(output, "distance_only", grid, baseline)
    _save_path(output, "communication_aware", grid, aware)
    categories = np.where(grid.obstacles, 2, grid.weak_signal.astype(int))
    np.savetxt(output / "grid_map.csv", categories, delimiter=",", fmt="%d")
    np.savetxt(output / "pathloss_map_db.csv", grid.pathloss_db, delimiter=",", fmt="%.3f")
    np.savez_compressed(output / "grid_data.npz", obstacles=grid.obstacles,
                        weak_signal=grid.weak_signal, pathloss_db=grid.pathloss_db,
                        communication_cost=grid.communication_cost,
                        baseline_expanded=baseline.expanded_mask,
                        aware_expanded=aware.expanded_mask,
                        extent=np.array(grid.extent), cell_size=grid.cell_size)
    effective_config = {**config, "communication_weight": weight, "connectivity": connectivity,
                        "start_m": list(start_xy), "goal_m": list(goal_xy)}
    (output / "used_config.json").write_text(json.dumps(effective_config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"requested_start_m": list(start_xy), "requested_goal_m": list(goal_xy),
               "actual_start_cell_center_m": list(grid.to_world(start)),
               "actual_goal_cell_center_m": list(grid.to_world(goal)),
               "grid_shape_rows_cols": list(grid.shape), "metrics": metrics}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 默认使用无窗口绘图，便于在 IDE、命令行或无界面环境运行。
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    from plot import make_figures
    make_figures(grid, start, goal, baseline, aware, metrics, sweep, output, args.show)

    print("\n二维栅格通信感知无人机路径规划完成。")
    print(f"地图：{grid.shape[1]} 列 x {grid.shape[0]} 行；栅格尺寸 {grid.cell_size:g} m")
    for record in metrics:
        print(f"{record['algorithm']}: 路径长度 {record['path_length_m']:.2f} m，"
              f"弱信号穿越长度 {record['weak_signal_length_m']:.2f} m，"
              f"弱信号占比 {record['weak_signal_ratio_percent']:.2f}%")
    print(f"图表与数据已保存：{output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (NoPathError, ValueError, KeyError, TypeError, OSError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
