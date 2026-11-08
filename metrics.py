"""统一计算两种 A* 路线的长度和弱信号穿越量。"""
import math

import numpy as np

from path_planner import edge_cost, neighbors


def evaluate(grid, result, weight, connectivity):
    path = result.path
    if not path or any(not grid.is_free(p) for p in path):
        raise ValueError("路径为空或穿越了障碍物。")
    length = weak_length = communication_integral = loss_integral = 0.0
    for a, b in zip(path, path[1:]):
        if b not in neighbors(grid, a, connectivity):
            raise ValueError("路径包含不合法移动或对角穿越障碍物。")
        distance = grid.cell_size * math.hypot(b[0] - a[0], b[1] - a[1])
        length += distance
        # 一条边的一半位于出发栅格，另一半位于到达栅格。
        weak_length += distance * (int(grid.weak_signal[a[1], a[0]])
                                   + int(grid.weak_signal[b[1], b[0]])) / 2
        communication_integral += (edge_cost(grid, a, b, 1.0) - distance)
        loss_integral += distance * (grid.pathloss_db[a[1], a[0]]
                                    + grid.pathloss_db[b[1], b[0]]) / 2
    if not math.isclose(result.objective_cost, length + weight * communication_integral,
                        rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("A* 代价与统计结果不一致。")
    return {
        "communication_weight": float(weight),
        "path_length_m": float(length),
        "weak_signal_cells_entered": sum(int(grid.weak_signal[y, x]) for x, y in path[1:]),
        "weak_signal_length_m": float(weak_length),
        "weak_signal_ratio_percent": float(100 * weak_length / length) if length else 0.0,
        "communication_integral_m": float(communication_integral),
        "mean_pathloss_db": float(loss_integral / length) if length else float(grid.pathloss_db[path[0][1], path[0][0]]),
        "objective_cost": float(result.objective_cost),
        "expanded_nodes": int(result.expanded_nodes),
        "elapsed_ms": float(result.elapsed_ms),
        "path_nodes": len(path),
    }


def profile(grid, result):
    xy = np.array([grid.to_world(p) for p in result.path])
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    loss = np.array([grid.pathloss_db[y, x] for x, y in result.path])
    return distance, loss
