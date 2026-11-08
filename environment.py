"""将论文的城市建筑投影到二维栅格；本文件不依赖原论文的三维模块。"""
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


@dataclass
class GridMap:
    # 数组用 [row, col] 索引；路径节点用 (col, row)，原点位于左下角。
    obstacles: np.ndarray
    pathloss_db: np.ndarray
    communication_cost: np.ndarray
    weak_signal: np.ndarray
    cell_size: float
    xmin: float
    ymin: float
    weak_threshold_db: float

    @property
    def shape(self):
        return self.obstacles.shape

    @property
    def extent(self):
        rows, cols = self.shape
        return (self.xmin, self.xmin + cols * self.cell_size,
                self.ymin, self.ymin + rows * self.cell_size)

    def contains(self, node):
        col, row = node
        rows, cols = self.shape
        return (isinstance(col, Integral) and isinstance(row, Integral)
                and 0 <= col < cols and 0 <= row < rows)

    def is_free(self, node):
        return self.contains(node) and not self.obstacles[node[1], node[0]]

    def to_world(self, node):
        col, row = node
        return (self.xmin + (col + 0.5) * self.cell_size,
                self.ymin + (row + 0.5) * self.cell_size)

    def to_grid(self, xy):
        x, y = map(float, xy)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("起终点坐标必须为有限数值。")
        xmin, xmax, ymin, ymax = self.extent
        if not (xmin <= x < xmax and ymin <= y < ymax):
            raise ValueError(f"坐标 {xy} 超出地图边界；右边界和上边界不包含在地图内。")
        return (int(math.floor((x - xmin) / self.cell_size)),
                int(math.floor((y - ymin) / self.cell_size)))


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须为数值。")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须为有限数值。")
    return value


def _rectangle(values, name):
    if len(values) != 4:
        raise ValueError(f"{name} 必须为 [xmin, xmax, ymin, ymax]。")
    x0, x1, y0, y1 = [_number(v, name) for v in values]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{name} 的最大坐标必须大于最小坐标。")
    return x0, x1, y0, y1


def build_grid(config):
    """建筑采用保守栅格化：格子与建筑内部有面积交集即禁止通行。"""
    xmin, xmax, ymin, ymax = _rectangle(config["bounds_m"], "bounds_m")
    size = _number(config["cell_size_m"], "cell_size_m")
    margin = _number(config.get("obstacle_margin_m", 0), "obstacle_margin_m")
    if size <= 0 or margin < 0:
        raise ValueError("栅格尺寸必须为正，障碍膨胀距离不能为负。")
    nx, ny = (xmax - xmin) / size, (ymax - ymin) / size
    if not math.isclose(nx, round(nx)) or not math.isclose(ny, round(ny)):
        raise ValueError("地图宽度和高度必须是栅格尺寸的整数倍。")
    cols, rows = int(round(nx)), int(round(ny))
    if cols < 1 or rows < 1 or cols * rows > 1000000:
        raise ValueError("地图必须至少包含一个格子，最多允许 100 万个格子。")
    xs = xmin + (np.arange(cols) + 0.5) * size
    ys = ymin + (np.arange(rows) + 0.5) * size
    xx, yy = np.meshgrid(xs, ys)
    obstacles = np.zeros((rows, cols), dtype=bool)
    for index, box in enumerate(config["buildings_xy_m"]):
        x0, x1, y0, y1 = _rectangle(box, f"建筑 {index + 1}")
        obstacles |= ((xx + size / 2 > x0 - margin)
                      & (xx - size / 2 < x1 + margin)
                      & (yy + size / 2 > y0 - margin)
                      & (yy - size / 2 < y1 + margin))

    # 本科题目直接设置高损耗栅格，不进行 HGNN 推理或射线追踪。
    normal = _number(config["normal_pathloss_db"], "normal_pathloss_db")
    threshold = _number(config["weak_threshold_db"], "weak_threshold_db")
    if not 0 <= normal < threshold:
        raise ValueError("正常损耗应非负，且必须小于弱信号阈值。")
    loss = np.full((rows, cols), normal, dtype=float)
    for index, region in enumerate(config["weak_signal_regions"]):
        x0, x1, y0, y1 = _rectangle(region["bounds_m"], f"弱信号区 {index + 1}")
        value = _number(region["pathloss_db"], "pathloss_db")
        if value < threshold:
            raise ValueError("弱信号区域的损耗必须不低于弱信号阈值。")
        mask = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        loss[mask] = np.maximum(loss[mask], value)
    weak = (loss >= threshold) & ~obstacles
    # 正常区域代价为 0；最强的人工高损耗区域代价为 1。
    scale = max(float(loss.max()) - normal, 1.0)
    cost = np.where(weak, (loss - normal) / scale, 0.0)
    return GridMap(obstacles, loss, cost, weak, size, xmin, ymin, threshold)
