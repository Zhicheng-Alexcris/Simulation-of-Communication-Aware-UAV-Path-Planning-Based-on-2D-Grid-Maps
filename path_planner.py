"""经典 A*：f(n)=g(n)+h(n)，通信因素只加入实际边代价 g。"""
from dataclasses import dataclass
import heapq
from itertools import count
import math
from time import perf_counter


class NoPathError(RuntimeError):
    """障碍物导致起点和终点之间没有可行路径。"""


@dataclass
class SearchResult:
    path: list
    objective_cost: float
    expanded_nodes: int
    elapsed_ms: float
    expanded_mask: object


def heuristic(node, goal, cell_size, connectivity):
    """四邻域用 Manhattan，八邻域用 octile；不高估剩余代价。"""
    dx, dy = abs(node[0] - goal[0]), abs(node[1] - goal[1])
    if connectivity == 4:
        return cell_size * (dx + dy)
    return cell_size * (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy))


def edge_cost(grid, current, neighbor, weight):
    distance = grid.cell_size * math.hypot(neighbor[0] - current[0],
                                          neighbor[1] - current[1])
    c0 = float(grid.communication_cost[current[1], current[0]])
    c1 = float(grid.communication_cost[neighbor[1], neighbor[0]])
    return distance * (1.0 + weight * (c0 + c1) / 2.0)


def neighbors(grid, current, connectivity):
    moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    if connectivity == 8:
        moves += [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    col, row = current
    for dc, dr in moves:
        neighbor = (col + dc, row + dr)
        if not grid.is_free(neighbor):
            continue
        # 对角移动时两侧格子都必须可通行，禁止从建筑角缝穿过。
        if dc and dr and (not grid.is_free((col + dc, row))
                          or not grid.is_free((col, row + dr))):
            continue
        yield neighbor


def astar(grid, start, goal, communication_weight=0.0, connectivity=8):
    weight = float(communication_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("通信权重必须是非负有限数值。")
    if connectivity not in (4, 8):
        raise ValueError("邻域只能选择 4 或 8。")
    for name, node in [("起点", start), ("终点", goal)]:
        if not grid.is_free(node):
            raise ValueError(f"{name} {node} 位于障碍物内或超出地图。")

    begin = perf_counter()
    expanded = grid.obstacles.copy()
    expanded[:] = False
    sequence = count()
    open_list = [(heuristic(start, goal, grid.cell_size, connectivity),
                  0.0, next(sequence), start)]
    g_score = {start: 0.0}
    parent = {}
    closed = set()

    while open_list:
        _, saved_g, _, current = heapq.heappop(open_list)
        # 一个节点可能多次进入堆；跳过旧条目和已经处理的节点。
        if current in closed or saved_g > g_score[current]:
            continue
        closed.add(current)
        expanded[current[1], current[0]] = True
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return SearchResult(path, g_score[goal], len(closed),
                                (perf_counter() - begin) * 1000, expanded)
        for neighbor in neighbors(grid, current, connectivity):
            if neighbor in closed:
                continue
            tentative = g_score[current] + edge_cost(grid, current, neighbor, weight)
            if tentative < g_score.get(neighbor, math.inf):
                parent[neighbor] = current
                g_score[neighbor] = tentative
                estimate = tentative + heuristic(neighbor, goal, grid.cell_size, connectivity)
                heapq.heappush(open_list, (estimate, tentative, next(sequence), neighbor))

    raise NoPathError("没有可行路径：请调整障碍物、起终点或栅格分辨率。")
