# path_planner.py
import numpy as np
import math
from scipy.interpolate import BSpline, PchipInterpolator
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
import heapq


# ------------------------------
# A* helpers (2D, building projection)
# ------------------------------
def _inflated_building_boxes(env, inflation=8.0):
    boxes = []
    for b in env.buildings:
        boxes.append((
            (b.x_bounds[0] - inflation, b.x_bounds[1] + inflation),
            (b.y_bounds[0] - inflation, b.y_bounds[1] + inflation)
        ))
    return boxes


def _is_xy_free(x, y, boxes):
    for (xb, yb) in boxes:
        if (xb[0] <= x <= xb[1]) and (yb[0] <= y <= yb[1]):
            return False
    return True


def _point_segment_dist_xy(p, a, b):
    p = np.asarray(p[:2], dtype=float)
    a = np.asarray(a[:2], dtype=float)
    b = np.asarray(b[:2], dtype=float)
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def astar_2d(env, start_xy, goal_xy, res=2.0, inflation=8.0,
             bounds=(-100, 100, -100, 100),
             forbidden_tubes=None, cruise_layer: int = 0,
             beta_channel: float = 0.0,
             channel_cost_fn=None):

    xs, ys = start_xy
    xg, yg = goal_xy

    if bounds is None:
        xmin, xmax = getattr(env, "xlim", (-100.0, 100.0))
        ymin, ymax = getattr(env, "ylim", (-100.0, 100.0))
    else:
        xmin, xmax, ymin, ymax = bounds

    xmin, xmax = (min(float(xmin), float(xmax)), max(float(xmin), float(xmax)))
    ymin, ymax = (min(float(ymin), float(ymax)), max(float(ymin), float(ymax)))

    boxes_inf = _inflated_building_boxes(env, inflation=inflation)
    boxes_true = _inflated_building_boxes(env, inflation=0.0)

    def to_grid(x, y):
        gx = int(round((x - xmin) / res))
        gy = int(round((y - ymin) / res))
        return gx, gy

    def to_world(gx, gy):
        x = xmin + gx * res
        y = ymin + gy * res
        return x, y

    gs = to_grid(xs, ys)
    gg = to_grid(xg, yg)

    def h(gp):
        x, y = to_world(*gp)
        return ((x - xg) ** 2 + (y - yg) ** 2) ** 0.5

    openq = []
    heapq.heappush(openq, (0.0, gs))
    came = {}
    gscore = {gs: 0.0}
    closed = set()

    moves = [(1, 0), (0, 1), (-1, 0), (0, -1),
             (1, 1), (-1, 1), (1, -1), (-1, -1)]

    def too_close_to_tubes(wx, wy):
        if not forbidden_tubes:
            return False
        for (a_xy, b_xy, r) in forbidden_tubes:
            d = _point_segment_dist_xy(
                (wx, wy, 0.0),
                (a_xy[0], a_xy[1], 0.0),
                (b_xy[0], b_xy[1], 0.0)
            )
            if d <= r:
                return True
        return False

    while openq:
        _, cur = heapq.heappop(openq)

        if cur == gg:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()

            ideal_z = env.layer_mid_z(cruise_layer)
            xy = [to_world(gx, gy) for gx, gy in path]
            poly = [(x, y, ideal_z) for x, y in xy]
            return poly

        if cur in closed:
            continue
        closed.add(cur)

        cx, cy = cur
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            wx, wy = to_world(nx, ny)

            if not (xmin <= wx <= xmax and ymin <= wy <= ymax):
                continue

            if not _is_xy_free(wx, wy, boxes_inf):
                if (nx, ny) == gs or (nx, ny) == gg:
                    if not _is_xy_free(wx, wy, boxes_true):
                        continue
                else:
                    continue

            if too_close_to_tubes(wx, wy):
                continue

            step_cost = ((dx * dx + dy * dy) ** 0.5) * res

            channel_cost = 0.0
            if beta_channel > 1e-12 and channel_cost_fn is not None:
                zc = env.layer_mid_z(cruise_layer)
                channel_cost = beta_channel * float(channel_cost_fn(wx, wy, zc))

            ng = gscore[cur] + step_cost + channel_cost

            if (nx, ny) not in gscore or ng < gscore[(nx, ny)]:
                gscore[(nx, ny)] = ng
                came[(nx, ny)] = cur
                heapq.heappush(openq, (ng + h((nx, ny)), (nx, ny)))

    ideal_z = env.layer_mid_z(cruise_layer)
    return [(start_xy[0], start_xy[1], ideal_z), (goal_xy[0], goal_xy[1], ideal_z)]


def _point_segment_dist_xy_keep_z(p, a, b):
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    pxy = p[:2]
    axy = a[:2]
    bxy = b[:2]
    ab = bxy - axy
    t = np.dot(pxy - axy, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj_xy = axy + t * ab

    d = float(np.linalg.norm(pxy - proj_xy))
    q = np.array([proj_xy[0], proj_xy[1], p[2]], dtype=float)  # ✅ keep z
    return d, q


def build_corridor_from_polyline(polyline, radius):
    caps = []
    for i in range(len(polyline) - 1):
        caps.append((tuple(polyline[i]), tuple(polyline[i + 1]), float(radius)))
    return caps


def project_point_into_corridor(pt, corridor_caps):
    p = np.asarray(pt, dtype=float)
    best = (1e18, None, None)

    for a, b, r in corridor_caps:
        d, q = _point_segment_dist_xy_keep_z(p, a, b)
        signed = d - r
        if signed < best[0]:
            best = (signed, q, r)

    signed, q, r = best
    if signed <= 0:
        return tuple(p), False

    v = p[:2] - q[:2]
    n = np.linalg.norm(v) + 1e-12
    new_xy = q[:2] + (v / n) * r
    newp = np.array([new_xy[0], new_xy[1], p[2]], dtype=float)
    return tuple(newp), True


def segment_segment_min_distance(a1, a2, b1, b2):
    p = np.asarray(a1)
    q = np.asarray(a2)
    r = np.asarray(b1)
    s = np.asarray(b2)
    u = q - p
    v = s - r
    w0 = p - r

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    D = a * c - b * b
    sc, sN, sD = 0.0, D, D
    tc, tN, tD = 0.0, D, D

    if D < 1e-12:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < 1e-12 else sN / sD
    tc = 0.0 if abs(tN) < 1e-12 else tN / tD
    dP = w0 + sc * u - tc * v
    return float(np.linalg.norm(dP))


def capsules_min_distance(seg1, r1, seg2, r2):
    a1, a2 = seg1
    b1, b2 = seg2
    d = segment_segment_min_distance(a1, a2, b1, b2)
    return d - (r1 + r2)


def batch_segment_segment_min_distance(seg_a1, seg_a2, seg_b1, seg_b2):
    """向量化批量线段-线段最短距离。

    seg_a1, seg_a2: (P, n_a, 3)  A 侧线段端点 (P 条候选路径, 每路径 n_a 段)
    seg_b1, seg_b2: (n_b, 3)      B 侧线段端点 (单条其他机路径, n_b 段)
    返回: (P, n_a, n_b) 逐对最短距离

    数学上与标量 segment_segment_min_distance 完全等价 (robust base method
    from 3D Game Engine Design / Real-Time Collision Detection)。
    """
    P, n_a, _ = seg_a1.shape
    n_b = seg_b1.shape[0]

    p = seg_a1                                     # (P, n_a, 3)
    q = seg_a2
    r = seg_b1                                     # (n_b, 3)
    s = seg_b2

    u = q - p                                      # (P, n_a, 3)
    v = s - r                                      # (n_b, 3)
    w0 = p[:, :, None, :] - r[None, None, :, :]   # (P, n_a, n_b, 3)

    a_ = np.sum(u * u, axis=2)                     # (P, n_a)
    b_ = np.sum(u[:, :, None, :] * v[None, None, :, :], axis=3)  # (P, n_a, n_b)
    c_ = np.sum(v * v, axis=1)                     # (n_b,)
    d_ = np.sum(u[:, :, None, :] * w0, axis=3)     # (P, n_a, n_b)
    e_ = np.sum(v[None, None, :, :] * w0, axis=3)  # (P, n_a, n_b)

    D = a_[:, :, None] * c_[None, None, :] - b_ * b_  # (P, n_a, n_b)

    # ---- 分支 1: 平行/近平行 vs 斜交 ----
    near_para = D < 1e-12

    sN = np.where(near_para, 0.0, b_ * e_ - c_[None, None, :] * d_)
    sD = np.where(near_para, 1.0, D)
    tN = np.where(near_para, e_, a_[:, :, None] * e_ - b_ * d_)
    tD = np.where(near_para, c_[None, None, :], D)

    # sN < 0 → clamp (仅斜交; 平行 sN=0 不受影响)
    sN_neg = (sN < 0.0) & ~near_para
    sN = np.where(sN_neg, 0.0, sN)
    tN = np.where(sN_neg, e_, tN)
    tD = np.where(sN_neg, c_[None, None, :], tD)

    # sN > sD → clamp (elif: 不重叠于上述 sN<0 条目)
    sN_hi = (sN > sD) & ~near_para
    sN = np.where(sN_hi, sD, sN)
    tN = np.where(sN_hi, e_ + b_, tN)
    tD = np.where(sN_hi, c_[None, None, :], tD)

    # ---- 分支 2: tN 修正 (对所有条目) ----
    tN_neg = tN < 0.0

    neg_d = -d_
    mask_lt0 = neg_d < 0.0
    mask_gt_a = neg_d > a_[:, :, None]
    sN_tn = np.where(mask_lt0, 0.0, np.where(mask_gt_a, sD, neg_d))
    sD_tn = np.where(~mask_lt0 & ~mask_gt_a, a_[:, :, None], sD)

    tN = np.where(tN_neg, 0.0, tN)
    sN = np.where(tN_neg, sN_tn, sN)
    sD = np.where(tN_neg, sD_tn, sD)

    # tN > tD (elif)
    tN_hi = ~tN_neg & (tN > tD)
    neg_dpb = -d_ + b_
    maskb_lt0 = neg_dpb < 0.0
    maskb_gt_a = neg_dpb > a_[:, :, None]
    sN1_tn = np.where(maskb_lt0, 0.0, np.where(maskb_gt_a, sD, neg_dpb))
    sD1_tn = np.where(~maskb_lt0 & ~maskb_gt_a, a_[:, :, None], sD)

    tN = np.where(tN_hi, tD, tN)
    sN = np.where(tN_hi, sN1_tn, sN)
    sD = np.where(tN_hi, sD1_tn, sD)

    # ---- 最终: sc = sN/sD, tc = tN/tD ----
    sc = np.where(np.abs(sN) < 1e-12, 0.0, sN / (sD + 1e-15))
    tc = np.where(np.abs(tN) < 1e-12, 0.0, tN / (tD + 1e-15))

    dP = w0 + sc[:, :, :, None] * u[:, :, None, :] - tc[:, :, :, None] * v[None, None, :, :]

    return np.linalg.norm(dP, axis=3)  # (P, n_a, n_b)


class SplinePathPlanner:
    def __init__(self, environment, degree=3, n_control_points=10, pipe_radius=2.0):
        self.env = environment
        self.degree = degree
        self.n_control_points = n_control_points

        self.other_drone_paths = []
        self.corridor_caps = None
        self.pipe_radius = float(pipe_radius)
        self.cruise_layer_idx = 0

        self.channel_cost_fn = None
        self.channel_weight = 0.0

    def set_channel_cost(self, channel_cost_fn, weight=1200.0, vec_fn=None):
        self.channel_cost_fn = channel_cost_fn
        self.channel_weight = float(weight)
        self.channel_cost_fn_vec = vec_fn

    def set_cruise_layer(self, layer_idx: int):
        self.cruise_layer_idx = int(layer_idx)

    def set_corridor(self, coarse_polyline, corridor_radius=6.0):
        self.corridor_caps = build_corridor_from_polyline(coarse_polyline, corridor_radius)
        if len(coarse_polyline) >= 2:
            p0, p1 = np.array(coarse_polyline[0]), np.array(coarse_polyline[1])
            q0, q1 = np.array(coarse_polyline[-2]), np.array(coarse_polyline[-1])
            d0 = p1 - p0
            d1 = q1 - q0
            self._endpoint_dir_start = d0 / (np.linalg.norm(d0) + 1e-12)
            self._endpoint_dir_end = d1 / (np.linalg.norm(d1) + 1e-12)
            self._endpoint_anchor_dist = 12.0

    def _enforce_corridor_on_control_points(self, cps):
        if not self.corridor_caps:
            return cps
        cps = [tuple(float(c) for c in p) for p in cps]
        n = len(cps)
        if n <= 2:
            return cps

        # 待投影点的索引：内部控制点（端点固定）。端点方向锚点（q1/qm）追加并写回相邻索引。
        proj_idx = [i for i in range(n) if i not in (0, n - 1)]
        extra = []  # (target_index, raw_point)
        zmin = float(self.env.min_height)
        layer_idx = self.cruise_layer_idx
        if hasattr(self, "_endpoint_dir_start") and n >= 4:
            sx, sy, sz = cps[0]
            d = self._endpoint_dir_start; L = self._endpoint_anchor_dist
            p1 = (sx + d[0] * L, sy + d[1] * L, self.env.clamp_to_layer(sz, layer_idx))
            extra.append((1, p1))
        if hasattr(self, "_endpoint_dir_end") and n >= 4:
            ex, ey, ez = cps[-1]
            d = self._endpoint_dir_end; L = self._endpoint_anchor_dist
            pm = (ex - d[0] * L, ey - d[1] * L, self.env.clamp_to_layer(ez, layer_idx))
            extra.append((n - 2, pm))

        new_cps = [list(p) for p in cps]
        src = [np.asarray(cps[i], dtype=float) for i in proj_idx]
        for _, p in extra:
            src.append(np.asarray(p, dtype=float))
        if src:
            # 单次批量投影（xy 投影、z 不变），替代逐点标量循环
            proj = self._project_points_into_corridor(np.stack(src, axis=0))
            for k, i in enumerate(proj_idx):
                x, y = float(proj[k, 0]), float(proj[k, 1])
                z = self.env.clamp_to_layer(max(float(cps[i][2]), zmin), layer_idx)
                new_cps[i] = [x, y, z]
            for m, (ti, p) in enumerate(extra):
                k = len(proj_idx) + m
                x, y = float(proj[k, 0]), float(proj[k, 1])
                z = self.env.clamp_to_layer(max(float(p[2]), zmin), layer_idx)
                new_cps[ti] = [x, y, z]
        return [tuple(p) for p in new_cps]

    def set_other_drone_paths(self, paths):
        if paths is None:
            self.other_drone_paths = []
        else:
            self.other_drone_paths = list(paths)

    def generate_control_points_bounds(self, start, end):
        def _ord(a, b):
            a = float(a)
            b = float(b)
            return (min(a, b), max(a, b))

        x_scene = _ord(*getattr(self.env, "xlim", (-110.0, 110.0)))
        y_scene = _ord(*getattr(self.env, "ylim", (-110.0, 110.0)))
        z_scene = _ord(*getattr(self.env, "zlim", (0.0, 40.0)))

        pad_xy = 50.0
        x_min = min(float(start[0]), float(end[0])) - pad_xy
        x_max = max(float(start[0]), float(end[0])) + pad_xy
        y_min = min(float(start[1]), float(end[1])) - pad_xy
        y_max = max(float(start[1]), float(end[1])) + pad_xy

        x_min = max(x_min, x_scene[0])
        x_max = min(x_max, x_scene[1])
        y_min = max(y_min, y_scene[0])
        y_max = min(y_max, y_scene[1])

        if x_max < x_min:
            x_min, x_max = x_scene
        if y_max < y_min:
            y_min, y_max = y_scene

        eps = 0.2
        if hasattr(self.env, "safe_layers") and len(self.env.safe_layers) > self.cruise_layer_idx:
            L = self.env.safe_layers[self.cruise_layer_idx]
            Lz = _ord(getattr(L, "z_min", z_scene[0]), getattr(L, "z_max", z_scene[1]))
            z_min = max(float(getattr(self.env, "min_height", 0.0)), z_scene[0] + eps, Lz[0] + eps)
            z_max = min(z_scene[1] - eps, Lz[1] - eps)
        else:
            z_min = max(float(getattr(self.env, "min_height", 0.0)), z_scene[0] + eps)
            z_max = z_scene[1] - eps

        if z_max < z_min:
            mid = 0.5 * (z_scene[0] + z_scene[1])
            mid = max(mid, float(getattr(self.env, "min_height", 0.0)) + 0.1)
            z_min = mid - 0.1
            z_max = mid + 0.1

        bounds = {}
        for i in range(self.n_control_points):
            bounds[f'x{i}'] = (float(x_min), float(x_max))
            bounds[f'y{i}'] = (float(y_min), float(y_max))
            bounds[f'z{i}'] = (float(z_min), float(z_max))
        return bounds

    def create_b_spline(self, control_points, n_samples=100):
        control_points = np.array(control_points, dtype=float)
        n = len(control_points)
        k = self.degree

        t = np.linspace(0, 1, n - k + 1)
        t = np.append([0] * k, t)
        t = np.append(t, [1] * k)

        spline = BSpline(t, control_points, k, extrapolate=False)
        u = np.linspace(0, 1, n_samples)
        path = spline(u)
        return path

    # ---- 向量化批量代价评估（Diffusion-ES 加速核心）----
    def _building_bounds_arrays(self):
        """预计算建筑 AABB：(raw, expanded) 两个 (B,6) 数组，B=建筑数。

        raw 用净空（到建筑表面距离），exp 用 safety_margin=pipe_radius 膨胀，
        供碰撞检测一次性向量化比较，避免原始 evaluate_path 里逐建筑 Python 循环。
        """
        blds = self.env.buildings
        n = len(blds)
        if n == 0:
            return np.zeros((0, 6), dtype=float), np.zeros((0, 6), dtype=float)
        xs = np.array([b.x_bounds for b in blds], dtype=float)
        ys = np.array([b.y_bounds for b in blds], dtype=float)
        zs = np.array([b.z_bounds for b in blds], dtype=float)
        raw = np.concatenate([xs, ys, zs], axis=1)
        m = float(self.pipe_radius)
        exp = raw.copy()
        exp[:, 0] -= m; exp[:, 1] += m
        exp[:, 2] -= m; exp[:, 3] += m
        exp[:, 4] -= m; exp[:, 5] += m
        return raw, exp

    def evaluate_paths(self, paths, start, end, skip_expensive_if_infeasible=False):
        """批量代价评估：把整批轨迹 (P,S,3) 一次性算完，collision/clearance 向量化。

        返回 (lengths, penalties, min_clearances, collision_counts)，均为 (P,) 数组。
        语义与单条 evaluate_path 完全一致：碰撞计数 = 落在任一(膨胀)AABB 内的采样点数；
        净空 = 到任一建筑表面的最小距离。区别：去掉了逐建筑 Python 循环，且对碰撞个体
        可短路掉昂贵的曲率/回退/单调/信道/多机代价（skip_expensive_if_infeasible）。

        paths: list of (S,3) array-like；start/end: 端点。
        """
        paths = [np.asarray(p, dtype=float) for p in paths]
        P = len(paths)
        if P == 0:
            return (np.array([]), np.array([]), np.array([]), np.array([]))
        S = min(len(p) for p in paths)
        pts = np.stack(
            [p[:S] if len(p) >= S else np.pad(p, ((0, S - len(p)), (0, 0)))
             for p in paths], axis=0
        ).astype(float)
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        # 长度
        seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
        lengths = seg.sum(axis=1)

        raw, exp = self._building_bounds_arrays()
        B = raw.shape[0]
        collision_counts = np.zeros(P, dtype=float)
        min_clearance = np.full(P, np.inf, dtype=float)
        if B > 0:
            # 碰撞：与 check_collisions 同粒度（max(8, seg_len/4) 点/段），取各段最大值统一采样，
            # 保证密度不低于 check_collisions —— 从而 ncol==0 必为真正无碰撞（保守）。
            a = pts[:, :-1, :]
            b = pts[:, 1:, :]
            seg_len = np.linalg.norm(b - a, axis=2)               # (P, S-1)
            sub = int(np.ceil(seg_len.max() / 4.0))
            sub = max(sub, 8)
            tt = np.linspace(0.0, 1.0, sub)
            samples = a[:, :, None, :] + tt[None, None, :, None] * (b - a)[:, :, None, :]
            samples = samples.reshape(P, (S - 1) * sub, 3)
            samples[:, :, 2] = np.maximum(samples[:, :, 2], self.env.min_height)
            x = samples[:, :, 0:1]; y = samples[:, :, 1:2]; z = samples[:, :, 2:3]
            inside = ((exp[:, 0] <= x) & (x <= exp[:, 1]) &
                      (exp[:, 2] <= y) & (y <= exp[:, 3]) &
                      (exp[:, 4] <= z) & (z <= exp[:, 5]))   # (P,Ns,B)
            collision_counts = inside.any(axis=2).sum(axis=1).astype(float)
            # 净空：到建筑表面最小距离（逐点 × 建筑）
            xp = pts[:, :, 0:1]; yp = pts[:, :, 1:2]; zp = pts[:, :, 2:3]
            dx = np.maximum(np.maximum(raw[:, 0] - xp, 0.0), xp - raw[:, 1])
            dy = np.maximum(np.maximum(raw[:, 2] - yp, 0.0), yp - raw[:, 3])
            dz = np.maximum(np.maximum(raw[:, 4] - zp, 0.0), zp - raw[:, 5])
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)        # (P,S,B)
            min_clearance = dist.min(axis=2).min(axis=1)

        collision_penalty = collision_counts * 20000.0

        # 结构项（向量化跨 P 批量计算；信道为黑盒逐点调用，仅对可行个体求值）
        ideal_height = float(self.env.layer_mid_z(self.cruise_layer_idx))
        height_penalty = np.abs(np.mean(pts[:, :, 2], axis=1) - ideal_height) * 10.0

        curvature_penalty = np.zeros(P, dtype=float)
        backtracking_penalty = np.zeros(P, dtype=float)
        monotone_penalty = np.zeros(P, dtype=float)
        other_path_penalty = np.zeros(P, dtype=float)
        channel_penalty = np.zeros(P, dtype=float)

        v = end[:2] - start[:2]
        nv = np.linalg.norm(v) + 1e-12
        vh = v / nv

        # ---- 曲率（向量化）----
        # 对每条路径 i=1..S-2 计算等价曲率半径惩罚，端点 20m 内加权 w_end。
        v1 = pts[:, 1:-1, :] - pts[:, :-2, :]                 # (P, S-2, 3)
        v2 = pts[:, 2:, :] - pts[:, 1:-1, :]                  # (P, S-2, 3)
        n1 = np.linalg.norm(v1, axis=2)                      # (P, S-2)
        n2 = np.linalg.norm(v2, axis=2)
        dot = np.sum(v1 * v2, axis=2)
        cosang = np.clip(dot / ((n1 + 1e-12) * (n2 + 1e-12)), -1.0, 1.0)
        ang = np.arccos(cosang)
        r_equiv = (n1 + n2) / (2.0 * (ang + 1e-6))
        base = np.clip(18.0 - r_equiv, 0.0, None)
        acc_s = np.cumsum(n1, axis=1)                        # (P, S-2) 累积弧长
        rem = lengths[:, None] - acc_s
        near_end = (acc_s < 20.0) | (rem < 20.0)
        w = np.where(near_end, 3.0, 1.0)
        curvature_penalty = np.sum(w * base * 8.0, axis=1)

        # ---- 回退（向量化）----
        dxy = np.diff(pts[:, :, :2], axis=1)                 # (P, S-1, 2)
        seg_xy = np.linalg.norm(dxy, axis=2) + 1e-12
        dxy_u = dxy / seg_xy[:, :, None]
        dots = dxy_u @ vh
        back = np.clip(-dots, 0.0, 1.0)
        backtracking_penalty = np.sum(back * seg_xy, axis=1) * 80.0

        # ---- 单调（向量化）----
        prog = (pts[:, :, :2] - start[:2]) @ vh              # (P, S)
        dprog = np.diff(prog, axis=1)
        monotone_penalty = np.sum(np.clip(-dprog, 0.0, None), axis=1) * 120.0

        # 不可行个体：跳过昂贵项（曲率/回退/单调已便宜，置零即可，避免计入排名）
        if skip_expensive_if_infeasible:
            keep = collision_counts == 0
            curvature_penalty = np.where(keep, curvature_penalty, 0.0)
            backtracking_penalty = np.where(keep, backtracking_penalty, 0.0)
            monotone_penalty = np.where(keep, monotone_penalty, 0.0)

        # ---- 多机避让（向量化：批量线段-线段胶囊距离）----
        if self.other_drone_paths:
            pipe_r = self.pipe_radius
            safe_margin = max(5.0, 1.5 * pipe_r)
            step_i = 2; step_j = 3
            seg_idx_a = list(range(0, S - 1, step_i))
            if seg_idx_a:
                seg_a1 = pts[:, seg_idx_a, :]            # (P, n_a, 3)
                seg_a2 = pts[:, [i + 1 for i in seg_idx_a], :]
                for other_path in self.other_drone_paths:
                    if other_path is None:
                        continue
                    other = np.asarray(other_path, dtype=float)
                    n_o = len(other)
                    if n_o < 2:
                        continue
                    seg_idx_b = list(range(0, n_o - 1, step_j))
                    if not seg_idx_b:
                        continue
                    seg_b1 = other[seg_idx_b, :]         # (n_b, 3)
                    seg_b2 = other[[i + 1 for i in seg_idx_b], :]
                    dist = batch_segment_segment_min_distance(
                        seg_a1, seg_a2, seg_b1, seg_b2)  # (P, n_a, n_b)
                    gap = dist - 2.0 * pipe_r             # 胶囊净距
                    pen = np.where(
                        gap < 0,
                        -gap * 500.0,
                        np.where(gap < safe_margin, (safe_margin - gap) * 50.0, 0.0)
                    )                                     # (P, n_a, n_b)
                    other_path_penalty += pen.sum(axis=(1, 2))
            # 不可行个体不计多机项
            if skip_expensive_if_infeasible:
                keep = collision_counts == 0
                other_path_penalty = np.where(keep, other_path_penalty, 0.0)

        # ---- 信道（向量化批量：对所有“需评估”个体的采样点一次性求值）----
        if self.channel_cost_fn is not None and self.channel_weight > 1e-12:
            if skip_expensive_if_infeasible:
                eval_mask = collision_counts == 0
            else:
                eval_mask = np.ones(P, dtype=bool)
            if np.any(eval_mask):
                sample_n = min(max(150, S // 2), S)
                idx = np.linspace(0, S - 1, sample_n).astype(int)
                samp = pts[eval_mask][:, idx, :]              # (F, sample_n, 3)
                F = samp.shape[0]
                flat = samp.reshape(F * sample_n, 3)          # (F*sample_n, 3)
                if self.channel_cost_fn_vec is not None:
                    cvals = np.asarray(self.channel_cost_fn_vec(flat), dtype=float)
                else:
                    cvals = np.array([float(self.channel_cost_fn(float(p[0]), float(p[1]), float(p[2])))
                                      for p in flat], dtype=float)
                avg_c = cvals.reshape(F, sample_n).mean(axis=1)   # (F,)
                lens = lengths[eval_mask]
                channel_penalty[eval_mask] = (
                    self.channel_weight * avg_c * np.maximum(1.0, lens))

        penalty = (collision_penalty + height_penalty + curvature_penalty
                   + backtracking_penalty + monotone_penalty
                   + other_path_penalty + channel_penalty)
        return lengths, penalty, min_clearance, collision_counts

    def evaluate_path(self, path, start, end):
        path = np.asarray(path, dtype=float)

        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        length = float(np.sum(seg))

        sample_n = min(max(150, len(path) // 2), len(path))
        idx = np.linspace(0, len(path) - 1, sample_n).astype(int)

        safety_margin = float(self.pipe_radius)

        # 碰撞/净空检测沿“路径段内插采样”，与 check_collisions 完全一致，
        # 避免只在顶点检测而漏掉顶点之间穿障的情况。
        collision_count = 0
        _seg_samples = 6
        path_arr = np.asarray(path, dtype=float)
        for i in range(len(path_arr) - 1):
            a = path_arr[i]
            b = path_arr[i + 1]
            for tt in np.linspace(0.0, 1.0, _seg_samples):
                x = a[0] + tt * (b[0] - a[0])
                y = a[1] + tt * (b[1] - a[1])
                z = max(a[2] + tt * (b[2] - a[2]), self.env.min_height)
                for bld in self.env.buildings:
                    if bld.contains_with_margin((x, y, z), safety_margin):
                        collision_count += 1
                        break
        collision_penalty = collision_count * 20000

        min_clearance = float("inf")
        for k in range(len(path)):
            point = path[k]
            for building in self.env.buildings:
                dx = max(building.x_bounds[0] - point[0], 0, point[0] - building.x_bounds[1])
                dy = max(building.y_bounds[0] - point[1], 0, point[1] - building.y_bounds[1])
                dz = max(building.z_bounds[0] - point[2], 0, point[2] - building.z_bounds[1])
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < min_clearance:
                    min_clearance = dist

        ideal_height = self.env.layer_mid_z(self.cruise_layer_idx)
        avg_height = float(np.mean(path[:, 2]))
        height_penalty = abs(avg_height - ideal_height) * 10

        curvature_penalty = 0.0
        R_soft = 18.0
        w_end = 3.0
        acc_s = 0.0
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i - 1]
            v2 = path[i + 1] - path[i]
            n1 = np.linalg.norm(v1) + 1e-12
            n2 = np.linalg.norm(v2) + 1e-12
            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            ang = math.acos(cosang)
            r_equiv = (n1 + n2) / (2.0 * (ang + 1e-6))
            base = max(0.0, (R_soft - r_equiv))

            acc_s += n1
            near_end = (acc_s < 20.0) or ((length - acc_s) < 20.0)
            w = (w_end if near_end else 1.0)
            curvature_penalty += w * base * 8.0

        v = np.asarray(end[:2], dtype=float) - np.asarray(start[:2], dtype=float)
        nv = np.linalg.norm(v) + 1e-12
        vh = v / nv

        dxy = np.diff(path[:, :2], axis=0)
        seg_xy = np.linalg.norm(dxy, axis=1) + 1e-12
        dxy_u = dxy / seg_xy[:, None]
        dots = dxy_u @ vh
        back = np.clip(-dots, 0.0, 1.0)
        backtracking_penalty = float(np.sum(back * seg_xy)) * 80.0

        prog = (path[:, :2] - np.asarray(start[:2], dtype=float)) @ vh
        dprog = np.diff(prog)
        monotone_penalty = float(np.sum(np.clip(-dprog, 0.0, None))) * 120.0

        other_path_penalty = 0.0
        if self.other_drone_paths:
            pipe_r = self.pipe_radius
            safe_margin = max(5.0, 1.5 * pipe_r)
            step_i = 2
            step_j = 3
            S = len(path)
            seg_idx_a = list(range(0, S - 1, step_i))
            if seg_idx_a:
                seg_a1 = path[seg_idx_a, :][None, :, :]        # (1, n_a, 3)
                seg_a2 = path[[i + 1 for i in seg_idx_a], :][None, :, :]
                for other_path in self.other_drone_paths:
                    if other_path is None:
                        continue
                    other = np.asarray(other_path, dtype=float)
                    n_o = len(other)
                    if n_o < 2:
                        continue
                    seg_idx_b = list(range(0, n_o - 1, step_j))
                    if not seg_idx_b:
                        continue
                    seg_b1 = other[seg_idx_b, :]              # (n_b, 3)
                    seg_b2 = other[[i + 1 for i in seg_idx_b], :]
                    dist = batch_segment_segment_min_distance(
                        seg_a1, seg_a2, seg_b1, seg_b2)       # (1, n_a, n_b)
                    gap = dist - 2.0 * pipe_r
                    pen = np.where(
                        gap < 0, -gap * 500.0,
                        np.where(gap < safe_margin, (safe_margin - gap) * 50.0, 0.0)
                    )
                    other_path_penalty += float(pen.sum())

        channel_penalty = 0.0
        if self.channel_cost_fn is not None and self.channel_weight > 1e-12:
            if self.channel_cost_fn_vec is not None:
                samp = path[idx]                                    # (sample_n, 3)
                cvals = np.asarray(self.channel_cost_fn_vec(samp), dtype=float)
                avg_c = float(np.mean(cvals))
            else:
                cs = [float(self.channel_cost_fn(float(path[k, 0]), float(path[k, 1]), float(path[k, 2])))
                      for k in idx]
                avg_c = float(np.mean(cs))  # 0..1
            channel_penalty = self.channel_weight * avg_c * max(1.0, length)

        penalty = (collision_penalty
                   + height_penalty
                   + curvature_penalty
                   + backtracking_penalty
                   + monotone_penalty
                   + other_path_penalty
                   + channel_penalty)

        return length, penalty, min_clearance

    def _check_segment_collision(self, start, end, buildings, other_paths=None,
                                 segment_samples=20, safety_margin=2.0):
        if other_paths is None:
            other_paths = self.other_drone_paths

        collision_points = []

        for t in np.linspace(0, 1, segment_samples):
            interp_pos = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                max(start[2] + t * (end[2] - start[2]), self.env.min_height)
            )
            for building in buildings:
                if building.contains_with_margin(interp_pos, safety_margin):
                    collision_points.append(interp_pos)
                    break

        return collision_points

    def check_collisions(self, trajectory, buildings, safety_margin=2.0):
        collision_points = []
        for i in range(len(trajectory) - 1):
            start = trajectory[i]
            end = trajectory[i + 1]
            segment_length = self.env.calculate_distance(start, end)
            samples = max(8, int(segment_length / 4.0))
            segment_collisions = self._check_segment_collision(
                start, end, buildings,
                other_paths=None,
                segment_samples=samples,
                safety_margin=safety_margin
            )
            collision_points.extend(segment_collisions)
        return collision_points

    def _has_collision_between(self, p1, p2, steps=20):
        collisions = self._check_segment_collision(
            p1, p2, self.env.buildings,
            other_paths=None,
            segment_samples=steps,
            safety_margin=float(self.pipe_radius)
        )
        return len(collisions) > 0

    def project_path_into_corridor(self, traj):
        if not self.corridor_caps:
            return traj
        traj = np.asarray(traj, dtype=float)
        proj = self._project_points_into_corridor(traj)
        zc = np.maximum(proj[:, 2], float(self.env.min_height))
        zc = np.array([self.env.clamp_to_layer(z, self.cruise_layer_idx) for z in zc])
        out = np.empty_like(proj)
        out[:, 0] = proj[:, 0]
        out[:, 1] = proj[:, 1]
        out[:, 2] = zc
        return [tuple(p) for p in out]

    def _project_points_into_corridor(self, pts):
        """向量化把 (N,3) 点投影进走廊（逐胶囊取最小 signed 距离），xy 投影、z 不变。

        与 project_point_into_corridor 语义一致：找到 signed=d-r 最小的胶囊，
        signed<=0（已在走廊内）则不移动，否则沿径向投影到半径 r 的圆柱面。
        """
        if not self.corridor_caps:
            return np.asarray(pts, dtype=float)
        pts = np.asarray(pts, dtype=float)
        caps = self.corridor_caps
        axy = np.array([c[0][:2] for c in caps], dtype=float)
        bxy = np.array([c[1][:2] for c in caps], dtype=float)
        r = np.array([c[2] for c in caps], dtype=float)
        pxy = pts[:, :2]
        ab = bxy - axy
        ab2 = np.sum(ab * ab, axis=1) + 1e-12
        t = np.clip(((pxy[:, None, :] - axy[None, :, :]) * ab[None, :, :]).sum(axis=2) / ab2[None, :],
                    0.0, 1.0)
        proj_xy = axy[None, :, :] + t[:, :, None] * ab[None, :, :]
        d_xy = np.linalg.norm(pxy[:, None, :] - proj_xy, axis=2)
        signed = d_xy - r[None, :]
        best = np.argmin(signed, axis=1)
        signed_min = signed[np.arange(pts.shape[0]), best]
        proj_best = proj_xy[np.arange(pts.shape[0]), best]
        out = pts.copy()
        move = signed_min > 0
        if np.any(move):
            idx = np.where(move)[0]
            q = proj_best[idx]
            v = pxy[idx] - q
            n = np.linalg.norm(v, axis=1) + 1e-12
            new_xy = q + (v / n[:, None]) * r[best[idx]][:, None]
            out[idx, 0] = new_xy[:, 0]
            out[idx, 1] = new_xy[:, 1]
        return out

    def project_paths_into_corridor(self, trajs):
        """批量走廊投影：list of (S,3) -> list of (S,3) 数组。z 处理与逐条版一致
        （max(p[2], min_height) 再 clamp_to_layer）。"""
        out = []
        min_height = float(self.env.min_height)
        layer_idx = self.cruise_layer_idx
        for t in trajs:
            pts = self._project_points_into_corridor(np.asarray(t, dtype=float))
            zc = np.maximum(pts[:, 2], min_height)
            zc = np.clip(zc, self.env.safe_layers[layer_idx].z_min,
                         self.env.safe_layers[layer_idx].z_max)
            pts[:, 2] = zc
            out.append(pts)
        return out

    def smooth_path(self, path, smoothing_factor=0.3, iters=3,
                    keep_endpoints=True, clamp_to_corridor=True):
        if len(path) < 3:
            return path

        pts = np.array(path, dtype=float)
        for _ in range(max(1, iters)):
            prev = pts.copy()
            for i in range(1, len(pts) - 1):
                pts[i] = (1 - smoothing_factor) * pts[i] + smoothing_factor * 0.5 * (prev[i - 1] + prev[i + 1])

            pts[:, 2] = np.maximum(pts[:, 2], self.env.min_height)

            if clamp_to_corridor and self.corridor_caps:
                pts = np.array(self.project_path_into_corridor(pts), dtype=float)

            if keep_endpoints:
                pts[0] = path[0]
                pts[-1] = path[-1]

        return pts.tolist()

    def smooth_path_with_bspline(self, trajectory, num_points=200,
                                 clamp_to_corridor=True, keep_endpoints=True):
        traj = np.asarray(trajectory, dtype=float)
        if len(traj) < 4:
            out = traj.copy()
            if clamp_to_corridor and self.corridor_caps:
                out = np.asarray(self.project_path_into_corridor(out), dtype=float)
            out[:, 2] = np.maximum(out[:, 2], self.env.min_height)
            if keep_endpoints:
                out[0] = trajectory[0]
                out[-1] = trajectory[-1]
            return out.tolist()

        clean = [traj[0]]
        for p in traj[1:]:
            if np.linalg.norm(p - clean[-1]) > 1e-6:
                clean.append(p)
        traj = np.asarray(clean, dtype=float)
        if len(traj) < 4:
            return traj.tolist()

        seg = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-9:
            return traj.tolist()
        t = s / s[-1]
        t_new = np.linspace(0, 1, num_points)

        fx = PchipInterpolator(t, traj[:, 0])
        fy = PchipInterpolator(t, traj[:, 1])
        fz = PchipInterpolator(t, traj[:, 2])
        out = np.column_stack([fx(t_new), fy(t_new), fz(t_new)])

        out[:, 2] = np.maximum(out[:, 2], self.env.min_height)

        if clamp_to_corridor and self.corridor_caps:
            out = np.asarray(self.project_path_into_corridor(out), dtype=float)

        if keep_endpoints:
            out[0] = trajectory[0]
            out[-1] = trajectory[-1]

        return out.tolist()

    def analyze_path(self, trajectory, buildings, safety_margin=None):
        if safety_margin is None:
            safety_margin = float(self.pipe_radius)

        min_distances = []
        for p in trajectory:
            distances = []
            for b in buildings:
                dx = max(b.x_bounds[0] - p[0], 0, p[0] - b.x_bounds[1])
                dy = max(b.y_bounds[0] - p[1], 0, p[1] - b.y_bounds[1])
                dz = max(b.z_bounds[0] - p[2], 0, p[2] - b.z_bounds[1])
                distances.append(math.sqrt(dx * dx + dy * dy + dz * dz))
            min_distances.append(min(distances))

        return {
            'avg_clearance': float(np.mean(min_distances)),
            'min_clearance': float(min(min_distances)),
            'collision_count': len(self.check_collisions(trajectory, buildings, safety_margin=safety_margin))
        }

    def _repair_trajectory(self, trajectory):
        if len(trajectory) < 2:
            return trajectory

        new_path = [trajectory[0]]
        for i in range(1, len(trajectory)):
            prev = new_path[-1]
            curr = trajectory[i]

            if not self._has_collision_between(prev, curr):
                new_path.append(curr)
                continue

            L = self.env.safe_layers[self.cruise_layer_idx]
            z_top = getattr(self.env, "zlim", (0.0, 40.0))[1] - 0.2
            safe_height = min(z_top, max(float(L.z_min), float(L.z_max)) - 0.2)

            bypass_points = [
                (prev[0], prev[1], safe_height),
                (curr[0], curr[1], safe_height),
                curr
            ]

            valid_bypass = [p for p in bypass_points
                            if not any(b.contains_with_margin(p, safety_margin=float(self.pipe_radius)) for b in self.env.buildings)]

            if len(valid_bypass) >= 2:
                new_path.extend(valid_bypass)
            else:
                new_path.append((curr[0], curr[1], safe_height))

        return new_path

    def select_best_solution(self, F):
        F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-10)
        weights = [0.3, 0.7]
        scores = F_norm @ np.array(weights)
        return int(np.argmin(scores))

    def optimize_path(self, drone_id, population_size=50, n_generations=100,
                      start=None, end=None, seed=None, verbose=False,
                      pareto_dir=None):
        if start is None or end is None:
            start, end = self.env.get_start_end(drone_id)

        class PathOptimizationProblem(Problem):
            def __init__(self, planner, start, end):
                self.planner = planner
                self.start = start
                self.end = end
                bounds = planner.generate_control_points_bounds(start, end)

                xl, xu = [], []
                for i in range(planner.n_control_points):
                    xl.extend([bounds[f'x{i}'][0], bounds[f'y{i}'][0], bounds[f'z{i}'][0]])
                    xu.extend([bounds[f'x{i}'][1], bounds[f'y{i}'][1], bounds[f'z{i}'][1]])

                super().__init__(
                    n_var=planner.n_control_points * 3,
                    n_obj=2,
                    n_constr=2,
                    xl=np.array(xl, dtype=float),
                    xu=np.array(xu, dtype=float)
                )

                bad = np.where(self.xu < self.xl)[0]
                if bad.size > 0:
                    k = int(bad[0])
                    raise ValueError(f"Invalid bounds at var[{k}]: xl={self.xl[k]}, xu={self.xu[k]}")

            def _evaluate(self, X, out, *args, **kwargs):
                F = []
                G = []
                r_safe = float(self.planner.pipe_radius)

                for i in range(len(X)):
                    control_points = []
                    for j in range(self.planner.n_control_points):
                        idx = j * 3
                        control_points.append([X[i, idx], X[i, idx + 1], X[i, idx + 2]])

                    control_points[0] = list(self.start)
                    control_points[-1] = list(self.end)

                    control_points = self.planner._enforce_corridor_on_control_points(control_points)

                    path = self.planner.create_b_spline(control_points)
                    path = self.planner.project_path_into_corridor(path.tolist())
                    path = np.asarray(path, dtype=float)

                    length, penalty, min_clearance = self.planner.evaluate_path(path, self.start, self.end)

                    collision_points = self.planner.check_collisions(
                        path.tolist(), self.planner.env.buildings, safety_margin=float(self.planner.pipe_radius)
                    )
                    has_collision = (len(collision_points) > 0)

                    F.append([length, penalty])

                    g1 = 1.0 if has_collision else 0.0
                    g2 = (r_safe - min_clearance)
                    G.append([g1, g2])

                out["F"] = np.array(F, dtype=float)
                out["G"] = np.array(G, dtype=float)

        problem = PathOptimizationProblem(self, start, end)

        algorithm = NSGA2(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", n_generations)

        res = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=verbose
        )

        if pareto_dir is not None:
            try:
                import os
                os.makedirs(pareto_dir, exist_ok=True)

                pop = res.pop
                F_all = pop.get("F")  # (pop,2) -> [length, penalty]
                CV_all = pop.get("CV")
                CV_all = np.asarray(CV_all).reshape(-1, 1)

                seed_tag = "None" if seed is None else str(seed)

                # (A) 原始 NSGA 目标：length vs penalty
                raw_path = os.path.join(pareto_dir, f"pop_F_drone{drone_id}_seed{seed_tag}.csv")
                np.savetxt(
                    raw_path,
                    np.hstack([np.asarray(F_all, dtype=float), CV_all]),
                    delimiter=",",
                    header="obj1_length,obj2_penalty,CV",
                    comments=""
                )
                print(f"[Pareto] Saved: {raw_path}")

                # (B) length vs comm_cost（0..1），更适合写“Pareto 航程-通信”那张图
                X_all = pop.get("X")
                rows = []

                if X_all is not None:
                    X_all = np.asarray(X_all, dtype=float)

                    for i in range(len(X_all)):
                        xvec = X_all[i]

                        cps = []
                        for j in range(self.n_control_points):
                            idx = j * 3
                            cps.append([xvec[idx], xvec[idx + 1], xvec[idx + 2]])
                        cps[0] = list(start)
                        cps[-1] = list(end)

                        cps = self._enforce_corridor_on_control_points(cps)
                        path = self.create_b_spline(cps)
                        path = self.project_path_into_corridor(path.tolist())
                        path = np.asarray(path, dtype=float)

                        # length
                        if len(path) >= 2:
                            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
                            length = float(np.sum(seg))
                        else:
                            length = 0.0

                        if self.channel_cost_fn is not None and len(path) > 0:
                            sample_n = min(120, len(path))
                            idxs = np.linspace(0, len(path) - 1, sample_n).astype(int)
                            cs = [
                                float(self.channel_cost_fn(
                                    float(path[k, 0]), float(path[k, 1]), float(path[k, 2])
                                ))
                                for k in idxs
                            ]
                            comm = float(np.mean(cs))
                        else:
                            comm = float("nan")

                        penalty = float(F_all[i, 1])
                        cv = float(CV_all[i, 0])
                        rows.append([length, comm, penalty, cv])

                    len_comm_path = os.path.join(pareto_dir, f"pop_len_comm_drone{drone_id}_seed{seed_tag}.csv")
                    np.savetxt(
                        len_comm_path,
                        np.array(rows, dtype=float),
                        delimiter=",",
                        header="length_m,comm_cost_0to1,penalty,CV",
                        comments=""
                    )
                    print(f"[Pareto] Saved: {len_comm_path}")

            except Exception as e:
                print(f"[Pareto] dump failed: {e}")

        if res.F is None or res.X is None:
            pop = res.pop
            Xs = pop.get("X")
            CVs = pop.get("CV")
            best_idx = int(np.argmin(CVs))
            best_solution = Xs[best_idx]
        else:
            best_idx = self.select_best_solution(res.F)
            best_solution = res.X[best_idx]

        control_points = []
        for j in range(self.n_control_points):
            idx = j * 3
            control_points.append([best_solution[idx], best_solution[idx + 1], best_solution[idx + 2]])

        control_points[0] = list(start)
        control_points[-1] = list(end)

        control_points = self._enforce_corridor_on_control_points(control_points)

        final_path = self.create_b_spline(control_points, n_samples=220).tolist()

        final_path = self.project_path_into_corridor(final_path)
        final_path = self.smooth_path(
            final_path, smoothing_factor=0.35, iters=4,
            keep_endpoints=True, clamp_to_corridor=True
        )
        final_path = self._repair_trajectory(final_path)
        final_path = self.project_path_into_corridor(final_path)
        final_path = [(p[0], p[1], max(p[2], self.env.min_height)) for p in final_path]

        return final_path
