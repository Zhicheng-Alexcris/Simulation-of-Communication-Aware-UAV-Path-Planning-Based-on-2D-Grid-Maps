# channel.py
import csv
import numpy as np
from collections import defaultdict


def _normalize_header(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s


def _sniff_delimiter(path: str, sample_bytes: int = 8192) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(sample_bytes)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return "\t" if sample.count("\t") > sample.count(",") else ","


def fspl_db(d_m: float, f_hz: float = 2.0e9) -> float:
    d_m = max(float(d_m), 1e-3)
    f_hz = max(float(f_hz), 1.0)
    d_km = d_m / 1000.0
    f_mhz = f_hz / 1e6
    return float(32.44 + 20.0 * np.log10(d_km) + 20.0 * np.log10(f_mhz))


def fspl_db_arr(d_m, f_hz: float = 2.0e9):
    """数组版 FSPL，供向量化信道函数使用（与 fspl_db 语义一致）。"""
    d = np.asarray(d_m, dtype=float)
    d = np.maximum(d, 1e-3)
    f_hz = max(float(f_hz), 1.0)
    d_km = d / 1000.0
    f_mhz = f_hz / 1e6
    return 32.44 + 20.0 * np.log10(d_km) + 20.0 * np.log10(f_mhz)


def load_pathloss_curve_csv(path, d_col="Straight Line Distance", pl_col="Path Loss (dB)",
                            distance_round=0.1):
    delim = _sniff_delimiter(path)
    bucket = defaultdict(list)

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        headers = {_normalize_header(h): h for h in reader.fieldnames}
        d_key = headers.get(_normalize_header(d_col))
        pl_key = headers.get(_normalize_header(pl_col))

        if d_key is None:
            raise KeyError(f"Distance column not found. Want '{d_col}'. Available: {list(headers.keys())[:20]} ...")
        if pl_key is None:
            raise KeyError(f"Pathloss column not found. Want '{pl_col}'. Available: {list(headers.keys())[:20]} ...")

        for r in reader:
            try:
                d = float(r[d_key])
                pl = float(r[pl_key])
            except Exception:
                continue
            if not np.isfinite(d) or not np.isfinite(pl):
                continue

            d = max(d, 0.0)
            d = round(d / distance_round) * distance_round
            bucket[d].append(pl)

    if not bucket:
        raise ValueError("No valid (distance, pathloss) pairs parsed from CSV.")

    d_arr = np.array(sorted(bucket.keys()), dtype=float)
    pl_arr = np.array([float(np.mean(bucket[d])) for d in d_arr], dtype=float)

    idx = np.argsort(d_arr)
    d_arr = d_arr[idx]
    pl_arr = pl_arr[idx]
    return d_arr, pl_arr


def pathloss_from_distance(d, d_arr, pl_arr, f_hz: float = 2.0e9):
    d = max(float(d), 0.0)

    # FSPL-only fallback
    if d_arr is None or pl_arr is None or len(d_arr) == 0:
        return fspl_db(d, f_hz=f_hz)

    d0 = float(d_arr[0])
    d1 = float(d_arr[-1])
    pl0 = float(pl_arr[0])
    pl1 = float(pl_arr[-1])

    # Extrapolate using FSPL with offset that matches curve at boundary
    if d <= d0:
        offset = pl0 - fspl_db(d0, f_hz=f_hz)
        return fspl_db(max(d, 1e-3), f_hz=f_hz) + offset
    if d >= d1:
        offset = pl1 - fspl_db(d1, f_hz=f_hz)
        return fspl_db(max(d, 1e-3), f_hz=f_hz) + offset

    return float(np.interp(d, d_arr, pl_arr))


def avg_pathloss_to_ues_db(x, y, z, ues, d_arr, pl_arr, f_hz: float = 2.0e9):
    if not ues:
        return 0.0
    pls = []
    for (ux, uy, uz) in ues:
        dx = x - ux
        dy = y - uy
        dz = z - uz
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        pls.append(pathloss_from_distance(d, d_arr, pl_arr, f_hz=f_hz))
    return float(np.mean(pls))


def estimate_pl_range(env, layer_idx, ues, d_arr, pl_arr, n=900, p_lo=5, p_hi=95, f_hz: float = 2.0e9):
    zc = env.layer_mid_z(layer_idx)

    xlim = getattr(env, "xlim", (-100, 100))
    ylim = getattr(env, "ylim", (-100, 100))
    xs = np.random.uniform(xlim[0], xlim[1], size=n)
    ys = np.random.uniform(ylim[0], ylim[1], size=n)

    pls = [avg_pathloss_to_ues_db(x, y, zc, ues, d_arr, pl_arr, f_hz=f_hz) for x, y in zip(xs, ys)]
    pl_min = float(np.percentile(pls, p_lo))
    pl_max = float(np.percentile(pls, p_hi))
    if pl_max - pl_min < 1e-6:
        pl_max = pl_min + 1.0
    return pl_min, pl_max


def make_channel_cost_fn(env, layer_idx, ues, d_arr, pl_arr, pl_min, pl_max, f_hz: float = 2.0e9):
    zc = env.layer_mid_z(layer_idx)

    def channel_cost_fn(x, y, z_unused=None):
        pl_db = avg_pathloss_to_ues_db(x, y, zc, ues, d_arr, pl_arr, f_hz=f_hz)
        c = (pl_db - pl_min) / (pl_max - pl_min)
        return float(np.clip(c, 0.0, 1.0))

    return channel_cost_fn


def make_channel_cost_fn_vec(env, layer_idx, ues, d_arr, pl_arr, pl_min, pl_max, f_hz: float = 2.0e9):
    """向量化信道代价：对 (M,3) 点一次性返回 (M,) 归一化代价，与逐点
    make_channel_cost_fn 语义完全一致（同样使用固定层高 zc、同样的插值/外推规则），
    但把所有 UE 距离用广播一次算完、并只做一次 np.interp，避免 Python 双重循环。

    与原标量版的对应：
      - 距离都取 3D 欧氏距离，但点的 z 固定为层高 zc（原 avg_pathloss_to_ues_db 也忽略 z）。
      - 距离在 [d0,d1] 内用 np.interp；越界用 FSPL + 边界偏移外推，与 pathloss_from_distance 一致。
    """
    zc = float(env.layer_mid_z(layer_idx))
    ues = np.asarray(ues, dtype=float).reshape(-1, 3)
    pl_range = max(float(pl_max - pl_min), 1e-12)
    d_arr = None if d_arr is None else np.asarray(d_arr, dtype=float)
    pl_arr = None if pl_arr is None else np.asarray(pl_arr, dtype=float)

    if d_arr is None or pl_arr is None or len(d_arr) == 0:
        # FSPL-only 退化分支
        def _vec_fspl(pts):
            pts = np.asarray(pts, dtype=float)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            dx = pts[:, 0:1] - ues[None, :, 0]
            dy = pts[:, 1:2] - ues[None, :, 1]
            dz = zc - ues[None, :, 2]
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            pl = np.array([fspl_db_arr(d[:, k], f_hz=f_hz) for k in range(ues.shape[0])]).mean(axis=0)
            c = (pl - pl_min) / pl_range
            return np.clip(c, 0.0, 1.0)
        return _vec_fspl

    d0 = float(d_arr[0]); d1 = float(d_arr[-1])
    pl0 = float(pl_arr[0]); pl1 = float(pl_arr[-1])
    off0 = pl0 - fspl_db(d0, f_hz=f_hz)
    off1 = pl1 - fspl_db(d1, f_hz=f_hz)

    def channel_cost_fn_vec(pts):
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        # 到所有 UE 的距离：(M, U)
        dx = pts[:, 0:1] - ues[None, :, 0]
        dy = pts[:, 1:2] - ues[None, :, 1]
        dz = zc - ues[None, :, 2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        d_flat = d.reshape(-1)
        pl_flat = np.interp(d_flat, d_arr, pl_arr).copy()
        below = d_flat <= d0
        above = d_flat >= d1
        if np.any(below):
            pl_flat[below] = fspl_db_arr(d_flat[below], f_hz=f_hz) + off0
        if np.any(above):
            pl_flat[above] = fspl_db_arr(d_flat[above], f_hz=f_hz) + off1
        pl = pl_flat.reshape(d.shape).mean(axis=1)          # (M,)
        c = (pl - pl_min) / pl_range
        return np.clip(c, 0.0, 1.0)

    return channel_cost_fn_vec
