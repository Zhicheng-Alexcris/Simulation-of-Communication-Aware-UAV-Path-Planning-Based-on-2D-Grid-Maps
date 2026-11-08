"""生成论文 Fig.14（每 UAV 平均路损，Geo-only vs Geo+Channel）与 Fig.2/3（平均路损 CDF，n=30）。

思路：对 hybrid 模式（CA-MOCO）规划 N=5，沿轨迹采样点用 channel.avg_pathloss_to_ues_db
计算到地面用户的平均 path loss (dB)：
  - Fig.14：单次规划（trials=1），取该次 5 个 UAV 的路损画分组柱状
  - Fig.2/3：n=30 次独立重复（不同 seed），每次取 5 UAV 平均路损作为一个样本，画 CDF

用法：
  # 采集（前台 trials=1 出 Fig.14；后台 trials=30 出 Fig.2/3）
  python make_fig_pathloss.py --mode collect --trials 1  --no_channel 0 --out pl_ch_t1.npz
  python make_fig_pathloss.py --mode collect --trials 1  --no_channel 1 --out pl_noch_t1.npz
  python make_fig_pathloss.py --mode collect --trials 30 --no_channel 0 --out pl_ch.npz &
  python make_fig_pathloss.py --mode collect --trials 30 --no_channel 1 --out pl_noch.npz &

  # 绘图
  python make_fig_pathloss.py --mode plot14 --ch pl_ch_t1.npz   --noch pl_noch_t1.npz
  python make_fig_pathloss.py --mode plot23 --ch pl_ch.npz      --noch pl_noch.npz
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import Environment, PIPE_RADIUS
from channel import (load_pathloss_curve_csv, estimate_pl_range,
                     make_channel_cost_fn, make_channel_cost_fn_vec,
                     avg_pathloss_to_ues_db)
from main import assign_layers, plan_one_drone

CHANNEL_CSV = "processed_crossroad_path_data.csv"
FHZ = 2.0e9


def build_env_and_channel(n, d_arr, pl_arr):
    env = Environment()
    env.starts = env.starts[:n]
    env.ends = env.ends[:n]
    if hasattr(env, "vertiports"):
        env.vertiports = env.vertiports[:n]
    lcc, lccv = {}, {}
    for li in range(len(env.safe_layers)):
        pl_min, pl_max = estimate_pl_range(env, li, env.ues, d_arr, pl_arr, f_hz=FHZ)
        lcc[li] = make_channel_cost_fn(env, li, env.ues, d_arr, pl_arr, pl_min, pl_max, f_hz=FHZ)
        lccv[li] = make_channel_cost_fn_vec(env, li, env.ues, d_arr, pl_arr, pl_min, pl_max, f_hz=FHZ)
    return env, lcc, lccv


def collect(trials, no_channel, seed0, n, d_arr, pl_arr, out):
    layers = assign_layers(Environment())
    pipe_radius = float(PIPE_RADIUS)
    r_safe = max(2.0, pipe_radius + 0.5)
    beta = 0.0 if no_channel else 10.0
    chw = 0.0 if no_channel else 1400.0
    all_pl = []      # (trials, n) per-UAV 平均路损，供 Fig.14
    all_avg = []     # (trials,) 每 trial 平均路损（所有 UAV 采样点均值）→ Fig.2
    all_worst = []   # (trials,) 每 trial 全段最坏路损（所有 UAV 采样点最大值）
    all_worst_mid = []  # (trials,) 每 trial 中间巡航段最坏路损（排除两端起降段）→ Fig.3
    for t in range(trials):
        seed = seed0 + t * 1000
        env, lcc, lccv = build_env_and_channel(n, d_arr, pl_arr)
        other = []
        per_uav = []
        trial_pts = []
        trial_worst_mid = []
        for drid in range(n):
            li = layers[drid]
            path, _, _ = plan_one_drone(
                env=env, drone_id=drid, layer_idx=li, other_paths=other,
                pipe_radius=pipe_radius, r_safe=r_safe, astar_res=2.0,
                channel_cost_fn=lcc[li], channel_cost_fn_vec=lccv[li],
                quick_first=True, nsga_seed=seed + drid * 100 + 1, mode="hybrid",
                astar_inflation=12.0, corridor_radius=6.0, pop=45, gens=35,
                beta_channel=beta, channel_weight=chw)
            other.append(np.asarray(path))
            pts = np.asarray(path, dtype=float)
            samp = pts[::max(1, len(pts) // 80)]
            pls = [avg_pathloss_to_ues_db(float(p[0]), float(p[1]), float(p[2]),
                                          env.ues, d_arr, pl_arr, f_hz=FHZ) for p in samp]
            per_uav.append(float(np.mean(pls)))
            trial_pts.extend(pls)
            # 中间巡航段最坏路损：排除每条 UAV 轨迹首尾各 10%（起降段）
            k = max(1, len(pls) // 10)
            mid = pls[k:-k] if len(pls) > 2 * k else pls
            trial_worst_mid.append(float(np.max(mid)))
        all_pl.append(per_uav)
        all_avg.append(float(np.mean(trial_pts)))
        all_worst.append(float(np.max(trial_pts)))
        all_worst_mid.append(float(np.max(trial_worst_mid)))
        print(f"[trial {t}] avg={all_avg[-1]:.2f} worst={all_worst[-1]:.2f} worst_mid={all_worst_mid[-1]:.2f} dB")
    np.savez(out, pl=np.array(all_pl), pl_avg=np.array(all_avg),
             pl_worst=np.array(all_worst), pl_worst_mid=np.array(all_worst_mid))
    print("saved", out)



def cdf_plot(ax, data, label, color):
    d = np.sort(np.asarray(data, float))
    m = len(d)
    y = np.arange(1, m + 1) / m
    ax.plot(d, y, label=label, color=color, lw=2)


def plot14(ch_npz, noch_npz):
    ch = np.load(ch_npz)["pl"]
    noch = np.load(noch_npz)["pl"]
    c, nc = ch[0], noch[0]
    x = np.arange(len(c))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, nc, w, label="Geo-only", color="C0")
    ax.bar(x + w / 2, c, w, label="Geo+Channel", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"UAV {i + 1}" for i in range(len(c))])
    ax.set_ylabel("Average path loss (dB)")
    ax.set_title("Fig.14 - Average path loss from UAVs to users (per UAV)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("figs/Fig14_avg_pathloss_strategies.png", dpi=300, bbox_inches="tight")
    print("[ok] Fig14_avg_pathloss_strategies.png")


def plot2(ch_npz, noch_npz):
    """Fig.2 - CDF of the AVERAGE path loss (n=30)，两策略对比。"""
    ca = np.load(ch_npz)["pl_avg"]
    na = np.load(noch_npz)["pl_avg"]
    fig, ax = plt.subplots(figsize=(8, 5))
    cdf_plot(ax, na, "Geo-only", "C0")
    cdf_plot(ax, ca, "Geo+Channel", "C2")
    ax.set_xlabel("Average path loss (dB)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Fig.2 - CDF of average path loss (n={len(ca)})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig("figs/Fig2_pathloss_cdf.png", dpi=300, bbox_inches="tight")
    print("[ok] Fig2_pathloss_cdf.png  | avg: Geo-only=%.2f  Geo+Channel=%.2f  (diff=%.2f)" % (
        na.mean(), ca.mean(), na.mean() - ca.mean()))


def plot3(ch_npz, noch_npz):
    """Fig.3 - CDF of the WORST-CASE (mid-cruise) path loss (n=30)，两策略对比。
    用排除两端起降段后的中间巡航段最差路损，避免最坏点被固定端点主导。"""
    cw = np.load(ch_npz)["pl_worst_mid"]
    nw = np.load(noch_npz)["pl_worst_mid"]
    fig, ax = plt.subplots(figsize=(8, 5))
    cdf_plot(ax, nw, "Geo-only", "C0")
    cdf_plot(ax, cw, "Geo+Channel", "C2")
    ax.set_xlabel("Worst-case (mid-cruise) path loss (dB)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Fig.3 - CDF of worst-case path loss (n={len(cw)})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig("figs/Fig3_pathloss_cdf.png", dpi=300, bbox_inches="tight")
    print("[ok] Fig3_pathloss_cdf.png  | worst_mid: Geo-only=%.2f  Geo+Channel=%.2f  (diff=%.2f)" % (
        nw.mean(), cw.mean(), nw.mean() - cw.mean()))


def plot23(ch_npz, noch_npz):
    """生成两张独立图：Fig.2=average path loss CDF，Fig.3=worst-case path loss CDF。"""
    plot2(ch_npz, noch_npz)
    plot3(ch_npz, noch_npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="collect")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--no_channel", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--out", default=None,
                    help="collect 输出 npz；默认按 no_channel 写到 artifacts/trials_*.npz")
    ap.add_argument("--ch", default="artifacts/trials_ch.npz")
    ap.add_argument("--noch", default="artifacts/trials_noch.npz")
    a = ap.parse_args()

    d_arr, pl_arr = load_pathloss_curve_csv(
        CHANNEL_CSV, d_col="Straight Line Distance", pl_col="Path Loss (dB)")
    if a.mode == "collect":
        out = a.out or ("artifacts/trials_noch.npz" if a.no_channel else "artifacts/trials_ch.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        collect(a.trials, a.no_channel, a.seed0, a.N, d_arr, pl_arr, out)
    elif a.mode == "plot14":
        plot14(a.ch, a.noch)
    elif a.mode == "plot23":
        plot23(a.ch, a.noch)


if __name__ == "__main__":
    main()
