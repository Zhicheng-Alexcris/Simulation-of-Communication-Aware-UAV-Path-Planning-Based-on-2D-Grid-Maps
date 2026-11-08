"""Per-evaluation feasible-hypervolume convergence for a single CA run (U=5).

Computes the running nondominated feasible hypervolume after EVERY
evaluation (not per generation) from candidates_drone*.csv, giving a
fine-grained convergence curve with visible per-step dynamics.

HV convention matches the paper's Fig. 4: objectives are (path length m,
communication cost), reference point (2000, 1.05), reference area 2100,
normalized by that area. Only feasible candidates enter the front.

Output: figs/convergence_hv_pereval_<trial>_<drone>.png/.pdf
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(r"D:\desktop\code")
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "ca_diffusion_es"))

from ca_diffusion_es.report import _apply_publication_axis_style  # noqa: E402

RUNS = CODE_ROOT / "runs" / "ca_v9_candidate_logs_n5_v1" / "ca_diffusion_es" / "N5"
OUT_DIR = CODE_ROOT / "figs"

REF_POINT = np.array([2000.0, 1.05])
REF_AREA = 2000.0 * 1.05

FS_LABEL = 11.5
FS_TICK = 10.0
FS_LEGEND = 10.0

C_RAW = "#9AA6B0"
C_LINE = "#D62F2F"


def load_candidates(trial: str, drone: int):
    f = RUNS / trial / f"candidates_drone{drone}.csv"
    rows = []
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((
                int(r["evaluation_id"]),
                float(r["length_m"]) if r["length_m"] else np.nan,
                float(r["communication_cost"]) if r["communication_cost"] else np.nan,
                float(r["is_feasible"]),
            ))
    rows.sort()
    ev = np.array([r[0] for r in rows])
    obj = np.array([(r[1], r[2]) for r in rows])
    feas = np.array([r[3] for r in rows]) >= 0.5
    return ev, obj, feas


def running_hv(ev, obj, feas):
    """HV of the nondominated feasible set after each evaluation."""
    n = len(ev)
    hv = np.full(n, np.nan)
    front = []  # list of (length, comm) nondominated, length asc -> comm desc
    for i in range(n):
        if feas[i]:
            p = obj[i]
            if np.isfinite(p).all():
                # dominated check against current front
                dominated = False
                new_front = []
                for q in front:
                    # q dominates p?
                    if q[0] <= p[0] and q[1] <= p[1]:
                        dominated = True
                    # p dominates q?  (strictly better in one, not worse in other)
                    if not (p[0] <= q[0] and p[1] <= q[1]):
                        new_front.append(q)
                if not dominated:
                    new_front.append((p[0], p[1]))
                    # sort by length, keep comm strictly decreasing
                    new_front.sort()
                    filt = []
                    best_comm = np.inf
                    for L, C in new_front:
                        if C < best_comm:
                            filt.append((L, C))
                            best_comm = C
                    front = filt
                # HV by layered rectangles (staircase integration)
                hv[i] = hv_of_front(front)
        else:
            hv[i] = hv[i - 1] if i > 0 else 0.0
    return hv


def hv_of_front(front):
    """Staircase HV w.r.t. reference point; front sorted by length asc."""
    if not front:
        return 0.0
    total = 0.0
    prev_len = 0.0
    # front: length increasing, comm decreasing -> each point contributes
    # (next_len - this_len) * (ref_comm - this_comm)... classic 2-D staircase
    pts = front
    for j, (L, C) in enumerate(pts):
        next_len = pts[j + 1][0] if j + 1 < len(pts) else REF_POINT[0]
        # rectangle covered by point j and all dominated region to its right
        # up to next point's length at this point's comm level is already
        # covered by better points; standard formulation:
        total += (min(next_len, REF_POINT[0]) - min(L, REF_POINT[0])) * max(REF_POINT[1] - C, 0.0)
    return total / REF_AREA


def main(trial="trial_02", drone=3):
    ev, obj, feas = load_candidates(trial, drone)
    hv = running_hv(ev, obj, feas)

    fig = plt.figure(figsize=(7.16, 3.4), facecolor="white")
    grid = fig.add_gridspec(1, 1, left=0.10, right=0.985, top=0.96, bottom=0.155)
    ax = fig.add_subplot(grid[0])

    ax.plot(ev, hv, color=C_LINE, linewidth=1.6, zorder=5,
            label="Feasible hypervolume (per evaluation)")
    # mark generation boundaries lightly to relate to the per-gen log
    for g in range(1, 40):
        ax.axvline(g * 100, color="#E4E8EC", linewidth=0.4, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(1, 4000)
    ax.set_xticks([1, 10, 100, 1000, 4000])
    ax.set_xticklabels(["1", "10", "100", "1000", "4000"])

    ax.set_xlabel("Evaluations", fontsize=FS_LABEL)
    ax.set_ylabel("Reference-normalized\nfeasible hypervolume", fontsize=FS_LABEL)
    ax.set_ylim(0.0, 0.78)
    _apply_publication_axis_style(ax, grid_axis="both")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(loc="lower right", fontsize=FS_LEGEND, frameon=True,
              framealpha=0.9, edgecolor="#C8CDD2", handlelength=1.8,
              borderpad=0.5).get_frame().set_linewidth(0.6)

    out = OUT_DIR / f"convergence_hv_pereval_{trial}_d{drone}"
    fig.savefig(out.with_suffix(".png"), dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved: {out}.png/.pdf")
    print(f"HV start={hv[~np.isnan(hv)][0]:.4f} end={hv[-1]:.4f} "
          f"max={np.nanmax(hv):.4f}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
