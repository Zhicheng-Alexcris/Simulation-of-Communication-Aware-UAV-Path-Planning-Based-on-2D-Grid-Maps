"""Generate formal tables and figures from a completed experiment directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "astar": "A*",
    "astar_nsga2": "A* + NSGA-II",
    "nsga2": "NSGA-II",
    "diffusion_only": "Diffusion-only",
    "ca_diffusion_es": "CA-Diffusion-ES",
}
METHOD_COLORS = {
    "astar": "#4E79A7",
    "astar_nsga2": "#76B7B2",
    "nsga2": "#B07AA1",
    "diffusion_only": "#9C755F",
    "ca_diffusion_es": "#D55E00",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

PROPOSED_METHOD = "ca_diffusion_es"

UE_COLOR = "#4A4F55"
BUILDING_FACE_2D = "#F5F6F7"
BUILDING_EDGE_2D = "#CDD3D8"
BUILDING_FACE_3D = "#E3E6E8"
BUILDING_EDGE_3D = "#A9B0B6"


def _uav_palette(max_count: int = 9) -> list[str]:
    base = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#8C564B",
        "#7F7F7F",
        "#BCBD22",
    ]
    return base[:max_count]




def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#4E79A7")


def _method_linewidth(method: str, proposed: float = 2.1, baseline: float = 1.7) -> float:
    return proposed if method == PROPOSED_METHOD else baseline


def _apply_publication_axis_style(axis, *, grid_axis: str = "y") -> None:
    if grid_axis == "both":
        axis.grid(color="#E6EAED", linewidth=0.6, alpha=0.8)
    elif grid_axis == "x":
        axis.grid(axis="x", color="#E6EAED", linewidth=0.6, alpha=0.8)
    elif grid_axis == "y":
        axis.grid(axis="y", color="#E6EAED", linewidth=0.6, alpha=0.8)

    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#A9B0B6")
    axis.spines["bottom"].set_color("#A9B0B6")
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(colors="#4E545A", width=0.8, length=3.0)


def _finalize_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    if frame is not None:
        frame.set_linewidth(0.0)
        frame.set_facecolor("none")




def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
def _active_methods(records: list[dict[str, str]]) -> list[str]:
    present = {row.get("method") for row in records}
    return [method for method in METHOD_LABELS if method in present]




def _formal_hv_reference(method_roots: dict[str, Path]) -> tuple[float, float]:
    """Load and verify the common Hypervolume reference from run manifests."""
    references: list[tuple[float, float]] = []
    sources: list[Path] = []
    for root in sorted(set(method_roots.values()), key=str):
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Formal manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = manifest.get("config", {}).get("hypervolume_reference")
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"Invalid Hypervolume reference in {manifest_path}: {raw}")
        reference = (float(raw[0]), float(raw[1]))
        if reference[0] <= 0.0 or reference[1] <= 0.0:
            raise ValueError(f"Hypervolume reference must be positive: {reference}")
        references.append(reference)
        sources.append(manifest_path)
    first = references[0]
    for reference, source in zip(references[1:], sources[1:]):
        if not np.allclose(reference, first, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Formal runs use inconsistent Hypervolume references: "
                f"{sources[0]}={first}, {source}={reference}"
            )
    return first


def _normalized_hv(value: float, reference: tuple[float, float]) -> float:
    """Normalize raw HV by the common reference rectangle area."""
    scale = float(reference[0] * reference[1])
    return float(np.clip(float(value) / scale, 0.0, 1.0))


def _write_summary(
    records: list[dict[str, str]],
    output: Path,
    method_roots: dict[str, Path],
    hypervolume_reference: tuple[float, float],
) -> Path:
    groups = defaultdict(list)
    for row in records:
        groups[(row["method"], int(row["uav_count"]))].append(row)
    fields = [
        "method", "uav_count", "success_rate", "feasible_samples",
        "path_length_mean_m",
        "path_length_std_m", "communication_cost_mean", "communication_cost_std",
        "mean_pathloss_db", "worst_pathloss_db", "outage_ratio",
        "min_clearance_m", "turn_p95_deg", "scenario_runtime_mean_s",
        "evaluation_count_mean", "hypervolume_final_mean",
        "normalized_hypervolume_final_mean",
    ]
    rows = []
    for (method, uav_count), group in sorted(groups.items()):
        feasible_group = [
            row for row in group if _float(row, "success") >= 0.5
        ]
        quality_group = feasible_group if feasible_group else group

        def values(key):
            return np.asarray(
                [_float(row, key) for row in quality_group], dtype=float
            )

        scenario_times = {}
        for row in group:
            scenario_times[int(row["trial"])] = _float(row, "scenario_runtime_s")
        final_hv = []
        history_paths = sorted(
            (method_roots[method] / method / f"N{uav_count}").glob(
                "trial_*/history_drone*.csv"
            )
        )
        for path in history_paths:
            history = _load_csv(path)
            if history:
                final_hv.append(_float(history[-1], "hypervolume"))
        rows.append(
            {
                "method": method,
                "uav_count": uav_count,
                "success_rate": float(
                    np.mean([_float(row, "success") for row in group])
                ),
                "feasible_samples": len(feasible_group),
                "path_length_mean_m": float(np.nanmean(values("path_length_m"))),
                "path_length_std_m": float(np.nanstd(values("path_length_m"))),
                "communication_cost_mean": float(
                    np.nanmean(values("communication_cost"))
                ),
                "communication_cost_std": float(
                    np.nanstd(values("communication_cost"))
                ),
                "mean_pathloss_db": float(np.nanmean(values("mean_pathloss_db"))),
                "worst_pathloss_db": float(np.nanmean(values("worst_pathloss_db"))),
                "outage_ratio": float(np.nanmean(values("outage_ratio"))),
                "min_clearance_m": float(np.nanmean(values("min_clearance_m"))),
                "turn_p95_deg": float(np.nanmean(values("turn_p95_deg"))),
                "scenario_runtime_mean_s": float(np.mean(list(scenario_times.values()))),
                "evaluation_count_mean": float(
                    np.nanmean(values("evaluation_count"))
                ),
                "hypervolume_final_mean": float(np.mean(final_hv)) if final_hv else 0.0,
                "normalized_hypervolume_final_mean": (
                    float(np.mean([_normalized_hv(value, hypervolume_reference) for value in final_hv]))
                    if final_hv else 0.0
                ),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _feasible_values(records, method: str, key: str) -> np.ndarray:
    values = np.asarray(
        [
            _float(row, key)
            for row in records
            if row["method"] == method and _float(row, "success") >= 0.5
        ],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _completed_values(records, method: str, key: str) -> np.ndarray:
    """Return finite metric values from every completed run, including failures."""
    values = np.asarray(
        [
            _float(row, key)
            for row in records
            if row["method"] == method and row.get("status", "complete") == "complete"
        ],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _sample_counts(records, method: str) -> tuple[int, int]:
    method_rows = [row for row in records if row["method"] == method]
    feasible = sum(_float(row, "success") >= 0.5 for row in method_rows)
    return int(feasible), len(method_rows)


def _method_tick_label(records, method: str) -> str:
    feasible, total = _sample_counts(records, method)
    return f"{METHOD_LABELS[method]}\n({feasible}/{total} feasible)"


def _plot_worst_pathloss_ecdf(axis, records, methods: list[str]) -> None:
    threshold_db = 110.0
    for method in methods:
        values = np.sort(_feasible_values(records, method, "worst_pathloss_db"))
        if len(values) == 0:
            continue
        probability = np.arange(1, len(values) + 1, dtype=float) / len(values)
        axis.step(
            values,
            probability,
            where="post",
            label=METHOD_LABELS[method],
            color=_method_color(method),
            linewidth=_method_linewidth(method, proposed=2.1, baseline=1.7),
        )
    axis.axvline(threshold_db, color="#7A7F85", linestyle="--", linewidth=1.0)
    axis.text(
        threshold_db,
        0.02,
        "110 dB",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=8,
        color="#6A7076",
    )
    axis.set_xlabel("Worst path loss (dB)")
    axis.set_ylabel("Empirical CDF")
    axis.set_ylim(0.0, 1.02)
    _apply_publication_axis_style(axis, grid_axis="both")
    legend = axis.legend(loc="lower right", frameon=False, handlelength=2.2)
    _finalize_legend(legend)


def _plot_outage_heatmap(axis, records, methods: list[str]) -> None:
    drones = sorted({int(row["drone_id"]) for row in records})
    values = np.full((len(methods), len(drones)), np.nan, dtype=float)

    for method_index, method in enumerate(methods):
        for drone_index, drone_id in enumerate(drones):
            outage = [
                _float(row, "outage_ratio")
                for row in records
                if row["method"] == method
                and int(row["drone_id"]) == drone_id
                and _float(row, "success") >= 0.5
            ]
            if outage:
                values[method_index, drone_index] = 100.0 * float(np.mean(outage))

    maximum = max(1.0, float(np.nanmax(values))) if np.isfinite(values).any() else 1.0
    image = axis.imshow(
        values,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=maximum,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(drones)))
    axis.set_xticklabels([f"UAV {drone_id + 1}" for drone_id in drones])
    axis.set_yticks(np.arange(len(methods)))
    axis.set_yticklabels([METHOD_LABELS[method] for method in methods])
    axis.set_xlabel("Vehicle")
    axis.set_ylabel("Method")

    for row_index in range(len(methods)):
        for column_index in range(len(drones)):
            value = values[row_index, column_index]
            color = "white" if np.isfinite(value) and value > 0.55 * maximum else "#222222"
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}" if np.isfinite(value) else "N/A",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0, colors="#4E545A")
    cbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    cbar.set_label("Outage ratio (%)")
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#A9B0B6")


def _plot_clearance_survival(axis, records, methods: list[str]) -> None:
    safety_radius_m = 3.5
    for method in methods:
        clearance = np.sort(_feasible_values(records, method, "min_clearance_m"))
        if len(clearance) == 0:
            continue
        survival = (len(clearance) - np.arange(len(clearance), dtype=float)) / len(clearance)
        plot_clearance = np.concatenate([[safety_radius_m], clearance])
        survival = np.concatenate([[1.0], survival])
        axis.step(
            plot_clearance,
            survival,
            where="post",
            label=METHOD_LABELS[method],
            color=_method_color(method),
            linewidth=_method_linewidth(method, proposed=2.1, baseline=1.7),
        )
    axis.axvline(safety_radius_m, color="#7A7F85", linestyle="--", linewidth=1.0)
    axis.text(
        safety_radius_m,
        0.02,
        "3.5 m",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=8,
        color="#6A7076",
    )
    axis.set_xlabel("Minimum clearance (m)")
    axis.set_ylabel("Survival probability")
    axis.set_ylim(0.0, 1.02)
    _apply_publication_axis_style(axis, grid_axis="both")
    legend = axis.legend(loc="upper right", frameon=False, handlelength=2.2)
    _finalize_legend(legend)


def _plot_turn_ecdf(axis, records, methods: list[str]) -> None:
    reference_deg = 35.0
    for method in methods:
        values = np.sort(_feasible_values(records, method, "turn_p95_deg"))
        if len(values) == 0:
            continue
        probability = np.arange(1, len(values) + 1, dtype=float) / len(values)
        axis.step(
            values,
            probability,
            where="post",
            label=METHOD_LABELS[method],
            color=_method_color(method),
            linewidth=_method_linewidth(method, proposed=2.1, baseline=1.7),
        )
    axis.axvline(reference_deg, color="#7A7F85", linestyle="--", linewidth=1.0)
    axis.text(
        reference_deg,
        0.02,
        "35°",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=8,
        color="#6A7076",
    )
    axis.set_xlabel("TurnP95 (deg)")
    axis.set_ylabel("Empirical CDF")
    axis.set_ylim(0.0, 1.02)
    _apply_publication_axis_style(axis, grid_axis="both")
    legend = axis.legend(loc="lower right", frameon=False, handlelength=2.2)
    _finalize_legend(legend)


def _boxplot_metric(
    axis,
    records,
    methods: list[str],
    key: str,
    label: str,
    *,
    mean_decimals=None,
) -> None:
    """Draw a publication-style distribution plot for one metric."""
    data = [_feasible_values(records, method, key) for method in methods]
    valid_methods = [method for method, values in zip(methods, data) if len(values)]
    valid_data = [values for values in data if len(values)]

    if not valid_data:
        axis.text(0.5, 0.5, "No feasible runs", ha="center", va="center", transform=axis.transAxes)
        _apply_publication_axis_style(axis, grid_axis="y")
        return

    plot = axis.boxplot(
        valid_data,
        patch_artist=True,
        widths=0.62,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#222222",
            "markersize": 5,
            "markeredgewidth": 0.8,
        },
        medianprops={"color": "#222222", "linewidth": 1.3},
        whiskerprops={"color": "#666B70", "linewidth": 1.0},
        capprops={"color": "#666B70", "linewidth": 1.0},
        boxprops={"edgecolor": "#666B70", "linewidth": 0.9},
        flierprops={"marker": ".", "markersize": 2.2, "alpha": 0.18},
    )

    for patch, method in zip(plot["boxes"], valid_methods):
        patch.set_facecolor(_method_color(method))
        patch.set_alpha(0.55)

    axis.set_xticks(range(1, len(valid_methods) + 1))
    axis.set_xticklabels([METHOD_LABELS[m] for m in valid_methods], rotation=15, ha="right")
    axis.set_ylabel(label)
    _apply_publication_axis_style(axis, grid_axis="y")


def _violin_metric(axis, records, methods: list[str], key: str, label: str) -> None:
    """Draw a compact distribution view with quartiles and mean/median markers."""
    data = [_feasible_values(records, method, key) for method in methods]
    valid = [(index, method, values) for index, (method, values) in
             enumerate(zip(methods, data), start=1) if len(values)]
    if not valid:
        axis.text(0.5, 0.5, "No feasible runs", ha="center", va="center",
                  transform=axis.transAxes, color="#666666")
        return

    positions = [item[0] for item in valid]
    values_list = [item[2] for item in valid]
    plot = axis.violinplot(
        values_list, positions=positions, widths=0.72,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body, (_, method, _) in zip(plot["bodies"], valid):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_edgecolor("#333333")
        body.set_linewidth(0.8)
        body.set_alpha(0.58)

    for position, (_, method, values) in zip(positions, valid):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        mean = float(np.mean(values))
        axis.vlines(position, q1, q3, color="#333333", linewidth=2.2, zorder=4)
        axis.hlines([q1, q3], position - 0.10, position + 0.10,
                    color="#333333", linewidth=1.2, zorder=4)
        axis.scatter(position, median, s=30, marker="o", facecolor="white",
                     edgecolor="#222222", linewidth=0.9, zorder=5)
        axis.scatter(position, mean, s=32, marker="D", facecolor="#222222",
                     edgecolor="white", linewidth=0.6, zorder=5)
        # A restrained deterministic jitter shows sample density without clutter.
        jitter = np.linspace(-0.16, 0.16, len(values))
        jitter += 0.008 * np.sin(np.arange(len(values), dtype=float) * 2.4)
        axis.scatter(position + jitter, values, s=7,
                     color=METHOD_COLORS[method], alpha=0.14,
                     linewidths=0, zorder=2)

    axis.set_xlim(0.4, len(methods) + 0.6)
    axis.set_xticks(range(1, len(methods) + 1))
    axis.set_xticklabels([_method_tick_label(records, method) for method in methods])
    axis.set_ylabel(label)
    axis.grid(axis="y", alpha=0.25)


def _bar_metric(axis, records, methods: list[str], key: str, label: str) -> None:
    """Draw method means with 95% confidence intervals for a scalar metric."""
    data = [_completed_values(records, method, key) for method in methods]
    means = np.asarray(
        [float(np.mean(values)) if len(values) else np.nan for values in data],
        dtype=float,
    )
    errors = np.asarray(
        [
            np.nan if len(values) == 0 else (
                0.0 if len(values) == 1
                else float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
            )
            for values in data
        ],
        dtype=float,
    )
    positions = np.arange(len(methods), dtype=float)
    colors = [_method_color(method) for method in methods]
    bars = axis.bar(
        positions,
        means,
        yerr=errors,
        capsize=3.0,
        color=colors,
        alpha=0.72,
        edgecolor="#4E545A",
        linewidth=0.7,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#4E545A"},
    )
    if np.isfinite(means).any():
        top = float(np.nanmax(means + np.nan_to_num(errors)))
        span = max(top, 1.0)
        axis.set_ylim(0.0, top + 0.12 * span)
        for bar, mean, error in zip(bars, means, errors):
            if np.isfinite(mean):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    mean + (error if np.isfinite(error) else 0.0) + 0.022 * span,
                    f"{mean:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    color="#222222",
                )
    axis.set_xticks(positions)
    axis.set_xticklabels([METHOD_LABELS[method] for method in methods], rotation=15, ha="right")
    axis.set_ylabel(label)
    _apply_publication_axis_style(axis, grid_axis="y")


def _path_length_point_range(records, report_dir: Path) -> None:
    """Journal-style continuous broken-axis forest plot for mean path length."""
    methods = _active_methods(records)
    values_by_method = {
        method: _completed_values(records, method, "path_length_m")
        for method in methods
    }
    valid = [method for method in methods if len(values_by_method[method])]
    if not valid:
        return

    means = np.asarray([float(np.mean(values_by_method[m])) for m in valid], dtype=float)
    errors = np.asarray(
        [
            0.0 if len(values_by_method[m]) == 1 else
            float(1.96 * np.std(values_by_method[m], ddof=1) / np.sqrt(len(values_by_method[m])))
            for m in valid
        ],
        dtype=float,
    )

    order = np.argsort(means)
    methods = [valid[i] for i in order]
    means = means[order]
    errors = errors[order]
    positions = np.arange(len(methods), dtype=float)

    gaps = np.diff(means)
    positive_gaps = gaps[gaps > 0.0]
    split_index = int(np.argmax(gaps) + 1) if len(gaps) else len(methods)
    median_gap = float(np.median(positive_gaps)) if len(positive_gaps) else 0.0
    use_break = bool(
        len(methods) >= 3
        and split_index < len(methods)
        and gaps[split_index - 1] > max(100.0, 4.0 * median_gap)
    )

    proposed_color = _method_color(PROPOSED_METHOD)
    baseline_color = "#365F7D"
    text_color = "#222222"

    def _nice_limits(local_means: np.ndarray, local_errors: np.ndarray, *, left_pad=0.10, right_pad=0.18):
        low = float(np.min(local_means - local_errors))
        high = float(np.max(local_means + local_errors))
        span = max(high - low, 1.0)
        lower = 10.0 * np.floor((low - left_pad * span) / 10.0)
        upper = 10.0 * np.ceil((high + right_pad * span) / 10.0)
        return float(lower), float(upper)

    if use_break:
        figure, (left_axis, right_axis) = plt.subplots(
            1,
            2,
            sharey=True,
            figsize=(8.6, 4.8),
            facecolor="white",
            gridspec_kw={"width_ratios": [4.9, 1.7]},
        )

        left_methods = methods[:split_index]
        right_methods = methods[split_index:]

        left_means = means[:split_index]
        left_errors = errors[:split_index]
        right_means = means[split_index:]
        right_errors = errors[split_index:]

        left_axis.set_xlim(*_nice_limits(left_means, left_errors, left_pad=0.12, right_pad=0.46))
        right_axis.set_xlim(*_nice_limits(right_means, right_errors, left_pad=0.18, right_pad=0.22))

        axes = (left_axis, right_axis)
        for axis in axes:
            axis.set_facecolor("white")
            axis.set_ylim(-0.6, len(methods) - 0.4)
            axis.set_yticks(positions)
            axis.grid(axis="x", color="#E4E7EB", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_visible(False)
            axis.spines["bottom"].set_color("#9AA1A8")
            axis.spines["bottom"].set_linewidth(0.9)
            axis.tick_params(axis="y", length=0, pad=7)
            axis.tick_params(axis="x", labelsize=8.8, colors="#5A6168")

        left_axis.set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=9.7)
        right_axis.tick_params(axis="y", left=False, labelleft=False)
        left_axis.invert_yaxis()

        for tick, method in zip(left_axis.get_yticklabels(), methods):
            if method == PROPOSED_METHOD:
                tick.set_color(proposed_color)
                tick.set_fontweight("semibold")
            else:
                tick.set_color("#41474D")

        for idx, method in enumerate(methods):
            color = proposed_color if method == PROPOSED_METHOD else baseline_color
            target_axis = left_axis if idx < split_index else right_axis
            target_axis.errorbar(
                means[idx],
                positions[idx],
                xerr=errors[idx],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.45 if method == PROPOSED_METHOD else 1.20,
                capsize=3.2,
                capthick=1.1,
                markersize=6.6 if method == PROPOSED_METHOD else 5.8,
                markeredgecolor="white",
                markeredgewidth=0.85,
                zorder=5,
            )
            target_axis.annotate(
                f"{means[idx]:.1f}",
                xy=(means[idx] + errors[idx], positions[idx]),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9.0,
                color=color if method == PROPOSED_METHOD else text_color,
                clip_on=False,
                zorder=6,
            )

        left_axis.spines["right"].set_visible(False)
        right_axis.spines["left"].set_visible(False)
        diagonal = 0.012
        mark_style = dict(color="#6B737A", clip_on=False, linewidth=1.1)
        left_axis.plot(
            (1 - diagonal, 1 + diagonal),
            (-diagonal, diagonal),
            transform=left_axis.transAxes,
            **mark_style,
        )
        left_axis.plot(
            (1 - diagonal, 1 + diagonal),
            (1 - diagonal, 1 + diagonal),
            transform=left_axis.transAxes,
            **mark_style,
        )
        right_axis.plot(
            (-diagonal, diagonal),
            (-diagonal, diagonal),
            transform=right_axis.transAxes,
            **mark_style,
        )
        right_axis.plot(
            (-diagonal, diagonal),
            (1 - diagonal, 1 + diagonal),
            transform=right_axis.transAxes,
            **mark_style,
        )

        left_ticks = np.arange(
            20.0 * np.ceil(left_axis.get_xlim()[0] / 20.0),
            left_axis.get_xlim()[1] + 1e-9,
            20.0,
        )
        left_axis.set_xticks(left_ticks)

        right_span = right_axis.get_xlim()[1] - right_axis.get_xlim()[0]
        if right_span <= 160:
            right_step = 50.0
        else:
            right_step = 100.0
        right_ticks = np.arange(
            right_step * np.ceil(right_axis.get_xlim()[0] / right_step),
            right_axis.get_xlim()[1] + 1e-9,
            right_step,
        )
        right_axis.set_xticks(right_ticks)

        figure.supxlabel("Mean path length (m)", fontsize=10.5, y=0.06)
        figure.subplots_adjust(left=0.23, right=0.98, bottom=0.16, top=0.98, wspace=0.04)

    else:
        figure, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
        axis.set_facecolor("white")
        axis.set_xlim(*_nice_limits(means, errors, left_pad=0.12, right_pad=0.24))
        axis.set_ylim(-0.6, len(methods) - 0.4)
        axis.set_yticks(positions)
        axis.set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=9.7)
        axis.invert_yaxis()
        axis.set_xlabel("Mean path length (m)")
        axis.grid(axis="x", color="#E4E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#9AA1A8")
        axis.spines["bottom"].set_linewidth(0.9)
        axis.tick_params(axis="y", length=0, pad=7)
        axis.tick_params(axis="x", labelsize=8.8, colors="#5A6168")
        axis.set_xticks(
            np.arange(
                20.0 * np.ceil(axis.get_xlim()[0] / 20.0),
                axis.get_xlim()[1] + 1e-9,
                20.0,
            )
        )

        for tick, method in zip(axis.get_yticklabels(), methods):
            if method == PROPOSED_METHOD:
                tick.set_color(proposed_color)
                tick.set_fontweight("semibold")
            else:
                tick.set_color("#41474D")

        for idx, method in enumerate(methods):
            color = proposed_color if method == PROPOSED_METHOD else baseline_color
            axis.errorbar(
                means[idx],
                positions[idx],
                xerr=errors[idx],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.45 if method == PROPOSED_METHOD else 1.20,
                capsize=3.2,
                capthick=1.1,
                markersize=6.6 if method == PROPOSED_METHOD else 5.8,
                markeredgecolor="white",
                markeredgewidth=0.85,
                zorder=5,
            )
            axis.annotate(
                f"{means[idx]:.1f}",
                xy=(means[idx] + errors[idx], positions[idx]),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9.0,
                color=color if method == PROPOSED_METHOD else text_color,
                clip_on=False,
                zorder=6,
            )

        figure.subplots_adjust(left=0.25, right=0.98, bottom=0.16, top=0.98)

    figure.savefig(
        report_dir / "formal_path_length.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _paired_path_length_axis(axis, records, methods: list[str]) -> None:
    """Show paired CA-minus-baseline length differences by UAV/trial."""
    ca = {
        (int(row["uav_count"]), int(row["trial"]), int(row["drone_id"])): _float(row, "path_length_m")
        for row in records
        if row["method"] == "ca_diffusion_es" and _float(row, "success") >= 0.5
    }
    comparisons = [method for method in methods if method != "ca_diffusion_es"]
    values_by_method = []
    labels = []
    for method in comparisons:
        deltas = []
        for row in records:
            if row["method"] != method or _float(row, "success") < 0.5:
                continue
            key = (int(row["uav_count"]), int(row["trial"]), int(row["drone_id"]))
            if key in ca:
                deltas.append(ca[key] - _float(row, "path_length_m"))
        values_by_method.append(np.asarray(deltas, dtype=float))
        labels.append(f"CA - {METHOD_LABELS[method]}")
    plot = axis.boxplot(values_by_method, patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#222222"})
    for patch in plot["boxes"]:
        patch.set_facecolor(METHOD_COLORS["ca_diffusion_es"])
        patch.set_alpha(0.72)
    axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.1)
    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(labels)
    axis.set_ylabel("Length difference (m)")
    axis.grid(axis="y", alpha=0.25)
    for index, values in enumerate(values_by_method, start=1):
        if len(values):
            axis.text(index, float(np.mean(values)), f"mean {np.mean(values):+.1f} m",
                      ha="center", va="bottom", fontsize=9)


def _save_standalone_metric_figures(records, report_dir: Path, methods: list[str]) -> None:
    """Save the main quantitative metrics as consistent standalone figures."""
    _path_length_point_range(records, report_dir)

    standalone = [
        ("formal_mean_pathloss.png", "mean_pathloss_db", "Mean path loss (dB)"),
        ("formal_communication_cost.png", "communication_cost", "Communication cost"),
    ]
    for filename, key, label in standalone:
        fig, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
        _boxplot_metric(axis, records, methods, key, label)
        fig.tight_layout()
        fig.savefig(report_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    ecdf_specs = [
        ("formal_worst_pathloss.png", _plot_worst_pathloss_ecdf),
        ("formal_min_clearance.png", _plot_clearance_survival),
        ("formal_turn_p95.png", _plot_turn_ecdf),
        ("formal_outage_ratio.png", _plot_outage_heatmap),
    ]
    for filename, plotter in ecdf_specs:
        fig, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
        plotter(axis, records, methods)
        fig.tight_layout()
        fig.savefig(report_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _metric_boxplots(records, report_dir: Path):
    methods = _active_methods(records)
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.4), facecolor="white")

    axes[0, 0].set_title("(a) Path length", fontsize=10.5, pad=6)
    _bar_metric(axes[0, 0], records, methods, "path_length_m", "Path length (m)")

    axes[0, 1].set_title("(b) Mean path loss", fontsize=10.5, pad=6)
    _boxplot_metric(axes[0, 1], records, methods, "mean_pathloss_db", "Mean path loss (dB)")

    axes[0, 2].set_title("(c) Worst path loss", fontsize=10.5, pad=6)
    _plot_worst_pathloss_ecdf(axes[0, 2], records, methods)

    axes[1, 0].set_title("(d) Minimum clearance", fontsize=10.5, pad=6)
    _plot_clearance_survival(axes[1, 0], records, methods)

    axes[1, 1].set_title("(e) TurnP95", fontsize=10.5, pad=6)
    _plot_turn_ecdf(axes[1, 1], records, methods)

    axes[1, 2].set_title("(f) Outage ratio", fontsize=10.5, pad=6)
    _plot_outage_heatmap(axes[1, 2], records, methods)

    fig.tight_layout()
    fig.savefig(report_dir / "formal_main_metrics.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _scalability(records, report_dir: Path):
    methods = _active_methods(records)
    counts = sorted({int(row["uav_count"]) for row in records})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), facecolor="white")
    offsets = np.linspace(-0.12, 0.12, len(methods))

    for method_index, method in enumerate(methods):
        runtimes = []
        runtime_std = []
        success = []
        for count in counts:
            subset = [
                row for row in records
                if row["method"] == method and int(row["uav_count"]) == count
            ]
            by_trial = {}
            for row in subset:
                by_trial.setdefault(int(row["trial"]), []).append(_float(row, "scenario_runtime_s"))
            trial_runtimes = np.asarray(
                [float(np.mean(values)) for values in by_trial.values()],
                dtype=float,
            ) if by_trial else np.asarray([], dtype=float)
            runtimes.append(float(np.mean(trial_runtimes)) if len(trial_runtimes) else np.nan)
            runtime_std.append(float(np.std(trial_runtimes)) if len(trial_runtimes) else np.nan)
            success.append(float(np.mean([_float(row, "success") for row in subset])) if subset else np.nan)

        axes[0].errorbar(
            counts,
            runtimes,
            yerr=runtime_std,
            marker="o",
            capsize=3,
            linewidth=_method_linewidth(method, proposed=2.0, baseline=1.6),
            markersize=5,
            label=METHOD_LABELS[method],
            color=_method_color(method),
        )
        axes[1].plot(
            np.asarray(counts, dtype=float) + offsets[method_index],
            success,
            marker="o",
            linewidth=_method_linewidth(method, proposed=2.0, baseline=1.6),
            markersize=5,
            label=METHOD_LABELS[method],
            color=_method_color(method),
        )

    axes[0].set_title("(a) Scenario runtime", fontsize=10.5, pad=6)
    axes[1].set_title("(b) Success rate", fontsize=10.5, pad=6)
    axes[0].set_ylabel("Scenario runtime (s)")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(0.0, 1.02)
    for axis in axes:
        axis.set_xlabel("Number of UAVs")
        axis.set_xticks(counts)
        _apply_publication_axis_style(axis, grid_axis="y")
    legend = axes[0].legend(loc="upper left", frameon=False)
    _finalize_legend(legend)
    fig.tight_layout()
    fig.savefig(report_dir / "formal_scalability.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _trial_hv_curves(
    method_root: Path,
    method: str,
    evaluation_grid: np.ndarray,
    hypervolume_reference: tuple[float, float],
) -> np.ndarray:
    """Aggregate drone histories within each trial before uncertainty analysis."""
    trial_curves = []
    for trial_dir in sorted((method_root / method / "N5").glob("trial_*")):
        drone_curves = []
        for path in sorted(trial_dir.glob("history_drone*.csv")):
            rows = _load_csv(path)
            if not rows:
                continue
            evaluations = np.asarray(
                [_float(row, "evaluations") for row in rows], dtype=float
            )
            hypervolume = np.asarray(
                [_float(row, "hypervolume") for row in rows], dtype=float
            )
            finite = np.isfinite(evaluations) & np.isfinite(hypervolume)
            evaluations = evaluations[finite]
            hypervolume = hypervolume[finite]
            if not len(evaluations):
                continue
            order = np.argsort(evaluations)
            evaluations = evaluations[order]
            normalized = np.asarray([
                _normalized_hv(value, hypervolume_reference)
                for value in hypervolume[order]
            ])
            normalized = np.maximum.accumulate(normalized)
            left = normalized[0] if method == "astar" else 0.0
            drone_curves.append(
                np.interp(
                    evaluation_grid, evaluations, normalized,
                    left=left, right=normalized[-1],
                )
            )
        if drone_curves:
            trial_curves.append(np.mean(np.asarray(drone_curves), axis=0))
    return np.asarray(trial_curves, dtype=float)


def _convergence(
    method_roots: dict[str, Path],
    report_dir: Path,
    hypervolume_reference: tuple[float, float],
) -> None:
    """Compact convergence plot with larger usable plotting area."""
    figure, axis = plt.subplots(figsize=(8.0, 5.4), facecolor="white")
    evaluation_grid = np.arange(0.0, 4000.0 + 1e-9, 100.0)
    methods = ("astar_nsga2", "nsga2", "diffusion_only", "ca_diffusion_es")

    global_top = 0.0
    for method in methods:
        matrix = _trial_hv_curves(
            method_roots[method], method, evaluation_grid, hypervolume_reference
        )
        if not len(matrix):
            continue

        mean_curve = np.mean(matrix, axis=0)
        if matrix.shape[0] == 1:
            ci = np.zeros_like(mean_curve)
        else:
            ci = 1.96 * np.std(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])

        global_top = max(global_top, float(np.max(mean_curve + ci)))
        color = _method_color(method)
        axis.plot(
            evaluation_grid,
            mean_curve,
            color=color,
            linewidth=_method_linewidth(method, proposed=2.3, baseline=1.8),
            label=METHOD_LABELS[method],
        )
        axis.fill_between(
            evaluation_grid,
            np.clip(mean_curve - ci, 0.0, 1.0),
            np.clip(mean_curve + ci, 0.0, 1.0),
            color=color,
            alpha=0.12 if method == PROPOSED_METHOD else 0.08,
            linewidth=0,
        )

    axis.set_xlabel("Evaluations")
    axis.set_ylabel("Normalized feasible hypervolume")
    axis.set_xlim(evaluation_grid[0], evaluation_grid[-1])

    # Reduce the empty space above the curves while keeping a formal axis range.
    upper = max(0.35, 0.05 * np.ceil((global_top + 0.03) / 0.05))
    upper = min(1.0, upper)
    axis.set_ylim(0.0, upper)

    _apply_publication_axis_style(axis, grid_axis="y")
    legend = axis.legend(loc="lower right", frameon=False, handlelength=2.4)
    _finalize_legend(legend)

    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.97)
    figure.savefig(
        report_dir / "formal_convergence.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _final_hypervolume(
    method_roots: dict[str, Path],
    report_dir: Path,
    hypervolume_reference: tuple[float, float],
) -> None:
    methods = []
    data = []

    for method in METHOD_LABELS:
        values = []
        root = method_roots[method] / method
        for path in root.glob("N*/trial_*/history_drone*.csv"):
            rows = _load_csv(path)
            if not rows:
                continue
            value = _float(rows[-1], "hypervolume")
            if np.isfinite(value):
                values.append(_normalized_hv(value, hypervolume_reference))
        if values:
            methods.append(method)
            data.append(np.asarray(values, dtype=float))

    if not data:
        return

    figure, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    plot = axis.boxplot(
        data,
        patch_artist=True,
        widths=0.62,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#222222",
            "markersize": 5,
            "markeredgewidth": 0.8,
        },
        medianprops={"color": "#222222", "linewidth": 1.3},
        whiskerprops={"color": "#666B70", "linewidth": 1.0},
        capprops={"color": "#666B70", "linewidth": 1.0},
        boxprops={"edgecolor": "#666B70", "linewidth": 0.9},
        flierprops={"marker": ".", "markersize": 2.2, "alpha": 0.18},
    )
    for patch, method in zip(plot["boxes"], methods):
        patch.set_facecolor(_method_color(method))
        patch.set_alpha(0.55)

    axis.set_xticks(range(1, len(methods) + 1))
    axis.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
    axis.set_ylabel("Final normalized feasible hypervolume")
    axis.set_ylim(0.0, 1.02)
    _apply_publication_axis_style(axis, grid_axis="y")
    figure.tight_layout()
    figure.savefig(report_dir / "formal_final_hypervolume.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _pareto(method_roots: dict[str, Path], report_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True, facecolor="white")
    missing_feasible = []

    for method in ("astar", "astar_nsga2", "nsga2", "diffusion_only", "ca_diffusion_es"):
        path = method_roots[method] / method / "N5" / "trial_00" / "pareto_drone0.csv"
        if not path.exists():
            continue

        rows = [
            row for row in _load_csv(path)
            if _float(row, "constraint_violation") <= 1e-9
        ]
        if not rows:
            missing_feasible.append(METHOD_LABELS[method])
            continue

        length = np.asarray([_float(row, "length_m") for row in rows], dtype=float)
        communication = np.asarray([_float(row, "communication_cost") for row in rows], dtype=float)
        finite = np.isfinite(length) & np.isfinite(communication)
        length = length[finite]
        communication = communication[finite]
        if len(length) == 0:
            missing_feasible.append(METHOD_LABELS[method])
            continue

        order = np.argsort(length)
        color = _method_color(method)
        for axis in axes:
            axis.plot(
                length[order],
                communication[order],
                marker="o",
                markersize=3.6 if method == PROPOSED_METHOD else 3.2,
                linewidth=_method_linewidth(method, proposed=1.8, baseline=1.4),
                label=METHOD_LABELS[method],
                color=color,
            )

    axes[0].set_title("(a) Full range", fontsize=10.5, pad=6)
    axes[1].set_title("(b) Zoomed view", fontsize=10.5, pad=6)
    axes[1].set_xlim(250.0, 650.0)
    for axis in axes:
        axis.set_xlabel("Path length (m)")
        _apply_publication_axis_style(axis, grid_axis="both")
    axes[0].set_ylabel("Communication cost")
    legend = axes[0].legend(loc="best", frameon=False, handlelength=2.0)
    _finalize_legend(legend)

    if missing_feasible:
        fig.text(
            0.5,
            0.01,
            "No feasible front: " + ", ".join(missing_feasible),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8B3A3A",
        )

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(report_dir / "formal_pareto_representative.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _trajectories(
    method_roots: dict[str, Path],
    report_dir: Path,
    legacy_root: Path,
):
    """Publication-style five-method trajectory comparison."""
    if str(legacy_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(legacy_root))

    from environment import Environment
    from matplotlib.lines import Line2D

    env = Environment()

    methods = [
        method
        for method in METHOD_LABELS
        if (method_roots[method] / method / "N5" / "trial_00").exists()
    ]
    if not methods:
        return

    # Journal-oriented 3+2 small-multiple layout.
    figure = plt.figure(figsize=(11.2, 7.2), facecolor="white")
    grid = figure.add_gridspec(
        2,
        6,
        left=0.075,
        right=0.985,
        bottom=0.145,
        top=0.955,
        wspace=0.28,
        hspace=0.24,
    )

    panel_locations = [
        grid[0, 0:2],
        grid[0, 2:4],
        grid[0, 4:6],
        grid[1, 1:3],
        grid[1, 3:5],
    ]
    axes = [
        figure.add_subplot(panel_locations[i])
        for i in range(min(len(methods), 5))
    ]

    users = np.asarray(getattr(env, "ues", []), dtype=float)

    # Keep UAV identity consistent across all methods.
    path_colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
    ]
    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for panel_index, (axis, method) in enumerate(zip(axes, methods)):
        axis.set_facecolor("white")

        success_by_drone = {}
        records_path = method_roots[method] / "records.csv"
        if records_path.exists():
            for row in _load_csv(records_path):
                if (
                    row["method"] == method
                    and int(row["uav_count"]) == 5
                    and int(row["trial"]) == 0
                ):
                    success_by_drone[int(row["drone_id"])] = (
                        _float(row, "success") >= 0.5
                    )

        # Buildings stay in the background: white/light neutral fill,
        # thin outline, no grey wash.
        for building in env.buildings:
            axis.add_patch(
                plt.Rectangle(
                    (
                        building.x_bounds[0],
                        building.y_bounds[0],
                    ),
                    building.x_bounds[1] - building.x_bounds[0],
                    building.y_bounds[1] - building.y_bounds[0],
                    facecolor="#F7F8F9",
                    edgecolor="#C8CDD2",
                    linewidth=0.55,
                    alpha=1.0,
                    zorder=1,
                )
            )

        # Ground users: compact neutral marker.
        if len(users):
            axis.scatter(
                users[:, 0],
                users[:, 1],
                marker="^",
                s=22,
                facecolor="#44484C",
                edgecolors="white",
                linewidths=0.50,
                alpha=0.92,
                zorder=5,
            )

        # UAV trajectories.
        for drone_id in range(5):
            path = (
                method_roots[method]
                / method
                / "N5"
                / "trial_00"
                / f"trajectory_drone{drone_id}.csv"
            )
            if not path.exists():
                continue

            trajectory_rows = _load_csv(path)
            points = np.asarray(
                [
                    [
                        _float(row, "x_m"),
                        _float(row, "y_m"),
                        _float(row, "z_m"),
                    ]
                    for row in trajectory_rows
                ],
                dtype=float,
            )

            is_feasible = success_by_drone.get(drone_id, True)

            axis.plot(
                points[:, 0],
                points[:, 1],
                linewidth=1.85,
                linestyle="-" if is_feasible else "--",
                alpha=0.98 if is_feasible else 0.72,
                color=path_colors[drone_id],
                solid_capstyle="round",
                dash_capstyle="round",
                zorder=4,
            )

            start = points[0]
            end = points[-1]

            # Takeoff: open circle. Landing: filled square.
            axis.scatter(
                start[0],
                start[1],
                marker="o",
                s=30,
                facecolor="white",
                edgecolor=path_colors[drone_id],
                linewidth=1.15,
                zorder=7,
            )
            axis.scatter(
                end[0],
                end[1],
                marker="s",
                s=28,
                facecolor=path_colors[drone_id],
                edgecolor="white",
                linewidth=0.65,
                zorder=7,
            )

        # No explanatory subtitles inside the figure.
        axis.set_title(
            f"{panel_letters[panel_index]} {METHOD_LABELS[method]}",
            fontsize=10.2,
            fontweight="semibold",
            pad=8,
            color="#20252B",
        )

        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-120, 120)
        axis.set_ylim(-120, 120)

        # Restrained ticks/grid suitable for a journal figure.
        axis.set_xticks([-100, 0, 100])
        axis.set_yticks([-100, 0, 100])
        axis.grid(
            color="#E7EAED",
            linewidth=0.50,
            alpha=0.72,
        )
        axis.set_axisbelow(True)

        axis.tick_params(
            axis="both",
            labelsize=7.8,
            length=2.4,
            width=0.7,
            colors="#555B60",
        )

        # Shared labels: suppress repeated inner tick labels.
        row = 0 if panel_index < 3 else 1
        col_in_row = panel_index if panel_index < 3 else panel_index - 3

        if panel_index not in (0, 3):
            axis.tick_params(axis="y", labelleft=False)
        if row == 0:
            axis.tick_params(axis="x", labelbottom=False)

        for spine in axis.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#AEB4B9")

    # Shared axis labels only.
    figure.text(
        0.50,
        0.105,
        "X position (m)",
        ha="center",
        va="center",
        fontsize=9.7,
        color="#20252B",
    )
    figure.text(
        0.025,
        0.55,
        "Y position (m)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.7,
        color="#20252B",
    )

    # Compact common legend with only essential visual semantics.
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=path_colors[index],
            linewidth=1.8,
            label=f"UAV {index + 1}",
        )
        for index in range(5)
    ]
    legend_handles.extend(
        [
            Line2D(
                [0], [0],
                marker="^",
                linestyle="None",
                markersize=5.8,
                markerfacecolor="#44484C",
                markeredgecolor="white",
                label="UE",
            ),
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=5.5,
                markerfacecolor="white",
                markeredgecolor="#555555",
                label="Takeoff",
            ),
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                markersize=5.5,
                markerfacecolor="#555555",
                markeredgecolor="white",
                label="Landing",
            ),
            Line2D(
                [0], [0],
                color="#555555",
                linewidth=1.5,
                linestyle="--",
                label="Infeasible",
            ),
        ]
    )

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=9,
        frameon=False,
        fontsize=7.8,
        handlelength=1.7,
        columnspacing=0.85,
        handletextpad=0.45,
    )

    figure.savefig(
        report_dir / "formal_trajectory_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

def _cylinder_surface(
    center: np.ndarray,
    direction: np.ndarray,
    radius: float,
    height: float,
):
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        direction = np.asarray([0.0, 0.0, 1.0])
    else:
        direction = direction / norm

    reference = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(direction[0]) < 0.9
        else np.asarray([0.0, 1.0, 0.0])
    )
    first = np.cross(direction, reference)
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-9:
        reference = np.asarray([0.0, 0.0, 1.0])
        first = np.cross(direction, reference)
        first_norm = np.linalg.norm(first)
    first = first / (first_norm + 1e-12)
    second = np.cross(direction, first)
    second = second / (np.linalg.norm(second) + 1e-12)

    axial = np.linspace(0.0, float(height), 10)
    angle = np.linspace(0.0, 2.0 * np.pi, 20)
    angle_grid, axial_grid = np.meshgrid(angle, axial)
    local = np.stack(
        [
            float(radius) * np.cos(angle_grid),
            float(radius) * np.sin(angle_grid),
            axial_grid,
        ]
    )
    rotation = np.column_stack([first, second, direction])
    rotated = np.tensordot(rotation, local, axes=1)
    translated = rotated + np.asarray(center, dtype=float).reshape(3, 1, 1)
    return translated[0], translated[1], translated[2]


def _sphere_surface(center: np.ndarray, radius: float):
    longitude = np.linspace(0.0, 2.0 * np.pi, 12)
    latitude = np.linspace(0.0, np.pi, 8)
    center = np.asarray(center, dtype=float)
    sphere_x = center[0] + float(radius) * np.outer(
        np.cos(longitude), np.sin(latitude)
    )
    sphere_y = center[1] + float(radius) * np.outer(
        np.sin(longitude), np.sin(latitude)
    )
    sphere_z = center[2] + float(radius) * np.outer(
        np.ones_like(longitude), np.cos(latitude)
    )
    return sphere_x, sphere_y, sphere_z


def _plot_segmented_pipe(axis, points, color, radius: float, alpha: float = 1.0):
    trajectory = [np.asarray(point, dtype=float) for point in points]
    if len(trajectory) < 2:
        return

    start_x, start_y, start_z = _cylinder_surface(
        trajectory[0],
        trajectory[1] - trajectory[0],
        radius,
        radius * 2.0,
    )
    axis.plot_surface(start_x, start_y, start_z, color=color, alpha=alpha)

    end_x, end_y, end_z = _cylinder_surface(
        trajectory[-1],
        trajectory[-1] - trajectory[-2],
        radius,
        radius * 2.0,
    )
    axis.plot_surface(end_x, end_y, end_z, color=color, alpha=alpha)

    for start, end in zip(trajectory[:-1], trajectory[1:]):
        direction = end - start
        length = np.linalg.norm(direction)
        if length <= 1e-12:
            continue
        pipe_x, pipe_y, pipe_z = _cylinder_surface(
            start,
            direction / length,
            radius,
            length,
        )
        axis.plot_surface(pipe_x, pipe_y, pipe_z, color=color, alpha=alpha)

    # Opaque spheres cover the open, differently oriented cylinder ends at turns.
    for point in trajectory:
        joint_x, joint_y, joint_z = _sphere_surface(point, radius)
        axis.plot_surface(
            joint_x,
            joint_y,
            joint_z,
            color=color,
            alpha=1.0,
            linewidth=0,
            antialiased=False,
            shade=True,
        )


def _plot_ground_ues_3d(axis, users: np.ndarray) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    users = np.asarray(users, dtype=float)
    if len(users) == 0:
        return

    # Render UEs as small dark neutral pyramids so they are clearly distinct
    # from the takeoff markers used for UAV trajectories.
    half_width = 1.55
    height = 3.8
    azimuth = np.deg2rad(-55.0)
    horizontal = np.asarray([-np.sin(azimuth), np.cos(azimuth)], dtype=float)
    triangles = []
    for user in users:
        center = np.asarray(user[:2], dtype=float)
        first = center - half_width * horizontal
        second = center + half_width * horizontal
        triangles.append(
            [
                (first[0], first[1], 0.0),
                (second[0], second[1], 0.0),
                (center[0], center[1], height),
            ]
        )

    collection = Poly3DCollection(
        triangles,
        facecolors="#4A4F55",
        edgecolors="white",
        linewidths=0.28,
        alpha=0.96,
    )
    collection.set_sort_zpos(0.0)
    axis.add_collection3d(collection)


def _ca_trajectories_3d(
    method_roots: dict[str, Path],
    report_dir: Path,
    legacy_root: Path,
    *,
    pipe_radius: float = 3.5,
):
    if str(legacy_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(legacy_root))
    from environment import Environment
    from matplotlib.lines import Line2D

    env = Environment()
    env.starts = list(env.starts[:5])
    env.ends = list(env.ends[:5])
    rebuild = getattr(env, "_build_vertiports_from_starts_ends", None)
    if callable(rebuild):
        env.vertiports = rebuild(z_min=0.0, z_max=40.0, nd=2)

    trajectory_root = (
        method_roots["ca_diffusion_es"]
        / "ca_diffusion_es"
        / "N5"
        / "trial_00"
    )
    fig = plt.figure(figsize=(14, 10))
    axis = fig.add_subplot(111, projection="3d")
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:red",
    ]

    for building in env.buildings:
        vertices = np.asarray(building.vertices, dtype=float)
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        axis.bar3d(
            minimum[0],
            minimum[1],
            minimum[2],
            maximum[0] - minimum[0],
            maximum[1] - minimum[1],
            maximum[2] - minimum[2],
            color="gray",
            alpha=0.4,
            edgecolor="k",
        )

    if getattr(env, "ues", None):
        users = np.asarray(env.ues, dtype=float)
        _plot_ground_ues_3d(axis, users)

    paths = []
    for drone_id in range(5):
        path = trajectory_root / f"trajectory_drone{drone_id}.csv"
        if not path.exists():
            continue
        rows = _load_csv(path)
        points = np.asarray(
            [
                [_float(row, "x_m"), _float(row, "y_m"), _float(row, "z_m")]
                for row in rows
            ],
            dtype=float,
        )
        paths.append(points)
        _plot_segmented_pipe(
            axis,
            points,
            colors[drone_id],
            float(pipe_radius),
            alpha=1.0,
        )
    if not paths:
        plt.close(fig)
        return

    for vertiport in getattr(env, "vertiports", []):
        drone_id = getattr(vertiport, "drone_id", None)
        color = (
            colors[int(drone_id) % len(colors)]
            if drone_id is not None
            else "#7e57c2"
        )
        z_min = max(0.0, float(getattr(vertiport, "z_min", 0.0)))
        z_max = min(40.0, float(getattr(vertiport, "z_max", 10.0)))
        cylinder_x, cylinder_y, cylinder_z = _cylinder_surface(
            (vertiport.x, vertiport.y, z_min),
            (0.0, 0.0, 1.0),
            float(pipe_radius),
            max(0.1, z_max - z_min),
        )
        axis.plot_surface(
            cylinder_x,
            cylinder_y,
            cylinder_z,
            color=color,
            alpha=1.0,
            edgecolor="none",
        )

    # Vertiports are rendered after the paths; redraw endpoint joints last so
    # each horizontal corridor visibly intersects its vertical access column.
    for drone_id, points in enumerate(paths):
        for endpoint in (points[0], points[-1]):
            joint_x, joint_y, joint_z = _sphere_surface(endpoint, pipe_radius)
            axis.plot_surface(
                joint_x,
                joint_y,
                joint_z,
                color=colors[drone_id],
                alpha=1.0,
                linewidth=0,
                antialiased=False,
                shade=True,
            )
    start_labels = {
        0: "Takeoff Zone 1",
        1: "Takeoff Zone 2",
    }
    end_labels = {
        0: "Landing Zone 1",
        1: "Landing Zone 2",
        2: "Landing Zone 3",
        3: "Landing Zone 4",
        4: "Landing Zone 5",
    }
    label_box = {
        "facecolor": "white",
        "alpha": 0.65,
        "edgecolor": "none",
        "pad": 1.2,
    }
    station_labeled = False
    for drone_id, points in enumerate(paths):
        color = colors[drone_id]
        start = np.asarray(points[0], dtype=float)
        end = np.asarray(points[-1], dtype=float)
        axis.scatter(*start, color="g", s=150, marker="^", label="_nolegend_")
        axis.scatter(*end, color="r", s=150, marker="v", label="_nolegend_")
        start_text = (start[0] + 2.0, start[1] + 2.0, 38.8)
        end_text = (end[0] + 2.0, end[1] + 2.0, 38.8)
        if drone_id in {2, 3, 4}:
            if not station_labeled:
                axis.text(
                    *start_text,
                    "UAV Takeoff Station",
                    fontsize=10,
                    color="#222222",
                    ha="left",
                    va="bottom",
                    bbox=label_box,
                )
                station_labeled = True
        else:
            axis.text(
                *start_text,
                start_labels[drone_id],
                fontsize=10,
                color=color,
                ha="left",
                va="bottom",
                bbox=label_box,
            )
        axis.text(
            *end_text,
            end_labels[drone_id],
            fontsize=10,
            color=color,
            ha="left",
            va="bottom",
            bbox=label_box,
        )

    handles = [
        Line2D([0], [0], color=colors[index], lw=2.6, label=f"Drone {index + 1}")
        for index in range(5)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="^",
            markersize=8,
            markerfacecolor="tab:green",
            markeredgecolor="k",
            label="UEs",
        )
    )
    axis.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        fontsize=10,
        frameon=True,
    )

    axis.set_xlim(-120.0, 120.0)
    axis.set_ylim(-120.0, 120.0)
    axis.set_zlim(0.0, 40.0)
    axis.set_zticks([0, 20, 40])
    axis.set_box_aspect((240.0, 240.0, 40.0))
    axis.set_xlabel("X (m)", fontsize=12, labelpad=15)
    axis.set_ylabel("Y (m)", fontsize=12, labelpad=15)
    axis.set_zlabel("Height (m)", fontsize=12, labelpad=15)
    try:
        axis.zaxis.set_label_position("left")
        axis.zaxis.set_label_coords(-0.08, 0.5)
    except (AttributeError, TypeError, ValueError):
        pass
    axis.view_init(elev=30, azim=-55)
    plt.tight_layout()
    fig.savefig(
        report_dir / "formal_ca_trajectory_3d.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _load_ca_paths_by_scale(
    method_roots: dict[str, Path],
    uav_count: int,
    *,
    trial: int = 0,
) -> list[tuple[int, np.ndarray]]:
    trajectory_root = (
        method_roots["ca_diffusion_es"]
        / "ca_diffusion_es"
        / f"N{uav_count}"
        / f"trial_{trial:02d}"
    )
    paths = []
    for drone_id in range(uav_count):
        path = trajectory_root / f"trajectory_drone{drone_id}.csv"
        if not path.exists():
            continue
        rows = _load_csv(path)
        points = np.asarray(
            [
                [_float(row, "x_m"), _float(row, "y_m"), _float(row, "z_m")]
                for row in rows
            ],
            dtype=float,
        )
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) >= 2:
            paths.append((drone_id, points))
    return paths



def _draw_ca_3d_panel(
    axis,
    env,
    paths: list[tuple[int, np.ndarray]],
    uav_count: int,
    colors: list[str],
    pipe_radius: float,
) -> None:
    from matplotlib.lines import Line2D

    for building in env.buildings:
        vertices = np.asarray(building.vertices, dtype=float)
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        axis.bar3d(
            minimum[0],
            minimum[1],
            minimum[2],
            maximum[0] - minimum[0],
            maximum[1] - minimum[1],
            maximum[2] - minimum[2],
            color="gray",
            alpha=0.4,
            edgecolor="k",
        )

    if getattr(env, "ues", None):
        users = np.asarray(env.ues, dtype=float)
        _plot_ground_ues_3d(axis, users)

    for drone_id, points in paths:
        _plot_segmented_pipe(
            axis,
            points,
            colors[drone_id],
            float(pipe_radius),
            alpha=1.0,
        )

    for vertiport in getattr(env, "vertiports", []):
        drone_id = getattr(vertiport, "drone_id", None)
        color = (
            colors[int(drone_id) % len(colors)]
            if drone_id is not None
            else "#7e57c2"
        )
        z_min = max(0.0, float(getattr(vertiport, "z_min", 0.0)))
        z_max = min(40.0, float(getattr(vertiport, "z_max", 10.0)))
        cylinder_x, cylinder_y, cylinder_z = _cylinder_surface(
            (vertiport.x, vertiport.y, z_min),
            (0.0, 0.0, 1.0),
            float(pipe_radius),
            max(0.1, z_max - z_min),
        )
        axis.plot_surface(
            cylinder_x,
            cylinder_y,
            cylinder_z,
            color=color,
            alpha=1.0,
            edgecolor="none",
        )

    for drone_id, points in paths:
        for endpoint in (points[0], points[-1]):
            joint_x, joint_y, joint_z = _sphere_surface(endpoint, pipe_radius)
            axis.plot_surface(
                joint_x,
                joint_y,
                joint_z,
                color=colors[drone_id],
                alpha=1.0,
                linewidth=0,
                antialiased=False,
                shade=True,
            )

    # Remove start/end markers and keep only the essentials, matching the 2D figure
    # more closely while preserving the 3D corridor rendering.
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[index],
            lw=2.6,
            label=f"UAV {index + 1}",
        )
        for index in range(uav_count)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="^",
            markersize=6.4,
            markerfacecolor="#4A4F55",
            markeredgecolor="white",
            label="UE",
        )
    )
    axis.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        fontsize=8,
        frameon=True,
    )

    axis.set_xlim(-120.0, 120.0)
    axis.set_ylim(-120.0, 120.0)
    axis.set_zlim(0.0, 40.0)
    axis.set_zticks([0, 20, 40])
    axis.set_box_aspect((240.0, 240.0, 40.0))
    axis.set_xlabel("X (m)", fontsize=12, labelpad=15)
    axis.set_ylabel("Y (m)", fontsize=12, labelpad=15)
    axis.set_zlabel("Height (m)", fontsize=12, labelpad=15)
    axis.set_title("", pad=0)
    try:
        axis.zaxis.set_label_position("left")
        axis.zaxis.set_label_coords(-0.08, 0.5)
    except (AttributeError, TypeError, ValueError):
        pass
    axis.view_init(elev=30, azim=-55)


def _ca_trajectories_by_scale(
    method_roots: dict[str, Path],
    report_dir: Path,
    legacy_root: Path,
    *,
    pipe_radius: float = 2.4,
) -> None:
    """Unified Fig. 2 / Fig. 3 export with the original 3D style and tighter layout."""
    if str(legacy_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(legacy_root))
    from environment import Environment
    from matplotlib.lines import Line2D

    env = Environment()
    counts = (5, 7, 9)
    panel_labels = ["(a)", "(b)", "(c)"]
    paths_by_count = {count: _load_ca_paths_by_scale(method_roots, count) for count in counts}
    colors = _uav_palette(9)

    figure_2d, axes_2d = plt.subplots(
        1, len(counts), figsize=(14.8, 5.3), sharex=True, sharey=True, facecolor="white",
    )
    users = np.asarray(getattr(env, "ues", []), dtype=float)

    for axis, count, panel_label in zip(axes_2d, counts, panel_labels):
        for building in env.buildings:
            axis.add_patch(
                plt.Rectangle(
                    (building.x_bounds[0], building.y_bounds[0]),
                    building.x_bounds[1] - building.x_bounds[0],
                    building.y_bounds[1] - building.y_bounds[0],
                    facecolor=BUILDING_FACE_2D,
                    edgecolor=BUILDING_EDGE_2D,
                    linewidth=0.60,
                    alpha=1.0,
                    zorder=1,
                )
            )

        if len(users):
            axis.scatter(
                users[:, 0], users[:, 1], marker="^", s=28,
                facecolor=UE_COLOR, edgecolors="white", linewidths=0.45,
                alpha=0.92, zorder=5,
            )

        start_groups = defaultdict(list)
        for drone_id, points in paths_by_count[count]:
            start = points[0]
            start_key = (round(float(start[0]), 2), round(float(start[1]), 2))
            start_groups[start_key].append(drone_id)

        labeled_shared_starts = set()

        for drone_id, points in paths_by_count[count]:
            color = colors[drone_id]
            axis.plot(
                points[:, 0], points[:, 1],
                color=color, linewidth=2.0, solid_capstyle="round", zorder=4,
            )
            start = np.asarray(points[0], dtype=float)
            end = np.asarray(points[-1], dtype=float)

            axis.scatter(
                start[0], start[1], marker="o", s=28,
                facecolor="white", edgecolor=color, linewidth=1.0, zorder=7,
            )
            axis.scatter(
                end[0], end[1], marker="s", s=26,
                facecolor=color, edgecolor="white", linewidth=0.6, zorder=7,
            )

            start_key = (round(float(start[0]), 2), round(float(start[1]), 2))
            start_dx = -5 if start[0] > 80.0 else 5
            start_ha = "right" if start[0] > 80.0 else "left"

            if len(start_groups[start_key]) >= 2:
                if start_key not in labeled_shared_starts:
                    axis.annotate(
                        "UAV Takeoff Station",
                        xy=(start[0], start[1]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        color="#222222",
                        fontsize=7.6,
                        ha="left",
                        va="bottom",
                        zorder=8,
                    )
                    labeled_shared_starts.add(start_key)
            else:
                axis.annotate(
                    f"Takeoff {drone_id + 1}",
                    xy=(start[0], start[1]),
                    xytext=(start_dx, 5),
                    textcoords="offset points",
                    color=color,
                    fontsize=7.4,
                    ha=start_ha,
                    va="bottom",
                    zorder=8,
                )

            end_dx = -5 if end[0] > 80.0 else 5
            end_ha = "right" if end[0] > 80.0 else "left"
            axis.annotate(
                f"Landing {drone_id + 1}",
                xy=(end[0], end[1]),
                xytext=(end_dx, 5),
                textcoords="offset points",
                color=color,
                fontsize=7.4,
                ha=end_ha,
                va="bottom",
                zorder=8,
            )

        axis.set_title(f"{panel_label} U={count}", fontsize=11, pad=8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-120.0, 120.0)
        axis.set_ylim(-120.0, 120.0)
        axis.set_xlabel("X position (m)")
        _apply_publication_axis_style(axis, grid_axis="both")

    axes_2d[0].set_ylabel("Y position (m)")

    handles_2d = [Line2D([0], [0], color=colors[index], linewidth=2.0, label=f"UAV {index + 1}")
                  for index in range(max(counts))]
    handles_2d.extend([
        Line2D([0], [0], linestyle="None", marker="^", markersize=6.2,
               markerfacecolor=UE_COLOR, markeredgecolor="white", label="UE"),
        Line2D([0], [0], linestyle="None", marker="o", markersize=6.0,
               markerfacecolor="white", markeredgecolor="#555555", label="Takeoff"),
        Line2D([0], [0], linestyle="None", marker="s", markersize=5.8,
               markerfacecolor="#555555", markeredgecolor="white", label="Landing"),
    ])
    legend_2d = figure_2d.legend(
        handles=handles_2d, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.01),
        fontsize=8.2, frameon=False, columnspacing=0.9, handlelength=1.9,
    )
    _finalize_legend(legend_2d)
    figure_2d.subplots_adjust(left=0.055, right=0.99, bottom=0.18, top=0.92, wspace=0.12)
    figure_2d.savefig(
        report_dir / "formal_ca_trajectory_2d_n579.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(figure_2d)

    figure_3d = plt.figure(figsize=(16.8, 5.8), facecolor="white")
    base_starts = list(env.starts)
    base_ends = list(env.ends)
    rebuild = getattr(env, "_build_vertiports_from_starts_ends", None)

    for panel_index, (count, panel_label) in enumerate(zip(counts, panel_labels), start=1):
        env.starts = base_starts[:count]
        env.ends = base_ends[:count]
        if callable(rebuild):
            env.vertiports = rebuild(z_min=0.0, z_max=40.0, nd=2)

        axis = figure_3d.add_subplot(1, len(counts), panel_index, projection="3d")
        _draw_ca_3d_panel(
            axis,
            env,
            paths_by_count[count],
            count,
            colors,
            pipe_radius,
        )

        local_legend = axis.get_legend()
        if local_legend is not None:
            local_legend.remove()

        axis.set_title("")
        axis.text2D(
            0.5,
            0.84,
            f"{panel_label} U={count}",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="#20252B",
        )
        axis.set_xlabel("X position (m)", fontsize=8.8, labelpad=4)
        axis.set_ylabel("Y position (m)", fontsize=8.8, labelpad=4)
        axis.set_zlabel("Altitude (m)", fontsize=8.8, labelpad=4)
        axis.tick_params(axis="both", which="major", labelsize=7.1, pad=1, colors="#555B60")
        try:
            axis.zaxis.set_tick_params(labelsize=7.1, pad=1)
        except Exception:
            pass

    handles_3d = [
        Line2D(
            [0],
            [0],
            color=colors[index],
            linewidth=2.4,
            label=f"UAV {index + 1}",
        )
        for index in range(max(counts))
    ]
    handles_3d.append(
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="^",
            markersize=6.3,
            markerfacecolor="#4A4F55",
            markeredgecolor="white",
            label="UE",
        )
    )

    legend_3d = figure_3d.legend(
        handles=handles_3d,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.145),
        fontsize=7.7,
        frameon=False,
        columnspacing=0.82,
        handlelength=1.7,
        handletextpad=0.5,
    )
    try:
        _finalize_legend(legend_3d)
    except Exception:
        pass

    figure_3d.subplots_adjust(
        left=0.02,
        right=0.99,
        bottom=0.115,
        top=0.995,
        wspace=0.01,
    )
    figure_3d.savefig(
        report_dir / "formal_ca_trajectory_3d_n579.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure_3d)


def _paired_tests(records: list[dict[str, str]], report_dir: Path) -> None:
    from scipy.stats import rankdata, wilcoxon

    metrics = [
        ("path_length_m", "lower"),
        ("communication_cost", "lower"),
        ("mean_pathloss_db", "lower"),
        ("worst_pathloss_db", "lower"),
        ("outage_ratio", "lower"),
        ("min_clearance_m", "higher"),
        ("turn_p95_deg", "lower"),
        ("runtime_s", "lower"),
    ]
    by_method = defaultdict(dict)
    for row in records:
        key = (int(row["uav_count"]), int(row["trial"]), int(row["drone_id"]))
        by_method[row["method"]][key] = row
    common = sorted(
        key
        for key in (
            set(by_method["astar_nsga2"])
            & set(by_method["ca_diffusion_es"])
        )
        if _float(by_method["astar_nsga2"][key], "success") >= 0.5
        and _float(by_method["ca_diffusion_es"][key], "success") >= 0.5
    )
    rng = np.random.default_rng(42)
    rows = []
    for metric, direction in metrics:
        baseline = np.asarray(
            [_float(by_method["astar_nsga2"][key], metric) for key in common],
            dtype=float,
        )
        proposed = np.asarray(
            [_float(by_method["ca_diffusion_es"][key], metric) for key in common],
            dtype=float,
        )
        finite = np.isfinite(baseline) & np.isfinite(proposed)
        baseline = baseline[finite]
        proposed = proposed[finite]
        difference = proposed - baseline
        nonzero = difference[np.abs(difference) > 1e-12]
        if len(nonzero) == 0:
            p_value = 1.0
            rank_biserial = 0.0
        else:
            p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
            ranks = rankdata(np.abs(nonzero))
            positive = float(np.sum(ranks[nonzero > 0]))
            negative = float(np.sum(ranks[nonzero < 0]))
            rank_biserial = (positive - negative) / max(positive + negative, 1e-12)
        if len(difference) > 0:
            indices = rng.integers(0, len(difference), size=(5000, len(difference)))
            boot = difference[indices].mean(axis=1)
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            win_rate = float(
                np.mean(proposed < baseline)
                if direction == "lower"
                else np.mean(proposed > baseline)
            )
        else:
            ci_low = ci_high = win_rate = float("nan")
        rows.append(
            {
                "metric": metric,
                "paired_samples": len(difference),
                "proposed_minus_baseline_mean": float(np.mean(difference)),
                "bootstrap_ci95_low": float(ci_low),
                "bootstrap_ci95_high": float(ci_high),
                "wilcoxon_p_value": p_value,
                "rank_biserial_effect": float(rank_biserial),
                "proposed_win_rate": win_rate,
            }
        )
    fields = list(rows[0])
    with (report_dir / "formal_paired_tests.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def generate_report(
    experiment_dir: Path,
    legacy_root: Path,
    ca_rerun_dir: Path | None = None,
    additional_dir: Path | None = None,
) -> Path:
    records_path = experiment_dir / "records.csv"
    if not records_path.exists():
        raise FileNotFoundError(
            f"Formal baseline records do not exist: {records_path}"
        )
    records = _load_csv(records_path)
    method_roots = {method: experiment_dir for method in METHOD_LABELS}
    report_root = experiment_dir
    if ca_rerun_dir is not None:
        rerun_records_path = ca_rerun_dir / "records.csv"
        if not rerun_records_path.exists():
            raise FileNotFoundError(
                f"CA rerun records do not exist yet: {rerun_records_path}"
            )
        rerun_records = [
            row for row in _load_csv(rerun_records_path)
            if row["method"] == "ca_diffusion_es"
        ]
        expected = sum({5, 7, 9}) * 10
        if len(rerun_records) != expected:
            raise ValueError(
                f"CA rerun is incomplete: expected {expected} records, "
                f"found {len(rerun_records)}"
            )
        records = [row for row in records if row["method"] != "ca_diffusion_es"]
        records.extend(rerun_records)
        method_roots["ca_diffusion_es"] = ca_rerun_dir
        report_root = ca_rerun_dir

    if additional_dir is not None:
        additional_records_path = additional_dir / "records.csv"
        if not additional_records_path.exists():
            raise FileNotFoundError(
                "Additional baseline records do not exist yet: "
                f"{additional_records_path}"
            )
        additional_methods = {"nsga2", "diffusion_only"}
        additional_records = [
            row
            for row in _load_csv(additional_records_path)
            if row["method"] in additional_methods
        ]
        expected = sum({5, 7, 9}) * 10
        counts = {
            method: sum(row["method"] == method for row in additional_records)
            for method in additional_methods
        }
        incomplete = {
            method: count for method, count in counts.items() if count != expected
        }
        if incomplete:
            raise ValueError(
                f"Additional baselines are incomplete; expected {expected} "
                f"records per method, found {incomplete}"
            )
        records = [row for row in records if row["method"] not in additional_methods]
        records.extend(additional_records)
        for method in additional_methods:
            method_roots[method] = additional_dir

    hypervolume_reference = _formal_hv_reference(method_roots)
    report_dir = report_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(
        records, report_dir / "formal_main_table.csv", method_roots,
        hypervolume_reference,
    )
    _paired_tests(records, report_dir)
    _metric_boxplots(records, report_dir)
    _save_standalone_metric_figures(records, report_dir, _active_methods(records))
    _scalability(records, report_dir)
    _convergence(method_roots, report_dir, hypervolume_reference)
    _final_hypervolume(method_roots, report_dir, hypervolume_reference)
    _pareto(method_roots, report_dir)
    _trajectories(method_roots, report_dir, legacy_root)
    _ca_trajectories_3d(method_roots, report_dir, legacy_root)
    _ca_trajectories_by_scale(method_roots, report_dir, legacy_root)
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--ca-rerun-dir", type=Path, default=None)
    parser.add_argument("--additional-dir", type=Path, default=None)
    args = parser.parse_args()
    print(
        generate_report(
            experiment_dir=args.experiment_dir.resolve(),
            legacy_root=args.legacy_root.resolve(),
            ca_rerun_dir=(
                None if args.ca_rerun_dir is None else args.ca_rerun_dir.resolve()
            ),
            additional_dir=(
                None if args.additional_dir is None else args.additional_dir.resolve()
            ),
        )
    )

if __name__ == "__main__":
    main()
