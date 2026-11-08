"""Create tables and figures for the CA-Diffusion-ES v9 parameter experiment."""

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


V9_ID = "v9_adaptive_anchor_050"

# -----------------------------------------------------------------------------
# Journal figure style
# Keep the same visual language as the main-paper result figures:
# white background, colorblind-safe distinct baseline identities, and a single
# warm accent for the final CA-Diffusion-ES parameter setting.
# -----------------------------------------------------------------------------
JOURNAL_ACCENT = "#D55E00"
JOURNAL_BLUE = "#365F7D"
JOURNAL_TEXT = "#20252B"
JOURNAL_TICK = "#50565C"
JOURNAL_AXIS = "#8B9299"
JOURNAL_GRID = "#DCE3E8"
JOURNAL_BASELINES = ["#0072B2", "#009E73", "#7A5195", "#56B4E9", "#5F6B73"]
JOURNAL_LINESTYLES = ["-", "-", "-", "-", "-"]
JOURNAL_MARKERS = ["s", "^", "D", "v", "P"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.labelsize": 10.0,
    "xtick.labelsize": 8.7,
    "ytick.labelsize": 8.7,
    "legend.fontsize": 8.4,
    "axes.linewidth": 0.8,
    "font.size": 9.0,
})


def _style_axis(axis, *, grid_axis: str = "both") -> None:
    """Apply the common clean journal styling used throughout the paper."""
    axis.set_facecolor("white")
    axis.set_axisbelow(True)
    axis.grid(
        axis=grid_axis, color=JOURNAL_GRID, linewidth=0.65, alpha=0.68
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(JOURNAL_AXIS)
    axis.spines["bottom"].set_color(JOURNAL_AXIS)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", colors=JOURNAL_TICK, width=0.7, length=3.0)
    axis.xaxis.label.set_color(JOURNAL_TEXT)
    axis.yaxis.label.set_color(JOURNAL_TEXT)


def _parameter_styles(ids: list[str]) -> dict[str, tuple[str, object]]:
    """Return consistent color/linestyle pairs for one parameter group."""
    styles: dict[str, tuple[str, object]] = {}
    baseline_index = 0
    for configuration_id in ids:
        if configuration_id == V9_ID:
            styles[configuration_id] = (JOURNAL_ACCENT, "-")
        else:
            styles[configuration_id] = (
                JOURNAL_BASELINES[baseline_index % len(JOURNAL_BASELINES)],
                JOURNAL_LINESTYLES[baseline_index % len(JOURNAL_LINESTYLES)],
            )
            baseline_index += 1
    return styles


def _parameter_markers(ids: list[str]) -> dict[str, str]:
    """Return sparse, publication-friendly marker identities for one parameter group."""
    markers: dict[str, str] = {}
    baseline_index = 0
    for configuration_id in ids:
        if configuration_id == V9_ID:
            markers[configuration_id] = "o"
        else:
            markers[configuration_id] = JOURNAL_MARKERS[baseline_index % len(JOURNAL_MARKERS)]
            baseline_index += 1
    return markers


def _display_label(config: dict, configuration_id: str, group_name: str) -> str:
    """Return a group-specific label for the shared v9 control setting."""
    for item in config["configurations"]:
        if str(item["id"]) != configuration_id:
            continue
        label = str(item["label"]).replace(" (v9)", " (CA-Diffusion-ES)")
        if configuration_id == V9_ID and group_name == "anchor":
            return "Anchor ratio 0.50 (CA-Diffusion-ES)"
        return label
    return configuration_id


def _plot_label(config: dict, configuration_id: str, group_name: str) -> str:
    """Return a compact label for plots; the final setting is highlighted graphically."""
    label = _display_label(config, configuration_id, group_name)
    label = label.replace(" (CA-Diffusion-ES)", "").replace(" (v9)", "")
    return label


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def _mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(array))
    ci = 0.0 if len(array) == 1 else float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))
    return mean, ci


def _configuration_ids(config: dict, group: str) -> list[str]:
    return [
        str(item["id"]) for item in config["configurations"]
        if group in item.get("groups", [])
    ]


def _labels(config: dict) -> dict[str, str]:
    return {str(item["id"]): str(item["label"]) for item in config["configurations"]}


def _pair_maxima(records: list[dict[str, str]]) -> dict[tuple[int, int], float]:
    maxima: dict[tuple[int, int], float] = defaultdict(float)
    for row in records:
        if row.get("status") != "complete":
            continue
        key = (int(row["seed"]), int(row["drone_id"]))
        feasible_hv = _number(row, "final_hypervolume", 0.0) if _number(row, "success", 0.0) >= 0.5 else 0.0
        maxima[key] = max(maxima[key], feasible_hv)
    return maxima


def _relative_final_hv(row: dict[str, str], maxima: dict[tuple[int, int], float]) -> float:
    if row.get("status") != "complete" or _number(row, "success", 0.0) < 0.5:
        return 0.0
    denominator = maxima.get((int(row["seed"]), int(row["drone_id"])), 0.0)
    return _number(row, "final_hypervolume", 0.0) / max(denominator, 1e-12)


def _write_summary(
    records: list[dict[str, str]], config: dict, output: Path
) -> list[dict]:
    maxima = _pair_maxima(records)
    expected = len(config["drones"]) * len(config["seeds"])
    rows = []
    for setting in config["configurations"]:
        configuration_id = str(setting["id"])
        group = [row for row in records if row["configuration_id"] == configuration_id]
        complete = [row for row in group if row.get("status") == "complete"]
        relative_hv = [_relative_final_hv(row, maxima) for row in complete]
        normalized_auc = []
        for row in complete:
            denominator = maxima.get((int(row["seed"]), int(row["drone_id"])), 0.0)
            normalized_auc.append(_number(row, "hypervolume_auc", 0.0) / max(denominator, 1e-12))
        final_mean, final_ci = _mean_ci(relative_hv)
        auc_mean, auc_ci = _mean_ci(normalized_auc)

        def mean(key: str) -> float:
            values = [_number(row, key) for row in complete]
            return float(np.nanmean(values)) if values else float("nan")

        rows.append({
            "configuration_id": configuration_id,
            "label": setting["label"],
            "groups": ";".join(setting["groups"]),
            "expected_runs": expected,
            "completed_runs": len(complete),
            "failed_runs": len(group) - len(complete),
            "success_rate": float(np.mean([_number(row, "success", 0.0) for row in complete])) if complete else 0.0,
            "relative_feasible_hv_mean": final_mean,
            "relative_feasible_hv_ci95": final_ci,
            "normalized_hv_auc_mean": auc_mean,
            "normalized_hv_auc_ci95": auc_ci,
            "raw_final_hv_mean": mean("final_hypervolume"),
            "initial_feasible_ratio_mean": mean("initial_feasible_ratio"),
            "raw_offspring_feasible_ratio_mean": mean("mean_raw_offspring_feasible_ratio"),
            "projection_per_100_evaluations_mean": mean("projection_per_100_evaluations"),
            "path_length_mean_m": mean("path_length_m"),
            "communication_cost_mean": mean("communication_cost"),
            "runtime_mean_s": mean("runtime_s"),
            "pareto_size_mean": mean("pareto_size"),
        })
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _history_curve(path: Path, budget: int, denominator: float, grid: np.ndarray) -> np.ndarray:
    history = _load_csv(path)
    evaluations = np.asarray([_number(row, "evaluations") for row in history], dtype=float)
    hypervolume = np.asarray([_number(row, "hypervolume", 0.0) for row in history], dtype=float)
    evaluations = np.concatenate([[0.0], evaluations])
    hypervolume = np.concatenate([[0.0], hypervolume]) / max(denominator, 1e-12)
    return np.interp(grid, evaluations, hypervolume, left=0.0, right=hypervolume[-1])


def _plot_convergence(
    records: list[dict[str, str]], config: dict, experiment_dir: Path,
    report_dir: Path, group_name: str,
) -> None:
    ids = _configuration_ids(config, group_name)
    maxima = _pair_maxima(records)
    budget = int(config["evaluation_budget"])
    grid = np.linspace(0.0, budget, 81)

    # Same aspect ratio and visual weight as the main convergence figure.
    figure, axis = plt.subplots(figsize=(9.4, 6.1))
    figure.patch.set_facecolor("white")

    curves_by_id: dict[str, np.ndarray] = {}
    ci_by_id: dict[str, np.ndarray] = {}
    parameter_styles = _parameter_styles(ids)
    parameter_markers = _parameter_markers(ids)

    for configuration_index, configuration_id in enumerate(ids):
        curves = []
        for row in records:
            if row["configuration_id"] != configuration_id or row.get("status") != "complete":
                continue
            denominator = maxima.get((int(row["seed"]), int(row["drone_id"])), 0.0)
            history_path = experiment_dir / row["history_path"]
            if denominator > 0.0 and history_path.exists():
                curves.append(_history_curve(history_path, budget, denominator, grid))

        if not curves:
            continue

        values = np.asarray(curves, dtype=float)
        mean = np.mean(values, axis=0)
        ci = (
            np.zeros_like(mean)
            if len(values) == 1
            else 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
        )
        curves_by_id[configuration_id] = mean
        ci_by_id[configuration_id] = ci

        color, linestyle = parameter_styles[configuration_id]
        marker = parameter_markers[configuration_id]
        is_selected = configuration_id == V9_ID

        # Use solid curves throughout. Sparse, phase-shifted hollow markers
        # distinguish nearly overlapping parameter settings without visual noise.
        marker_phase = (3 * configuration_index) % 12
        axis.plot(
            grid,
            mean,
            label=_plot_label(config, configuration_id, group_name),
            color=color,
            linewidth=2.00 if is_selected else 1.28,
            linestyle="-",
            marker=marker,
            markevery=(marker_phase, 12),
            markersize=3.8 if is_selected else 3.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.85 if is_selected else 0.65,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5 if is_selected else 3,
        )
        axis.fill_between(
            grid,
            np.maximum(0.0, mean - ci),
            np.minimum(1.0, mean + ci),
            color=color,
            alpha=0.065 if is_selected else 0.035,
            linewidth=0.0,
            zorder=1,
        )

    axis.set_xlabel("Black-box evaluations")
    axis.set_ylabel("Relative feasible hypervolume")
    axis.set_xlim(0.0, budget)

    # Tighten the vertical extent to the actual result range, avoiding the large
    # empty top region that looks weak after IEEE single-column scaling.
    if curves_by_id:
        upper_values = []
        for configuration_id, mean in curves_by_id.items():
            ci = ci_by_id[configuration_id]
            upper_values.append(np.minimum(1.0, mean + ci))
        observed_top = float(np.nanmax(np.concatenate(upper_values)))
        y_top = min(1.02, max(0.10, observed_top + max(0.018, 0.055 * observed_top)))
        axis.set_ylim(0.0, y_top)
    else:
        axis.set_ylim(0.0, 1.0)

    _style_axis(axis, grid_axis="y")

    if curves_by_id:
        # Keep legend and zoom inset as one compact lower-band composition.
        # Their vertical centers are aligned so the legend reads as the left
        # companion of the zoom panel rather than as an element attached to x-axis.
        inset_bounds = (0.540, 0.205, 0.420, 0.375)
        inset_center_y = inset_bounds[1] + 0.5 * inset_bounds[3]
        selected_label = _plot_label(config, V9_ID, group_name)
        legend_handles, legend_labels = axis.get_legend_handles_labels()
        legend_order = (
            [i for i, label in enumerate(legend_labels) if label == selected_label]
            + [i for i, label in enumerate(legend_labels) if label != selected_label]
        )
        legend = axis.legend(
            [legend_handles[i] for i in legend_order],
            [legend_labels[i] for i in legend_order],
            frameon=False,
            loc="center left",
            bbox_to_anchor=(0.125, inset_center_y),
            borderaxespad=0.0,
            ncol=2 if len(curves_by_id) >= 4 else 1,
            fontsize=8.6,
            labelspacing=0.38,
            columnspacing=1.30,
            handlelength=2.00,
            handletextpad=0.55,
            markerscale=1.05,
        )
        for text_item in legend.get_texts():
            if text_item.get_text() == selected_label:
                text_item.set_color(JOURNAL_ACCENT)
                text_item.set_fontweight("semibold")
            else:
                text_item.set_color(JOURNAL_TEXT)

        # Late-budget inset is retained because the parameter curves can be very
        # close near convergence; it is styled as a restrained statistical inset.
        zoom_start = budget * 0.58
        late_mask = grid >= zoom_start
        late_values = np.concatenate([
            curve[late_mask] for curve in curves_by_id.values()
        ])
        raw_low = float(np.nanmin(late_values))
        raw_high = float(np.nanmax(late_values))
        margin = max(0.0025, 0.16 * max(raw_high - raw_low, 1e-6))
        zoom_low = max(0.0, raw_low - margin)
        zoom_high = min(1.0, raw_high + margin)
        if zoom_high - zoom_low < 0.02:
            center = 0.5 * (zoom_low + zoom_high)
            zoom_low = max(0.0, center - 0.01)
            zoom_high = min(1.0, center + 0.01)

        # A larger but still subordinate inset makes the late-budget differences
        # readable after single-column scaling without obscuring the main curves.
        inset = axis.inset_axes(inset_bounds)
        inset.set_facecolor("white")
        for configuration_index, configuration_id in enumerate(ids):
            if configuration_id not in curves_by_id:
                continue
            color, _ = parameter_styles[configuration_id]
            marker = parameter_markers[configuration_id]
            is_selected = configuration_id == V9_ID
            marker_phase = (2 * configuration_index) % 10
            inset.plot(
                grid,
                curves_by_id[configuration_id],
                color=color,
                linewidth=1.45 if is_selected else 0.98,
                linestyle="-",
                marker=marker,
                markevery=(marker_phase, 10),
                markersize=2.9 if is_selected else 2.4,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.65 if is_selected else 0.50,
                solid_capstyle="round",
                zorder=4 if is_selected else 2,
            )
        inset.set_xlim(zoom_start, budget)
        inset.set_ylim(zoom_low, zoom_high)
        from matplotlib.ticker import MaxNLocator
        inset.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
        inset.tick_params(
            labelsize=7.3, colors=JOURNAL_TICK, width=0.60, length=2.1, pad=2.0
        )
        inset.grid(
            axis="y", color=JOURNAL_GRID, linewidth=0.45, alpha=0.52
        )
        for spine in inset.spines.values():
            spine.set_color(JOURNAL_AXIS)
            spine.set_linewidth(0.65)

        # Journal-style zoom indication: draw the selected late-budget rectangle
        # on the main axis and connect its upper corners to the inset.  Keep the
        # geometry light gray so it explains the magnification without competing
        # with the experimental curves.
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        zoom_patch, connector_left, connector_right = mark_inset(
            axis,
            inset,
            loc1=2,
            loc2=1,
            fc="none",
            ec="#A7ADB2",
            lw=0.82,
        )
        for artist in (zoom_patch, connector_left, connector_right):
            artist.set_alpha(0.78)
            artist.set_zorder(1.6)
    else:
        axis.text(
            0.5, 0.5, "No completed runs", ha="center", va="center",
            transform=axis.transAxes, color=JOURNAL_TICK,
        )

    figure.subplots_adjust(left=0.115, right=0.985, bottom=0.130, top=0.975)
    figure.savefig(
        report_dir / f"{group_name}_hv_convergence.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _boxplot(
    records: list[dict[str, str]], config: dict, report_dir: Path,
    group_name: str, metric: str, ylabel: str, filename: str,
    *, relative_hv: bool = False,
) -> None:
    ids = _configuration_ids(config, group_name)
    maxima = _pair_maxima(records)
    data = []

    for configuration_id in ids:
        group = [
            row for row in records
            if row["configuration_id"] == configuration_id and row.get("status") == "complete"
        ]
        if relative_hv:
            values = [_relative_final_hv(row, maxima) for row in group]
        else:
            values = [_number(row, metric) for row in group]
            values = [value for value in values if np.isfinite(value)]
        data.append(values)

    # Match the main-paper quantitative figures: generous vertical room, but no
    # oversized horizontal canvas that becomes tiny after single-column scaling.
    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    figure.patch.set_facecolor("white")
    parameter_styles = _parameter_styles(ids)

    if any(values for values in data):
        plot = axis.boxplot(
            data,
            patch_artist=True,
            showmeans=True,
            widths=0.58,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": JOURNAL_TEXT,
                "markeredgewidth": 0.85,
                "markersize": 5.0,
            },
            medianprops={"color": JOURNAL_TEXT, "linewidth": 1.25},
            whiskerprops={"color": "#626A70", "linewidth": 0.95},
            capprops={"color": "#626A70", "linewidth": 0.95},
            flierprops={
                "marker": ".",
                "markersize": 2.6,
                "markerfacecolor": "#626A70",
                "markeredgecolor": "none",
                "alpha": 0.22,
            },
        )

        for index, patch in enumerate(plot["boxes"]):
            configuration_id = ids[index]
            color, _ = parameter_styles[configuration_id]
            is_selected = configuration_id == V9_ID
            edge_color = color if is_selected else "#5F6B73"

            patch.set_facecolor(color)
            patch.set_alpha(0.82 if is_selected else 0.56)
            patch.set_edgecolor(edge_color)
            patch.set_linewidth(1.30 if is_selected else 0.85)

            # Keep a collapsed/highly concentrated selected box visible as orange.
            plot["medians"][index].set_color(color if is_selected else JOURNAL_TEXT)
            plot["medians"][index].set_linewidth(1.55 if is_selected else 1.25)
            if index < len(plot.get("means", [])):
                plot["means"][index].set_markeredgecolor(color if is_selected else JOURNAL_TEXT)
            for line in plot["whiskers"][2 * index: 2 * index + 2]:
                line.set_color(color if is_selected else "#626A70")
                line.set_linewidth(1.10 if is_selected else 0.95)
            for line in plot["caps"][2 * index: 2 * index + 2]:
                line.set_color(color if is_selected else "#626A70")
                line.set_linewidth(1.10 if is_selected else 0.95)
    else:
        axis.text(
            0.5, 0.5, "No completed runs", ha="center", va="center",
            transform=axis.transAxes, color=JOURNAL_TICK,
        )

    axis.set_xticks(range(1, len(ids) + 1))
    axis.set_xticklabels(
        [_plot_label(config, item, group_name) for item in ids],
        rotation=0,
    )
    axis.set_ylabel(ylabel)

    # Highlight only the final selected configuration; do not add explanatory
    # annotations inside the figure.
    for tick, configuration_id in zip(axis.get_xticklabels(), ids):
        if configuration_id == V9_ID:
            tick.set_color(JOURNAL_ACCENT)
            tick.set_fontweight("semibold")
        else:
            tick.set_color(JOURNAL_TEXT)

    if relative_hv:
        finite = [np.asarray(values, dtype=float) for values in data if len(values)]
        if finite:
            all_values = np.concatenate(finite)
            observed_low = float(np.nanmin(all_values))
            observed_high = float(np.nanmax(all_values))
            span = max(observed_high - observed_low, 0.02)
            lower = max(0.0, observed_low - max(0.018, 0.16 * span))
            upper = min(1.02, observed_high + max(0.018, 0.16 * span))
            if upper - lower < 0.08:
                center = 0.5 * (upper + lower)
                lower = max(0.0, center - 0.04)
                upper = min(1.02, center + 0.04)
            axis.set_ylim(lower, upper)
        else:
            axis.set_ylim(0.0, 1.0)

    _style_axis(axis, grid_axis="y")
    axis.tick_params(axis="x", pad=6)

    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.19, top=0.97)
    figure.savefig(
        report_dir / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _write_recommendation(summary: list[dict], report_dir: Path) -> None:
    complete = [row for row in summary if int(row["completed_runs"]) > 0]
    complete.sort(
        key=lambda row: (
            float(row["success_rate"]),
            float(row["relative_feasible_hv_mean"]),
            float(row["normalized_hv_auc_mean"]),
            -float(row["runtime_mean_s"]),
        ),
        reverse=True,
    )
    recommendation = {
        "selection_rule": "success rate, then relative feasible HV@4000, then normalized HV-AUC, then runtime",
        "recommended_configuration": complete[0]["configuration_id"] if complete else None,
        "ranking": [row["configuration_id"] for row in complete],
        "warning": "This ranking is valid only for the configured validation scenarios and parameter ranges.",
    }
    (report_dir / "parameter_recommendation.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )


def create_report(root: Path, config_path: Path, experiment_dir: Path | None = None) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_dir = experiment_dir or (root / config["output_dir"])
    records_path = experiment_dir / "records.csv"
    if not records_path.exists():
        raise FileNotFoundError(f"Parameter records not found: {records_path}")
    records = _load_csv(records_path)
    report_dir = experiment_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = _write_summary(records, config, report_dir / "parameter_summary.csv")
    for group in ("truncation", "anchor"):
        _plot_convergence(records, config, experiment_dir, report_dir, group)
        _boxplot(
            records, config, report_dir, group, "final_hypervolume",
            "Relative feasible hypervolume",
            f"{group}_final_feasible_hv.png", relative_hv=True,
        )
    _boxplot(
        records, config, report_dir, "truncation",
        "mean_raw_offspring_feasible_ratio", "Raw offspring feasible ratio",
        "truncation_raw_feasibility.png",
    )
    _boxplot(
        records, config, report_dir, "truncation",
        "projection_per_100_evaluations", "Projected candidates per 100 evaluations",
        "truncation_projection_rate.png",
    )
    _boxplot(
        records, config, report_dir, "anchor",
        "initial_feasible_ratio", "Initial population feasible ratio",
        "anchor_initial_feasibility.png",
    )
    _write_recommendation(summary, report_dir)
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()
    root = args.legacy_root.resolve()
    experiment_dir = args.experiment_dir.resolve() if args.experiment_dir else None
    print(create_report(root, args.config.resolve(), experiment_dir))


if __name__ == "__main__":
    main()
