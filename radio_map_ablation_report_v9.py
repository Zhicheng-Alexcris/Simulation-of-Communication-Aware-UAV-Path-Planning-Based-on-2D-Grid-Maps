"""Create paired true-channel figures for the CA v9 Radio Map ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


VARIANT_ORDER = ("without_radio_map", "with_radio_map")
METRICS = (
    ("true_mean_pathloss_db", "True mean path loss", "dB", 1.0),
    ("true_outage_ratio", "True outage ratio", "%", 100.0),
    ("path_length_m", "Path length", "m", 1.0),
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def paired_records(
    records: list[dict[str, str]], *, require_success: bool = True
) -> list[tuple[dict[str, str], dict[str, str]]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, str]]] = {}
    for row in records:
        if row.get("status") != "complete":
            continue
        key = (int(row["seed"]), int(row["drone_id"]))
        grouped.setdefault(key, {})[str(row["variant"])] = row
    pairs = []
    for key in sorted(grouped):
        variants = grouped[key]
        if any(variant not in variants for variant in VARIANT_ORDER):
            continue
        without = variants["without_radio_map"]
        with_radio = variants["with_radio_map"]
        if require_success and (
            _number(without, "success") < 0.5
            or _number(with_radio, "success") < 0.5
        ):
            continue
        pairs.append((without, with_radio))
    return pairs


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    ci = 0.0 if len(values) == 1 else float(
        1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
    )
    return mean, ci


def _wilcoxon_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0 or np.all(np.abs(differences) <= 1e-12):
        return 1.0
    return float(wilcoxon(differences, alternative="two-sided").pvalue)


def _write_outputs(
    pairs: list[tuple[dict[str, str], dict[str, str]]], report_dir: Path
) -> list[dict]:
    summary = []
    paired_rows = []
    for key, label, unit, scale in METRICS:
        without = np.asarray([_number(pair[0], key) for pair in pairs]) * scale
        with_radio = np.asarray([_number(pair[1], key) for pair in pairs]) * scale
        finite = np.isfinite(without) & np.isfinite(with_radio)
        metric_pairs = [pair for pair, keep in zip(pairs, finite) if keep]
        without = without[finite]
        with_radio = with_radio[finite]
        differences = with_radio - without
        without_mean, without_ci = _mean_ci(without)
        with_mean, with_ci = _mean_ci(with_radio)
        delta_mean, delta_ci = _mean_ci(differences)
        summary.append(
            {
                "metric": key,
                "label": label,
                "unit": unit,
                "paired_samples": len(differences),
                "without_radio_mean": without_mean,
                "without_radio_ci95": without_ci,
                "with_radio_mean": with_mean,
                "with_radio_ci95": with_ci,
                "paired_delta_mean_with_minus_without": delta_mean,
                "paired_delta_ci95": delta_ci,
                "paired_delta_median": float(np.median(differences)),
                "improved_pair_fraction": float(np.mean(differences < 0.0)),
                "wilcoxon_pvalue": _wilcoxon_pvalue(differences),
            }
        )
        for pair_index, (pair, left, right, difference) in enumerate(
            zip(metric_pairs, without, with_radio, differences)
        ):
            paired_rows.append(
                {
                    "metric": key,
                    "pair_index": pair_index,
                    "seed": int(pair[0]["seed"]),
                    "drone_id": int(pair[0]["drone_id"]),
                    "without_radio": float(left),
                    "with_radio": float(right),
                    "with_minus_without": float(difference),
                }
            )

    with (report_dir / "radio_map_ablation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (report_dir / "radio_map_ablation_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    return summary


def _write_run_summary(records: list[dict[str, str]], report_dir: Path) -> None:
    rows = []
    for variant in VARIANT_ORDER:
        group = [row for row in records if row.get("variant") == variant]
        completed = [row for row in group if row.get("status") == "complete"]
        rows.append(
            {
                "variant": variant,
                "total_records": len(group),
                "completed_records": len(completed),
                "successful_records": sum(
                    _number(row, "success") >= 0.5 for row in completed
                ),
                "success_rate": float(np.mean([
                    _number(row, "success") >= 0.5 for row in completed
                ])) if completed else 0.0,
            }
        )
    with (report_dir / "radio_map_ablation_run_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def _plot_pairs(
    pairs: list[tuple[dict[str, str], dict[str, str]]], report_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    rng = np.random.default_rng(2026)
    for axis, (key, label, unit, scale) in zip(axes, METRICS):
        without = np.asarray([_number(pair[0], key) for pair in pairs]) * scale
        with_radio = np.asarray([_number(pair[1], key) for pair in pairs]) * scale
        finite = np.isfinite(without) & np.isfinite(with_radio)
        differences = with_radio[finite] - without[finite]
        mean, ci = _mean_ci(differences)
        jitter = rng.uniform(-0.12, 0.12, size=len(differences))
        point_colors = np.where(
            differences < -1e-12, "#2A9D8F",
            np.where(differences > 1e-12, "#D95F59", "#777777"),
        )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axis.scatter(
            jitter, differences, s=24, color=point_colors, alpha=0.62,
            linewidths=0, zorder=2,
        )
        axis.errorbar(
            [0.28], [mean], yerr=[ci], fmt="D", markersize=6.0,
            color="#202020", markerfacecolor="white", markeredgewidth=1.1,
            capsize=4.0, linewidth=1.3, zorder=4,
        )
        decimals = 2 if unit == "dB" else 1
        axis.annotate(
            f"{mean:+.{decimals}f}", xy=(0.28, mean), xytext=(8, 0),
            textcoords="offset points", ha="left", va="center", fontsize=8,
        )
        lower = min(float(np.min(differences)), mean - ci, 0.0)
        upper = max(float(np.max(differences)), mean + ci, 0.0)
        span = max(upper - lower, 1e-6)
        axis.set_xlim(-0.18, 0.45)
        axis.set_ylim(lower - 0.10 * span, upper + 0.10 * span)
        axis.set_xticks([0.0, 0.28])
        axis.set_xticklabels(["Paired runs", "Mean and\n95% CI"])
        difference_unit = "percentage points" if key == "true_outage_ratio" else unit
        axis.set_ylabel(f"{label}\nWith - without ({difference_unit})")
        axis.grid(axis="y", alpha=0.24)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(
        report_dir / "radio_map_ablation_true_channel.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_report(
    config_path: Path,
    experiment_dir: Path,
    *,
    allow_incomplete: bool = False,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records_path = experiment_dir / "records.csv"
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    records = _load_csv(records_path)
    expected = len(config["drones"]) * len(config["seeds"]) * 2
    completed = [row for row in records if row.get("status") == "complete"]
    if not allow_incomplete and len(completed) != expected:
        raise ValueError(
            f"Radio Map ablation is incomplete: expected {expected} complete records, "
            f"found {len(completed)}"
        )
    pairs = paired_records(records, require_success=True)
    if not pairs:
        raise ValueError(
            "The Radio Map ablation contains no pair in which both trajectories "
            "are geometrically feasible"
        )
    report_dir = experiment_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_run_summary(records, report_dir)
    _write_outputs(pairs, report_dir)
    _plot_pairs(pairs, report_dir)
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(
        create_report(
            args.config.resolve(),
            args.experiment_dir.resolve(),
            allow_incomplete=bool(args.allow_incomplete),
        )
    )


if __name__ == "__main__":
    main()
