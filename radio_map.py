"""Three-dimensional radio-map lookup backed by HGNN predictions."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


_XYZ_RE = re.compile(
    r"\(?\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)?"
)


def _normalize_header(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.replace("\ufeff", "").strip().split())


def _sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
    except csv.Error:
        return "\t" if sample.count("\t") > sample.count(",") else ","


def _parse_xyz(value: object) -> tuple[float, float, float]:
    match = _XYZ_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid 3D coordinate: {value!r}")
    return tuple(float(match.group(index)) for index in range(1, 4))


class HGNNRadioMap3D:
    """Black-box 3D lookup over HGNN-predicted path-loss samples.

    Several ray records can correspond to the same transmitter-receiver pair.
    They are reduced to one robust prediction before interpolation. Each
    transmitter represents one ground link; the point cost uses the maximum
    predicted path loss across links, as required by the paper's worst-link
    aggregation model.
    """

    def __init__(
        self,
        samples_by_tx: dict[tuple[float, float, float], tuple[list, list]],
        *,
        k_neighbors: int = 8,
        idw_power: float = 2.0,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
    ) -> None:
        if not samples_by_tx:
            raise ValueError("The HGNN radio map contains no valid samples")
        self.k_neighbors = max(1, int(k_neighbors))
        self.idw_power = float(idw_power)
        self.transmitters: list[np.ndarray] = []
        self._maps: list[tuple[np.ndarray, np.ndarray]] = []

        for transmitter, (coordinates, values) in sorted(samples_by_tx.items()):
            coordinates_array = np.asarray(coordinates, dtype=float).reshape(-1, 3)
            values_array = np.asarray(values, dtype=float).reshape(-1)
            if len(coordinates_array) == 0 or len(coordinates_array) != len(values_array):
                continue
            self.transmitters.append(np.asarray(transmitter, dtype=float))
            self._maps.append((coordinates_array, values_array))
        self._trees = [cKDTree(coordinates) for coordinates, _ in self._maps]
        if not self._maps:
            raise ValueError("The HGNN radio map contains no usable grid points")

        all_altitudes = np.concatenate([coordinates[:, 2] for coordinates, _ in self._maps])
        self.altitudes = np.unique(np.round(all_altitudes, decimals=6))
        self._layer_bounds: dict[float, tuple[float, float]] = {}
        for altitude in self.altitudes:
            layer_points = []
            for coordinates, _ in self._maps:
                mask = np.isclose(coordinates[:, 2], altitude, atol=1e-6)
                if np.any(mask):
                    layer_points.append(coordinates[mask])
            points = np.unique(np.concatenate(layer_points, axis=0), axis=0)
            aggregate_pathloss = self.pathloss(points)
            lower = float(np.percentile(aggregate_pathloss, lower_percentile))
            upper = float(np.percentile(aggregate_pathloss, upper_percentile))
            if upper - lower < 1e-6:
                upper = lower + 1.0
            self._layer_bounds[float(altitude)] = (lower, upper)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        predicted_column: str = "Predicted_PathLoss",
        transmitter_column: str = "Transmitter",
        receiver_column: str = "Receiver",
        duplicate_reducer: str = "median",
        **kwargs,
    ) -> "HGNNRadioMap3D":
        path = Path(path)
        delimiter = _sniff_delimiter(path)
        grouped: dict[tuple[tuple, tuple], list[float]] = defaultdict(list)
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            headers = {_normalize_header(header): header for header in (reader.fieldnames or [])}
            transmitter_key = headers.get(_normalize_header(transmitter_column))
            receiver_key = headers.get(_normalize_header(receiver_column))
            prediction_key = headers.get(_normalize_header(predicted_column))
            if transmitter_key is None or receiver_key is None or prediction_key is None:
                raise KeyError(
                    "HGNN CSV must contain Transmitter, Receiver, and "
                    f"{predicted_column!r}; available={list(headers)}"
                )
            for row in reader:
                try:
                    transmitter = _parse_xyz(row[transmitter_key])
                    receiver = _parse_xyz(row[receiver_key])
                    pathloss = float(row[prediction_key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(pathloss):
                    grouped[(transmitter, receiver)].append(pathloss)

        if duplicate_reducer not in {"median", "mean"}:
            raise ValueError("duplicate_reducer must be 'median' or 'mean'")
        reduce_function = np.median if duplicate_reducer == "median" else np.mean
        samples_by_tx: dict[tuple, tuple[list, list]] = defaultdict(lambda: ([], []))
        for (transmitter, receiver), values in grouped.items():
            samples_by_tx[transmitter][0].append(receiver)
            samples_by_tx[transmitter][1].append(float(reduce_function(values)))
        return cls(dict(samples_by_tx), **kwargs)

    @staticmethod
    def _as_points(points: np.ndarray | list) -> np.ndarray:
        points_array = np.asarray(points, dtype=float)
        if points_array.ndim == 1:
            points_array = points_array.reshape(1, -1)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {points_array.shape}")
        return points_array

    def _interpolate_one(
        self,
        points: np.ndarray,
        coordinates: np.ndarray,
        values: np.ndarray,
        tree: cKDTree,
    ) -> np.ndarray:
        points = self._as_points(points)
        output = np.empty(len(points), dtype=float)
        neighbors = min(self.k_neighbors, coordinates.shape[0])
        for start in range(0, len(points), 4096):
            query = points[start : start + 4096]
            distances, indices = tree.query(query, k=neighbors, workers=1)
            distances = np.asarray(distances, dtype=float)
            indices = np.asarray(indices, dtype=int)
            if neighbors == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            squared_distance = distances * distances
            near_values = values[indices]
            exact = np.min(squared_distance, axis=1) <= 1e-12
            weights = 1.0 / np.maximum(squared_distance, 1e-12) ** (
                0.5 * self.idw_power
            )
            interpolated = np.sum(weights * near_values, axis=1) / np.sum(
                weights, axis=1
            )
            if np.any(exact):
                nearest = np.argmin(squared_distance[exact], axis=1)
                interpolated[exact] = near_values[exact, nearest]
            output[start : start + len(query)] = interpolated
        return output

    def pathloss_per_transmitter(self, points: np.ndarray | list) -> np.ndarray:
        points_array = self._as_points(points)
        per_link = [
            self._interpolate_one(points_array, coordinates, pathloss, tree)
            for (coordinates, pathloss), tree in zip(self._maps, self._trees)
        ]
        return np.stack(per_link, axis=1)

    def pathloss(self, points: np.ndarray | list) -> np.ndarray:
        """Worst-link HGNN-predicted path loss in dB for each point."""
        return np.max(self.pathloss_per_transmitter(points), axis=1)

    def _normalize_pathloss(
        self, points: np.ndarray, pathloss: np.ndarray
    ) -> np.ndarray:
        points_array = self._as_points(points)
        pathloss = np.asarray(pathloss, dtype=float).reshape(-1)
        if len(pathloss) != len(points_array):
            raise ValueError("Path-loss values must match the point count")
        nearest_layer = np.argmin(
            np.abs(points_array[:, 2:3] - self.altitudes.reshape(1, -1)), axis=1
        )
        costs = np.empty(len(points_array), dtype=float)
        for layer_index, altitude in enumerate(self.altitudes):
            mask = nearest_layer == layer_index
            if not np.any(mask):
                continue
            lower, upper = self._layer_bounds[float(altitude)]
            costs[mask] = (pathloss[mask] - lower) / (upper - lower)
        return np.clip(costs, 0.0, 1.0)

    def pathloss_and_normalized_cost(
        self, points: np.ndarray | list
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return both radio metrics from one nearest-neighbor query."""
        points_array = self._as_points(points)
        pathloss = self.pathloss(points_array)
        return pathloss, self._normalize_pathloss(points_array, pathloss)

    def normalized_cost(self, points: np.ndarray | list) -> np.ndarray:
        """Layer-normalized worst-link communication cost in [0, 1]."""
        _, costs = self.pathloss_and_normalized_cost(points)
        return costs

    def trajectory_metrics(self, path: np.ndarray | list) -> dict[str, float]:
        points = self._as_points(path)
        predicted, normalized = self.pathloss_and_normalized_cost(points)
        return {
            "communication_cost": float(np.mean(normalized)),
            "mean_predicted_pathloss_db": float(np.mean(predicted)),
            "worst_predicted_pathloss_db": float(np.max(predicted)),
        }

    def make_cost_functions(self):
        def scalar(x: float, y: float, z: float) -> float:
            return float(self.normalized_cost([[x, y, z]])[0])

        def vector(points: np.ndarray | list) -> np.ndarray:
            return self.normalized_cost(points)

        return scalar, vector


def load_hgnn_radio_map(
    path: str | Path, *, predicted_column: str = "Predicted_PathLoss", **kwargs
) -> HGNNRadioMap3D:
    return HGNNRadioMap3D.from_csv(
        path, predicted_column=predicted_column, **kwargs
    )
