"""Canonical local-frame representation for conditional trajectory priors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalTrajectoryFrame:
    """Translate, rotate and scale a trajectory by its endpoint geometry."""

    start: np.ndarray
    end: np.ndarray
    basis: np.ndarray
    scale: float

    @classmethod
    def from_endpoints(
        cls, start: np.ndarray, end: np.ndarray
    ) -> "LocalTrajectoryFrame":
        start = np.asarray(start, dtype=float).reshape(3)
        end = np.asarray(end, dtype=float).reshape(3)
        horizontal = end[:2] - start[:2]
        horizontal_distance = float(np.linalg.norm(horizontal))
        if horizontal_distance <= 1e-9:
            direction = np.asarray([1.0, 0.0], dtype=float)
        else:
            direction = horizontal / horizontal_distance
        basis = np.asarray(
            [
                [direction[0], -direction[1], 0.0],
                [direction[1], direction[0], 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return cls(
            start=start,
            end=end,
            basis=basis,
            scale=max(horizontal_distance, 1.0),
        )

    @property
    def condition(self) -> np.ndarray:
        """Rotation-invariant endpoint condition used by the diffusion model."""
        return np.asarray(
            [
                np.linalg.norm(self.end[:2] - self.start[:2]),
                0.5 * (self.start[2] + self.end[2]),
                self.end[2] - self.start[2],
            ],
            dtype=np.float32,
        )

    def world_to_local(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return ((points - self.start) @ self.basis) / self.scale

    def local_to_world(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return self.start + (points * self.scale) @ self.basis.T

    def baseline(self, count: int) -> np.ndarray:
        progress = np.linspace(0.0, 1.0, int(count))
        local_end = self.world_to_local(self.end)
        return progress[:, None] * local_end[None, :]

    def encode_residual(self, controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls, dtype=float).reshape(-1, 3)
        return (self.world_to_local(controls) - self.baseline(len(controls))).reshape(-1)

    def decode_residual(self, residual: np.ndarray) -> np.ndarray:
        residual = np.asarray(residual, dtype=float).reshape(-1, 3)
        return self.local_to_world(residual + self.baseline(len(residual)))
