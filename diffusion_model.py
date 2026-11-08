"""Conditional DDPM used as the trajectory-manifold prior."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass


def sinusoidal_embedding(time_step: torch.Tensor, dimension: int = 64) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=time_step.device)
        / max(half, 1)
    )
    arguments = time_step[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
    if dimension % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros(time_step.size(0), 1, device=time_step.device)],
            dim=-1,
        )
    return embedding


class ConditionalTrajectoryDDPM(nn.Module):
    """Noise-prediction DDPM conditioned on start and destination in 3D."""

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        dimension_x: int = 30,
        dimension_condition: int = 6,
        hidden: int = 256,
        time_dimension: int = 64,
        diffusion_steps: int = 50,
    ) -> None:
        super().__init__()
        self.dimension_x = int(dimension_x)
        self.dimension_condition = int(dimension_condition)
        self.hidden = int(hidden)
        self.time_dimension = int(time_dimension)
        self.diffusion_steps = int(diffusion_steps)

        beta = np.linspace(1e-4, 0.02, self.diffusion_steps).astype(np.float32)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("alpha_bar", torch.tensor(alpha_bar))

        self.time_embedding = nn.Sequential(
            nn.Linear(self.time_dimension, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.network = nn.Sequential(
            nn.Linear(
                self.dimension_x + self.dimension_condition + self.hidden,
                self.hidden,
            ),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.dimension_x),
        )

    def forward(
        self, noisy_trajectory: torch.Tensor, time_step: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(
            sinusoidal_embedding(time_step, self.time_dimension)
        )
        features = torch.cat([noisy_trajectory, condition, time_embedding], dim=-1)
        return self.network(features)

    def q_sample(
        self, clean: torch.Tensor, time_step: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bar[time_step].unsqueeze(-1)
        noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
        return noisy, noise

    def training_loss(self, clean: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = clean.size(0)
        time_step = torch.randint(
            0, self.diffusion_steps, (batch_size,), device=clean.device
        )
        noisy, target_noise = self.q_sample(clean, time_step)
        predicted_noise = self.forward(noisy, time_step, condition)
        return ((predicted_noise - target_noise) ** 2).mean()

    def _reverse_step(
        self,
        noisy: torch.Tensor,
        condition: torch.Tensor,
        time_step: int,
        *,
        stochastic: bool,
        stochastic_scale: float = 1.0,
    ) -> torch.Tensor:
        batch_size = noisy.size(0)
        time_vector = torch.full(
            (batch_size,), time_step, device=noisy.device, dtype=torch.long
        )
        predicted_noise = self.forward(noisy, time_vector, condition)
        alpha = self.alpha[time_step]
        beta = self.beta[time_step]
        alpha_bar = self.alpha_bar[time_step]
        mean = (noisy - beta * predicted_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
        if time_step > 0 and stochastic:
            mean = (
                mean
                + float(stochastic_scale)
                * torch.sqrt(beta)
                * torch.randn_like(mean)
            )
        return mean

    @torch.no_grad()
    def sample(self, condition: np.ndarray, population_size: int) -> np.ndarray:
        condition_tensor = torch.tensor(
            np.asarray(condition, dtype=np.float32), device=DEVICE
        ).reshape(1, -1).repeat(population_size, 1)
        trajectory = torch.randn(population_size, self.dimension_x, device=DEVICE)
        self.eval()
        for time_step in range(self.diffusion_steps - 1, -1, -1):
            trajectory = self._reverse_step(
                trajectory, condition_tensor, time_step, stochastic=True
            )
        return trajectory.cpu().numpy()

    @torch.no_grad()
    def truncated_denoise(
        self,
        elite: np.ndarray,
        condition: np.ndarray,
        truncation_step: int,
        *,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """Forward-noise elites to ``t`` and run the truncated reverse process."""
        elite_tensor = torch.tensor(np.asarray(elite, dtype=np.float32), device=DEVICE)
        condition_tensor = torch.tensor(
            np.asarray(condition, dtype=np.float32), device=DEVICE
        )
        if condition_tensor.ndim == 1:
            condition_tensor = condition_tensor.unsqueeze(0)
        if condition_tensor.size(0) == 1 and elite_tensor.size(0) > 1:
            condition_tensor = condition_tensor.repeat(elite_tensor.size(0), 1)
        truncation_step = int(
            np.clip(truncation_step, 1, self.diffusion_steps - 1)
        )
        noise_scale = float(noise_scale)
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative")
        alpha_bar = self.alpha_bar[truncation_step]
        noisy = (
            torch.sqrt(alpha_bar) * elite_tensor
            + noise_scale
            * torch.sqrt(1.0 - alpha_bar)
            * torch.randn_like(elite_tensor)
        )
        self.eval()
        for time_step in range(truncation_step, -1, -1):
            noisy = self._reverse_step(
                noisy,
                condition_tensor,
                time_step,
                stochastic=time_step > 0,
                stochastic_scale=noise_scale,
            )
        return noisy.cpu().numpy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": self.CHECKPOINT_VERSION,
                "state_dict": self.state_dict(),
                "dimension_x": self.dimension_x,
                "dimension_condition": self.dimension_condition,
                "hidden": self.hidden,
                "time_dimension": self.time_dimension,
                "diffusion_steps": self.diffusion_steps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device = DEVICE):
        checkpoint = torch.load(Path(path), map_location=device)
        if checkpoint.get("version") != cls.CHECKPOINT_VERSION:
            raise ValueError("Checkpoint is not an A*-free CA-Diffusion-ES prior")
        model = cls(
            dimension_x=checkpoint["dimension_x"],
            dimension_condition=checkpoint["dimension_condition"],
            hidden=checkpoint["hidden"],
            time_dimension=checkpoint["time_dimension"],
            diffusion_steps=checkpoint["diffusion_steps"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model


def train_prior(
    model: ConditionalTrajectoryDDPM,
    trajectories: np.ndarray,
    conditions: np.ndarray,
    *,
    epochs: int = 300,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    seed_everything(seed)
    trajectory_tensor = torch.tensor(
        np.asarray(trajectories, dtype=np.float32), device=DEVICE
    )
    condition_tensor = torch.tensor(
        np.asarray(conditions, dtype=np.float32), device=DEVICE
    )
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    model.train()
    sample_count = trajectory_tensor.size(0)
    for _ in range(int(epochs)):
        permutation = torch.randperm(sample_count, device=DEVICE)
        losses = []
        for start in range(0, sample_count, batch_size):
            indices = permutation[start : start + batch_size]
            loss = model.training_loss(
                trajectory_tensor[indices], condition_tensor[indices]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)))
    model.eval()
    return history
