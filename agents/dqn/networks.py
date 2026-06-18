from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    """Small MLP that maps a normalized state vector to one Q-value per action."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")

        layers: list[nn.Module] = []
        in_features = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, int(hidden_size)))
            layers.append(nn.ReLU())
            in_features = int(hidden_size)
        layers.append(nn.Linear(in_features, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)
