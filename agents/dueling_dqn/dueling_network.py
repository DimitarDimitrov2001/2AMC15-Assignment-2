from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    """Dueling Q-network with a shared MLP trunk and separate value/action heads.

    The network returns Q-values using the standard aggregation
    ``Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)``. Subtracting the mean keeps
    the split between the value and advantage streams well-defined.
    """

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
        hidden_sizes = tuple(hidden_sizes)
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one layer")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")

        # Shared feature extractor.
        trunk_layers: list[nn.Module] = []
        in_features = state_dim
        for hidden_size in hidden_sizes:
            trunk_layers.append(nn.Linear(in_features, int(hidden_size)))
            trunk_layers.append(nn.ReLU())
            in_features = int(hidden_size)
        self.trunk = nn.Sequential(*trunk_layers)

        # Separate value and advantage projections.
        self.value_stream = nn.Linear(in_features, 1)
        self.advantage_stream = nn.Linear(in_features, n_actions)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return Q-values for one state ``(state_dim,)`` or a batch ``(B, state_dim)``."""
        single_input = states.dim() == 1
        if single_input:
            states = states.unsqueeze(0)

        features = self.trunk(states)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        if single_input:
            return q_values.squeeze(0)
        return q_values