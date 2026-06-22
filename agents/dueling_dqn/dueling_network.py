from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    """
    Dueling architecture (Wang et al., 2016).

    Shares the ENTIRE hidden trunk (matching plain QNetwork's depth exactly),
    then splits only at the final projection into:
      - value_stream:      features -> scalar V(s)
      - advantage_stream:  features -> one A(s, a) per action

    Recombined as:
        Q(s, a) = V(s) + ( A(s, a) - mean_a' A(s, a') )

    The mean-subtraction is what makes V and A separately identifiable --
    without it, Q = V + A has infinitely many (V, A) decompositions that
    produce the same Q, so the two streams would carry no individual meaning.

    Design note: an earlier version held back the *last* hidden layer from
    the shared trunk and gave each stream its own extra hidden layer. That
    weakened the shared feature extractor (one fewer shared layer than
    QNetwork) and doubled the from-scratch parameters each stream had to
    learn, which let the advantage head collapse toward a near
    state-independent ordering -- i.e. the greedy policy picked roughly the
    same action everywhere, even though V(s) converged fine. Sharing the
    full trunk and keeping the heads to a single linear layer each avoids
    that: both streams reuse the same well-trained features QNetwork would
    have used for its single output layer.

    Same constructor signature and input/output shapes as a plain QNetwork
    (state_dim in, n_actions out) -- this is intentional so it can be used
    anywhere a QNetwork is expected.
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

        # Shared trunk: ALL hidden layers, same depth as QNetwork.
        trunk_layers: list[nn.Module] = []
        in_features = state_dim
        for hidden_size in hidden_sizes:
            trunk_layers.append(nn.Linear(in_features, int(hidden_size)))
            trunk_layers.append(nn.ReLU())
            in_features = int(hidden_size)
        self.trunk = nn.Sequential(*trunk_layers)

        # Thin heads: a single linear layer each, splitting only at the
        # very last projection -- matches the paper's split point.
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