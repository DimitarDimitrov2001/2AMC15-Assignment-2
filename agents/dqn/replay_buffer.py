from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity replay buffer with uniform random sampling."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: torch.device | str,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self._pos = 0
        self._size = 0

        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        if state_arr.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},), got {state_arr.shape}")
        if next_state_arr.shape != (self.state_dim,):
            raise ValueError(
                f"next_state must have shape ({self.state_dim},), got {next_state_arr.shape}"
            )

        self._states[self._pos] = state_arr
        self._actions[self._pos] = int(action)
        self._rewards[self._pos] = float(reward)
        self._next_states[self._pos] = next_state_arr
        self._dones[self._pos] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError("not enough transitions in replay buffer")

        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(
            states=torch.as_tensor(self._states[indices], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(self._actions[indices], dtype=torch.long, device=self.device),
            rewards=torch.as_tensor(self._rewards[indices], dtype=torch.float32, device=self.device),
            next_states=torch.as_tensor(
                self._next_states[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            dones=torch.as_tensor(self._dones[indices], dtype=torch.float32, device=self.device),
        )
