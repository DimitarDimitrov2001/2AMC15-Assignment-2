"""Replay buffer used by the DQN-style agents.

Transitions are kept in fixed-size numpy arrays, then sampled back as torch
tensors on the agent's device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agents.base_agent import Transition
from agents.defaults import REPLAY_DEFAULT_CAPACITY, REPLAY_DEFAULT_START_SIZE

@dataclass(frozen=True)
class Batch:
    """One replay minibatch, onverted to torch tensors."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Uniform replay buffer with a fixed capacity.

    Once the buffer is full, new transitions overwrite the oldest ones.
    """

    _capacity: int
    _replay_start_size: int
    _obs_dim: int
    _device: torch.device
    _rng: np.random.Generator
    _states: np.ndarray
    _next_states: np.ndarray
    _actions: np.ndarray
    _rewards: np.ndarray
    _dones: np.ndarray
    _size: int
    _next_idx: int

    def __init__(
        self,
        obs_dim: int,
        capacity: int | None = None,
        replay_start_size: int | None = None,
        seed: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Allocate storage for up to ``capacity`` transitions."""
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")

        capacity = capacity if capacity is not None else REPLAY_DEFAULT_CAPACITY
        self._capacity = capacity
        self._replay_start_size = replay_start_size if replay_start_size else REPLAY_DEFAULT_START_SIZE
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._obs_dim = obs_dim
        self._device = torch.device(device)
        self._rng = np.random.default_rng(seed)

        self._states = np.zeros((self._capacity, obs_dim), dtype=np.float32)
        self._next_states = np.zeros((self._capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(self._capacity, dtype=np.int64)
        self._rewards = np.zeros(self._capacity, dtype=np.float32)
        self._dones = np.zeros(self._capacity, dtype=np.float32)

        self._size = 0
        self._next_idx = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition and advance the ring-buffer pointer."""
        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        if state_arr.shape != (self._obs_dim,):
            raise ValueError(f"state must have shape ({self._obs_dim},), got {state_arr.shape}")
        if next_state_arr.shape != (self._obs_dim,):
            raise ValueError(
                f"next_state must have shape ({self._obs_dim},), got {next_state_arr.shape}"
            )

        idx = self._next_idx
        self._states[idx] = state_arr
        self._next_states[idx] = next_state_arr
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._dones[idx] = float(done)

        self._next_idx = (idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def add_transition(self, transition: Transition) -> None:
        """Convenience wrapper for the BaseAgent transition object."""
        self.add(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            done=transition.terminated,
        )

    def sample(self, batch_size: int) -> Batch:
        """Draw a minibatch once the warmup threshold has been reached."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.can_sample(batch_size):
            raise ValueError(
                f"cannot sample {batch_size} transitions from buffer of size {self._size}"
            )

        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return Batch(
            states=torch.as_tensor(self._states[indices], dtype=torch.float32, device=self._device),
            actions=torch.as_tensor(self._actions[indices], dtype=torch.int64, device=self._device),
            rewards=torch.as_tensor(self._rewards[indices], dtype=torch.float32, device=self._device),
            next_states=torch.as_tensor(
                self._next_states[indices], dtype=torch.float32, device=self._device
            ),
            dones=torch.as_tensor(self._dones[indices], dtype=torch.float32, device=self._device),
        )

    def can_sample(self, batch_size: int) -> bool:
        """Return whether the buffer is warm enough for a training batch."""
        if batch_size > self._replay_start_size:
            return self._size >= batch_size
        else:
            return self._size >= self._replay_start_size

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._size
