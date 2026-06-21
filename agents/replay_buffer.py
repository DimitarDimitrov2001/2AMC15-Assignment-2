"""Experience replay buffer for off-policy deep RL agents.

The buffer is a reusable component owned by the agent (e.g. DQN), not the
Trainer. It stores transitions in preallocated numpy ring buffers and samples
uniform random minibatches as torch tensors placed on the configured device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agents.base_agent import Transition
from agents.defaults import REPLAY_DEFAULT_CAPACITY, REPLAY_DEFAULT_START_SIZE

@dataclass(frozen=True)
class Batch:
    """A sampled minibatch of transitions as device-placed torch tensors."""

    states: torch.Tensor       # (batch, obs_dim) float32
    actions: torch.Tensor      # (batch,) int64
    rewards: torch.Tensor      # (batch,) float32
    next_states: torch.Tensor  # (batch, obs_dim) float32
    dones: torch.Tensor        # (batch,) float32


class ReplayBuffer:
    """Fixed-capacity uniform experience replay buffer.

    Transitions are stored in preallocated numpy arrays that are overwritten
    in a ring once capacity is reached, so memory usage is bounded and adds
    are O(1).
    """

    # Private fields declared with types up front.
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
        """Create an empty buffer.

        Args:
            obs_dim: Length of the (flat) state vector.
            capacity: Maximum number of transitions retained.
            replay_start_size: Minimum fill before sampling is allowed.
            seed: Optional seed for reproducible sampling.
            device: Device that sampled tensors are placed on.
        """
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
        """Store a single transition, overwriting the oldest when full."""
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
        """Store a :class:`Transition`. ``terminated`` is treated as ``done``."""
        self.add(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            done=transition.terminated,
        )

    def sample(self, batch_size: int) -> Batch:
        """Sample a uniform random minibatch without replacement.

        Args:
            batch_size: Number of transitions to draw.

        Returns:
            A :class:`Batch` of torch tensors placed on the buffer's device.

        Raises:
            ValueError: If fewer than ``batch_size`` transitions are stored.
        """
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
        """True if the buffer holds at least ``replay_start_size`` transitions or batch size if batch size > replay start size."""
        if batch_size > self._replay_start_size:
            return self._size >= batch_size
        else:
            return self._size >= self._replay_start_size

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._size
