from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

"""
Agent Base.
We define the base class for all agents in this file.
"""
@dataclass(frozen=True)
class Transition:
    # One interaction (state -> action -> reward -> next_state) between the agent and environment
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class BaseAgent(ABC):
    # Base interface for all agents. Trainer only knows this interface
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        # Choose one discrete action
        raise NotImplementedError

    def observe(self, transition: Transition) -> None:
        # Receive one transition from trainer
        return None

    def update(self) -> dict[str, float]:
        # Do one learning update
        return {}

    def on_episode_start(self, episode: int) -> None:
        # Option to do something at the start of each episode
        return None

    def on_episode_end(self, episode: int, episode_metrics: dict[str, float]) -> dict[str, float]:
        # Option to do something at the end of each episode
        return {}

    def save_checkpoint(self, path: str) -> None:
        # Persist agent state (e.g. network weights). No-op for stateless agents.
        return None

    def load_checkpoint(self, path: str) -> None:
        # Restore agent state from a checkpoint. No-op for stateless agents.
        return None