from abc import ABC, abstractmethod
from typing import Hashable
import numpy as np

from agents.defaults import (
    EPSILON_DEFAULT_MAX,
    EPSILON_DEFAULT_MIN,
    EPSILON_DEFAULT_DECAY,
    EPSILON_ANNEAL_DURATION,
    EPSILON_ANNEAL_START_STEP,
)

class EpsilonSchedule(ABC):
    @abstractmethod
    def epsilon(self, state: Hashable | np.ndarray | None) -> float:
        """Return exploration rate for this decision (global or state-specific)"""

    def step(self) -> None:
        """Advance per-step schedules."""
        return None

    def on_episode_end(self) -> None:
        """Advance per-episode schedules; no-op for fixed/state-only schedules by default."""
        return None

class ConstantEpsilon(EpsilonSchedule):
    def __init__(self, epsilon: float = EPSILON_DEFAULT_MAX) -> None:
        super().__init__()
        self._epsilon = epsilon

    def epsilon(self, state: Hashable | np.ndarray | None) -> float:
        return self._epsilon

class LinearEpsilonAnnealing(EpsilonSchedule):
    """Linearly anneal epsilon from max to min over a fixed number of steps.
    
    Annealing only starts after `start_step`.
    """
    def __init__(
        self,
        duration: int = EPSILON_ANNEAL_DURATION,
        start_step: int = EPSILON_ANNEAL_START_STEP,
        epsilon_max: float = EPSILON_DEFAULT_MAX,
        epsilon_min: float = EPSILON_DEFAULT_MIN,
    ) -> None:
        super().__init__()
        self._duration = duration
        self._start_step = start_step
        self._epsilon_max = epsilon_max
        self._epsilon_min = epsilon_min
        self._steps = 0
        self._epsilon = epsilon_max

    def epsilon(self, state: Hashable | np.ndarray | None = None) -> float:
        return self._epsilon

    def step(self) -> None:
        self._steps += 1
        if self._steps < self._start_step:
            self._epsilon = self._epsilon_max
        elif self._steps < self._start_step + self._duration:
            progress = (self._steps - self._start_step) / self._duration
            self._epsilon = self._epsilon_max - progress * (self._epsilon_max - self._epsilon_min)
        else:
            self._epsilon = self._epsilon_min

class ExponentialEpsilonDecay(EpsilonSchedule):
    """Multiply epsilon by decay factor each episode, clamp to epsilon_min."""
    def __init__(
        self,
        decay: float = EPSILON_DEFAULT_DECAY,
        epsilon_max: float = EPSILON_DEFAULT_MAX,
        epsilon_min: float = EPSILON_DEFAULT_MIN,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._epsilon_max = epsilon_max
        self._epsilon_min = epsilon_min
        self._epsilon = epsilon_max

    def epsilon(self, state: Hashable | np.ndarray | None = None) -> float:
        return self._epsilon

    def on_episode_end(self) -> None:
        self._epsilon = max(self._epsilon_min, self._epsilon * self._decay)
