from abc import ABC, abstractmethod
from typing import Hashable
import numpy as np

_DEFAULT_EPSILON_MAX = 1.0
_DEFAULT_EPSILON_MIN = 0.05
_DEFAULT_EPSILON_DECAY = 0.95
_DEFAULT_EPSILON = 0.1

class EpsilonSchedule(ABC):
    @abstractmethod
    def epsilon(self, state: Hashable | np.ndarray | None) -> float:
        """Return exploration rate for this decision (global or state-specific)"""

    def on_episode_end(self) -> None:
        """Advance per-episode schedules; no-op for fixed/state-only schedules."""
        return None

class ConstantEpsilon(EpsilonSchedule):
    def __init__(self, epsilon: float = _DEFAULT_EPSILON) -> None:
        super().__init__()
        self._epsilon = epsilon

    def epsilon(self, state: Hashable | np.ndarray | None) -> float:
        return self._epsilon

class ExponentialEpsilonDecay(EpsilonSchedule):
    """Multiply epsilon by decay factor each episode, clamp to epsilon_min."""
    def __init__(self, decay: float = _DEFAULT_EPSILON_DECAY, epsilon_max: float = _DEFAULT_EPSILON_MAX, epsilon_min: float = _DEFAULT_EPSILON_MIN) -> None:
        super().__init__()
        self._decay = decay
        self._epsilon_max = epsilon_max
        self._epsilon_min = epsilon_min
        self._epsilon = epsilon_max

    def epsilon(self, state: Hashable | np.ndarray | None = None) -> float:
        return self._epsilon

    def on_episode_end(self) -> None:
        self._epsilon = max(self._epsilon_min, self._epsilon * self._decay)