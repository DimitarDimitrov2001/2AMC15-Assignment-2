import abc
import math
import numpy as np
from typing import Tuple, Union

from agents.defaults import BETA_DEFAULT

class IntrinsicMotivation(abc.ABC):
    """
    Abstract Base Class for all intrinsic motivation (curiosity) modules.
    Any new exploration strategy must implement these methods.
    """
    def __init__(self, beta: float):
        """
        Args:
            beta: Scaling factor for the intrinsic reward.
        """
        self.beta = beta

    @abc.abstractmethod
    def get_bonus(self, state: Union[np.ndarray, Tuple[float, float]]) -> float:
        """Calculates the intrinsic reward for a given state without updating counts."""
        pass

    @abc.abstractmethod
    def update(self, state: Union[np.ndarray, Tuple[float, float]]) -> None:
        """Updates the internal visitation counts or density model for the state."""
        pass

    def get_bonus_and_update(self, state: Union[np.ndarray, Tuple[float, float]]) -> float:
        """
        Convenience method: gets the bonus for the current state, 
        then updates the count for future visits.
        """
        bonus = self.get_bonus(state)
        self.update(state)
        return bonus
    
class NoMotivation(IntrinsicMotivation):
    """
    No intrinsic motivation, always returns 0
    """
    def __init__(self, beta: float| None = None):
        """
        Args:
            beta: Scaling factor for the intrinsic reward.
        """
        super().__init__(0.0)

    def get_bonus(self, state: Union[np.ndarray, Tuple[float, float]]) -> float:
        """Calculates the intrinsic reward for a given state without updating counts."""
        return 0.0

    def update(self, state: Union[np.ndarray, Tuple[float, float]]) -> None:
        """Updates the internal visitation counts or density model for the state."""
        pass

    def get_bonus_and_update(self, state: Union[np.ndarray, Tuple[float, float]]) -> float:
        """
        Convenience method: gets the bonus for the current state, 
        then updates the count for future visits.
        """
        return 0.0

class GridCountMotivation(IntrinsicMotivation):
    """
    Highly optimized exploration module for bounded grid environments.
    Uses contiguous NumPy memory for O(1) lookups and avoids Python C-API overhead.
    """
    def __init__(self, max_x: float, max_y: float, step_size: float, beta: float = BETA_DEFAULT):
        super().__init__(beta)
        self.step_size = step_size
        
        # Calculate the integer dimensions of the grid (+1 for inclusive bounds)
        self.shape_x = int((max_x / step_size)) + 1
        self.shape_y = int((max_y / step_size)) + 1
        
        # Pre-allocate memory using NumPy for maximum speed
        self.visit_counts = np.zeros((self.shape_x, self.shape_y), dtype=np.int32)

    def _get_indices(self, state: Union[np.ndarray, Tuple[float, float]]) -> Tuple[int, int]:
        """Helper to safely discretize continuous states into grid indices."""
        idx_x = int(state[0] / self.step_size)
        idx_y = int(state[1] / self.step_size)
        
        # Ensure we don't crash if the agent steps slightly out of bounds
        idx_x = max(0, min(idx_x, self.shape_x - 1))
        idx_y = max(0, min(idx_y, self.shape_y - 1))
        return idx_x, idx_y

    def get_bonus(self, state: Union[np.ndarray, Tuple[float, float]]) -> float:
        idx_x, idx_y = self._get_indices(state)
        count = self.visit_counts[idx_x, idx_y]
        
        # If never visited, return a large initial bonus to encourage stepping there
        if count == 0:
            return self.beta * 2.0 
            
        return self.beta / math.sqrt(count)

    def update(self, state: Union[np.ndarray, Tuple[float, float]]) -> None:
        idx_x, idx_y = self._get_indices(state)
        self.visit_counts[idx_x, idx_y] += 1