"""
Reward functions for grid-world training.
"""

from collections.abc import Callable

import numpy as np

from world.grid_codes import BOUNDARY_WALL_CELL, EMPTY_CELL, OBSTACLE_CELL, TARGET_CELL

RewardFunction = Callable[[np.ndarray, tuple[int, int]], int]

STEP_REWARD = -1
TARGET_REWARD = 10


def build_basic_reward_function() -> RewardFunction:
    """Build the reward function described in the assignment specification.

    Every step (including bumping into a wall or obstacle, where the agent
    stays in place) gives -1. Reaching the delivery destination gives +10.
    """

    def reward_function(grid: np.ndarray, agent_pos: tuple[int, int]) -> int:
        cell_value = int(grid[agent_pos])
        if cell_value == TARGET_CELL:
            return TARGET_REWARD
        if cell_value in (EMPTY_CELL, BOUNDARY_WALL_CELL, OBSTACLE_CELL):
            return STEP_REWARD
        raise ValueError(f"Grid cell should not have value {cell_value} at position {agent_pos}.")

    return reward_function
