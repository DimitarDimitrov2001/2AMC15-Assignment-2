"""
Reward functions for grid-world training.
"""

from collections.abc import Callable

import numpy as np

from world.grid_codes import BOUNDARY_WALL_CELL, EMPTY_CELL, OBSTACLE_CELL, TARGET_CELL

RewardFunction = Callable[[np.ndarray, tuple[int, int]], int]

STEP_REWARD = -1
WALL_OR_OBSTACLE_REWARD = -5
MIN_TARGET_REWARD = 10


def _manhattan_distance(start_pos: tuple[int, int], target_pos: tuple[int, int]) -> int:
    return abs(start_pos[0] - target_pos[0]) + abs(start_pos[1] - target_pos[1])


def find_target_position(grid: np.ndarray) -> tuple[int, int]:
    """Find the single target cell in a grid."""

    target_cells = np.argwhere(grid == TARGET_CELL)
    if len(target_cells) != 1:
        raise ValueError(f"Expected exactly one target cell, found {len(target_cells)}.")

    target_col, target_row = target_cells[0]
    return int(target_col), int(target_row)


def build_manhattan_reward_function(start_pos: tuple[int, int], target_pos: tuple[int, int]) -> RewardFunction:
    """Build a reward function scaled to the start-target Manhattan distance."""

    target_reward = max(MIN_TARGET_REWARD, 2 * _manhattan_distance(start_pos, target_pos))

    def reward_function(grid: np.ndarray, agent_pos: tuple[int, int]) -> int:
        cell_value = int(grid[agent_pos])
        if cell_value == EMPTY_CELL:
            return STEP_REWARD
        if cell_value in (BOUNDARY_WALL_CELL, OBSTACLE_CELL):
            return WALL_OR_OBSTACLE_REWARD
        if cell_value == TARGET_CELL:
            return target_reward

        raise ValueError(f"Grid cell should not have value {cell_value} at position {agent_pos}.")

    return reward_function
