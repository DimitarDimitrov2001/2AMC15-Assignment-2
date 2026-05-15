"""
Reward functions for grid-world training.
"""

from collections.abc import Callable

import numpy as np

from world.grid_codes import BOUNDARY_WALL_CELL, EMPTY_CELL, OBSTACLE_CELL, TARGET_CELL

RewardFunction = Callable[[np.ndarray, tuple[int, int]], int]

STEP_REWARD = -3
TARGET_REWARD = 10

WALL_OR_OBSTACLE_REWARD = -4
MIN_TARGET_REWARD = 10
DISTANCE_MULTIPLIER = 5.0
DISTANCE_FROM_START_REWARD = 3.0


def _manhattan_distance(start_pos: tuple[int, int], target_pos: tuple[int, int]) -> int:
    return abs(start_pos[0] - target_pos[0]) + abs(start_pos[1] - target_pos[1])


def find_target_position(grid: np.ndarray) -> tuple[int, int]:
    """Find the single target cell in a grid."""

    target_cells = np.argwhere(grid == TARGET_CELL)
    if len(target_cells) != 1:
        raise ValueError(f"Expected exactly one target cell, found {len(target_cells)}.")

    target_col, target_row = target_cells[0]
    return int(target_col), int(target_row)


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


def build_manhattan_reward_function(start_pos: tuple[int, int], target_pos: tuple[int, int]) -> RewardFunction:
    """Build a reward function scaled to the start-target Manhattan distance.

    Unlike the basic assignment reward, this variant penalises wall/obstacle
    hits more heavily (``WALL_OR_OBSTACLE_REWARD``, currently -4 vs the basic
    step reward of -3) and scales the target reward with the start-target
    Manhattan distance to encourage faster convergence on larger grids. Empty
    cells get an additional distance-from-start shaping term so making
    forward progress is rewarded over staying put.
    """
    full_distance = _manhattan_distance(start_pos, target_pos)
    target_reward = max(MIN_TARGET_REWARD, DISTANCE_MULTIPLIER * full_distance)

    def reward_function(grid: np.ndarray, agent_pos: tuple[int, int]) -> int:
        cell_value = int(grid[agent_pos])
        if cell_value == EMPTY_CELL:
            distance_from_start = _manhattan_distance(start_pos=start_pos, target_pos=agent_pos)
            return STEP_REWARD + (distance_from_start / full_distance) * DISTANCE_FROM_START_REWARD
        if cell_value in (BOUNDARY_WALL_CELL, OBSTACLE_CELL):
            return WALL_OR_OBSTACLE_REWARD
        if cell_value == TARGET_CELL:
            return target_reward

        raise ValueError(f"Grid cell should not have value {cell_value} at position {agent_pos}.")

    return reward_function
