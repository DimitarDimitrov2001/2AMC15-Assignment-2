from pathlib import Path

from world.grid import Grid
from world.grid_codes import (
    BOUNDARY_WALL_CELL,
    EMPTY_CELL,
    OBSTACLE_CELL,
    START_CELL,
    TARGET_CELL,
)
from world.gui import GUI
from world.environment import Environment
from world.rewards import build_basic_reward_function, build_manhattan_reward_function, find_target_position


GRID_CONFIGS_FP = Path(__file__).parents[1].resolve() / Path("grid_configs")
GRID_CONFIGS_FP.mkdir(parents=True, exist_ok=True)

__all__ = [
    "BOUNDARY_WALL_CELL",
    "EMPTY_CELL",
    "Environment",
    "GRID_CONFIGS_FP",
    "GUI",
    "Grid",
    "OBSTACLE_CELL",
    "START_CELL",
    "TARGET_CELL",
    "build_basic_reward_function",
    "build_manhattan_reward_function",
    "find_target_position",
]

