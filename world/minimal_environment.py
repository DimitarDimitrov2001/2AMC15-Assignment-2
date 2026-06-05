"""
Continuous-space grid-world environment for deep reinforcement learning.

The Assignment 1 grid is reused as the map: each integer cell (i, j) becomes
a 1x1 unit square in continuous space.  The agent navigates as a point with
real-valued position (x, y).

Coordinate convention (mirrors the discrete Environment)
---------------------------------------------------------
  pos[0] (x)  <->  first grid-array index
  pos[1] (y)  <->  second grid-array index
  grid[floor(x), floor(y)]  ->  cell the agent currently occupies

Action encoding (same integers as the discrete environment)
-----------------------------------------------------------
  0  down   (+y)
  1  up     (-y)
  2  left   (-x)
  3  right  (+x)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from world.grid import Grid
from world.grid_codes import (
    BOUNDARY_WALL_CELL,
    EMPTY_CELL,
    OBSTACLE_CELL,
    START_CELL,
    TARGET_CELL,
)


# Unit direction vectors per action, scaled by step_size at runtime.
# Encoding matches helpers.ACTIONS_TO_DIRECTIONS so action integers are
# interchangeable between the discrete and continuous environments.
ACTION_DELTAS: dict[int, np.ndarray] = {
    0: np.array([0.0,  1.0]),   # down
    1: np.array([0.0, -1.0]),   # up
    2: np.array([-1.0, 0.0]),   # left
    3: np.array([1.0,  0.0]),   # right
}
N_ACTIONS: int = len(ACTION_DELTAS)

# Default reward values (pre-assignment 2 spec)
_GOAL_REWARD: float = 1.0
_LIVING_PENALTY: float = -0.01
_COLLISION_PENALTY: float = -1.0


class MinimalEnvironment:
    """Grid-world with continuous (x, y) agent position.

    Loads the same .npy grid files as Assignment 1. Exposes the same
    (state, reward, terminated, info) step interface, so training loops
    written for the discrete Environment require minimal changes.

    The state returned by reset() and step() is np.ndarray of shape (2,):
        [x, y]  -- both floats in [0, grid_dim).
    """

    def __init__(
        self,
        grid_fp: Path,
        step_size: float = 0.5,
        sigma: float = 0.0,
        agent_start_pos: tuple[float, float] | None = None,
        reward_fn: callable | None = None,
        random_seed: int = 0,
    ):
        """
        Args:
            grid_fp:          Path to a .npy grid file (Assignment 1 format).
            step_size:        Distance delta the agent moves per action.
            sigma:            Probability that a random action replaces the
                              chosen one (same semantics as the discrete env).
            agent_start_pos:  Optional fixed (x, y) start. If None, the centre
                              of the grid's START_CELL (value 4) is used; if no
                              start cell is marked a random empty cell is chosen.
            reward_fn:        Custom reward function with signature
                                fn(grid, pos, new_pos, collision) -> float
                              Defaults to the pre-assignment spec rewards.
            random_seed:      Seed for the internal RNG.
        """
        if not Path(grid_fp).exists():
            raise FileNotFoundError(f"Grid file not found: {grid_fp}")

        self.grid_fp = Path(grid_fp)
        self.step_size = step_size
        self.sigma = sigma
        self.agent_start_pos = agent_start_pos
        self.reward_fn = reward_fn if reward_fn is not None else _default_reward
        self._rng = random.Random(random_seed)

        # Populated on reset()
        self.grid: np.ndarray | None = None
        self.pos: np.ndarray | None = None
        self.terminal: bool = False
        self.world_stats: dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reload the grid and place the agent at the start position.

        Returns:
            Initial state [x, y] as np.ndarray of shape (2,).
        """
        self.grid = Grid.load_grid(self.grid_fp).cells
        self.terminal = False
        self.world_stats = {
            "cumulative_reward": 0.0,
            "total_steps": 0,
            "total_collisions": 0,
            "total_agent_moves": 0,
            "targets_reached": 0,
        }

        self.pos = (
            np.array(self.agent_start_pos, dtype=float)
            if self.agent_start_pos is not None
            else self._find_start()
        )
        return self.pos.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Move the agent one step in the chosen direction.

        If the new position would land inside a wall or outside the grid the
        agent stays in place and receives the collision penalty.

        Args:
            action: Integer in {0, 1, 2, 3}.

        Returns:
            state:      np.ndarray([x, y]) after the step.
            reward:     Scalar float.
            terminated: True when all targets have been collected.
            info:       Dict with keys "collision", "action", "pos".
        """
        assert self.grid is not None, "Call reset() before step()."

        self.world_stats["total_steps"] += 1

        # Optional action stochasticity (same sigma semantics as A1)
        if self._rng.random() < self.sigma:
            action = self._rng.randrange(N_ACTIONS)

        new_pos = self.pos + ACTION_DELTAS[action] * self.step_size
        collision = self._is_collision(new_pos)

        reward = self.reward_fn(self.grid, self.pos, new_pos, collision)

        if collision:
            self.world_stats["total_collisions"] += 1
        else:
            self.pos = new_pos
            self.world_stats["total_agent_moves"] += 1

            i, j = _cell(self.pos)
            if self.grid[i, j] == TARGET_CELL:
                self.grid[i, j] = EMPTY_CELL
                self.world_stats["targets_reached"] += 1
                if not np.any(self.grid == TARGET_CELL):
                    self.terminal = True

        self.world_stats["cumulative_reward"] += reward

        return (
            self.pos.copy(),
            reward,
            self.terminal,
            {"collision": collision, "action": action, "pos": self.pos.copy()},
        )

    # ------------------------------------------------------------------
    # Properties useful for building the DQN network
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector (2 for the minimal environment)."""
        return 2

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def grid_shape(self) -> tuple[int, int]:
        assert self.grid is not None, "Call reset() first."
        return tuple(self.grid.shape)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_start(self) -> np.ndarray:
        """Return the continuous centre of the START_CELL, or a random
        empty cell if no start marker is present in the grid."""
        starts = np.argwhere(self.grid == START_CELL)
        if len(starts):
            i, j = starts[0]
            self.grid[i, j] = EMPTY_CELL
            return np.array([i + 0.5, j + 0.5], dtype=float)

        empty = np.argwhere(self.grid == EMPTY_CELL)
        i, j = empty[self._rng.randrange(len(empty))]
        return np.array([i + 0.5, j + 0.5], dtype=float)

    def _is_collision(self, pos: np.ndarray) -> bool:
        """True if pos is out of bounds or inside a wall/obstacle cell."""
        x, y = pos
        dim_i, dim_j = self.grid.shape
        if x < 0.0 or y < 0.0 or x >= dim_i or y >= dim_j:
            return True
        i, j = _cell(pos)
        return int(self.grid[i, j]) in (BOUNDARY_WALL_CELL, OBSTACLE_CELL)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _cell(pos: np.ndarray) -> tuple[int, int]:
    """Grid indices (i, j) for a continuous position (x, y)."""
    return int(np.floor(pos[0])), int(np.floor(pos[1]))


def _default_reward(
    grid: np.ndarray,
    pos: np.ndarray,
    new_pos: np.ndarray,
    collision: bool,
) -> float:
    """
      +1.0   reaching the target
      -0.01  living penalty each step
      -1.0   collision with wall or obstacle
    """
    if collision:
        return _COLLISION_PENALTY
    i, j = _cell(new_pos)
    if grid[i, j] == TARGET_CELL:
        return _GOAL_REWARD
    return _LIVING_PENALTY
