"""
Minimal continuous-space grid-world environment for deep reinforcement learning.

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

from pathlib import Path
from typing import Any

import numpy as np

from world.defaults import DEFAULT_MINIMAL_ENVIRONMENT_SIGMA, DEFAULT_MINIMAL_ENVIRONMENT_STEP_SIZE
from world.environment_base import BaseGridEnvironment, RewardFn, cell_index
from world.grid_codes import EMPTY_CELL, TARGET_CELL

# Unit direction vectors per action, scaled by step_size at runtime.
# Encoding matches helpers.ACTIONS_TO_DIRECTIONS so action integers are
# interchangeable between the discrete and continuous environments.
ACTION_DELTAS: dict[int, np.ndarray] = {
    0: np.array([0.0, 1.0]),    # down
    1: np.array([0.0, -1.0]),   # up
    2: np.array([-1.0, 0.0]),   # left
    3: np.array([1.0, 0.0]),    # right
}
N_ACTIONS: int = len(ACTION_DELTAS)


class MinimalEnvironment(BaseGridEnvironment):
    """Grid-world with continuous (x, y) agent position.

    Loads the same .npy grid files as Assignment 1. The state returned by
    reset() and step() is np.ndarray of shape (2,):
        [x, y]  -- both floats in [0, grid_dim).
    """

    sigma: float

    def __init__(
        self,
        grid_fp: Path,
        step_size: float = DEFAULT_MINIMAL_ENVIRONMENT_STEP_SIZE,
        sigma: float = DEFAULT_MINIMAL_ENVIRONMENT_SIGMA,
        agent_start_pos: tuple[float, float] | None = None,
        reward_fn: RewardFn | None = None,
        random_seed: int = 0,
    ) -> None:
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
        super().__init__(
            grid_fp,
            step_size=step_size,
            agent_start_pos=agent_start_pos,
            reward_fn=reward_fn,
            random_seed=random_seed,
        )
        self.sigma = sigma

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
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
        assert self.grid is not None and self.pos is not None, "Call reset() before step()."

        self.world_stats["total_steps"] += 1

        # Optional action stochasticity (same sigma semantics as A1).
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

            i, j = cell_index(self.pos)
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

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector (2 for the minimal environment)."""
        return 2

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def observation_high(self) -> np.ndarray:
        """Upper bound of (x, y): the grid (rows, cols), so x/dim, y/dim in [0, 1)."""
        dim_i, dim_j = self._grid_dims()
        return np.array([dim_i, dim_j], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_state(self) -> np.ndarray:
        """Return the (x, y) observation vector."""
        assert self.pos is not None, "Call reset() first."
        return self.pos.copy()

    def _is_collision(self, pos: np.ndarray) -> bool:
        """True if pos is out of bounds or inside a wall/obstacle cell."""
        assert self.grid is not None, "Call reset() first."
        x, y = pos
        dim_i, dim_j = self.grid.shape
        if x < 0.0 or y < 0.0 or x >= dim_i or y >= dim_j:
            return True
        return self._is_obstacle_cell(*cell_index(pos))
