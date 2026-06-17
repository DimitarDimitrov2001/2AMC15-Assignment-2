"""Shared base class for the grid-world environments.

Both deep-RL environments (the minimal point-mass and the continuous robot)
load the same Assignment 1 ``.npy`` grids and share the episode scaffolding:
grid loading, start placement, collision-by-cell checks, reward defaults, and
the ``reset``/``step`` contract consumed by the Trainer.

The base applies the Template Method pattern: ``reset`` fixes the invariant
episode-reset sequence and delegates the parts that differ between
environments (RNG setup, per-episode statistics, observation construction) to
overridable hooks. Subclasses implement ``step`` and ``_make_state``; the rest
is shared so the two environments cannot drift apart.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np

from world.grid import Grid
from world.grid_codes import (
    BOUNDARY_WALL_CELL,
    EMPTY_CELL,
    OBSTACLE_CELL,
    START_CELL,
    TARGET_CELL,
)

# Reward function contract shared by every grid environment:
# fn(grid, pos, new_pos, collision) -> reward.
RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, bool], float]

# Default reward values (shared pre-assignment spec).
GOAL_REWARD: float = 5.0
LIVING_PENALTY: float = -0.01
COLLISION_PENALTY: float = -0.01


def cell_index(pos: np.ndarray) -> tuple[int, int]:
    """Return the grid indices (i, j) for a continuous position (x, y)."""
    return int(np.floor(pos[0])), int(np.floor(pos[1]))


def default_reward(
    grid: np.ndarray,
    pos: np.ndarray,
    new_pos: np.ndarray,
    collision: bool,
) -> float:
    """Return the default step reward.
    """
    if collision:
        return COLLISION_PENALTY
    i, j = cell_index(new_pos)
    if grid[i, j] == TARGET_CELL:
        return GOAL_REWARD
    return LIVING_PENALTY


class BaseGridEnvironment(ABC):
    """Abstract grid-world environment with shared episode scaffolding.

    Subclasses must implement ``step``, ``_make_state``, ``state_dim`` and
    ``n_actions``. They may override the ``_reseed``, ``_init_world_stats`` and
    ``_on_reset`` hooks to extend the reset sequence with environment-specific
    state (extra RNGs, heading, additional counters).
    """

    # Configuration set in __init__.
    grid_fp: Path
    step_size: float
    agent_start_pos: tuple[float, float] | None
    reward_fn: RewardFn

    # Episode state populated on reset().
    grid: np.ndarray | None
    pos: np.ndarray | None
    terminal: bool
    world_stats: dict[str, float]

    # RNG for start placement and action stochasticity.
    _rng: random.Random

    def __init__(
        self,
        grid_fp: Path,
        step_size: float,
        agent_start_pos: tuple[float, float] | None = None,
        reward_fn: RewardFn | None = None,
        random_seed: int = 0,
    ) -> None:
        """Validate the grid path and initialise shared configuration.

        Args:
            grid_fp: Path to a ``.npy`` grid file (Assignment 1 format).
            step_size: Distance delta the agent moves per move action.
            agent_start_pos: Optional fixed (x, y) start. If None, the grid's
                START_CELL is used, falling back to a random empty cell.
            reward_fn: Custom reward function; defaults to ``default_reward``.
            random_seed: Seed for the internal RNG.
        """
        if not Path(grid_fp).exists():
            raise FileNotFoundError(f"Grid file not found: {grid_fp}")

        self.grid_fp = Path(grid_fp)
        self.step_size = step_size
        self.agent_start_pos = agent_start_pos
        self.reward_fn = reward_fn if reward_fn is not None else default_reward
        self._rng = random.Random(random_seed)

        self.grid = None
        self.pos = None
        self.terminal = False
        self.world_stats = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reload the grid, place the agent, and return the initial observation.

        Template method: the reset sequence is fixed here while environment
        specifics are delegated to the ``_reseed``, ``_init_world_stats`` and
        ``_on_reset`` hooks and the ``_make_state`` observation builder.

        Args:
            seed: Optional per-episode seed; reseeds the environment RNG(s).

        Returns:
            The initial observation produced by ``_make_state``.
        """
        if seed is not None:
            self._reseed(seed)

        self.grid = Grid.load_grid(self.grid_fp).cells
        self.terminal = False
        self.world_stats = self._init_world_stats()
        self.pos = (
            np.array(self.agent_start_pos, dtype=float)
            if self.agent_start_pos is not None
            else self._find_start()
        )
        self._on_reset()
        return self._make_state()

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Return the dimensionality of the observation vector."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Return the number of discrete actions available to the agent."""

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply one action and return (observation, reward, terminated, info)."""

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return the (rows, cols) shape of the loaded grid."""
        assert self.grid is not None, "Call reset() first."
        return (int(self.grid.shape[0]), int(self.grid.shape[1]))

    @property
    def observation_high(self) -> np.ndarray:
        """Per-dimension upper bound of the observation, for input normalization.

        Defaults to ones (a no-op scaling). Subclasses override with the real
        bounds so agents can normalise inputs to roughly [0, 1].
        """
        return np.ones(self.state_dim, dtype=np.float32)

    def _grid_dims(self) -> tuple[int, int]:
        """Return (rows, cols) of the grid, loading from file if not yet reset.

        Reset-independent so callers (e.g. an agent's normaliser) can query
        bounds before the first ``reset``.
        """
        if self.grid is not None:
            return (int(self.grid.shape[0]), int(self.grid.shape[1]))
        cells = Grid.load_grid(self.grid_fp).cells
        return (int(cells.shape[0]), int(cells.shape[1]))

    # ------------------------------------------------------------------
    # Reset hooks (overridable by subclasses)
    # ------------------------------------------------------------------

    def _reseed(self, seed: int) -> None:
        """Reseed the environment RNG. Subclasses extend for extra RNGs."""
        self._rng = random.Random(seed)

    def _init_world_stats(self) -> dict[str, float]:
        """Return a fresh per-episode statistics dict with the common counters."""
        return {
            "cumulative_reward": 0.0,
            "total_steps": 0,
            "total_collisions": 0,
            "total_agent_moves": 0,
            "targets_reached": 0,
        }

    def _on_reset(self) -> None:
        """Hook for subclass-specific reset state (e.g. heading). Default no-op."""
        return None

    # ------------------------------------------------------------------
    # Shared internal helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def _make_state(self) -> np.ndarray:
        """Build and return the current observation vector."""

    def _find_start(self) -> np.ndarray:
        """Return the continuous centre of START_CELL, else a random empty cell."""
        assert self.grid is not None, "Call reset() first."
        starts = np.argwhere(self.grid == START_CELL)
        if len(starts):
            i, j = starts[0]
            self.grid[i, j] = EMPTY_CELL
            return np.array([i + 0.5, j + 0.5], dtype=float)

        empty = np.argwhere(self.grid == EMPTY_CELL)
        i, j = empty[self._rng.randrange(len(empty))]
        return np.array([i + 0.5, j + 0.5], dtype=float)

    def _is_obstacle_cell(self, i: int, j: int) -> bool:
        """True if cell (i, j) is out of bounds or a wall/obstacle."""
        assert self.grid is not None, "Call reset() first."
        dim_i, dim_j = self.grid.shape
        if i < 0 or j < 0 or i >= dim_i or j >= dim_j:
            return True
        return int(self.grid[i, j]) in (BOUNDARY_WALL_CELL, OBSTACLE_CELL)
