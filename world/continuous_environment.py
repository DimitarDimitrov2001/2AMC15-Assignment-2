"""
Continuous environment with realistic robot action space and sensory data.

Builds on the minimal continuous environment with two additions:

Addition 1 — Realistic action & state space
    Actions:  rotate_left, rotate_right, move_forward
    State:    (x, y, theta)  where theta is the heading angle in degrees

Addition 2 — 8-direction distance sensors
    State:    (x, y, theta, d0, d1, d2, d3, d4, d5, d6, d7)
    d0..d7 are distances to the nearest wall/obstacle in 8 directions
    (0=East, 45=NE, 90=North, 135=NW, 180=West, 225=SW, 270=South, 315=SE)

Action encoding
---------------
    0  rotate_left    theta -= rotation_step
    1  rotate_right   theta += rotation_step
    2  move_forward   x += step_size * cos(theta)
                      y += step_size * sin(theta)
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


N_ACTIONS: int = 3          # rotate_left, rotate_right, move_forward
N_SENSORS: int = 8          # one distance reading per 45 degrees

# Reward values
_GOAL_REWARD: float       =  1.0
_LIVING_PENALTY: float    = -0.01
_COLLISION_PENALTY: float = -1.0

# Sensor ray directions in degrees (absolute, East = 0)
_SENSOR_ANGLES = np.arange(0, 360, 45)   # [0, 45, 90, 135, 180, 225, 270, 315]
_RAY_STEP      = 0.1                      # resolution when stepping along each ray


class ContinuousEnvironment:
    """Continuous grid-world with rotate-then-move actions and distance sensors.

    State vector returned by reset() and step():
        [x, y, theta, d0, d1, d2, d3, d4, d5, d6, d7]
        shape (11,), all floats

    theta is in degrees [0, 360).
    d0..d7 are distances to the nearest wall in 8 directions.
    """

    def __init__(
        self,
        grid_fp: Path,
        step_size: float = 0.5,
        rotation_step: float = 30.0,
        max_sensor_range: float = 3.0,
        sigma: float = 0.0,
        agent_start_pos: tuple[float, float] | None = None,
        initial_heading: float = 0.0,
        reward_fn: callable | None = None,
        random_seed: int = 0,
    ):
        """
        Args:
            grid_fp:          Path to a .npy grid file.
            step_size:        Distance moved per move_forward action.
            rotation_step:    Degrees rotated per rotate action.
            max_sensor_range: Maximum distance each ray can travel. If no wall
                              is found within this range the sensor returns
                              max_sensor_range (meaning "clear ahead").
            sigma:            Probability of a random action replacing the chosen one.
            agent_start_pos:  Optional fixed (x, y) start position.
            initial_heading:  Starting heading angle in degrees.
            reward_fn:        Custom reward function with signature
                                fn(grid, pos, new_pos, collision) -> float
            random_seed:      Seed for the internal RNG.
        """
        if not Path(grid_fp).exists():
            raise FileNotFoundError(f"Grid file not found: {grid_fp}")

        self.grid_fp         = Path(grid_fp)
        self.step_size       = step_size
        self.rotation_step   = rotation_step
        self.max_sensor_range = max_sensor_range
        self.sigma           = sigma
        self.agent_start_pos = agent_start_pos
        self.initial_heading = initial_heading
        self.reward_fn      = reward_fn if reward_fn is not None else _default_reward
        self._rng           = random.Random(random_seed)

        # Populated on reset()
        self.grid:     np.ndarray | None = None
        self.pos:      np.ndarray | None = None   # [x, y]
        self.theta:    float             = initial_heading
        self.terminal: bool              = False
        self.world_stats: dict           = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reload the grid and place the agent.

        Returns:
            Initial state [x, y, theta, d0..d7] as np.ndarray of shape (11,).
        """
        self.grid     = Grid.load_grid(self.grid_fp).cells
        self.terminal = False
        self.theta    = self.initial_heading
        self.world_stats = {
            "cumulative_reward":  0.0,
            "total_steps":        0,
            "total_collisions":   0,
            "total_agent_moves":  0,
            "total_rotations":    0,
            "targets_reached":    0,
        }

        self.pos = (
            np.array(self.agent_start_pos, dtype=float)
            if self.agent_start_pos is not None
            else self._find_start()
        )

        return self._make_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Apply one action and return the new state.

        Args:
            action: 0=rotate_left, 1=rotate_right, 2=move_forward

        Returns:
            state:      np.ndarray of shape (11,)  [x, y, theta, d0..d7]
            reward:     scalar float
            terminated: True when all targets collected
            info:       dict with keys collision, action, pos, theta
        """
        assert self.grid is not None, "Call reset() before step()."

        self.world_stats["total_steps"] += 1

        # Optional stochasticity
        if self._rng.random() < self.sigma:
            action = self._rng.randrange(N_ACTIONS)

        collision = False

        if action == 0:                                      # rotate left
            self.theta = (self.theta - self.rotation_step) % 360.0
            self.world_stats["total_rotations"] += 1
            reward = _LIVING_PENALTY

        elif action == 1:                                    # rotate right
            self.theta = (self.theta + self.rotation_step) % 360.0
            self.world_stats["total_rotations"] += 1
            reward = _LIVING_PENALTY

        else:                                                # move forward
            rad       = np.deg2rad(self.theta)
            delta     = np.array([np.cos(rad), np.sin(rad)]) * self.step_size
            new_pos   = self.pos + delta
            collision = self._is_collision(new_pos)

            reward = self.reward_fn(self.grid, self.pos, new_pos, collision)

            if collision:
                self.world_stats["total_collisions"] += 1
            else:
                self.pos = new_pos
                self.world_stats["total_agent_moves"] += 1

                # Check for goal
                i, j = _cell(self.pos)
                if self.grid[i, j] == TARGET_CELL:
                    self.grid[i, j] = EMPTY_CELL
                    self.world_stats["targets_reached"] += 1
                    if not np.any(self.grid == TARGET_CELL):
                        self.terminal = True

        self.world_stats["cumulative_reward"] += reward

        return (
            self._make_state(),
            reward,
            self.terminal,
            {"collision": collision, "action": action,
             "pos": self.pos.copy(), "theta": self.theta},
        )

    # ------------------------------------------------------------------
    # Properties for building the DQN
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """x, y, theta + 8 sensor readings = 11."""
        return 2 + 1 + N_SENSORS

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

    def _make_state(self) -> np.ndarray:
        """Build the full state vector [x, y, theta, d0..d7]."""
        sensors = self._cast_rays()
        return np.concatenate([self.pos, [self.theta], sensors]).astype(np.float32)

    def _cast_rays(self) -> np.ndarray:
        """Cast 8 rays up to max_sensor_range and return hit distances.

        If no wall is found within max_sensor_range the sensor returns
        max_sensor_range, meaning the path is clear up to the range limit.
        """
        distances = np.full(N_SENSORS, self.max_sensor_range)

        for i, angle_deg in enumerate(_SENSOR_ANGLES):
            rad       = np.deg2rad(angle_deg)
            direction = np.array([np.cos(rad), np.sin(rad)])

            dist = _RAY_STEP
            while dist <= self.max_sensor_range:
                ray_pos = self.pos + direction * dist
                if self._is_collision(ray_pos):
                    distances[i] = dist
                    break
                dist += _RAY_STEP

        return distances

    def _find_start(self) -> np.ndarray:
        """Return the continuous centre of the START_CELL or a random empty cell."""
        starts = np.argwhere(self.grid == START_CELL)
        if len(starts):
            i, j = starts[0]
            self.grid[i, j] = EMPTY_CELL
            return np.array([i + 0.5, j + 0.5], dtype=float)

        empty = np.argwhere(self.grid == EMPTY_CELL)
        i, j  = empty[self._rng.randrange(len(empty))]
        return np.array([i + 0.5, j + 0.5], dtype=float)

    def _is_collision(self, pos: np.ndarray) -> bool:
        """True if pos is out of bounds or inside a wall/obstacle cell."""
        x, y      = pos
        dim_i, dim_j = self.grid.shape
        if x < 0.0 or y < 0.0 or x >= dim_i or y >= dim_j:
            return True
        i, j = _cell(pos)
        return int(self.grid[i, j]) in (BOUNDARY_WALL_CELL, OBSTACLE_CELL)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _cell(pos: np.ndarray) -> tuple[int, int]:
    return int(np.floor(pos[0])), int(np.floor(pos[1]))


def _default_reward(
    grid: np.ndarray,
    pos: np.ndarray,
    new_pos: np.ndarray,
    collision: bool,
) -> float:
    if collision:
        return _COLLISION_PENALTY
    i, j = _cell(new_pos)
    if grid[i, j] == TARGET_CELL:
        return _GOAL_REWARD
    return _LIVING_PENALTY
