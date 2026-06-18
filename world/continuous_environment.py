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

from dataclasses import dataclass
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

# Action identifiers
type Action             = int
ROTATE_LEFT: Action     = 0
ROTATE_RIGHT: Action    = 1
MOVE_FORWARD: Action    = 2

# Default environment parameters
DEFAULT_STEP_SIZE: float        = 0.5
DEFAULT_ROTATION_STEP: float    = 30.0
DEFAULT_MAX_SENSOR_RANGE: float = 3.0
DEFAULT_ACTION_SIGMA: float     = 0.0
DEFAULT_SENSORY_SIGMA: float    = 0.0
DEFAULT_INITIAL_HEADING: float  = 0.0
DEFAULT_SENSOR_ANGLES: np.ndarray = np.arange(0, 360, 45)
DEFAULT_RAY_STEP: float           = 0.1
DEFAULT_RANDOM_SEED: int        = 0

# Reward values
_GOAL_REWARD: float       =  1.0
_LIVING_PENALTY: float    = -0.01
_COLLISION_PENALTY: float = -0.2

class ContinuousEnvironment:
    """Continuous grid-world with rotate-then-move actions and distance sensors.

    State vector returned by reset() and step():
        [x, y, theta, d0, d1, ..., dN]
        shape (3 + n_sensors,), all floats

    theta is in degrees [0, 360).
    d0..dN are distances to the nearest wall in the configured directions.
    """

    def __init__(
        self,
        grid_fp: Path,
        step_size: float = DEFAULT_STEP_SIZE,
        rotation_step: float = DEFAULT_ROTATION_STEP,
        max_sensor_range: float = DEFAULT_MAX_SENSOR_RANGE,
        action_sigma: float | None = None,
        sensory_sigma: float = DEFAULT_SENSORY_SIGMA,
        agent_start_pos: tuple[float, float] | None = None,
        initial_heading: float = DEFAULT_INITIAL_HEADING,
        sensor_angles: np.ndarray = DEFAULT_SENSOR_ANGLES,
        ray_step: float = DEFAULT_RAY_STEP,
        reward_fn: callable | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        sigma: float | None = None,
    ):
        """
        Args:
            grid_fp:          Path to a .npy grid file.
            step_size:        Distance moved per move_forward action.
            rotation_step:    Degrees rotated per rotate action.
            max_sensor_range: Maximum distance each ray can travel. If no wall
                              is found within this range the sensor returns
                              max_sensor_range (meaning "clear ahead").
            action_sigma:     Standard deviation of action noise.
            sigma:            Backwards-compatible alias for action_sigma.
            agent_start_pos:  Optional fixed (x, y) start position.
            initial_heading:  Starting heading angle in degrees.
            sensor_angles:    Angles at which distance sensors are pointed.
            ray_step:         Distance increment when checking for collisions along a ray.
            reward_fn:        Custom reward function with signature
                                fn(grid, pos, new_pos, collision) -> float
            random_seed:      Seed for the internal RNG.
        """
        if not Path(grid_fp).exists():
            raise FileNotFoundError(f"Grid file not found: {grid_fp}")

        if action_sigma is None:
            action_sigma = DEFAULT_ACTION_SIGMA if sigma is None else sigma

        self.grid_fp         = Path(grid_fp)
        self.step_size       = step_size
        self.rotation_step   = rotation_step
        self.max_sensor_range = max_sensor_range
        self.action_sigma    = float(action_sigma)
        self.sensory_sigma   = sensory_sigma
        self.agent_start_pos = agent_start_pos
        self.initial_heading = initial_heading
        self.sensor_angles   = sensor_angles
        self.ray_step        = ray_step
        self.reward_fn       = reward_fn if reward_fn is not None else _default_reward
        self._rng            = random.Random(random_seed)
        self._np_rng         = np.random.default_rng(random_seed)

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
            Initial state [x, y, theta, d0..dN] as np.ndarray.
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
            state:      np.ndarray of shape (3 + n_sensors,) [x, y, theta, d0..dN]
            reward:     scalar float
            terminated: True when all targets collected
            info:       dict with keys success, collision, action, pos, theta
        """
        assert self.grid is not None, "Call reset() before step()."

        self.world_stats["total_steps"] += 1

        collision = False
        action_noise = self._np_rng.normal(0, self.action_sigma) if self.action_sigma > 0 else 0

        if action == ROTATE_LEFT:                                      # rotate left
            self.theta = (self.theta - self.rotation_step + action_noise) % 360.0
            self.world_stats["total_rotations"] += 1
            reward = _LIVING_PENALTY

        elif action == ROTATE_RIGHT:                                    # rotate right
            self.theta = (self.theta + self.rotation_step + action_noise) % 360.0
            self.world_stats["total_rotations"] += 1
            reward = _LIVING_PENALTY

        elif action == MOVE_FORWARD:                                                # move forward
            noisy_step = self.step_size + action_noise
            
            collision, new_pos = self._move_forward(noisy_step)

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
        else:
            raise RuntimeError(f"Invalid action taken by agent: {action}")

        self.world_stats["cumulative_reward"] += reward

        reached_target = self.terminal and not collision

        return (
            self._make_state(),
            reward,
            self.terminal,
            {"success": bool(reached_target), "collision": collision, "action": action,
             "pos": self.pos.copy(), "theta": self.theta},
        )

    # ------------------------------------------------------------------
    # Properties for building the DQN
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """x, y, theta + sensor readings."""
        return 2 + 1 + len(self.sensor_angles)

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
        """Build the full state vector [x, y, theta, d0..dN]."""
        sensors = self._cast_rays()
        state = np.concatenate([self.pos, [self.theta], sensors]).astype(np.float32)
        
        if self.sensory_sigma > 0:
            noise = self._np_rng.normal(0, self.sensory_sigma, size=state.shape)
            state = state + noise
            n_sensors = len(self.sensor_angles)
            state[-n_sensors:] = np.clip(state[-n_sensors:], 0, self.max_sensor_range)
            
        return state.astype(np.float32)

    def _dda_raycast(self, start_pos: np.ndarray, dx: float, dy: float, max_dist: float) -> tuple[bool, float, np.ndarray]:
        """
        Implementation of the Amanatides & Woo Fast Voxel Traversal algorithm.
        Returns: (hit_wall: bool, exact_distance: float, exact_hit_position: np.ndarray)
        """
        x, y = start_pos
        current_x, current_y = int(np.floor(x)), int(np.floor(y))

        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
        t_delta_x = abs(1.0 / dx) if dx != 0 else float('inf')
        t_delta_y = abs(1.0 / dy) if dy != 0 else float('inf')

        if dx > 0:
            t_max_x = (current_x + 1.0 - x) * t_delta_x
        elif dx < 0:
            t_max_x = (x - current_x) * t_delta_x
        else:
            t_max_x = float('inf')
            
        if dy > 0:
            t_max_y = (current_y + 1.0 - y) * t_delta_y
        elif dy < 0:
            t_max_y = (y - current_y) * t_delta_y
        else:
            t_max_y = float('inf')

        if self._is_collision_cell(current_x, current_y):
            return True, 0.0, start_pos

        dist = 0.0
        while True:
            if t_max_x < t_max_y:
                dist = t_max_x
                current_x += step_x
                t_max_x += t_delta_x
            else:
                dist = t_max_y
                current_y += step_y
                t_max_y += t_delta_y

            if dist > max_dist:
                final_pos = start_pos + np.array([dx, dy]) * max_dist
                return False, max_dist, final_pos

            if self._is_collision_cell(current_x, current_y):
                exact_hit_pos = start_pos + np.array([dx, dy]) * dist
                return True, dist, exact_hit_pos
            
    def _is_collision_cell(self, i: int, j: int) -> bool:
        """True if the grid cell (i, j) is out of bounds or an obstacle."""
        dim_i, dim_j = self.grid.shape
        if i < 0 or j < 0 or i >= dim_i or j >= dim_j:
            return True
        return int(self.grid[i, j]) in (BOUNDARY_WALL_CELL, OBSTACLE_CELL)

    def _cast_rays(self) -> np.ndarray:
        """Cast rays and return exact hit distances using Amanatides & Woo DDA."""
        distances = np.full(len(self.sensor_angles), self.max_sensor_range, dtype=float)

        for i, angle_deg in enumerate(self.sensor_angles):
            world_angle = (self.theta + angle_deg) % 360.0
            rad = np.deg2rad(world_angle)
            dx, dy = np.cos(rad), np.sin(rad)
            
            is_hit, dist, _ = self._dda_raycast(self.pos, dx, dy, self.max_sensor_range)
            if is_hit:
                distances[i] = dist

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
    
    def _move_forward(self, step_size: float) -> tuple[bool, np.ndarray]:
        """Calculates the exact next position using Continuous Collision Detection (CCD)."""
        rad = np.deg2rad(self.theta)
        dx, dy = np.cos(rad), np.sin(rad)
        
        is_hit, dist, target_pos = self._dda_raycast(self.pos, dx, dy, step_size)
        
        if is_hit:
            # Stop slightly short of the wall boundary to avoid numeric rounding putting us inside the wall
            epsilon = 1e-4
            safe_dist = max(0.0, dist - epsilon)
            safe_pos = self.pos + np.array([dx, dy]) * safe_dist
            return True, safe_pos
        else:
            return False, target_pos

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
