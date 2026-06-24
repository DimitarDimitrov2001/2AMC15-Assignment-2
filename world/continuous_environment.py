"""
Continuous environment with realistic robot action space and sensory data.

Builds on the minimal continuous environment with two additions:

Addition 1 — Realistic action & state space
    Actions:  rotate_left, rotate_right, move_forward
    State:    (x, y, theta)  where theta is the heading angle in degrees

Addition 2 — 8-direction distance sensors (optional, on by default)
    State:    (x, y, theta, d0, d1, d2, d3, d4, d5, d6, d7)
    d0..d7 are distances to the nearest wall/obstacle in 8 directions
    (0=East, 45=NE, 90=North, 135=NW, 180=West, 225=SW, 270=South, 315=SE)
    Disable via ``use_sensors=False`` to fall back to the bare (x, y, theta)
    state.

Action encoding
---------------
    0  rotate_left    theta -= rotation_step
    1  rotate_right   theta += rotation_step
    2  move_forward   x += step_size * cos(theta)
                      y += step_size * sin(theta)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from world.defaults import *
from world.environment_base import (
    LIVING_PENALTY,
    BaseGridEnvironment,
    RewardFn,
    cell_index,
)
from world.grid_codes import EMPTY_CELL, TARGET_CELL


class ContinuousEnvironment(BaseGridEnvironment):
    """Continuous grid-world with rotate-then-move actions and distance sensors.

    State vector returned by reset() and step():
        [x, y, theta, d0, d1, ..., dN]  when ``use_sensors`` is True
        [x, y, theta]                   when ``use_sensors`` is False
        shape (3 + n_sensors,) or (3,), all floats

    theta is in degrees [0, 360).
    d0..dN are distances to the nearest wall in the configured directions.
    """

    # Heading angle in degrees, updated on rotate/move actions.
    theta: float

    # When False the 8-direction distance sensors are dropped from the
    # observation, leaving the bare (x, y, theta) kinematic state.
    use_sensors: bool

    # NumPy RNG for Gaussian action/sensor noise (separate from the base RNG).
    _np_rng: np.random.Generator

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
        use_sensors: bool = True,
        reward_fn: RewardFn | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        sigma: float | None = None,
    ) -> None:
        """
        Args:
            grid_fp:          Path to a .npy grid file.
            step_size:        Distance moved per move_forward action.
            rotation_step:    Degrees rotated per rotate action.
            max_sensor_range: Maximum distance each ray can travel. If no wall
                              is found within this range the sensor returns
                              max_sensor_range (meaning "clear ahead").
            action_sigma:     Std-dev of Gaussian noise added to actions.
            sensory_sigma:    Std-dev of Gaussian noise added to sensor readings.
            sigma:            Backwards-compatible alias for action_sigma.
            agent_start_pos:  Optional fixed (x, y) start position.
            initial_heading:  Starting heading angle in degrees.
            sensor_angles:    Angles at which distance sensors are pointed.
            ray_step:         Distance increment when checking for collisions along a ray.
            use_sensors:      If True (default) append the distance-sensor
                              readings to the observation; if False the state is
                              just (x, y, theta) and no rays are cast.
            reward_fn:        Custom reward function with signature
                                fn(grid, pos, new_pos, collision) -> float
            random_seed:      Seed for the internal RNG.
        """
        if action_sigma is None:
            action_sigma = DEFAULT_ACTION_SIGMA if sigma is None else sigma

        super().__init__(
            grid_fp,
            step_size=step_size,
            agent_start_pos=agent_start_pos,
            reward_fn=reward_fn,
            random_seed=random_seed,
        )
        self.rotation_step = rotation_step
        self.max_sensor_range = max_sensor_range
        self.action_sigma = float(action_sigma)
        self.sensory_sigma = sensory_sigma
        self.initial_heading = initial_heading
        self.sensor_angles = sensor_angles
        self.ray_step = ray_step
        self.use_sensors = use_sensors
        self.theta = initial_heading
        self._np_rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply one action and return the new state.

        Args:
            action: 0=rotate_left, 1=rotate_right, 2=move_forward

        Returns:
            state:      np.ndarray of shape (3 + n_sensors,) [x, y, theta, d0..dN]
            reward:     scalar float
            terminated: True when all targets collected
            info:       dict with keys success, collision, action, pos, theta
        """
        assert self.grid is not None and self.pos is not None, "Call reset() before step()."

        self.world_stats["total_steps"] += 1

        collision = False
        action_noise = self._np_rng.normal(0, self.action_sigma) if self.action_sigma > 0 else 0

        if action == ROTATE_LEFT:                                       # rotate left
            self.theta = (self.theta - self.rotation_step + action_noise) % 360.0
            self.world_stats["total_rotations"] += 1
            reward = LIVING_PENALTY

        elif action == ROTATE_RIGHT:                                    # rotate right
            self.theta = (self.theta + self.rotation_step + action_noise) % 360.0
            self.world_stats["total_rotations"] += 1
            reward = LIVING_PENALTY

        elif action == MOVE_FORWARD:                                    # move forward
            noisy_step = self.step_size + action_noise

            collision, new_pos = self._move_forward(noisy_step)

            reward = self.reward_fn(self.grid, self.pos, new_pos, collision)

            if collision:
                self.world_stats["total_collisions"] += 1
            else:
                self.pos = new_pos
                self.world_stats["total_agent_moves"] += 1

                # Check for goal
                i, j = cell_index(self.pos)
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

    @property
    def state_dim(self) -> int:
        """x, y, theta (+ sensor readings when ``use_sensors`` is True)."""
        n_sensors = len(self.sensor_angles) if self.use_sensors else 0
        return 2 + 1 + n_sensors

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def observation_high(self) -> np.ndarray:
        """Per-dimension upper bound for normalizing [x, y, theta(, d0..dN)].

        x, y are bounded by the grid (rows, cols); theta by 360 degrees; each
        sensor by ``max_sensor_range``. Lets agents scale every input to
        roughly [0, 1] instead of leaving theta (~360) and distances (~range)
        unscaled next to the small x, y values.
        """
        dim_i, dim_j = self._grid_dims()
        bounds = [float(dim_i), float(dim_j), 360.0]
        if self.use_sensors:
            bounds.extend([float(self.max_sensor_range)] * len(self.sensor_angles))
        return np.array(bounds, dtype=np.float32)

    @property
    def angular_dims(self) -> tuple[int, ...]:
        """theta is the heading angle (index 2) in the [x, y, theta, ...] state."""
        return (2,)

    # ------------------------------------------------------------------
    # Reset hooks
    # ------------------------------------------------------------------

    def _reseed(self, seed: int) -> None:
        """Reseed both the base RNG and the NumPy noise generator."""
        super()._reseed(seed)
        self._np_rng = np.random.default_rng(seed)

    def _init_world_stats(self) -> dict[str, float]:
        """Extend the common counters with rotation tracking."""
        stats = super()._init_world_stats()
        stats["total_rotations"] = 0
        return stats

    def _on_reset(self) -> None:
        """Reset the heading to the configured initial value."""
        self.theta = self.initial_heading

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_state(self) -> np.ndarray:
        """Build the state vector [x, y, theta] (+ [d0..dN] when sensors are on)."""
        assert self.pos is not None, "Call reset() first."
        if not self.use_sensors:
            return np.concatenate([self.pos, [self.theta]]).astype(np.float32)

        sensors = self._cast_rays()
        state = np.concatenate([self.pos, [self.theta], sensors]).astype(np.float32)

        if self.sensory_sigma > 0:
            noise = self._np_rng.normal(0, self.sensory_sigma, size=state.shape)
            state = state + noise
            n_sensors = len(self.sensor_angles)
            state[-n_sensors:] = np.clip(state[-n_sensors:], 0, self.max_sensor_range)

        return state.astype(np.float32)

    def _dda_raycast(
        self, start_pos: np.ndarray, dx: float, dy: float, max_dist: float
    ) -> tuple[bool, float, np.ndarray]:
        """
        Implementation of the Amanatides & Woo Fast Voxel Traversal algorithm.
        Returns: (hit_wall: bool, exact_distance: float, exact_hit_position: np.ndarray)
        """
        max_dist = max(0.0, float(max_dist))
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

        if self._is_obstacle_cell(current_x, current_y):
            return True, 0.0, start_pos

        tolerance = 1e-9
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

            if dist > max_dist + tolerance:
                final_pos = start_pos + np.array([dx, dy]) * max_dist
                return False, max_dist, final_pos

            if self._is_obstacle_cell(current_x, current_y):
                hit_dist = min(dist, max_dist)
                exact_hit_pos = start_pos + np.array([dx, dy]) * hit_dist
                return True, hit_dist, exact_hit_pos

            if dist > max_dist:
                final_pos = start_pos + np.array([dx, dy]) * max_dist
                return False, max_dist, final_pos

    def _cast_rays(self) -> np.ndarray:
        """Cast rays and return exact hit distances using Amanatides & Woo DDA."""
        assert self.pos is not None, "Call reset() first."
        distances = np.full(len(self.sensor_angles), self.max_sensor_range, dtype=float)

        for i, angle_deg in enumerate(self.sensor_angles):
            world_angle = (self.theta + angle_deg) % 360.0
            rad = np.deg2rad(world_angle)
            dx, dy = np.cos(rad), np.sin(rad)

            is_hit, dist, _ = self._dda_raycast(self.pos, dx, dy, self.max_sensor_range)
            if is_hit:
                distances[i] = dist

        return distances

    def _move_forward(self, step_size: float) -> tuple[bool, np.ndarray]:
        """Calculate the exact next position using Continuous Collision Detection (CCD)."""
        assert self.pos is not None, "Call reset() first."
        rad = np.deg2rad(self.theta)
        direction = 1.0 if step_size >= 0.0 else -1.0
        dx, dy = np.cos(rad) * direction, np.sin(rad) * direction
        travel_distance = abs(float(step_size))

        is_hit, dist, target_pos = self._dda_raycast(self.pos, dx, dy, travel_distance)

        if is_hit:
            # Stop slightly short of the wall boundary to avoid numeric rounding
            # putting us inside the wall.
            epsilon = 1e-4
            safe_dist = max(0.0, dist - epsilon)
            safe_pos = self.pos + np.array([dx, dy]) * safe_dist
            return True, safe_pos
        if self._is_collision_position(target_pos):
            return True, self.pos.copy()
        return False, target_pos

    def _is_collision_position(self, pos: np.ndarray) -> bool:
        """Return True when a continuous position is outside the grid or blocked."""
        assert self.grid is not None, "Call reset() first."
        x, y = pos
        dim_i, dim_j = self.grid.shape
        if x < 0.0 or y < 0.0 or x >= dim_i or y >= dim_j:
            return True
        return self._is_obstacle_cell(*cell_index(pos))
