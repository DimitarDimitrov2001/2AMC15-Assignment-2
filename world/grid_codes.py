"""Canonical grid cell codes.

These integer codes are the on-disk representation of a grid cell. They are
shared between the environment, the reward function, the agents, and the
plotting helpers, so they live in their own module rather than next to any
one of those concerns.

``START_CELL`` is the optional start-position marker some grids carry; the
environment and agents normalise it to ``EMPTY_CELL`` once the agent has
been placed.
"""

from __future__ import annotations

EMPTY_CELL: int = 0
BOUNDARY_WALL_CELL: int = 1
OBSTACLE_CELL: int = 2
TARGET_CELL: int = 3
START_CELL: int = 4
