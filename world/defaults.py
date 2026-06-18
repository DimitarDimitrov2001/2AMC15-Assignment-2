# Minimal environment defaults
import numpy as np


DEFAULT_MINIMAL_ENVIRONMENT_STEP_SIZE: float = 0.5
DEFAULT_MINIMAL_ENVIRONMENT_SIGMA: float = 0.0

# Continuous environment defaults
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

# Default reward values (shared pre-assignment spec).
GOAL_REWARD: float = 1.0
LIVING_PENALTY: float = -0.01
COLLISION_PENALTY: float = -0.2