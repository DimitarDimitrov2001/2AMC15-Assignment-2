"""Default hyperparameters and configuration constants for RL agents."""

# DQN Agent Defaults
DQN_N_HIDDEN_NODES: int = 64
DQN_DEFAULT_BATCH_SIZE: int = 32
DQN_DEFAULT_LEARNING_RATE: float = 2.5e-4
DQN_DEFAULT_GAMMA: float = 0.99
DQN_DEFAULT_NO_OBS_IN_STATE: int = 1
DQN_DEFAULT_UPDATE_FREQ: int = 4
DQN_DEFAULT_TARGET_UPDATE_FREQ: int = 10_000
DQN_DEFAULT_CHECKPOINT_PATH: str = "models/dqn/best_model.pt"

# A3C Agent Defaults
A3C_N_WORKERS: int = 4
A3C_T_MAX: int = 5
A3C_GAMMA: float = 0.99
A3C_LEARNING_RATE: float = 1e-4
A3C_ENTROPY_BETA: float = 0.01
A3C_VALUE_COEF: float = 0.5
A3C_MAX_GRAD_NORM: float = 40.0
A3C_N_HIDDEN_NODES: int = 128
A3C_DEFAULT_CHECKPOINT_PATH: str = "models/a3c/best_model.pt"
A3C_DEFAULT_TOTAL_STEPS: int = 200_000

# Replay Buffer Defaults
REPLAY_DEFAULT_CAPACITY: int = 100_000
REPLAY_DEFAULT_START_SIZE: int = 50_000

# Epsilon Schedule Defaults
EPSILON_SCHEDULER_DEFAULT: str = "linear_annealing"
EPSILON_DEFAULT_MAX: float = 1.0
EPSILON_DEFAULT_MIN: float = 0.05
EPSILON_DEFAULT_DECAY: float = 0.95
EPSILON_ANNEAL_DURATION: int = 200_000
EPSILON_ANNEAL_START_STEP: int = 1_000

# Curiosity Defaults
BETA_DEFAULT: float = 0.5
# Counting resolution (world units) for count-based curiosity
CURIOSITY_RESOLUTION_DEFAULT: float = 1.0
