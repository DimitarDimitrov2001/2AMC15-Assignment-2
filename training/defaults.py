"""Default configurations for the trainer and training loop."""

# Training loop defaults
DEFAULT_TOTAL_EPISODES: int = 3_000
DEFAULT_MAX_STEPS_PER_EPISODE: int = 500
DEFAULT_SEED: int = 0

# Evaluation defaults
DEFAULT_EVAL_INTERVAL: int = 10
DEFAULT_EVAL_EPISODES: int = 5

# Logging defaults
DEFAULT_LOG_INTERVAL: int = 50
DEFAULT_WANDB_PROJECT: str = "rl-in-practice-assignment-2"

# Checkpointing defaults
DEFAULT_BEST_METRIC: str = "eval/mean_reward"

# Visualization defaults
DEFAULT_VIZ_MAX_STEPS: int = 500
