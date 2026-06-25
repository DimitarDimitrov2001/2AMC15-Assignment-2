"""Default configurations for the trainer and training loop."""

# Training loop defaults
DEFAULT_ENV_NAME: str = "continuous"
DEFAULT_GRID_FILENAME: str = "realistic_super_hard_cave.npy"
DEFAULT_TOTAL_EPISODES: int = 6_000
DEFAULT_MAX_STEPS_PER_EPISODE: int = 500
DEFAULT_SEED: int = 0

# Evaluation defaults
DEFAULT_EVAL_INTERVAL: int = 25
DEFAULT_EVAL_EPISODES: int = 10
DEFAULT_FINAL_EVAL_RUNS: int = 10

# Logging defaults (aligned with eval cadence so eval/* metrics reach W&B every eval)
DEFAULT_LOG_INTERVAL: int = 1
DEFAULT_WANDB_PROJECT: str = "rl-in-practice-assignment-2"

# Checkpointing defaults
DEFAULT_BEST_METRIC: str = "eval/mean_reward"
DEFAULT_OUTPUT_ROOT: str = "results"

# Visualization defaults
DEFAULT_VIZ_MAX_STEPS: int = 500
DEFAULT_WANDB_VIZ_INTERVAL: int = 100
