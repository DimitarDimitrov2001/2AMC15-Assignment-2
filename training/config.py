from __future__ import annotations

from dataclasses import dataclass
from training.defaults import (
    DEFAULT_TOTAL_EPISODES,
    DEFAULT_MAX_STEPS_PER_EPISODE,
    DEFAULT_SEED,
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_BEST_METRIC,
)


@dataclass
class TrainerConfig:
    """
    Settings of the trainer

    We group all training settings into one dataclass. (No need to pass separate arguments into trainer)
    """
    
    # Training loop
    total_episodes: int = DEFAULT_TOTAL_EPISODES # Number of training episodes
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE # Maximum number of steps
    seed: int = DEFAULT_SEED # Random seed for reproducibility

    # Optional environment-step budget.
    max_env_steps: int | None = None

    # Evaluation loop
    eval_interval: int = DEFAULT_EVAL_INTERVAL # How often to evaluate the policy
    eval_episodes: int = DEFAULT_EVAL_EPISODES # Number of episodes used for one evaluation (For example: Trainer runs 10 test episodes and gives the mean)

    # Logging
    log_interval: int = DEFAULT_LOG_INTERVAL # How often to print training metrics

    # Checkpointing
    checkpoint_dir: str | None = None # Directory for agent checkpoints (None disables)
    save_best: bool = False # Save best.pt whenever best_metric improves at eval
    save_last: bool = False # Save last.pt at the end of training
    best_metric: str = DEFAULT_BEST_METRIC # Eval metric key used to decide "best"

    # History persistence
    history_path: str | None = None # If set, dump per-episode history to this JSON file

    # Full configuration for logging (e.g. CLI args)
    full_config: dict | None = None

    # W&B
    use_wandb: bool = False # Whether to use W&B
    wandb_project: str = DEFAULT_WANDB_PROJECT # Name of the project (Can remain unchanged)
    wandb_group: str | None = None # Group name (To group related runs) (Optional)
    run_name: str | None = None # Run name (Optional)
    finish_wandb_on_train_end: bool = True # Keep run open for post-training artifacts when False