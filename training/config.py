from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """
    Settings of the trainer

    We group all training settings into one dataclass. (No need to pass separate arguments into trainer)
    """
    
    # Training loop
    total_episodes: int = 500 # Number of training episodes
    max_steps_per_episode: int = 500 # Maximum number of steps
    seed: int = 0 # Random seed for reproducibility

    # Evaluation loop
    eval_interval: int = 25 # How often to evaluate the policy
    eval_episodes: int = 10 # Number of episodes used for one evaluation (For example: Trainer runs 10 test episodes and gives the mean)

    # Logging
    log_interval: int = 1 # How often to print training metrics

    # W&B
    use_wandb: bool = False # Whether to use W&B
    wandb_project: str = "rl-in-practice-assignment-2" # Name of the project (Can remain unchanged)
    wandb_group: str | None = None # Group name (To group related runs) (Optional)
    run_name: str | None = None # Run name (Optional)