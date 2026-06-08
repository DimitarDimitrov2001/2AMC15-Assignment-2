from pathlib import Path

from agents import RandomAgent
from training import Trainer, TrainerConfig
from world.continuous_environment import ContinuousEnvironment


def main() -> None:
    env = ContinuousEnvironment(
        grid_fp=Path("grid_configs/small_grid.npy"),
        step_size=0.5,
        rotation_step=30.0,
        max_sensor_range=3.0,
        sigma=0.0,
        initial_heading=0.0,
        random_seed=0,
    )

    agent = RandomAgent(
        num_actions=env.n_actions,
        seed=0,
    )

    config = TrainerConfig(
        total_episodes=5,
        max_steps_per_episode=50,
        seed=0,
        eval_interval=1,
        eval_episodes=2,
        log_interval=1,
        use_wandb=False,
    )

    trainer = Trainer(
        env=env,
        agent=agent,
        config=config,
    )

    history = trainer.train()

    print()
    print("Continuous test finished.")
    print("Logged episodes:", len(history))
    print("Last episode metrics:")
    print(history[-1])


if __name__ == "__main__":
    main()