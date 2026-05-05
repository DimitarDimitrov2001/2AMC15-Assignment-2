"""Evaluation helpers for fixed grid-world policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from world import Environment


def evaluate_policy_metrics(
    grid: Path,
    agent: Any,
    max_steps: int,
    sigma: float,
    agent_start_pos: tuple[int, int],
    reward_fn,
    gamma: float,
    random_seed: int,
    n_eval_episodes: int,
) -> dict:
    """Evaluate a fixed policy over repeated rollouts without learning."""
    if n_eval_episodes < 1:
        raise ValueError("n_eval_episodes must be at least 1")

    discounted_returns: list[float] = []
    undiscounted_returns: list[float] = []
    episode_lengths: list[int] = []
    successes: list[bool] = []

    for episode in range(n_eval_episodes):
        env = Environment(
            grid,
            no_gui=True,
            reward_fn=reward_fn,
            sigma=sigma,
            target_fps=-1,
            agent_start_pos=agent_start_pos,
            random_seed=random_seed + episode,
        )
        state = env.reset()
        discounted_return = 0.0
        undiscounted_return = 0.0
        terminated = False

        for step_idx in range(max_steps):
            action = agent.take_action(state)
            state, reward, terminated, _ = env.step(action)
            undiscounted_return += reward
            discounted_return += (gamma**step_idx) * reward
            if terminated:
                break

        discounted_returns.append(discounted_return)
        undiscounted_returns.append(undiscounted_return)
        episode_lengths.append(env.world_stats["total_steps"])
        successes.append(bool(terminated))

    success_lengths = [length for length, success in zip(episode_lengths, successes, strict=True) if success]
    return {
        "n_eval_episodes": n_eval_episodes,
        "max_steps": max_steps,
        "success_rate": sum(successes) / n_eval_episodes,
        "mean_discounted_return": sum(discounted_returns) / n_eval_episodes,
        "mean_undiscounted_return": sum(undiscounted_returns) / n_eval_episodes,
        "mean_episode_length": sum(episode_lengths) / n_eval_episodes,
        "mean_success_episode_length": (
            sum(success_lengths) / len(success_lengths) if success_lengths else None
        ),
        "discounted_returns": discounted_returns,
        "undiscounted_returns": undiscounted_returns,
        "episode_lengths": episode_lengths,
        "terminal_successes": successes,
    }
