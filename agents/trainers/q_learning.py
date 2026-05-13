"""Trainer for the Q-Learning agent."""

from __future__ import annotations

import random

import numpy as np
from tqdm import trange

from agents.q_learning_agent import QLearningAgent
from agents.trainers.common import (
    Policy,
    RewardFunction,
    TrainConfig,
    policy_disagreement_from_q_table,
)
from utils.plotting import TrainingHistory
from utils.training_logger import ConsoleTrainingLogger
from world import Environment
from world.grid_codes import EMPTY_CELL


def _q_table_as_array(q_table: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Convert the sparse position-indexed Q-table into logger-friendly rows."""
    if not q_table:
        return np.zeros((0, 4), dtype=float)
    return np.vstack([q_table[state] for state in sorted(q_table)])


def _empty_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return empty cells that can be used as training starts."""
    cols, rows = np.where(grid == EMPTY_CELL)
    return [(int(col), int(row)) for col, row in zip(cols, rows, strict=True)]


def train(
    env: Environment,
    reward_fn: RewardFunction,
    cfg: TrainConfig,
    *,
    optimal_policy: Policy | None = None,
) -> tuple[QLearningAgent, TrainingHistory]:
    """Train a Q-learning agent on ``env`` and return the agent plus history.

    Records ``avg_reward`` (per-episode sum of rewards) and ``epsilon``
    (post-episode exploration rate) for downstream plotting. When
    ``optimal_policy`` is provided, also records ``policy_diff`` per
    episode — fraction of optimal-policy states the agent disagrees with.
    The agent is switched to evaluation mode before returning so
    subsequent rollouts are greedy.
    """
    if cfg.ql_episodes is None:
        raise ValueError("TrainConfig.ql_episodes is required for Q-learning")
    if cfg.log_interval < 0:
        raise ValueError("TrainConfig.log_interval must be >= 0")
    training_start_positions = _empty_positions(env.grid) if cfg.exploring_starts else []
    if cfg.exploring_starts and not training_start_positions:
        raise ValueError("No empty cells available for exploring starts")
    start_rng = random.Random(cfg.random_seed + 1)

    agent = QLearningAgent(
        alpha=cfg.alpha if cfg.alpha is not None else 0.5,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon if cfg.epsilon is not None else 1.0,
        epsilon_min=cfg.epsilon_min if cfg.epsilon_min is not None else 0.05,
        epsilon_decay=cfg.epsilon_decay if cfg.epsilon_decay is not None else 0.995,
        alpha_min=cfg.alpha_min if cfg.alpha_min is not None else 0.05,
        alpha_decay=cfg.alpha_decay if cfg.alpha_decay is not None else 0.999,
        decaying_epsilon=not cfg.fixed_epsilon,
        decaying_alpha=not cfg.fixed_alpha,
        n_actions=4,
    )

    episode_rewards: list[float] = []
    episode_deltas: list[float] = []
    episode_epsilons: list[float] = []
    episode_alphas: list[float] = []
    episode_policy_diffs: list[float] = []

    logger = None
    if cfg.log_interval > 0:
        logger = ConsoleTrainingLogger(
            show_q_table=cfg.log_q_table,
            redraw_mode="scroll",
        )

    episode_iter = (
        range(cfg.ql_episodes)
        if logger is not None
        else trange(cfg.ql_episodes, desc="Q-learning", leave=False)
    )

    for episode_idx in episode_iter:
        episode_start = (
            start_rng.choice(training_start_positions)
            if cfg.exploring_starts
            else cfg.start_pos
        )
        state = env.reset(agent_start_pos=episode_start)
        env.reward_fn = reward_fn
        agent.start_episode()
        ep_reward = 0.0
        ep_delta = 0.0
        for _ in range(cfg.max_steps):
            action = agent.take_action(state)
            next_state, reward, terminated, _info = env.step(action)
            previous_state = agent._last_state
            old_q_value = (
                float(agent.q_table[previous_state][action])
                if previous_state is not None
                else 0.0
            )
            agent.update(next_state, reward, action, terminated=terminated)
            if previous_state is not None:
                new_q_value = float(agent.q_table[previous_state][action])
                ep_delta = max(ep_delta, abs(new_q_value - old_q_value))
            state = next_state
            ep_reward += reward
            if terminated:
                break
        agent.end_episode()
        episode_rewards.append(ep_reward)
        episode_deltas.append(ep_delta)
        episode_epsilons.append(agent.epsilon)
        episode_alphas.append(agent.alpha)

        episode_num = episode_idx + 1
        if logger is not None and (
            episode_num % cfg.log_interval == 0 or episode_num == cfg.ql_episodes
        ):
            logger.log_iteration(
                episode=episode_num,
                q_values=_q_table_as_array(agent.q_table),
                q_delta=ep_delta,
                converged=False,
                current_alpha=agent.alpha,
                current_epsilon=agent.epsilon,
            )

        if optimal_policy is not None:
            episode_policy_diffs.append(
                policy_disagreement_from_q_table(optimal_policy, agent.q_table)
            )

    if logger is not None:
        logger.close()

    if cfg.exploring_starts:
        env.agent_start_pos = cfg.start_pos

    agent.set_eval_mode()

    metrics: dict[str, list[float]] = {
        "avg_reward": episode_rewards,
        "delta_q": episode_deltas,
        "epsilon": episode_epsilons,
        "alpha": episode_alphas,
    }
    if optimal_policy is not None:
        metrics["policy_diff"] = episode_policy_diffs

    history = TrainingHistory(
        episodes=list(range(1, cfg.ql_episodes + 1)),
        metrics=metrics,
        hyperparams={
            "alpha": cfg.alpha,
            "alpha_decay": cfg.alpha_decay,
            "epsilon": cfg.epsilon,
            "epsilon_decay": cfg.epsilon_decay,
            "gamma": cfg.gamma,
            "sigma": cfg.sigma,
            "log_interval": cfg.log_interval,
            "log_q_table": cfg.log_q_table,
            "exploring_starts": cfg.exploring_starts,
        },
    )
    return agent, history
