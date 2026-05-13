"""Trainer for off-policy Monte Carlo control."""

from __future__ import annotations

import random

import numpy as np
from tqdm import trange

from agents.off_policy_mc_agent import OffPolicyMCAgent
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

_DEFAULT_MAX_EPISODE_LENGTH = 2000
_DEFAULT_EPSILON = 0.3
_DEFAULT_EPSILON_MIN = 0.02
_DEFAULT_EPSILON_DECAY = 0.9998
_DEFAULT_ALPHA = 0.2
_DEFAULT_ALPHA_MIN = 0.02
_DEFAULT_ALPHA_DECAY = 0.9998


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
) -> tuple[OffPolicyMCAgent, TrainingHistory]:
    """Train weighted-importance-sampling off-policy MC control."""
    if cfg.mc_episodes is None:
        raise ValueError("TrainConfig.mc_episodes is required for off-policy Monte Carlo")
    if cfg.start_pos is None:
        raise ValueError("TrainConfig.start_pos is required for off-policy Monte Carlo")
    if cfg.log_interval < 0:
        raise ValueError("TrainConfig.log_interval must be >= 0")
    if cfg.off_policy_update not in {"weighted", "alpha"}:
        raise ValueError("TrainConfig.off_policy_update must be 'weighted' or 'alpha'")

    max_episode_length = (
        cfg.max_episode_length if cfg.max_episode_length is not None else _DEFAULT_MAX_EPISODE_LENGTH
    )
    epsilon_decay = 1.0 if cfg.fixed_epsilon else (
        cfg.epsilon_decay if cfg.epsilon_decay is not None else _DEFAULT_EPSILON_DECAY
    )
    alpha_decay = 1.0 if cfg.fixed_alpha else (
        cfg.alpha_decay if cfg.alpha_decay is not None else _DEFAULT_ALPHA_DECAY
    )

    agent = OffPolicyMCAgent(
        gamma=cfg.gamma,
        epsilon=cfg.epsilon if cfg.epsilon is not None else _DEFAULT_EPSILON,
        epsilon_min=cfg.epsilon_min if cfg.epsilon_min is not None else _DEFAULT_EPSILON_MIN,
        epsilon_decay=epsilon_decay,
        alpha=cfg.alpha if cfg.alpha is not None else _DEFAULT_ALPHA,
        alpha_min=cfg.alpha_min if cfg.alpha_min is not None else _DEFAULT_ALPHA_MIN,
        alpha_decay=alpha_decay,
        decaying_alpha=not cfg.fixed_alpha,
        q_init=cfg.q_init,
        q_init_noise=cfg.q_init_noise,
        update_mode=cfg.off_policy_update,
        importance_weight_clip=cfg.importance_weight_clip,
        random_seed=cfg.random_seed,
    )
    training_start_positions = _empty_positions(env.grid) if cfg.exploring_starts else []
    if cfg.exploring_starts and not training_start_positions:
        raise ValueError("No empty cells available for exploring starts")
    start_rng = random.Random(cfg.random_seed + 1)

    episode_rewards: list[float] = []
    episode_deltas: list[float] = []
    episode_epsilons: list[float] = []
    episode_alphas: list[float] = []
    episode_importance_weights: list[float] = []
    episode_policy_diffs: list[float] = []

    logger = None
    if cfg.log_interval > 0:
        logger = ConsoleTrainingLogger(
            show_q_table=cfg.log_q_table,
            redraw_mode="scroll",
        )

    episode_iter = (
        range(cfg.mc_episodes)
        if logger is not None
        else trange(cfg.mc_episodes, desc="Off-policy MC", leave=False)
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

        for _ in range(max_episode_length):
            action = agent.take_action(state)
            next_state, reward, terminated, _info = env.step(action)
            agent.record_step(state, action, reward)
            state = next_state
            if terminated:
                break

        result = agent.end_episode()
        episode_rewards.append(result.total_reward)
        episode_deltas.append(result.delta_q)
        episode_epsilons.append(result.epsilon)
        if result.alpha is not None:
            episode_alphas.append(result.alpha)
        episode_importance_weights.append(result.importance_weight)

        episode_num = episode_idx + 1
        if logger is not None and (
            episode_num % cfg.log_interval == 0 or episode_num == cfg.mc_episodes
        ):
            logger.log_iteration(
                episode=episode_num,
                q_values=_q_table_as_array(agent.q_table),
                q_delta=result.delta_q,
                converged=False,
                current_alpha=result.alpha,
                current_epsilon=result.epsilon,
            )

        if optimal_policy is not None:
            episode_policy_diffs.append(
                policy_disagreement_from_q_table(optimal_policy, agent.q_table)
            )

    if logger is not None:
        logger.close()

    agent.build_value_and_policy()

    metrics: dict[str, list[float]] = {
        "avg_reward": episode_rewards,
        "delta_q": episode_deltas,
        "epsilon": episode_epsilons,
        "importance_weight": episode_importance_weights,
    }
    if episode_alphas:
        metrics["alpha"] = episode_alphas
    if optimal_policy is not None:
        metrics["policy_diff"] = episode_policy_diffs

    history = TrainingHistory(
        episodes=list(range(1, len(episode_rewards) + 1)),
        metrics=metrics,
        hyperparams={
            "algorithm": "off_policy_mc",
            "gamma": cfg.gamma,
            "epsilon": cfg.epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": cfg.epsilon_min,
            "alpha": cfg.alpha,
            "alpha_decay": alpha_decay,
            "alpha_min": cfg.alpha_min,
            "q_init": cfg.q_init,
            "q_init_noise": cfg.q_init_noise,
            "off_policy_update": cfg.off_policy_update,
            "importance_weight_clip": cfg.importance_weight_clip,
            "sigma": cfg.sigma,
            "max_episode_length": max_episode_length,
            "log_interval": cfg.log_interval,
            "log_q_table": cfg.log_q_table,
            "exploring_starts": cfg.exploring_starts,
        },
    )
    return agent, history
