"""Trainer for the Q-Learning agent."""

from __future__ import annotations

import numpy as np

from agents.q_learning_agent import QLearningAgent
from agents.trainers.common import (
    Policy,
    RewardFunction,
    TrainConfig,
    build_episode_iter,
    build_logger,
    policy_disagreement_from_q_table,
    q_table_as_array,
    should_log,
    validate_log_interval,
)
from utils.plotting import TrainingHistory
from world import Environment


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
    validate_log_interval(cfg)

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
        q_init=cfg.q_init,
        q_init_noise=cfg.q_init_noise,
        random_seed=cfg.random_seed,
    )

    episode_rewards: list[float] = []
    episode_discounted_rewards: list[float] = []
    episode_deltas: list[float] = []
    episode_epsilons: list[float] = []
    episode_alphas: list[float] = []
    episode_policy_diffs: list[float] = []

    logger, log_interval = build_logger(cfg)
    episode_iter = build_episode_iter(cfg.ql_episodes, logger, "Q-learning")

    for episode_idx in episode_iter:
        state = env.reset(agent_start_pos=cfg.start_pos)
        env.reward_fn = reward_fn
        agent.start_episode()
        ep_reward = 0.0
        ep_discounted_reward = 0.0
        ep_delta = 0.0
        gamma_power = 1.0
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
            ep_discounted_reward += gamma_power * reward
            gamma_power *= cfg.gamma
            if terminated:
                break
        agent.end_episode()
        episode_rewards.append(ep_reward)
        episode_discounted_rewards.append(ep_discounted_reward)
        episode_deltas.append(ep_delta)
        episode_epsilons.append(agent.epsilon)
        episode_alphas.append(agent.alpha)

        if optimal_policy is not None:
            episode_policy_diffs.append(
                policy_disagreement_from_q_table(optimal_policy, agent.q_table)
            )

        episode_num = episode_idx + 1
        if logger is not None and should_log(episode_num, log_interval, cfg.ql_episodes):
            live_values = {s: float(np.max(q)) for s, q in agent.q_table.items()}
            live_policy = {s: int(np.argmax(q)) for s, q in agent.q_table.items()}
            logger.log_iteration(
                episode=episode_num,
                q_values=q_table_as_array(agent.q_table),
                q_delta=ep_delta,
                converged=False,
                current_alpha=agent.alpha,
                current_epsilon=agent.epsilon,
                policy_diff=episode_policy_diffs[-1] if optimal_policy is not None else None,
                discounted_return=ep_discounted_reward,
                env_grid=env.grid,
                optimal_policy=optimal_policy,
                agent_start_pos=cfg.start_pos,
                agent_values=live_values,
                agent_policy=live_policy,
            )

    agent.set_eval_mode()

    metrics: dict[str, list[float]] = {
        "avg_reward": episode_rewards,
        "discounted_return": episode_discounted_rewards,
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
            "q_init": cfg.q_init,
            "q_init_noise": cfg.q_init_noise,
            "log_interval": log_interval,
            "log_q_table": cfg.log_q_table,
        },
    )
    return agent, history
