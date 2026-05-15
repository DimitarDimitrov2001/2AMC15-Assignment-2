"""Trainer for the Q-Learning agent."""

from __future__ import annotations

import numpy as np

from agents.learning_rates import build_lr_schedule
from agents.q_learning_agent import QLearningAgent
from agents.trainers.common import (
    OptimalActionSets,
    RewardFunction,
    TrainConfig,
    build_episode_iter,
    build_logger,
    mean_tail,
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
    optimal_policy: OptimalActionSets | None = None,
) -> tuple[QLearningAgent, TrainingHistory]:
    """Train a Q-learning agent on ``env`` and return the agent plus history.

    Records ``discounted_return`` (per-episode discounted return) and ``epsilon``
    (post-episode exploration rate) for downstream plotting. When
    ``optimal_policy`` is provided, also records ``policy_diff`` per
    episode — fraction of optimal-policy states the agent disagrees with.
    The agent is switched to evaluation mode before returning so
    subsequent rollouts are greedy.
    """
    if cfg.ql_episodes is None:
        raise ValueError("TrainConfig.ql_episodes is required for Q-learning")
    validate_log_interval(cfg)

    lr_schedule = build_lr_schedule(
        cfg.lr_schedule,
        alpha=cfg.alpha if cfg.alpha is not None else 0.5,
        alpha_decay=cfg.alpha_decay if cfg.alpha_decay is not None else 0.999,
        alpha_min=cfg.alpha_min if cfg.alpha_min is not None else 0.05,
        visit_count_c=cfg.visit_count_c,
    )

    agent = QLearningAgent(
        gamma=cfg.gamma,
        epsilon=cfg.epsilon if cfg.epsilon is not None else 1.0,
        epsilon_min=cfg.epsilon_min if cfg.epsilon_min is not None else 0.05,
        epsilon_decay=cfg.epsilon_decay if cfg.epsilon_decay is not None else 0.995,
        decaying_epsilon=not cfg.fixed_epsilon,
        n_actions=4,
        q_init=cfg.q_init,
        q_init_noise=cfg.q_init_noise,
        random_seed=cfg.random_seed,
        lr_schedule=lr_schedule,
    )

    episode_discounted_rewards: list[float] = []
    episode_deltas: list[float] = []
    episode_epsilons: list[float] = []
    episode_alphas: list[float] = []
    episode_alpha_mins: list[float] = []
    episode_alpha_maxs: list[float] = []
    episode_policy_diffs: list[float] = []

    schedule_has_global_rate = lr_schedule.get_global_rate() is not None

    logger, log_interval = build_logger(cfg, cfg.ql_episodes)
    episode_iter = build_episode_iter(cfg.ql_episodes, logger, "Q-learning")

    for episode_idx in episode_iter:
        state = env.reset(agent_start_pos=cfg.start_pos)
        env.reward_fn = reward_fn
        agent.start_episode()
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
            ep_discounted_reward += gamma_power * reward
            gamma_power *= cfg.gamma
            if terminated:
                break
        agent.end_episode()
        episode_discounted_rewards.append(ep_discounted_reward)
        episode_deltas.append(ep_delta)
        episode_epsilons.append(agent.epsilon)
        if agent.last_episode_mean_alpha is not None:
            episode_alphas.append(agent.last_episode_mean_alpha)
        if not schedule_has_global_rate:
            if agent.last_episode_alpha_min is not None:
                episode_alpha_mins.append(agent.last_episode_alpha_min)
            if agent.last_episode_alpha_max is not None:
                episode_alpha_maxs.append(agent.last_episode_alpha_max)

        if optimal_policy is not None:
            episode_policy_diffs.append(
                policy_disagreement_from_q_table(optimal_policy, agent.q_table)
            )

        episode_num = episode_idx + 1
        if logger is not None and should_log(episode_num, log_interval, cfg.ql_episodes):
            live_values = {s: float(np.max(q)) for s, q in agent.q_table.items()}
            live_policy = {s: int(np.argmax(q)) for s, q in agent.q_table.items()}
            mean_discounted = mean_tail(episode_discounted_rewards, log_interval)
            mean_delta = mean_tail(episode_deltas, log_interval)
            mean_pdiff = (
                mean_tail(episode_policy_diffs, log_interval)
                if optimal_policy is not None
                else None
            )
            logger.log_iteration(
                episode=episode_num,
                q_values=q_table_as_array(agent.q_table),
                q_delta=ep_delta,
                mean_q_delta=mean_delta,
                converged=False,
                current_alpha=agent.last_episode_mean_alpha,
                current_epsilon=agent.epsilon,
                policy_diff=mean_pdiff,
                discounted_return=mean_discounted,
                env_grid=env.grid,
                optimal_policy=optimal_policy,
                agent_start_pos=cfg.start_pos,
                agent_values=live_values,
                agent_policy=live_policy,
            )

    agent.set_eval_mode()

    metrics: dict[str, list[float]] = {
        "discounted_return": episode_discounted_rewards,
        "delta_q": episode_deltas,
        "epsilon": episode_epsilons,
    }
    if episode_alphas:
        metrics["alpha"] = episode_alphas
    if episode_alpha_mins:
        metrics["alpha_min"] = episode_alpha_mins
    if episode_alpha_maxs:
        metrics["alpha_max"] = episode_alpha_maxs
    if optimal_policy is not None:
        metrics["policy_diff"] = episode_policy_diffs

    history = TrainingHistory(
        episodes=list(range(1, cfg.ql_episodes + 1)),
        metrics=metrics,
        hyperparams={
            "lr_schedule": lr_schedule.describe(),
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
