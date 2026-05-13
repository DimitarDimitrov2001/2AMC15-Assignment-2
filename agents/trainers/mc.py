"""Trainer for the on-policy first-visit Monte Carlo agent.

Owns the training loop the same way ``agents/trainers/q_learning.py`` does:
the trainer drives episode iteration, environment interaction, and history
construction. The agent itself only knows how to take actions, record steps,
and finalise an episode.
"""

from __future__ import annotations

from agents.mc_agent import MCAgent
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

_DEFAULT_MAX_EPISODE_LENGTH = 2000
_DEFAULT_EPSILON_DECAY = 0.9995
_DEFAULT_ALPHA_DECAY = 0.9995
_DEFAULT_ALPHA_MIN = 0.01
_DEFAULT_ALPHA = 0.1
_DEFAULT_EPSILON = 0.2
_DEFAULT_EPSILON_MIN = 0.01


def train(
    env: Environment,
    reward_fn: RewardFunction,
    cfg: TrainConfig,
    *,
    optimal_policy: Policy | None = None,
) -> tuple[MCAgent, TrainingHistory]:
    """Train an on-policy first-visit Monte Carlo agent on ``env``.

    When ``optimal_policy`` is provided, also records ``policy_diff`` per
    episode — fraction of optimal-policy states the agent disagrees with.
    """
    if cfg.mc_episodes is None:
        raise ValueError("TrainConfig.mc_episodes is required for Monte Carlo")
    if cfg.start_pos is None:
        raise ValueError("TrainConfig.start_pos is required for Monte Carlo")
    validate_log_interval(cfg)

    max_episode_length = (
        cfg.max_episode_length if cfg.max_episode_length is not None else _DEFAULT_MAX_EPISODE_LENGTH
    )
    epsilon_decay = 1.0 if cfg.fixed_epsilon else (cfg.epsilon_decay if cfg.epsilon_decay is not None else _DEFAULT_EPSILON_DECAY)
    alpha_arg = cfg.alpha if cfg.alpha is not None else _DEFAULT_ALPHA
    alpha_decay_arg = 1.0 if cfg.fixed_alpha else (cfg.alpha_decay if cfg.alpha_decay is not None else _DEFAULT_ALPHA_DECAY)
    alpha_min_arg = cfg.alpha_min if cfg.alpha_min is not None else _DEFAULT_ALPHA_MIN

    agent = MCAgent(
        gamma=cfg.gamma,
        epsilon=cfg.epsilon if cfg.epsilon is not None else _DEFAULT_EPSILON,
        epsilon_min=cfg.epsilon_min if cfg.epsilon_min is not None else _DEFAULT_EPSILON_MIN,
        epsilon_decay=epsilon_decay,
        alpha=alpha_arg,
        alpha_min=alpha_min_arg,
        alpha_decay=alpha_decay_arg,
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
    episode_iter = build_episode_iter(cfg.mc_episodes, logger, "MC")

    for episode_idx in episode_iter:
        state = env.reset(agent_start_pos=cfg.start_pos)
        env.reward_fn = reward_fn
        agent.start_episode()
        ep_discounted_reward = 0.0
        gamma_power = 1.0

        for _ in range(max_episode_length):
            action = agent.take_action(state)
            next_state, reward, terminated, _info = env.step(action)
            agent.record_step(state, action, reward)
            ep_discounted_reward += gamma_power * reward
            gamma_power *= cfg.gamma
            state = next_state
            if terminated:
                break

        result = agent.end_episode()
        episode_rewards.append(result.total_reward)
        episode_discounted_rewards.append(ep_discounted_reward)
        episode_deltas.append(result.delta_q)
        episode_epsilons.append(result.epsilon)
        episode_alphas.append(result.alpha)

        if optimal_policy is not None:
            episode_policy_diffs.append(
                policy_disagreement_from_q_table(optimal_policy, agent.q_table)
            )

        episode_num = episode_idx + 1
        if logger is not None and should_log(episode_num, log_interval, cfg.mc_episodes):
            agent.build_value_and_policy()
            logger.log_iteration(
                episode=episode_num,
                q_values=q_table_as_array(agent.q_table),
                q_delta=result.delta_q,
                converged=False,
                current_alpha=result.alpha,
                current_epsilon=result.epsilon,
                policy_diff=episode_policy_diffs[-1] if optimal_policy is not None else None,
                discounted_return=ep_discounted_reward,
                env_grid=env.grid,
                optimal_policy=optimal_policy,
                agent_start_pos=cfg.start_pos,
                agent_values=agent.values,
                agent_policy=agent.policy,
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
        episodes=list(range(1, len(episode_rewards) + 1)),
        metrics=metrics,
        hyperparams={
            "gamma": cfg.gamma,
            "alpha": alpha_arg,
            "alpha_decay": alpha_decay_arg,
            "epsilon": cfg.epsilon,
            "epsilon_decay": epsilon_decay,
            "sigma": cfg.sigma,
            "max_episode_length": max_episode_length,
            "q_init": cfg.q_init,
            "q_init_noise": cfg.q_init_noise,
            "log_interval": log_interval,
            "log_q_table": cfg.log_q_table,
        },
    )
    return agent, history
