"""Trainer for the Q-Learning agent."""

from __future__ import annotations

from tqdm import trange

from agents.q_learning_agent import QLearningAgent
from agents.trainers.common import (
    Policy,
    RewardFunction,
    TrainConfig,
    policy_disagreement_from_q_table,
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
    episode_epsilons: list[float] = []
    episode_policy_diffs: list[float] = []

    for _ in trange(cfg.ql_episodes, desc="Q-learning", leave=False):
        state = env.reset()
        env.reward_fn = reward_fn
        agent.start_episode()
        ep_reward = 0.0
        for _ in range(cfg.max_steps):
            action = agent.take_action(state)
            next_state, reward, terminated, _info = env.step(action)
            agent.update(next_state, reward, action, terminated=terminated)
            state = next_state
            ep_reward += reward
            if terminated:
                break
        agent.end_episode()
        episode_rewards.append(ep_reward)
        episode_epsilons.append(agent.epsilon)
        if optimal_policy is not None:
            episode_policy_diffs.append(
                policy_disagreement_from_q_table(optimal_policy, agent.q_table)
            )

    agent.set_eval_mode()

    metrics: dict[str, list[float]] = {
        "avg_reward": episode_rewards,
        "epsilon": episode_epsilons,
    }
    if optimal_policy is not None:
        metrics["policy_diff"] = episode_policy_diffs

    history = TrainingHistory(
        episodes=list(range(1, cfg.ql_episodes + 1)),
        metrics=metrics,
        hyperparams={
            "alpha": cfg.alpha,
            "epsilon": cfg.epsilon,
            "gamma": cfg.gamma,
            "sigma": cfg.sigma,
        },
    )
    return agent, history
