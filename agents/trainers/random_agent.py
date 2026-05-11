"""Trainer for the Random baseline agent.

Random has no learning step, so this trainer just constructs the agent
and returns ``None`` for the history. Module name avoids shadowing the
``random`` stdlib module.
"""

from __future__ import annotations

from agents.random_agent import RandomAgent
from agents.trainers.common import Policy, RewardFunction, TrainConfig
from utils.plotting import TrainingHistory
from world import Environment


def train(
    env: Environment,
    reward_fn: RewardFunction,
    cfg: TrainConfig,
    *,
    optimal_policy: Policy | None = None,
) -> tuple[RandomAgent, TrainingHistory | None]:
    """Return a fresh RandomAgent. Random does not learn, so history is ``None``.

    ``optimal_policy`` is accepted for trainer-dispatch uniformity but
    ignored — the random baseline has no learnable policy to compare.
    """
    del env, reward_fn, cfg, optimal_policy
    return RandomAgent(), None
