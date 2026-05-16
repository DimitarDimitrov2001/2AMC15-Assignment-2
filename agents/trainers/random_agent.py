"""Trainer for the Random baseline agent.

Random has no learning step, so this trainer just constructs the agent
and returns ``None`` for the history. Module name avoids shadowing the
``random`` stdlib module.
"""

from __future__ import annotations

from agents.random_agent import RandomAgent
from agents.trainers.common import OptimalActionSets, RewardFunction, TrainConfig
from utils.plotting import TrainingHistory
from world import Environment


def train(
    env: Environment,
    reward_fn: RewardFunction,
    cfg: TrainConfig,
    *,
    optimal_policy: OptimalActionSets | None = None,
    optimal_values: dict[tuple[int, int], float] | None = None,
) -> tuple[RandomAgent, TrainingHistory | None]:
    """Return a fresh RandomAgent. Random does not learn, so history is ``None``.

    ``optimal_policy`` and ``optimal_values`` are accepted for
    trainer-dispatch uniformity but ignored — the random baseline has no
    learnable policy or value function to compare.
    """
    del env, reward_fn, cfg, optimal_policy, optimal_values
    return RandomAgent(), None
