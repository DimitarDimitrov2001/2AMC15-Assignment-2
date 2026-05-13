"""Trainer dispatch for the unified training CLI and the sweep script.

All trainers share the signature::

    train(env, reward_fn, cfg, *, optimal_policy=None)
        -> tuple[BaseAgent, TrainingHistory | None]

The ``optimal_policy`` keyword argument is optional and, when set, lets
QL and MC record a per-episode policy-disagreement metric against the
reference policy (typically VI's). VI and the random baseline accept
the argument for dispatch uniformity but ignore it.

Callers select a trainer by agent name through the ``TRAINERS`` dict.
"""

from __future__ import annotations

from typing import Protocol

from agents.base_agent import BaseAgent
from agents.trainers import mc, off_policy_mc, q_learning, random_agent, value_iteration
from agents.trainers.common import (
    Policy,
    RewardFunction,
    TrainConfig,
    make_artifact_prefix,
    parse_start_pos,
    policy_disagreement,
    policy_disagreement_from_q_table,
    save_run_artifacts,
    setup_grid_run,
)
from utils.plotting import TrainingHistory
from world import Environment


class TrainerFn(Protocol):
    """Callable shape implemented by every trainer in this package."""

    def __call__(
        self,
        env: Environment,
        reward_fn: RewardFunction,
        cfg: TrainConfig,
        *,
        optimal_policy: Policy | None = None,
    ) -> tuple[BaseAgent, TrainingHistory | None]: ...


TRAINERS: dict[str, TrainerFn] = {
    "value_iteration": value_iteration.train,
    "q_learning": q_learning.train,
    "mc": mc.train,
    "off_policy_mc": off_policy_mc.train,
    "random": random_agent.train,
}

__all__ = [
    "Policy",
    "RewardFunction",
    "TRAINERS",
    "TrainConfig",
    "TrainerFn",
    "make_artifact_prefix",
    "parse_start_pos",
    "policy_disagreement",
    "policy_disagreement_from_q_table",
    "save_run_artifacts",
    "setup_grid_run",
]
