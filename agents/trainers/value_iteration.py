"""Trainer for the Value Iteration agent."""

from __future__ import annotations

from agents.trainers.common import OptimalActionSets, RewardFunction, TrainConfig
from agents.value_iteration_agent import ValueIterationAgent
from utils.plotting import TrainingHistory
from world import Environment


def train(
    env: Environment,
    reward_fn: RewardFunction,
    cfg: TrainConfig,
    *,
    optimal_policy: OptimalActionSets | None = None,
    optimal_values: dict[tuple[int, int], float] | None = None,
) -> tuple[ValueIterationAgent, TrainingHistory | None]:
    """Run value iteration to convergence and return the agent plus history.

    VI is dynamic programming over the known grid model, so the loop does
    not interact with the environment after construction. The ``env``
    argument is only used to read the grid array. ``optimal_policy`` and
    ``optimal_values`` are accepted for trainer-dispatch uniformity but
    ignored — VI is the reference itself.
    """
    del optimal_policy, optimal_values
    agent = ValueIterationAgent(
        grid=env.grid,
        reward_fn=reward_fn,
        sigma=cfg.sigma,
        gamma=cfg.gamma,
        theta=cfg.theta if cfg.theta is not None else 1e-6,
        max_iterations=cfg.vi_max_iter if cfg.vi_max_iter is not None else 1000,
    )
    agent.train()
    return agent, agent.history
