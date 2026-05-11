"""Shared utilities for trainer modules.

This module is the single place where the bootstrap dance for a grid run
lives, and the single place where per-run artifacts are written. Trainer
functions in sibling modules stay pure (no I/O, no plotting, no prints).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from agents.base_agent import BaseAgent
from agents.value_iteration_agent import ValueIterationAgent
from utils.artifacts import (
    save_evaluation_summary_artifact,
    save_policy_disagreement_artifact,
    save_training_curves_artifact,
    save_value_iteration_artifacts,
    write_json,
)
from utils.plotting import TrainingHistory
from world import Environment, build_manhattan_reward_function, find_target_position

Position = tuple[int, int]
RewardFunction = Callable[[np.ndarray, Position], float]
Policy = dict[Position, int]
QTable = dict[Position, np.ndarray]


@dataclass
class TrainConfig:
    """Hyperparameters and run-time options consumed by trainer functions.

    Agent-specific fields default to ``None`` so a single dataclass works
    for every trainer. Each trainer reads only the fields it needs and
    expects them to be populated by the caller (CLI or sweep).
    """

    sigma: float
    gamma: float
    max_steps: int
    random_seed: int
    eval_episodes: int
    start_pos: Position | None = None
    alpha: float | None = None
    alpha_min: float | None = None
    alpha_decay: float | None = None
    epsilon: float | None = None
    epsilon_min: float | None = None
    epsilon_decay: float | None = None
    fixed_alpha: bool = False
    fixed_epsilon: bool = False
    ql_episodes: int | None = None
    mc_episodes: int | None = None
    max_episode_length: int | None = None
    theta: float | None = None
    vi_max_iter: int | None = None


def policy_disagreement_from_q_table(
    optimal_policy: Policy,
    q_table: QTable,
) -> float:
    """Fraction of ``optimal_policy`` states where greedy(q_table) disagrees.

    Mid-training learned policy is derived from ``q_table`` directly so the
    metric can be sampled without calling ``set_eval_mode()`` or
    ``build_value_and_policy()``, which mutate agent state. Unvisited states
    default to action 0, matching the convention in ``policy_disagreement``.
    """
    if not optimal_policy:
        return float("nan")
    mismatches = 0
    for state, optimal_action in optimal_policy.items():
        q_values = q_table.get(state)
        learned_action = 0 if q_values is None else int(np.argmax(q_values))
        if learned_action != optimal_action:
            mismatches += 1
    return mismatches / len(optimal_policy)


def policy_disagreement(optimal_policy: Policy, agent: BaseAgent) -> float:
    """End-of-training fraction of optimal-policy states the agent disagrees on.

    Reads ``agent.policy`` (a ``{state: action}`` dict on QL/MC/VI after
    training). Returns NaN when no reference policy is provided.
    Unvisited states default to action 0 — same as a zero Q-array's argmax.
    """
    if not optimal_policy or not hasattr(agent, "policy"):
        return float("nan")
    learned_policy: Policy = agent.policy  # type: ignore[attr-defined]
    mismatches = sum(
        1 for state, action in optimal_policy.items() if learned_policy.get(state, 0) != action
    )
    return mismatches / len(optimal_policy)


def parse_start_pos(raw: str | None) -> Position | None:
    """Parse a ``col,row`` string into a tuple, or return ``None``."""
    if raw is None:
        return None
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError("start_pos must be formatted as col,row")
    return int(parts[0]), int(parts[1])


def _placeholder_reward_function(_grid: np.ndarray, _agent_pos: Position) -> float:
    """Stand-in until the real reward function is built after ``env.reset()``."""
    return 0.0


def setup_grid_run(
    grid_path: Path,
    sigma: float,
    fps: int,
    no_gui: bool,
    start_pos: Position | None,
    random_seed: int,
) -> tuple[Environment, Position, RewardFunction]:
    """Construct the environment, choose the start position, build the reward.

    The Manhattan-scaled reward function needs the agent's start position,
    which itself depends on ``env.reset()``. This helper performs the
    bootstrap dance once and patches the env with the final reward function.
    """
    env = Environment(
        grid_fp=grid_path,
        no_gui=no_gui,
        reward_fn=_placeholder_reward_function,
        sigma=sigma,
        target_fps=fps,
        agent_start_pos=start_pos,
        random_seed=random_seed,
    )
    initial_pos = env.reset()
    env.agent_start_pos = initial_pos
    target_pos = find_target_position(env.grid)
    reward_fn = build_manhattan_reward_function(initial_pos, target_pos)
    env.reward_fn = reward_fn
    return env, initial_pos, reward_fn


def make_artifact_prefix(grid_path: Path, agent_name: str) -> str:
    """Build the timestamped artifact prefix used for output files."""
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    return f"{grid_path.stem}_{agent_name}_{timestamp}"


def save_run_artifacts(
    out_dir: Path,
    artifact_prefix: str,
    grid_path: Path,
    agent: BaseAgent,
    env_grid: np.ndarray,
    initial_pos: Position,
    evaluation_metrics: dict,
    reward_fn: RewardFunction,
    cfg: TrainConfig,
    optimal_policy: Policy | None = None,
    policy_diff_scalar: float | None = None,
    history: TrainingHistory | None = None,
) -> None:
    """Write per-run metrics, evaluation summary, and the path visualisation.

    For Value Iteration, also emits the value/policy PNG via the existing
    ``save_value_iteration_artifacts`` helper. When ``optimal_policy`` is
    provided and the agent exposes a learned ``policy`` attribute, also
    emits a spatial policy-disagreement heatmap. The disagreement scalar,
    if supplied, is included in the human-readable evaluation summary.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(agent, ValueIterationAgent):
        save_value_iteration_artifacts(
            out_dir=out_dir,
            artifact_prefix=artifact_prefix,
            grid=env_grid,
            initial_pos=initial_pos,
            agent=agent,
            evaluation_metrics=evaluation_metrics,
        )
    else:
        write_json(out_dir / f"{artifact_prefix}_metrics.json", {"evaluation": evaluation_metrics})

    save_evaluation_summary_artifact(
        out_dir, artifact_prefix, evaluation_metrics, policy_difference=policy_diff_scalar
    )

    # Spatial disagreement plot only makes sense for non-VI agents that
    # have a learned policy to compare against the reference.
    if optimal_policy is not None and not isinstance(agent, ValueIterationAgent):
        learned_policy = getattr(agent, "policy", None)
        if learned_policy:
            save_policy_disagreement_artifact(
                out_dir=out_dir,
                artifact_prefix=artifact_prefix,
                grid=env_grid,
                optimal_policy=optimal_policy,
                learned_policy=learned_policy,
                agent_start_pos=initial_pos,
            )

    # Per-episode training curves (avg_reward, epsilon, policy_diff, ...).
    # VI's value/policy plot already covers its history; non-VI runs with
    # a captured history get a generic training-curves PNG.
    if history is not None and not isinstance(agent, ValueIterationAgent):
        save_training_curves_artifact(
            out_dir=out_dir,
            artifact_prefix=artifact_prefix,
            history=history,
        )

    Environment.evaluate_agent(
        grid_fp=grid_path,
        agent=agent,
        max_steps=cfg.max_steps,
        sigma=cfg.sigma,
        agent_start_pos=initial_pos,
        reward_fn=reward_fn,
        random_seed=cfg.random_seed,
        out_dir=out_dir,
        file_name=f"{artifact_prefix}_path",
    )
