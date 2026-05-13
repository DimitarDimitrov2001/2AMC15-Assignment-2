"""Shared utilities for trainer modules.

This module is the single place where the bootstrap dance for a grid run
lives, and the single place where per-run artifacts are written. Trainer
functions in sibling modules stay pure (no I/O, no plotting, no prints).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from agents.base_agent import BaseAgent
from agents.value_iteration_agent import ValueIterationAgent

if TYPE_CHECKING:
    from utils.training_logger import TrainingLogger
from utils.artifacts import (
    save_evaluation_summary_artifact,
    save_policy_disagreement_artifact,
    save_training_curves_artifact,
    save_value_iteration_artifacts,
    write_json,
)
from utils.plotting import TrainingHistory
from world import Environment, build_basic_reward_function, build_manhattan_reward_function, find_target_position

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
    log_interval: int = 0
    log_q_table: bool = False
    q_init: float = 0.0
    q_init_noise: float = 1e-6
    # exploring_starts: bool = False
    off_policy_update: str = "alpha"
    importance_weight_clip: float | None = 10.0
    soft_target_epsilon: float = 0.0
    theta: float | None = None
    vi_max_iter: int | None = None
    reward_function: str = "manhattan"
    wandb: bool = False
    wandb_project: str = "rl-in-practice"


# ---------------------------------------------------------------------------
# Shared trainer helpers — deduplicated from mc / q_learning / off_policy_mc
# ---------------------------------------------------------------------------


def q_table_as_array(q_table: QTable) -> np.ndarray:
    """Convert a sparse position-indexed Q-table into a 2-D array for the logger."""
    if not q_table:
        return np.zeros((0, 4), dtype=float)
    return np.vstack([q_table[state] for state in sorted(q_table)])


def build_logger(cfg: TrainConfig) -> tuple[TrainingLogger | None, int]:
    """Create the appropriate training logger from *cfg*.

    Returns ``(logger, effective_log_interval)`` so callers never need to
    mutate *cfg* as a side-effect.  When W&B is enabled but
    ``cfg.log_interval`` is 0, the effective interval defaults to 100.

    For W&B the caller must have already called ``wandb.init()``; the
    logger only records metrics to the active run.
    """
    if cfg.wandb:
        from utils.training_logger import WandbTrainingLogger

        return WandbTrainingLogger(), cfg.log_interval if cfg.log_interval > 0 else 100
    if cfg.log_interval > 0:
        from utils.training_logger import ConsoleTrainingLogger

        return ConsoleTrainingLogger(
            show_q_table=cfg.log_q_table, redraw_mode="scroll",
        ), cfg.log_interval
    return None, cfg.log_interval


def build_episode_iter(
    n_episodes: int, logger: TrainingLogger | None, desc: str,
) -> Iterable[int]:
    """Return a ``range`` (when a logger provides its own output) or a ``trange`` progress bar."""
    if logger is not None:
        return range(n_episodes)
    from tqdm import trange

    return trange(n_episodes, desc=desc, leave=False)


def should_log(episode_num: int, log_interval: int, total_episodes: int) -> bool:
    """Whether to emit a log entry for *episode_num* (1-based)."""
    return log_interval > 0 and (
        episode_num % log_interval == 0 or episode_num == total_episodes
    )


def validate_log_interval(cfg: TrainConfig) -> None:
    """Raise if ``cfg.log_interval`` is negative."""
    if cfg.log_interval < 0:
        raise ValueError("TrainConfig.log_interval must be >= 0")


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


REWARD_FUNCTIONS = ("manhattan", "basic")


def setup_grid_run(
    grid_path: Path,
    sigma: float,
    fps: int,
    no_gui: bool,
    start_pos: Position | None,
    random_seed: int,
    reward_function: str = "manhattan",
) -> tuple[Environment, Position, RewardFunction]:
    """Construct the environment, choose the start position, build the reward.

    ``reward_function`` selects which reward scheme to use:
      * ``"manhattan"`` — -1 per step, -5 for walls/obstacles, target reward
        scaled to 2 * Manhattan(start, target) (min 10).
      * ``"basic"`` — -1 for every step (including wall bumps), +10 for
        reaching the target (the assignment spec default).
    """
    if reward_function not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function {reward_function!r}; "
            f"choose from {REWARD_FUNCTIONS}"
        )

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

    if reward_function == "basic":
        reward_fn = build_basic_reward_function()
    else:
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
    wandb_log: bool = False,
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
            wandb_log=wandb_log,
        )
    else:
        training_payload: dict[str, object] = {}
        if history is not None:
            training_payload = history.to_dict()
        write_json(
            out_dir / f"{artifact_prefix}_metrics.json",
            {"training": training_payload, "evaluation": evaluation_metrics},
        )

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
                wandb_log=wandb_log,
            )

    # Per-episode training curves (avg_reward, epsilon, policy_diff, ...).
    # VI's value/policy plot already covers its history; non-VI runs with
    # a captured history get a generic training-curves PNG.
    if history is not None and not isinstance(agent, ValueIterationAgent):
        save_training_curves_artifact(
            out_dir=out_dir,
            artifact_prefix=artifact_prefix,
            history=history,
            wandb_log=wandb_log,
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
