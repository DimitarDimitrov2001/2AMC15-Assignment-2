"""Execution and CSV writing for assignment experiments."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from agents.trainers import TRAINERS, TrainConfig, policy_disagreement, setup_grid_run
from agents.value_iteration_agent import ValueIterationAgent
from experiments.specs import ALGORITHMS, CSV_FIELDS, ExperimentCase, group_names
from utils.evaluation import evaluate_policy_metrics
from utils.plotting import TrainingHistory


# ---------------------------------------------------------------------------
# Per-run result container
#
# ``run_one`` returns this object so downstream plotting and overview code can
# use both the CSV row and richer in-memory artifacts such as histories,
# learned value tables, policies, and the VI reference policy.
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Collected data from one algorithm/case/seed run."""

    row: dict[str, Any]
    history: dict[str, Any] | None
    values: dict | None
    policy: dict | None
    optimal_policy: dict | None
    start_pos: tuple[int, int]
    grid_path: Path


# ---------------------------------------------------------------------------
# Cached Value Iteration references
#
# Q-learning and MC compare their learned policies against a VI optimal-action
# reference. Many experiment rows share the same grid/start/sigma/gamma/theta,
# so caching avoids recomputing VI repeatedly for those identical settings.
# ---------------------------------------------------------------------------


_vi_cache: dict[tuple[Any, ...], ValueIterationAgent] = {}


def _history_to_dict(history: TrainingHistory | dict | None) -> dict | None:
    """Normalize trainer history objects to plain dictionaries for plotting."""
    if history is None:
        return None
    if isinstance(history, TrainingHistory):
        return history.to_dict()
    return history


def _train_config(cfg: dict[str, Any], start_pos: tuple[int, int]) -> TrainConfig:
    """Convert an experiment config dictionary into the trainer dataclass."""
    return TrainConfig(
        # Environment/evaluation settings.
        sigma=cfg["sigma"],
        gamma=cfg["gamma"],
        eval_max_steps=cfg["eval_max_steps"],
        random_seed=cfg["random_seed"],
        eval_episodes=cfg["eval_episodes"],
        start_pos=start_pos,

        # Model-free learning-rate settings.
        alpha=cfg["alpha"],
        alpha_min=cfg["alpha_min"],
        alpha_decay=cfg["alpha_decay"],
        lr_schedule=cfg["lr_schedule"],
        visit_count_c=cfg["visit_count_c"],

        # Model-free exploration settings.
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        fixed_epsilon=cfg["fixed_epsilon"],

        # Training budgets.
        ql_episodes=cfg["ql_episodes"],
        mc_episodes=cfg["mc_episodes"],
        max_episode_length=cfg["max_episode_length"],

        # Value Iteration solver settings.
        theta=cfg["theta"],
        vi_max_iter=cfg["vi_max_iter"],

        # Training-start and stopping settings for Q-learning/MC.
        exploring_starts=cfg.get("exploring_starts", False),
        policy_stable_patience=cfg["policy_stable_patience"],
    )


def _get_vi_reference(
    grid_path: Path,
    cfg: dict[str, Any],
    start_pos: tuple[int, int],
    reward_fn,
    env,
) -> ValueIterationAgent:
    """Return the cached VI reference agent for a comparable run setting."""
    key = (
        str(grid_path),
        start_pos,
        cfg["sigma"],
        cfg["gamma"],
        cfg["theta"],
        cfg["vi_max_iter"],
    )
    if key not in _vi_cache:
        # Train VI once for this exact setting. The returned agent exposes both
        # values and optimal-action sets used by model-free comparisons.
        vi_cfg = _train_config(cfg, start_pos)
        vi_agent, _ = TRAINERS["value_iteration"](env, reward_fn, vi_cfg)
        _vi_cache[key] = vi_agent
    return _vi_cache[key]


def _fmt_optional(value: Any) -> Any:
    """Format optional/numeric values for stable CSV output."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return round(value, 6)
    return value


def run_one(
    case: ExperimentCase,
    algorithm: str,
    base_cfg: dict[str, Any],
    seed: int,
) -> RunResult:
    """Train and evaluate one algorithm for one assignment condition."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Assignment experiments do not run algorithm {algorithm!r}")

    # ------------------------------------------------------------------
    # Merge baseline config, case override, and seed
    # ------------------------------------------------------------------
    # ``case.overrides`` changes one experimental factor while all other
    # settings remain at the shared defaults.
    cfg = {**base_cfg, **case.overrides, "random_seed": seed}

    # ------------------------------------------------------------------
    # Build environment and trainer config
    # ------------------------------------------------------------------
    # ``setup_grid_run`` resolves the actual start position after env reset;
    # that resolved start is then pinned into TrainConfig and evaluation.
    env, start_pos, reward_fn = setup_grid_run(
        grid_path=case.grid_path,
        sigma=cfg["sigma"],
        fps=-1,
        no_gui=True,
        start_pos=None,
        random_seed=seed,
    )
    train_cfg = _train_config(cfg, start_pos)

    # ------------------------------------------------------------------
    # Optional VI reference for model-free algorithms
    # ------------------------------------------------------------------
    # VI is itself the reference, so only MC/Q-learning need optimal actions
    # for policy-disagreement metrics.
    optimal_policy = None
    if algorithm != "value_iteration":
        vi_agent = _get_vi_reference(case.grid_path, cfg, start_pos, reward_fn, env)
        optimal_policy = vi_agent.optimal_action_sets()

    # ------------------------------------------------------------------
    # Train selected algorithm
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    agent, history = TRAINERS[algorithm](
        env,
        reward_fn,
        train_cfg,
        optimal_policy=optimal_policy,
    )
    training_time = time.perf_counter() - t0
    history_dict = _history_to_dict(history)

    # ------------------------------------------------------------------
    # Evaluate trained policy
    # ------------------------------------------------------------------
    # Evaluation uses the same start position and reward function as training,
    # but runs fresh rollouts to measure success and returns.
    metrics = evaluate_policy_metrics(
        grid=case.grid_path,
        agent=agent,
        eval_max_steps=cfg["eval_max_steps"],
        sigma=cfg["sigma"],
        agent_start_pos=start_pos,
        reward_fn=reward_fn,
        gamma=cfg["gamma"],
        random_seed=seed,
        n_eval_episodes=cfg["eval_episodes"],
    )

    # ------------------------------------------------------------------
    # Build CSV row
    # ------------------------------------------------------------------
    # The row stores the experimental factors, performance metrics, and the
    # final policy disagreement scalar for summary tables.
    policy_diff = 0.0 if algorithm == "value_iteration" else policy_disagreement(optimal_policy, agent)
    row = {
        "setup_group": case.group,
        "condition": case.condition,
        "algorithm": algorithm,
        "grid": case.grid_path.stem,
        "seed": seed,
        "start_pos": f"{start_pos[0]},{start_pos[1]}",
        "sigma": cfg["sigma"],
        "gamma": cfg["gamma"],
        "alpha": cfg["alpha"],
        "alpha_min": cfg["alpha_min"],
        "alpha_decay": cfg["alpha_decay"],
        "lr_schedule": cfg["lr_schedule"],
        "visit_count_c": cfg["visit_count_c"],
        "epsilon": cfg["epsilon"],
        "epsilon_min": cfg["epsilon_min"],
        "epsilon_decay": cfg["epsilon_decay"],
        "fixed_epsilon": cfg["fixed_epsilon"],
        "ql_episodes": cfg["ql_episodes"],
        "mc_episodes": cfg["mc_episodes"],
        "max_episode_length": cfg["max_episode_length"],
        "success_rate": _fmt_optional(metrics["success_rate"]),
        "mean_discounted_return": _fmt_optional(metrics["mean_discounted_return"]),
        "mean_undiscounted_return": _fmt_optional(metrics["mean_undiscounted_return"]),
        "mean_episode_length": _fmt_optional(metrics["mean_episode_length"]),
        "mean_success_episode_length": _fmt_optional(metrics["mean_success_episode_length"]),
        "policy_difference_from_optimal": _fmt_optional(policy_diff),
        "training_time_s": round(training_time, 3),
    }

    # Keep richer artifacts alongside the CSV row so plotting can avoid
    # reparsing files or retraining agents.
    return RunResult(
        row=row,
        history=history_dict,
        values=getattr(agent, "values", None),
        policy=getattr(agent, "policy", None),
        optimal_policy=optimal_policy,
        start_pos=start_pos,
        grid_path=case.grid_path,
    )


def run_suite(
    cases: list[ExperimentCase],
    base_cfg: dict[str, Any],
    out_dir: Path,
    seeds: list[int],
) -> list[RunResult]:
    """Run all assignment cases, writing master and per-group CSVs."""
    # ------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    for group in group_names(cases):
        (out_dir / group).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Prepare run counters and CSV writers
    # ------------------------------------------------------------------
    results: list[RunResult] = []
    total = len(cases) * len(ALGORITHMS) * len(seeds)
    tqdm.write(f"Running {len(cases)} conditions x {len(ALGORITHMS)} algorithms x {len(seeds)} seeds = {total} runs")
    tqdm.write(f"Results -> {out_dir}")

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as master:
        master_writer = csv.DictWriter(master, fieldnames=CSV_FIELDS)
        master_writer.writeheader()

        # Each setup group gets its own results.csv in addition to the master
        # CSV, using the exact same schema.
        group_files = {
            group: (out_dir / group / "results.csv").open("w", newline="", encoding="utf-8")
            for group in group_names(cases)
        }
        try:
            group_writers = {
                group: csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                for group, handle in group_files.items()
            }
            for writer in group_writers.values():
                writer.writeheader()

            # ------------------------------------------------------------------
            # Main experiment loop
            # ------------------------------------------------------------------
            # Iterate in case -> seed -> algorithm order so progress output and
            # CSV rows stay grouped by report condition.
            with tqdm(total=total, unit="run") as pbar:
                for case in cases:
                    if not case.grid_path.exists():
                        tqdm.write(f"[SKIP] grid not found: {case.grid_path}")
                        pbar.update(len(ALGORITHMS) * len(seeds))
                        continue
                    for seed in seeds:
                        for algorithm in ALGORITHMS:
                            pbar.set_description(f"{case.group}/{case.condition}/{algorithm}/seed={seed}")
                            try:
                                result = run_one(case, algorithm, base_cfg, seed)
                            except Exception as exc:
                                tqdm.write(f"[ERROR] {case.group}/{case.condition}/{algorithm}/seed={seed}: {exc}")
                                pbar.update(1)
                                continue

                            # Write immediately and flush so long experiment
                            # runs leave usable partial CSVs if interrupted.
                            results.append(result)
                            master_writer.writerow(result.row)
                            group_writers[case.group].writerow(result.row)
                            master.flush()
                            group_files[case.group].flush()
                            pbar.update(1)
        finally:
            # Ensure group CSV handles close even if one run fails.
            for handle in group_files.values():
                handle.close()

    return results
