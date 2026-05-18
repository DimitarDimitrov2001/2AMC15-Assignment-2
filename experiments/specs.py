"""Experiment definitions for the assignment report suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALGORITHMS = ("value_iteration", "mc", "q_learning")

DEFAULT_GRIDS = (
    Path("grid_configs/A1_grid.npy"),
    Path("grid_configs/super_hard.npy"),
)

DEFAULTS: dict[str, Any] = {
    "sigma": 0,
    "gamma": 0.99,
    "eval_episodes": 50,
    "eval_max_steps": 1000,
    "random_seed": 0,
    "reward_function": "basic",
    "exploring_starts": True,
    "alpha": 0.2,
    "alpha_min": 0.01,
    "alpha_decay": 0.9995,
    "lr_schedule": "visit_count",
    "visit_count_c": 50,
    "epsilon": 0.7,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.99995,
    "fixed_epsilon": False,
    "ql_episodes": 100000,
    "mc_episodes": 100000,
    "max_episode_length": 1500,
    "theta": 1e-6,
    "vi_max_iter": 1000,
    "policy_stable_patience": 1000,
}

QUICK_OVERRIDES: dict[str, Any] = {
    "eval_episodes": 2,
    "eval_max_steps": 100,
    "ql_episodes": 40,
    "mc_episodes": 40,
    "max_episode_length": 100,
    "vi_max_iter": 200,
}

CSV_FIELDS = [
    "setup_group",
    "condition",
    "algorithm",
    "grid",
    "seed",
    "start_pos",
    "sigma",
    "gamma",
    "alpha",
    "alpha_min",
    "alpha_decay",
    "lr_schedule",
    "visit_count_c",
    "epsilon",
    "epsilon_min",
    "epsilon_decay",
    "fixed_epsilon",
    "ql_episodes",
    "mc_episodes",
    "max_episode_length",
    "success_rate",
    "mean_discounted_return",
    "mean_undiscounted_return",
    "mean_episode_length",
    "mean_success_episode_length",
    "policy_difference_from_optimal",
    "training_time_s",
]

METRIC_FIELDS = [
    "success_rate",
    "mean_discounted_return",
    "mean_undiscounted_return",
    "mean_episode_length",
    "policy_difference_from_optimal",
]


@dataclass(frozen=True)
class ExperimentCase:
    """One report condition to run for all selected algorithms."""

    group: str
    condition: str
    grid_path: Path
    overrides: dict[str, Any]


def defaults(*, quick: bool = False) -> dict[str, Any]:
    """Return the default run configuration, optionally shortened for smoke tests."""
    cfg = dict(DEFAULTS)
    if quick:
        cfg.update(QUICK_OVERRIDES)
    return cfg


def build_cases(grids: list[Path] | tuple[Path, ...] = DEFAULT_GRIDS) -> list[ExperimentCase]:
    """Build the assignment setup groups."""
    primary_grid = grids[0]
    cases: list[ExperimentCase] = [
        ExperimentCase(
            group="default",
            condition="default",
            grid_path=primary_grid,
            overrides={},
        )
    ]

    for grid in grids:
        cases.append(
            ExperimentCase(
                group="grid_comparison",
                condition=grid.stem,
                grid_path=grid,
                overrides={},
            )
        )

    cases.extend(
        [
            ExperimentCase("discount_factor", "gamma=0.6", primary_grid, {"gamma": 0.6}),
            ExperimentCase("discount_factor", "gamma=0.9", primary_grid, {"gamma": 0.9}),
            ExperimentCase("stochasticity", "sigma=0.02", primary_grid, {"sigma": 0.02}),
            ExperimentCase("stochasticity", "sigma=0.5", primary_grid, {"sigma": 0.5}),
            ExperimentCase(
                "exploration_epsilon",
                "low_fixed_epsilon",
                primary_grid,
                {"epsilon": 0.1, "fixed_epsilon": True},
            ),
            ExperimentCase(
                "exploration_epsilon",
                "high_fixed_epsilon",
                primary_grid,
                {"epsilon": 0.5, "fixed_epsilon": True},
            ),
            ExperimentCase(
                "exploration_epsilon",
                "decaying_epsilon",
                primary_grid,
                {
                    "epsilon": 1.0,
                    "epsilon_decay": 0.9995,
                    "epsilon_min": 0.01,
                    "fixed_epsilon": False,
                },
            ),
            ExperimentCase(
                "learning_rate",
                "low_fixed_alpha",
                primary_grid,
                {"alpha": 0.1, "lr_schedule": "constant"},
            ),
            ExperimentCase(
                "learning_rate",
                "high_fixed_alpha",
                primary_grid,
                {"alpha": 0.5, "lr_schedule": "constant"},
            ),
            ExperimentCase(
                "learning_rate",
                "decaying_alpha",
                primary_grid,
                {
                    "alpha": 0.5,
                    "alpha_decay": 0.9995,
                    "alpha_min": 0.01,
                    "lr_schedule": "exponential",
                },
            ),
            ExperimentCase(
                "learning_rate",
                "visit_count",
                primary_grid,
                {
                    "alpha": 0.5,
                    "lr_schedule": "visit_count",
                    "visit_count_c": 10.0,
                },
            ),
            ExperimentCase(
                "mc_episode_length",
                "max_episode_length=500",
                primary_grid,
                {"max_episode_length": 500},
            ),
            ExperimentCase(
                "mc_episode_length",
                "max_episode_length=5000",
                primary_grid,
                {"max_episode_length": 5000},
            ),
        ]
    )
    return cases


def group_names(cases: list[ExperimentCase]) -> list[str]:
    """Return setup-group names in first-seen order."""
    seen: set[str] = set()
    names: list[str] = []
    for case in cases:
        if case.group not in seen:
            seen.add(case.group)
            names.append(case.group)
    return names
