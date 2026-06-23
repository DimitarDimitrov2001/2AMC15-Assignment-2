"""Create final report tables and learning-curve figures from merged CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_TRAINING_CURVES_PATH = DEFAULT_RESULTS_DIR / "training_curves.csv"
DEFAULT_EVALUATION_RESULTS_PATH = DEFAULT_RESULTS_DIR / "evaluation_results.csv"
DEFAULT_EVALUATION_TABLE_PATH = DEFAULT_RESULTS_DIR / "final_evaluation_table.md"
DEFAULT_LEARNING_CURVES_PATH = DEFAULT_RESULTS_DIR / "final_learning_curves.png"
AGENT_LABELS = {"dqn": "DQN", "ddqn": "Dueling DQN"}
AGENT_COLORS = {"dqn": "#1f77b4", "ddqn": "#d62728"}
GRID_LAYOUT = (
    ("simple_cave_grid", "Simple Grid"),
    ("A1_grid", "Medium Difficulty Grid"),
    ("realistic_super_hard_cave", "Difficult Grid"),
)
SCENARIO_LAYOUT = (
    ("baseline", "Baseline", {"sensor_mode": "sensor", "sigma": 0.0}),
    ("no_sensors", "Experiment 1 - Without Sensors", {"sensor_mode": "no_sensor", "sigma": 0.0}),
    (
        "stochastic",
        "Experiment 2 - Stochastic Robustness (sigma 0.5)",
        {"sensor_mode": "sensor", "sigma": 0.5},
    ),
)
EVALUATION_METRICS = (
    (
        "Average return",
        "final_eval_mean_reward_mean",
        "final_eval_mean_reward_variance",
        "{:.3f}",
    ),
    (
        "Success rate",
        "final_eval_success_rate_mean",
        "final_eval_success_rate_variance",
        "{:.3f}",
    ),
    (
        "Average total training collisions",
        "training_total_collisions_mean",
        "training_total_collisions_variance",
        "{:.1f}",
    ),
    (
        "Average episode length",
        "final_eval_mean_steps_mean",
        "final_eval_mean_steps_variance",
        "{:.1f}",
    ),
)
ROLLING_WINDOW_EPISODES = 25
MAX_LEARNING_CURVE_EPISODE = 6000
LEARNING_CURVE_Y_LIMITS = (-12.0, 2.0)


@dataclass(frozen=True)
class Scenario:
    """Report scenario selector and display label."""

    key: str
    label: str
    sensor_mode: str
    sigma: float


def parse_args() -> argparse.Namespace:
    """Parse command-line options for final artifact generation."""
    parser = argparse.ArgumentParser(
        description="Create final evaluation table and learning-curve figure."
    )
    parser.add_argument(
        "--training-curves",
        type=Path,
        default=DEFAULT_TRAINING_CURVES_PATH,
        help="Input training_curves.csv created by scripts/merge_experiments.py.",
    )
    parser.add_argument(
        "--evaluation-results",
        type=Path,
        default=DEFAULT_EVALUATION_RESULTS_PATH,
        help="Input evaluation_results.csv created by scripts/merge_experiments.py.",
    )
    parser.add_argument(
        "--evaluation-table-output",
        type=Path,
        default=DEFAULT_EVALUATION_TABLE_PATH,
        help="Output Markdown file containing the grouped evaluation table.",
    )
    parser.add_argument(
        "--learning-curves-output",
        type=Path,
        default=DEFAULT_LEARNING_CURVES_PATH,
        help="Output PNG file containing the final learning-curve grid.",
    )
    return parser.parse_args()


def main() -> None:
    """Create final report artifacts from the aggregated experiment CSVs."""
    args = parse_args()
    scenarios = _build_scenarios()
    evaluation_rows = _read_csv(args.evaluation_results)
    training_rows = _read_csv(args.training_curves)

    _write_evaluation_table(evaluation_rows, scenarios, args.evaluation_table_output)
    _write_learning_curves(training_rows, scenarios, args.learning_curves_output)

    print(f"Wrote evaluation table to {args.evaluation_table_output}")
    print(f"Wrote learning curves to {args.learning_curves_output}")


def _build_scenarios() -> tuple[Scenario, ...]:
    return tuple(
        Scenario(
            key=key,
            label=label,
            sensor_mode=str(selector["sensor_mode"]),
            sigma=float(selector["sigma"]),
        )
        for key, label, selector in SCENARIO_LAYOUT
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_evaluation_table(
    rows: Sequence[Mapping[str, str]],
    scenarios: Sequence[Scenario],
    output_path: Path,
) -> None:
    indexed_rows = {
        (
            row["grid"],
            row["agent"],
            row["sensor_mode"],
            _to_float(row["sigma"]),
        ): row
        for row in rows
    }
    lines = [
        "# Final Evaluation Results",
        "",
        "Cells report `mean +/- std` across retained training seeds; `var` shows the seed variance used to compute the standard deviation.",
        "Average total training collisions is computed from per-episode training histories because final greedy collision counts are not emitted in `evaluation_summary.txt`.",
        "",
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="3">Experiment Setting</th>',
    ]
    for _, grid_label in GRID_LAYOUT:
        lines.append(
            f'      <th colspan="{len(EVALUATION_METRICS) * len(AGENT_LABELS)}">{grid_label}</th>'
        )
    lines.extend(["    </tr>", "    <tr>"])

    for _ in GRID_LAYOUT:
        for metric_label, _, _, _ in EVALUATION_METRICS:
            lines.append(f'      <th colspan="{len(AGENT_LABELS)}">{metric_label}</th>')
    lines.extend(["    </tr>", "    <tr>"])

    for _ in GRID_LAYOUT:
        for _ in EVALUATION_METRICS:
            for _, agent_label in AGENT_LABELS.items():
                lines.append(f"      <th>{agent_label}</th>")
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])

    for scenario in scenarios:
        lines.append("    <tr>")
        lines.append(f"      <td><strong>{scenario.label}</strong></td>")
        for grid_key, _ in GRID_LAYOUT:
            for _, mean_column, variance_column, format_spec in EVALUATION_METRICS:
                for agent_key in AGENT_LABELS:
                    row = indexed_rows.get(
                        (grid_key, agent_key, scenario.sensor_mode, scenario.sigma)
                    )
                    lines.append(
                        f"      <td>{_format_metric(row, mean_column, variance_column, format_spec)}</td>"
                    )
        lines.append("    </tr>")

    lines.extend(["  </tbody>", "</table>", ""])
    _write_text(output_path, "\n".join(lines))


def _write_learning_curves(
    rows: Sequence[Mapping[str, str]],
    scenarios: Sequence[Scenario],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        nrows=len(GRID_LAYOUT),
        ncols=len(scenarios),
        figsize=(16, 10),
        sharex=True,
        sharey=False,
    )
    figure.suptitle(
        f"Training Curves: {ROLLING_WINDOW_EPISODES}-Episode Rolling Mean Episodic Return +/- 1 Std Across Seeds",
        fontsize=15,
    )

    for row_index, (grid_key, grid_label) in enumerate(GRID_LAYOUT):
        for col_index, scenario in enumerate(scenarios):
            axis = axes[row_index][col_index]
            _plot_learning_cell(axis, rows, grid_key, scenario)
            if row_index == 0:
                axis.set_title(scenario.label, fontsize=10)
            if col_index == 0:
                axis.set_ylabel(f"{grid_label}\nEpisode return")
            if row_index == len(GRID_LAYOUT) - 1:
                axis.set_xlabel("Training episode")

    algorithm_handles = [
        plt.Line2D([0], [0], color=color, linewidth=2, label=AGENT_LABELS[agent])
        for agent, color in AGENT_COLORS.items()
    ]
    figure.legend(
        handles=algorithm_handles,
        title="Algorithm Colors",
        loc="lower center",
        bbox_to_anchor=(0.32, 0.025),
        ncol=len(algorithm_handles),
        frameon=False,
    )
    uncertainty_handles = [
        Patch(facecolor="gray", alpha=0.15, label="+/- 1 seed std. dev."),
        plt.Line2D(
            [0],
            [0],
            color="none",
            label=(
                f"Y clipped to [{LEARNING_CURVE_Y_LIMITS[0]:.0f}, "
                f"{LEARNING_CURVE_Y_LIMITS[1]:.0f}]"
            ),
        ),
    ]
    figure.legend(
        handles=uncertainty_handles,
        title="Band And Scale",
        loc="lower center",
        bbox_to_anchor=(0.7, 0.025),
        ncol=len(uncertainty_handles),
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_learning_cell(
    axis: Axes,
    rows: Sequence[Mapping[str, str]],
    grid_key: str,
    scenario: Scenario,
) -> None:
    for agent_key, color in AGENT_COLORS.items():
        curve_rows = sorted(
            (
                row
                for row in rows
                if row["grid"] == grid_key
                and row["agent"] == agent_key
                and row["sensor_mode"] == scenario.sensor_mode
                and _to_float(row["sigma"]) == scenario.sigma
                and _to_float(row["episode"]) <= MAX_LEARNING_CURVE_EPISODE
            ),
            key=lambda row: _to_float(row["episode"]),
        )
        if not curve_rows:
            continue

        episodes = [_to_float(row["episode"]) for row in curve_rows]
        means = [_to_float(row["rollout/episode_reward_mean"]) for row in curve_rows]
        stds = [
            _std_from_variance(_to_float(row["rollout/episode_reward_variance"]))
            for row in curve_rows
        ]
        means = _rolling_average(means, ROLLING_WINDOW_EPISODES)
        stds = _rolling_average(stds, ROLLING_WINDOW_EPISODES)
        lower = [mean - std for mean, std in zip(means, stds, strict=True)]
        upper = [mean + std for mean, std in zip(means, stds, strict=True)]

        axis.plot(episodes, means, color=color, linewidth=1.5, label=AGENT_LABELS[agent_key])
        axis.fill_between(episodes, lower, upper, color=color, alpha=0.15, linewidth=0)

    axis.grid(alpha=0.25)
    axis.set_xlim(left=0, right=MAX_LEARNING_CURVE_EPISODE)
    axis.set_ylim(*LEARNING_CURVE_Y_LIMITS)


def _rolling_average(values: Sequence[float], window_size: int) -> list[float]:
    averaged_values = []
    for index in range(len(values)):
        window = values[max(0, index - window_size + 1) : index + 1]
        finite_values = [value for value in window if math.isfinite(value)]
        averaged_values.append(sum(finite_values) / len(finite_values) if finite_values else math.nan)

    return averaged_values


def _format_metric(
    row: Mapping[str, str] | None,
    mean_column: str | None,
    variance_column: str | None,
    format_spec: str,
) -> str:
    if row is None or mean_column is None or variance_column is None:
        return "N/A"

    mean = _to_float(row.get(mean_column, ""))
    variance = _to_float(row.get(variance_column, ""))
    if not math.isfinite(mean) or not math.isfinite(variance):
        return "N/A"

    std = _std_from_variance(variance)
    return f"{format_spec.format(mean)} +/- {format_spec.format(std)}<br><small>var={format_spec.format(variance)}</small>"


def _std_from_variance(variance: float) -> float:
    if not math.isfinite(variance):
        return 0.0

    return math.sqrt(max(variance, 0.0))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
