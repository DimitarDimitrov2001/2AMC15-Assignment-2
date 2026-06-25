"""Create final report tables and learning-curve figures from merged CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from utils.rl_plots import _draw_grid_background
from world.grid_codes import EMPTY_CELL


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_TRAINING_CURVES_PATH = DEFAULT_RESULTS_DIR / "training_curves.csv"
DEFAULT_EVALUATION_RESULTS_PATH = DEFAULT_RESULTS_DIR / "evaluation_results.csv"
DEFAULT_EVALUATION_TABLE_PATH = DEFAULT_RESULTS_DIR / "final_evaluation_table.md"
DEFAULT_LEARNING_CURVES_PATH = DEFAULT_RESULTS_DIR / "final_learning_curves.png"
DEFAULT_POLICY_ROLLOUTS_PATH = DEFAULT_RESULTS_DIR / "final_policy_rollouts_ddqn_stoch_seed0.png"
DEFAULT_ROLLOUT_EXPERIMENT = "experiment_3"
DEFAULT_ROLLOUT_AGENT = "ddqn"
DEFAULT_ROLLOUT_SEED = 0
ROLLOUT_PATH_COLOR = "#E69F00"
ROLLOUT_PATH_EDGE_COLOR = "#FFFFFF"
ROLLOUT_START_COLOR = "#009E73"
ROLLOUT_END_COLOR = "#D55E00"
ROLLOUT_CONSENSUS_COLOR = "#0072B2"
ROLLOUT_NOISY_PATH_ALPHA = 0.22
ROLLOUT_NOISY_LINE_WIDTH = 1.1
ROLLOUT_CONSENSUS_MAX_ALPHA = 0.45
ROLLOUT_MULTI_PATH_ALPHA = 0.95
ROLLOUT_LINE_WIDTH = 2.6
ROLLOUT_MARKER_SIZE = 2.6
ROLLOUT_PANEL_FIGSIZE = (12.0, 4.6)
ROLLOUT_PANEL_WSPACE = 0.0
ROLLOUT_PANEL_COL_GAP = 0.004
ROLLOUT_HEADER_TITLE_FONTSIZE = 10.0
ROLLOUT_HEADER_SUBTITLE_FONTSIZE = 10.0
ROLLOUT_HEADER_TITLE_OFFSET = 0.074
ROLLOUT_HEADER_LINE1_OFFSET = 0.043
AGENT_LABELS = {"dqn": "DQN", "ddqn": "Dueling DQN"}
AGENT_COLORS = {"dqn": "#0072B2", "ddqn": "#E69F00"}
GRID_LAYOUT = (
    ("simple_cave_grid", "Simple Grid", "Simple"),
    ("big_spaces_cave", "Medium Difficulty Grid", "Medium"),
    ("realistic_super_hard_cave", "Difficult Grid", "Difficult"),
)
SCENARIO_LAYOUT = (
    ("baseline", "Baseline", {"sensor_mode": "sensor", "sigma": 0.0}),
    ("no_sensors", "No LiDAR", {"sensor_mode": "no_sensor", "sigma": 0.0}),
    (
        "stochastic",
        r"Stochastic ($\sigma=0.5$)",
        {"sensor_mode": "sensor", "sigma": 0.5},
    ),
)
LEARNING_CURVE_LINE_WIDTH = 2.0
LEARNING_CURVE_STD_ALPHA = 0.18
EVALUATION_METRICS = (
    (
        "Eval return",
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
        "Train collisions",
        "training_total_collisions_mean",
        "training_total_collisions_variance",
        "{:.1f}",
    ),
    (
        "Eval length",
        "final_eval_mean_steps_mean",
        "final_eval_mean_steps_variance",
        "{:.1f}",
    ),
)
ROLLING_WINDOW_EPISODES = 25
MAX_LEARNING_CURVE_EPISODE = 6000
LEARNING_CURVE_Y_LIMITS = (-2.0, 1.0)


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
    parser.add_argument(
        "--policy-rollouts-output",
        type=Path,
        default=DEFAULT_POLICY_ROLLOUTS_PATH,
        help="Output PNG file containing the combined greedy-policy rollout figure.",
    )
    parser.add_argument(
        "--rollout-experiment",
        default=DEFAULT_ROLLOUT_EXPERIMENT,
        help="Experiment folder under results/ that holds per-run rollout JSON files.",
    )
    parser.add_argument(
        "--rollout-agent",
        default=DEFAULT_ROLLOUT_AGENT,
        help="Agent subdirectory prefix used when locating rollout JSON files.",
    )
    parser.add_argument(
        "--rollout-seed",
        type=int,
        default=DEFAULT_ROLLOUT_SEED,
        help="Training seed used when locating rollout JSON files.",
    )
    parser.add_argument(
        "--skip-policy-rollouts",
        action="store_true",
        help="Skip generating the combined policy-rollout showcase figure.",
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

    if not args.skip_policy_rollouts:
        _write_policy_rollout_showcase(
            results_dir=DEFAULT_RESULTS_DIR,
            experiment=args.rollout_experiment,
            agent=args.rollout_agent,
            seed=args.rollout_seed,
            output_path=args.policy_rollouts_output,
        )
        print(f"Wrote policy rollouts to {args.policy_rollouts_output}")


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
    for _, grid_label, _ in GRID_LAYOUT:
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
        for grid_key, _, _ in GRID_LAYOUT:
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
        figsize=(10.5, 7.5),
        sharex=True,
        sharey=True,
    )
    if len(GRID_LAYOUT) == 1:
        axes = axes.reshape(1, -1)
    if len(scenarios) == 1:
        axes = axes.reshape(-1, 1)

    panel_index = 0
    for row_index, (grid_key, _, grid_plot_label) in enumerate(GRID_LAYOUT):
        for col_index, scenario in enumerate(scenarios):
            axis = axes[row_index, col_index]
            _plot_learning_cell(axis, rows, grid_key, scenario)
            axis.text(
                0.03,
                0.97,
                f"({chr(ord('a') + panel_index)})",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                fontweight="bold",
            )
            panel_index += 1
            if row_index == 0:
                axis.set_title(scenario.label, fontsize=10, pad=6)
            if col_index == 0:
                axis.set_ylabel(f"{grid_plot_label}\nEpisode return", fontsize=9)
            if row_index == len(GRID_LAYOUT) - 1:
                axis.set_xlabel("Training episode", fontsize=9)

    algorithm_handles = [
        plt.Line2D([0], [0], color=color, linewidth=LEARNING_CURVE_LINE_WIDTH, label=AGENT_LABELS[agent])
        for agent, color in AGENT_COLORS.items()
    ]
    uncertainty_handles = [
        Patch(facecolor="gray", alpha=LEARNING_CURVE_STD_ALPHA, label=r"$\pm 1$ seed std. dev."),
    ]
    figure.legend(
        handles=[*algorithm_handles, *uncertainty_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
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

        axis.plot(
            episodes,
            means,
            color=color,
            linewidth=LEARNING_CURVE_LINE_WIDTH,
            label=AGENT_LABELS[agent_key],
        )
        axis.fill_between(
            episodes,
            lower,
            upper,
            color=color,
            alpha=LEARNING_CURVE_STD_ALPHA,
            linewidth=0,
        )

    axis.grid(alpha=0.25, linewidth=0.5)
    axis.tick_params(labelsize=8)
    axis.set_xlim(left=0, right=MAX_LEARNING_CURVE_EPISODE)
    axis.set_ylim(*LEARNING_CURVE_Y_LIMITS)


def _write_policy_rollout_showcase(
    *,
    results_dir: Path,
    experiment: str,
    agent: str,
    seed: int,
    output_path: Path,
) -> None:
    """Write a three-panel figure of greedy policy rollouts across grid difficulties."""
    panel_specs: list[tuple[str, str, list[dict[str, Any]]]] = []
    for grid_key, _, grid_plot_label in GRID_LAYOUT:
        rollout_path = _resolve_rollout_json_path(
            results_dir=results_dir,
            experiment=experiment,
            grid_key=grid_key,
            agent=agent,
            seed=seed,
        )
        rollouts = _load_rollout_json(rollout_path)
        panel_specs.append((grid_plot_label, chr(ord("a") + len(panel_specs)), rollouts))

    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(panel_specs),
        figsize=ROLLOUT_PANEL_FIGSIZE,
        gridspec_kw={"wspace": ROLLOUT_PANEL_WSPACE},
    )
    if len(panel_specs) == 1:
        axes = np.asarray([axes])

    legend_handles: list[Any] = []
    panel_subtitles: list[str] = []
    for panel_index, (_, _, rollouts) in enumerate(panel_specs):
        axis = axes[panel_index]
        handles, subtitle = _plot_rollout_panel(
            axis=axis,
            rollouts=rollouts,
        )
        panel_subtitles.append(subtitle)
        if panel_index == 0:
            legend_handles = handles

    figure.subplots_adjust(left=0.008, right=0.998, top=0.82, bottom=0.10)
    n_panels = len(axes)
    left_margin = 0.008
    right_margin = 0.998
    bottom_margin = 0.10
    top_margin = 0.82
    plot_height = top_margin - bottom_margin
    total_width = right_margin - left_margin
    panel_width = (total_width - ROLLOUT_PANEL_COL_GAP * (n_panels - 1)) / n_panels
    for panel_index, axis in enumerate(axes):
        axis.set_position(
            [
                left_margin + panel_index * (panel_width + ROLLOUT_PANEL_COL_GAP),
                bottom_margin,
                panel_width,
                plot_height,
            ]
        )

    for panel_index, (grid_plot_label, panel_label, _) in enumerate(panel_specs):
        axis = axes[panel_index]
        position = axis.get_position()
        subtitle = panel_subtitles[panel_index]
        figure.text(
            position.x0 + position.width / 2,
            position.y1 + ROLLOUT_HEADER_TITLE_OFFSET,
            f"({panel_label}) {grid_plot_label}",
            ha="center",
            va="bottom",
            fontsize=ROLLOUT_HEADER_TITLE_FONTSIZE,
            fontweight="bold",
        )
        figure.text(
            position.x0 + position.width / 2,
            position.y1 + ROLLOUT_HEADER_LINE1_OFFSET,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=ROLLOUT_HEADER_SUBTITLE_FONTSIZE,
        )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=7.5,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, pad_inches=0.03)
    plt.close(figure)


def _resolve_rollout_json_path(
    *,
    results_dir: Path,
    experiment: str,
    grid_key: str,
    agent: str,
    seed: int,
) -> Path:
    experiment_dir = results_dir / experiment
    candidates = sorted(
        experiment_dir.glob(f"{grid_key}_{agent}_*seed{seed}/policy_rollout.json")
    )
    if not candidates:
        candidates = sorted(
            experiment_dir.glob(f"{grid_key}_{agent}_seed{seed}/policy_rollout.json")
        )
    if not candidates:
        raise FileNotFoundError(
            f"No policy_rollout.json found for grid={grid_key}, agent={agent}, seed={seed} "
            f"under {experiment_dir}"
        )
    return candidates[0]


def _load_rollout_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return [payload]


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16) / 255.0
    green = int(hex_color[2:4], 16) / 255.0
    blue = int(hex_color[4:6], 16) / 255.0
    return (red, green, blue, alpha)


def _rollout_step_count(rollout: Mapping[str, Any]) -> int:
    if "steps" in rollout:
        return int(rollout["steps"])
    positions = rollout.get("positions", [])
    return max(len(positions) - 1, 0)


def _select_representative_rollout(rollouts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    step_counts = [_rollout_step_count(item) for item in rollouts]
    median_steps = sorted(step_counts)[len(step_counts) // 2]
    return min(
        rollouts,
        key=lambda item: abs(_rollout_step_count(item) - median_steps),
    )


def _compute_visit_density(
    grid: np.ndarray,
    rollouts: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return per-cell visit counts across rollouts, shape ``(n_rows, n_cols)``."""
    n_cols, n_rows = grid.shape
    density = np.zeros((n_rows, n_cols), dtype=float)
    for rollout in rollouts:
        for position in rollout["positions"]:
            col = int(round(float(position[0])))
            row = int(round(float(position[1])))
            if 0 <= col < n_cols and 0 <= row < n_rows:
                density[row, col] += 1.0
    return density


def _draw_visit_consensus_overlay(
    axis: Axes,
    grid: np.ndarray,
    rollouts: Sequence[Mapping[str, Any]],
) -> None:
    """Highlight cells repeatedly visited across noisy evaluation rollouts."""
    density = _compute_visit_density(grid, rollouts)
    max_count = float(density.max())
    if max_count <= 1.0:
        return

    n_cols, n_rows = grid.shape
    overlay = np.zeros((n_rows, n_cols, 4), dtype=float)
    base_rgb = _hex_to_rgba(ROLLOUT_CONSENSUS_COLOR)[:3]
    for row in range(n_rows):
        for col in range(n_cols):
            if grid[col, row] != EMPTY_CELL:
                continue
            visit_count = density[row, col]
            if visit_count <= 0.0:
                continue
            fraction = visit_count / max_count
            alpha = 0.06 + ROLLOUT_CONSENSUS_MAX_ALPHA * fraction
            overlay[row, col] = (*base_rgb, alpha)

    axis.imshow(
        overlay,
        aspect="equal",
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )


def _plot_rollout_path(
    *,
    axis: Axes,
    positions: np.ndarray,
    path_color: str,
    linewidth: float,
    alpha: float,
    markers: bool,
    zorder: int,
) -> None:
    if len(positions) == 0:
        return

    if not markers:
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            color=path_color,
            alpha=alpha,
            linewidth=linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )
        return

    axis.plot(
        positions[:, 0],
        positions[:, 1],
        color=ROLLOUT_PATH_EDGE_COLOR,
        alpha=min(alpha + 0.2, 1.0),
        linewidth=linewidth + 1.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder - 1,
    )
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        color=path_color,
        alpha=alpha,
        linewidth=linewidth,
        marker="o",
        markersize=ROLLOUT_MARKER_SIZE,
        markerfacecolor=path_color,
        markeredgecolor=ROLLOUT_PATH_EDGE_COLOR,
        markeredgewidth=0.35,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )


def _configure_rollout_panel_axes(axis: Axes, n_cols: int, n_rows: int) -> None:
    """Configure rollout axes with tick marks only (no axis titles or numeric labels)."""
    axis.set_xlim(-0.5, n_cols - 0.5)
    axis.set_ylim(n_rows - 0.5, -0.5)
    axis.set_xticks(range(n_cols))
    axis.set_yticks(range(n_rows))
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="both", which="both", length=2, width=0.4, labelbottom=False, labelleft=False)


def _plot_rollout_panel(
    *,
    axis: Axes,
    rollouts: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], str]:
    """Draw one rollout panel and return legend handles."""
    grid = np.asarray(rollouts[0]["grid"])
    n_cols, n_rows = grid.shape
    _draw_grid_background(axis, grid)

    multi = len(rollouts) > 1
    if multi:
        _draw_visit_consensus_overlay(axis, grid, rollouts)
        representative = _select_representative_rollout(rollouts)
        for rollout in rollouts:
            positions = np.asarray(rollout["positions"], dtype=float) - 0.5
            _plot_rollout_path(
                axis=axis,
                positions=positions,
                path_color=ROLLOUT_PATH_COLOR,
                linewidth=ROLLOUT_NOISY_LINE_WIDTH,
                alpha=ROLLOUT_NOISY_PATH_ALPHA,
                markers=False,
                zorder=2,
            )

        representative_positions = np.asarray(representative["positions"], dtype=float) - 0.5
        _plot_rollout_path(
            axis=axis,
            positions=representative_positions,
            path_color=ROLLOUT_PATH_COLOR,
            linewidth=ROLLOUT_LINE_WIDTH,
            alpha=ROLLOUT_MULTI_PATH_ALPHA,
            markers=True,
            zorder=4,
        )
        start_col, start_row = representative_positions[0]
        end_points = np.asarray(
            [np.asarray(item["positions"], dtype=float)[-1] - 0.5 for item in rollouts],
            dtype=float,
        )
    else:
        representative = rollouts[0]
        representative_positions = np.asarray(representative["positions"], dtype=float) - 0.5
        _plot_rollout_path(
            axis=axis,
            positions=representative_positions,
            path_color=ROLLOUT_PATH_COLOR,
            linewidth=ROLLOUT_LINE_WIDTH,
            alpha=ROLLOUT_MULTI_PATH_ALPHA,
            markers=True,
            zorder=4,
        )
        start_col, start_row = representative_positions[0]
        end_points = representative_positions[-1:]

    start_circle = plt.Circle(
        (start_col, start_row),
        0.34,
        facecolor=ROLLOUT_START_COLOR,
        edgecolor="black",
        linewidth=1.2,
        zorder=6,
    )
    axis.add_patch(start_circle)
    axis.text(
        start_col,
        start_row,
        "S",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="white",
        zorder=7,
    )
    if multi:
        axis.scatter(
            end_points[:, 0],
            end_points[:, 1],
            color=ROLLOUT_END_COLOR,
            s=36,
            marker="*",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.75,
            zorder=6,
        )
        axis.scatter(
            end_points[:, 0].mean(),
            end_points[:, 1].mean(),
            color=ROLLOUT_END_COLOR,
            s=90,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )
    else:
        axis.scatter(
            end_points[0, 0],
            end_points[0, 1],
            color=ROLLOUT_END_COLOR,
            s=72,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )

    if multi:
        rewards = [float(item.get("total_reward", 0.0)) for item in rollouts]
        step_counts = [_rollout_step_count(item) for item in rollouts]
        subtitle = f"return={np.mean(rewards):.3f} · steps={min(step_counts)}–{max(step_counts)}"
    else:
        rollout = rollouts[0]
        subtitle = f"return={float(rollout.get('total_reward', 0.0)):.3f} · steps={_rollout_step_count(rollout)}"

    _configure_rollout_panel_axes(axis, n_cols, n_rows)
    axis.set_aspect("equal", adjustable="box")

    if multi:
        return (
            [
                plt.Line2D(
                    [0],
                    [0],
                    color=ROLLOUT_START_COLOR,
                    marker="o",
                    linestyle="None",
                    markersize=8,
                    label="Start",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color=ROLLOUT_PATH_COLOR,
                    linewidth=ROLLOUT_NOISY_LINE_WIDTH,
                    alpha=0.65,
                    label=rf"{len(rollouts)} noisy paths ($\sigma=0.5$)",
                ),
                Patch(
                    facecolor=_hex_to_rgba(ROLLOUT_CONSENSUS_COLOR, 0.35),
                    edgecolor="none",
                    label="Shared route",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color=ROLLOUT_PATH_COLOR,
                    linewidth=ROLLOUT_LINE_WIDTH,
                    marker="o",
                    markersize=4,
                    label="Median path",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color=ROLLOUT_END_COLOR,
                    marker="*",
                    linestyle="None",
                    markersize=11,
                    label="Goal reached",
                ),
            ],
            subtitle,
        )

    return (
        [
            plt.Line2D([0], [0], color=ROLLOUT_START_COLOR, marker="o", linestyle="None", markersize=8, label="Start"),
            plt.Line2D(
                [0],
                [0],
                color=ROLLOUT_PATH_COLOR,
                linewidth=ROLLOUT_LINE_WIDTH,
                marker="o",
                markersize=4,
                label="Greedy path",
            ),
            plt.Line2D(
                [0],
                [0],
                color=ROLLOUT_END_COLOR,
                marker="*",
                linestyle="None",
                markersize=11,
                label="Goal reached",
            ),
        ],
        subtitle,
    )


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
