"""Report-oriented plots for assignment experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.runner import RunResult
from utils.rl_plots import (
    plot_algorithm_comparison,
    plot_policy_disagreement,
    plot_value_and_policy,
)


# ---------------------------------------------------------------------------
# Plot styling and output folders
#
# These constants keep labels, colors, and subdirectory names consistent
# across all figures produced for the report suite.
# ---------------------------------------------------------------------------


ALGO_LABELS = {
    "value_iteration": "Value Iteration",
    "mc": "On-policy MC",
    "q_learning": "Q-learning",
}

ALGO_COLORS = {
    "value_iteration": "#1B6CA8",
    "mc": "#2F8F2F",
    "q_learning": "#C26A1B",
}

PLOT_DIRS = {
    "learning_curves": "learning_curves",
    "combined_learning_curves": "combined_learning_curves",
    "combined_policy_diff_curves": "combined_policy_diff_curves",
    "vi_convergence": "vi_convergence",
    "value_policy": "value_policy",
    "policy_disagreement": "policy_disagreement",
}

LEARNING_CURVE_SMOOTHING_WINDOW = 500
LEARNING_CURVE_METRICS = ["undiscounted_return", "delta_q", "policy_diff"]
LEARNING_ALGORITHMS = {"mc", "q_learning"}
LINE_STYLES = ["-", "--", ":", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

METRIC_TITLES = {
    "undiscounted_return": "Undiscounted Return",
    "delta_q": "Q-value Change",
    "policy_diff": "Policy Difference (%)",
}

GROUP_TITLES = {
    "default": "Default Configuration",
    "grid_comparison": "Grid Comparison",
    "discount_factor": "Discount Factor",
    "stochasticity": "Stochasticity",
    "exploration_epsilon": "Exploration (Epsilon)",
    "learning_rate": "Learning Rate",
    "mc_episode_length": "MC Episode Length",
}

GRID_TITLES = {
    "A1_grid": "A1 grid",
    "super_hard": "Super-hard grid",
}

CONDITION_TITLES = {
    "default": "default configuration",
    "low_fixed_epsilon": "low fixed epsilon",
    "high_fixed_epsilon": "high fixed epsilon",
    "decaying_epsilon": "decaying epsilon",
    "low_fixed_alpha": "low fixed alpha",
    "high_fixed_alpha": "high fixed alpha",
    "decaying_alpha": "decaying alpha",
    "visit_count": "visit-count schedule",
}


# ---------------------------------------------------------------------------
# Human-readable labels and filenames
# ---------------------------------------------------------------------------


def _pretty_grid(grid: str) -> str:
    """Human-readable grid name."""
    if grid in GRID_TITLES:
        return GRID_TITLES[grid]
    return grid.replace("_", " ").strip().capitalize()


def _pretty_condition(group: str, condition: str) -> str:
    """Human-readable condition label for a (group, condition) pair."""
    if condition in CONDITION_TITLES:
        return CONDITION_TITLES[condition]
    if "=" in condition:
        key, _, value = condition.partition("=")
        return f"{key.strip()} = {value.strip()}"
    if group == "grid_comparison":
        return _pretty_grid(condition)
    return condition.replace("_", " ")


# Maps (group, condition) to a short hyperparameter label drawn near the curve
# or as the legend title on per-scene figures. Values mirror the hard-coded
# overrides in experiments.specs.build_cases; keeping the mapping here avoids
# a runtime dependency on RunResult.row inside the figure builders.
_GREEK_LETTERS = {
    "alpha": "\u03b1",
    "gamma": "\u03b3",
    "sigma": "\u03c3",
    "epsilon": "\u03b5",
}

_HYPERPARAM_LABELS: dict[tuple[str, str], str] = {
    ("exploration_epsilon", "low_fixed_epsilon"): "\u03b5=0.1 (fixed)",
    ("exploration_epsilon", "high_fixed_epsilon"): "\u03b5=0.5 (fixed)",
    ("exploration_epsilon", "decaying_epsilon"): "\u03b5: 1.0\u21920.01 decay",
    ("learning_rate", "low_fixed_alpha"): "\u03b1=0.1 (const)",
    ("learning_rate", "high_fixed_alpha"): "\u03b1=0.5 (const)",
    ("learning_rate", "decaying_alpha"): "\u03b1: 0.5\u21920.01 exp",
    ("learning_rate", "visit_count"): "\u03b1 visit-count (c=10)",
}


def _hyperparam_annotation(group: str, condition: str) -> str:
    """Short label describing the hyperparameter setting for ``condition``.

    Returns an empty string for the ``default`` group (nothing to annotate).
    """
    if group == "default":
        return ""
    key = (group, condition)
    if key in _HYPERPARAM_LABELS:
        return _HYPERPARAM_LABELS[key]
    if group == "grid_comparison":
        return _pretty_grid(condition)
    if group == "mc_episode_length" and condition.startswith("max_episode_length="):
        return f"T_max={condition.split('=', 1)[1]}"
    if "=" in condition:
        key_name, _, value = condition.partition("=")
        symbol = _GREEK_LETTERS.get(key_name.strip(), key_name.strip())
        return f"{symbol}={value.strip()}"
    return condition.replace("_", " ")


def _pretty_scene_title(
    group: str,
    grid: str | None = None,
    *,
    condition: str | None = None,
    seed: int | None = None,
    all_seeds: bool = False,
    descriptor: str | None = None,
) -> str:
    """Build a human-readable figure suptitle.

    ``descriptor`` is an optional middle clause such as
    ``"algorithm and condition comparison"``. ``condition`` is appended when set
    (used by per-scene plots that pin a specific condition). ``grid`` may be
    omitted for figures that span multiple grids (e.g. grid comparison plots).
    """
    head = GROUP_TITLES.get(group, group.replace("_", " ").title())
    parts: list[str] = [head]
    middle_segments: list[str] = []
    if condition is not None:
        middle_segments.append(_pretty_condition(group, condition))
    if descriptor:
        middle_segments.append(descriptor)
    if middle_segments:
        parts.append(": " + ", ".join(middle_segments))
    paren_segments: list[str] = []
    if grid is not None:
        paren_segments.append(_pretty_grid(grid))
    if all_seeds:
        paren_segments.append("all seeds")
    elif seed is not None:
        paren_segments.append(f"seed={seed}")
    if paren_segments:
        parts.append(f" ({', '.join(paren_segments)})")
    return "".join(parts)


def _slug(text: str) -> str:
    """Make a condition/group label safe for filenames."""
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("=", "-")
        .replace(",", "-")
    )


def _plot_dir(out_dir: Path, group: str, plot_type: str) -> Path:
    """Return and create the output directory for one plot family."""
    path = out_dir / group / PLOT_DIRS[plot_type]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _first_seed_results(results: list[RunResult]) -> list[RunResult]:
    """Select one representative seed for spatial/value plots."""
    if not results:
        return []
    first_seed = min(int(r.row["seed"]) for r in results)
    return [r for r in results if int(r.row["seed"]) == first_seed]


def _history_metrics(history: dict[str, Any], desired: list[str]) -> list[str]:
    """Return desired metric names that are available in a history."""
    available = history.get("metrics", {})
    return [metric for metric in desired if metric in available]


def _smooth(values: list[float], window: int) -> np.ndarray:
    """Smooth a metric trace with a simple moving average."""
    if window <= 1:
        return np.array(values, dtype=float)
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _history_metric_values(history: dict[str, Any], metric: str) -> tuple[list[int], list[float]] | None:
    """Extract aligned episode and metric arrays from a TrainingHistory dict."""
    metrics = history.get("metrics", {})
    if metric not in metrics:
        return None
    values = metrics[metric]
    episodes = history.get("episodes", list(range(1, len(values) + 1)))
    n = min(len(episodes), len(values))
    if n == 0:
        return None
    return list(episodes[:n]), list(values[:n])


def _condition_line_styles(conditions: list[str]) -> dict[str, Any]:
    """Assign stable line styles to conditions within one figure."""
    styles: dict[str, Any] = {}
    ordered = ["default"] if "default" in conditions else []
    ordered.extend(condition for condition in conditions if condition not in ordered)
    for idx, condition in enumerate(ordered):
        styles[condition] = LINE_STYLES[idx % len(LINE_STYLES)]
    return styles


def _smoothed_metric_matrix(
    histories: list[dict[str, Any]],
    metric: str,
) -> tuple[list[int], np.ndarray] | None:
    """Build a seed-by-episode matrix for one smoothed metric."""
    series = [
        metric_values
        for history in histories
        if (metric_values := _history_metric_values(history, metric)) is not None
    ]
    if not series:
        return None
    min_len = min(len(values) for _episodes, values in series)
    if min_len == 0:
        return None
    episodes = series[0][0][:min_len]
    smoothing = min(LEARNING_CURVE_SMOOTHING_WINDOW, min_len)
    matrix = np.array(
        [_smooth(values[:min_len], smoothing) for _episodes, values in series],
        dtype=float,
    )
    return episodes, matrix


def _save_combined_curve_figure(
    curves: list[dict[str, Any]],
    *,
    group: str,
    title: str,
    path: Path,
    show_std: bool,
    metrics_filter: list[str] | None = None,
) -> None:
    """Render the combined learning-curve figure in a 1xN horizontal layout.

    ``metrics_filter`` restricts which metrics get plotted (e.g. a single-metric
    policy_diff variant). ``group`` is needed so per-curve hyperparameter
    annotations can be derived from the (group, condition) pair.
    """
    desired_metrics = metrics_filter or LEARNING_CURVE_METRICS
    available_metrics = [
        metric
        for metric in desired_metrics
        if any(
            metric in (history.get("metrics") or {})
            for curve in curves
            for history in curve["histories"]
        )
    ]
    if not available_metrics:
        return

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    # Combined figures show one subplot per metric. Policy-diff-only figures
    # use a narrower layout and stack legends vertically.
    n_metrics = len(available_metrics)
    if n_metrics == 1:
        figsize = (5.8, 3.8)
        right_margin = 0.82
    else:
        figsize = (4.5 * n_metrics, 3.6)
        right_margin = 0.96
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, constrained_layout=False)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = list(axes)

    algos_in_figure: list[str] = []
    # Track conditions in their first-seen order, with their assigned linestyle,
    # so the bottom legend can show one entry per condition mapped to its
    # linestyle and hyperparameter label.
    condition_styles: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Draw mean curves and optional seed variance
    # ------------------------------------------------------------------
    for ax, metric in zip(axes, available_metrics):
        for curve in curves:
            matrix_data = _smoothed_metric_matrix(curve["histories"], metric)
            if matrix_data is None:
                continue
            episodes, matrix = matrix_data
            mu = matrix.mean(axis=0)
            algorithm = curve["algorithm"]
            color = ALGO_COLORS[algorithm]
            linestyle = curve["linestyle"]
            ax.plot(
                episodes,
                mu,
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
            )
            if show_std and len(matrix) > 1:
                sigma = matrix.std(axis=0)
                ax.fill_between(
                    episodes,
                    mu - sigma,
                    mu + sigma,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
            if algorithm not in algos_in_figure:
                algos_in_figure.append(algorithm)
            condition = curve["condition"]
            if condition not in condition_styles:
                condition_styles[condition] = linestyle

        ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=10)
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=11)

    # ------------------------------------------------------------------
    # Build legends
    # ------------------------------------------------------------------
    # Colors encode algorithms. Linestyles encode conditions/hyperparameters.
    algos_present = [a for a in sorted(LEARNING_ALGORITHMS) if a in algos_in_figure]
    algo_handles = [
        Line2D([0], [0], color=ALGO_COLORS[a], lw=1.8, label=ALGO_LABELS[a])
        for a in algos_present
    ]

    # Build condition handles using each condition's linestyle in a neutral
    # color so the line icon (style) is what carries meaning, not the color.
    # Labels are the hyperparameter strings (e.g. "alpha=0.5 (const)") with a
    # readable fallback for any condition lacking a hyperparam mapping.
    condition_handles: list[Line2D] = []
    for condition, linestyle in condition_styles.items():
        label = _hyperparam_annotation(group, condition)
        if not label:
            label = "baseline" if condition == "default" else condition.replace("_", " ")
        condition_handles.append(
            Line2D(
                [0],
                [0],
                color="#444444",
                lw=1.6,
                linestyle=linestyle,
                label=label,
            )
        )

    # Narrow single-metric figures stack the two legends vertically (algo on
    # top, conditions below); wider multi-metric figures place them side by
    # side. Layout is built bottom-up in figure coordinates: condition legend
    # sits at the very bottom, then optionally the algo legend, then the
    # shared "Episode" supxlabel, then the axes themselves.
    stack_legends = n_metrics == 1
    condition_cols = min(3, max(1, len(condition_handles)))
    condition_rows = (
        max(1, (len(condition_handles) + condition_cols - 1) // condition_cols)
        if condition_handles
        else 0
    )
    row_height = 0.05
    gap = 0.015

    y_bottom_anchor = 0.02
    cond_top = y_bottom_anchor + row_height * condition_rows
    if stack_legends and algo_handles and condition_handles:
        algo_bottom = cond_top + gap
        algo_top = algo_bottom + row_height
        legend_top = algo_top
    elif algo_handles and condition_handles:
        # Side by side: both align at y_bottom_anchor; the tallest determines
        # the overall legend block height.
        legend_top = y_bottom_anchor + row_height * max(1, condition_rows)
        algo_bottom = y_bottom_anchor
    elif algo_handles:
        legend_top = y_bottom_anchor + row_height
        algo_bottom = y_bottom_anchor
    else:
        legend_top = y_bottom_anchor
        algo_bottom = y_bottom_anchor

    bottom_margin = legend_top + 0.08

    if algo_handles and condition_handles:
        if stack_legends:
            algo_anchor = (0.5, algo_bottom)
            algo_loc = "lower center"
            cond_anchor = (0.5, y_bottom_anchor)
            cond_loc = "lower center"
        else:
            algo_anchor = (0.48, y_bottom_anchor)
            algo_loc = "lower right"
            cond_anchor = (0.52, y_bottom_anchor)
            cond_loc = "lower left"
        # add_artist preserves the first legend when adding the second.
        algo_legend = fig.legend(
            handles=algo_handles,
            loc=algo_loc,
            bbox_to_anchor=algo_anchor,
            ncol=len(algo_handles),
            frameon=False,
            fontsize=9,
        )
        fig.add_artist(algo_legend)
        fig.legend(
            handles=condition_handles,
            loc=cond_loc,
            bbox_to_anchor=cond_anchor,
            ncol=condition_cols,
            frameon=False,
            fontsize=9,
        )
    elif algo_handles:
        fig.legend(
            handles=algo_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, y_bottom_anchor),
            ncol=len(algo_handles),
            frameon=False,
            fontsize=9,
        )

    wspace = 0.28 if n_metrics == 1 else 0.40
    fig.subplots_adjust(
        left=0.08, right=right_margin, top=0.84, bottom=bottom_margin, wspace=wspace
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _save_combined_with_policy_diff(
    curves: list[dict[str, Any]],
    *,
    group: str,
    title: str,
    out_dir: Path,
    filename: str,
    show_std: bool,
) -> None:
    """Save both the 3-metric combined figure and the policy_diff-only variant.

    The standalone variant is written under the ``combined_policy_diff_curves``
    sibling directory with the same filename minus the ``_combined_learning_curves``
    suffix.
    """
    # Main combined figure: returns, Q-value movement, and policy disagreement
    # when those metrics are present in the histories.
    combined_path = _plot_dir(out_dir, group, "combined_learning_curves") / filename
    _save_combined_curve_figure(
        curves,
        group=group,
        title=title,
        path=combined_path,
        show_std=show_std,
    )

    # Companion figure: policy disagreement only. This is easier to inspect in
    # the report when the combined 3-panel figure is too dense.
    policy_diff_filename = filename.replace(
        "_combined_learning_curves.png", "_policy_diff.png"
    )
    if policy_diff_filename == filename:
        policy_diff_filename = filename.replace(".png", "_policy_diff.png")
    policy_diff_path = (
        _plot_dir(out_dir, group, "combined_policy_diff_curves") / policy_diff_filename
    )
    _save_combined_curve_figure(
        curves,
        group=group,
        title=title,
        path=policy_diff_path,
        show_std=show_std,
        metrics_filter=["policy_diff"],
    )


def _save_learning_curves(results: list[RunResult], out_dir: Path) -> None:
    """Save per-condition learning curves for MC and Q-learning."""
    # ------------------------------------------------------------------
    # Group histories by scene and algorithm
    # ------------------------------------------------------------------
    # A scene is one setup group + condition + grid. Each algorithm can have
    # multiple histories when the suite was run with multiple seeds.
    curve_results = [
        r for r in results
        if r.row["algorithm"] in LEARNING_ALGORITHMS and r.history is not None
    ]
    # group by (setup_group, condition, grid, algorithm) -> list of histories (one per seed)
    seed_histories: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in curve_results:
        key = (
            result.row["setup_group"],
            result.row["condition"],
            result.row["grid"],
            result.row["algorithm"],
        )
        seed_histories[key].append(result.history)

    # epsilon/alpha are per-algo hyperparameter schedules, not performance metrics
    desired = LEARNING_CURVE_METRICS

    # collect all (group, condition, grid) combos to iterate over
    scene_keys: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for (group, condition, grid, algo) in seed_histories:
        if algo not in scene_keys[(group, condition, grid)]:
            scene_keys[(group, condition, grid)].append(algo)

    for (group, condition, grid), algos in scene_keys.items():
        # ------------------------------------------------------------------
        # Determine available metrics and figure shape
        # ------------------------------------------------------------------
        # find which metrics are available across all seeds for any algo
        available_metrics: list[str] = []
        for m in desired:
            for algo in algos:
                histories = seed_histories[(group, condition, grid, algo)]
                if any(m in (h.get("metrics") or {}) for h in histories):
                    if m not in available_metrics:
                        available_metrics.append(m)
                    break
        if not available_metrics:
            continue

        n_cols = len(available_metrics)
        if n_cols == 1:
            figsize = (5.8, 3.8)
        else:
            figsize = (4.5 * n_cols, 3.6)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, constrained_layout=False)
        if n_cols == 1:
            axes = [axes]
        else:
            axes = list(axes)

        # ------------------------------------------------------------------
        # Plot one curve per algorithm, averaged across seeds
        # ------------------------------------------------------------------
        algos_in_figure: list[str] = []
        for ax, metric in zip(axes, available_metrics):
            for algo in algos:
                histories = seed_histories[(group, condition, grid, algo)]
                seed_arrays = [
                    h["metrics"][metric]
                    for h in histories
                    if metric in (h.get("metrics") or {})
                ]
                if not seed_arrays:
                    continue
                min_len = min(len(a) for a in seed_arrays)
                clipped = np.array([a[:min_len] for a in seed_arrays], dtype=float)
                smoothing = min(LEARNING_CURVE_SMOOTHING_WINDOW, min_len)
                smoothed = np.array([_smooth(row.tolist(), smoothing) for row in clipped])
                mu = smoothed.mean(axis=0)
                sigma = smoothed.std(axis=0) if len(smoothed) > 1 else np.zeros_like(mu)
                episodes = np.arange(1, min_len + 1)
                color = ALGO_COLORS[algo]
                ax.plot(episodes, mu, color=color, linewidth=1.4)
                ax.fill_between(episodes, mu - sigma, mu + sigma, color=color, alpha=0.2)
                if algo not in algos_in_figure:
                    algos_in_figure.append(algo)

            ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=10)
            ax.grid(alpha=0.25)

        fig.suptitle(
            _pretty_scene_title(group, grid, condition=condition),
            fontsize=11,
        )

        algos_present = [a for a in sorted(LEARNING_ALGORITHMS) if a in algos_in_figure]
        handles = [
            Line2D([0], [0], color=ALGO_COLORS[a], lw=1.8, label=ALGO_LABELS[a])
            for a in algos_present
        ]
        if handles:
            # The legend title names the hyperparameter setting for this
            # condition, while colors identify algorithms.
            annotation = _hyperparam_annotation(group, condition)
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=len(handles),
                frameon=False,
                fontsize=9,
                title=annotation or None,
                title_fontsize=9,
            )

        fig.subplots_adjust(left=0.08, right=0.96, top=0.84, bottom=0.30, wspace=0.28)
        try:
            # A failed plot should not abort the whole report run; later plot
            # families and CSV/overview outputs are still useful.
            path = (
                _plot_dir(out_dir, group, "learning_curves")
                / f"{_slug(condition)}_{grid}_learning_curves.png"
            )
            fig.savefig(path, dpi=130, bbox_inches="tight")
        except Exception:
            pass
        plt.close(fig)


def _save_combined_learning_curves(results: list[RunResult], out_dir: Path) -> None:
    """Save cross-condition learning-curve comparison figures."""
    # ------------------------------------------------------------------
    # Index histories by group/condition/grid/algorithm/seed
    # ------------------------------------------------------------------
    # Combined figures need to assemble curves from multiple experiment cases,
    # so this lookup keeps every history addressable by its experimental keys.
    curve_results = [
        r for r in results
        if r.row["algorithm"] in LEARNING_ALGORITHMS and r.history is not None
    ]
    if not curve_results:
        return

    by_run: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    group_order: list[str] = []
    for result in curve_results:
        group = result.row["setup_group"]
        if group not in group_order:
            group_order.append(group)
        key = (
            group,
            result.row["condition"],
            result.row["grid"],
            result.row["algorithm"],
            int(result.row["seed"]),
        )
        by_run[key] = result.history

    default_histories: dict[tuple[str, str, int], dict[str, Any]] = {}
    for (group, _condition, grid, algorithm, seed), history in by_run.items():
        if group == "default":
            default_histories[(grid, algorithm, seed)] = history

    def make_curve(
        *,
        condition: str,
        algorithm: str,
        histories: list[dict[str, Any]],
        styles: dict[str, Any],
    ) -> dict[str, Any]:
        """Package curve metadata in the shape expected by the renderer."""
        return {
            "label": f"{ALGO_LABELS[algorithm]} | {condition}",
            "condition": condition,
            "algorithm": algorithm,
            "histories": histories,
            "linestyle": styles[condition],
        }

    for group in group_order:
        if group == "default":
            # Default group: compare MC and Q-learning on the baseline setting,
            # both per seed and aggregated across all seeds.
            seeds = sorted({seed for _grid, _algo, seed in default_histories})
            grids = sorted({grid for grid, _algo, _seed in default_histories})
            styles = _condition_line_styles(["default"])
            for grid in grids:
                for seed in seeds:
                    curves = [
                        make_curve(
                            condition="default",
                            algorithm=algorithm,
                            histories=[history],
                            styles=styles,
                        )
                        for (hist_grid, algorithm, hist_seed), history in default_histories.items()
                        if hist_grid == grid and hist_seed == seed
                    ]
                    if len(curves) < 2:
                        continue
                    _save_combined_with_policy_diff(
                        curves,
                        group=group,
                        title=_pretty_scene_title(
                            group,
                            grid,
                            descriptor="algorithm comparison",
                            seed=seed,
                        ),
                        out_dir=out_dir,
                        filename=f"default_{grid}_seed-{seed}_algorithms_combined_learning_curves.png",
                        show_std=False,
                    )
                summary_curves = []
                for algorithm in sorted(LEARNING_ALGORITHMS):
                    histories = [
                        history
                        for (hist_grid, hist_algorithm, _seed), history in default_histories.items()
                        if hist_grid == grid and hist_algorithm == algorithm
                    ]
                    if histories:
                        summary_curves.append(
                            make_curve(
                                condition="default",
                                algorithm=algorithm,
                                histories=histories,
                                styles=styles,
                            )
                        )
                if len(summary_curves) >= 2:
                    _save_combined_with_policy_diff(
                        summary_curves,
                        group=group,
                        title=_pretty_scene_title(
                            group,
                            grid,
                            descriptor="algorithm comparison",
                            all_seeds=True,
                        ),
                        out_dir=out_dir,
                        filename=f"default_{grid}_all-seeds_algorithms_combined_learning_curves.png",
                        show_std=True,
                    )
            continue

        group_keys = [key for key in by_run if key[0] == group]
        if group == "grid_comparison":
            # Grid-comparison group: condition is effectively the grid name, so
            # linestyles identify grids while colors still identify algorithms.
            combos = sorted({(algorithm, seed) for _group, _condition, _grid, algorithm, seed in group_keys})
            grid_conditions = sorted({_grid for _group, _condition, _grid, _algorithm, _seed in group_keys})
            styles = _condition_line_styles(grid_conditions)
            seeds = sorted({seed for _group, _condition, _grid, _algorithm, seed in group_keys})
            for seed in seeds:
                curves = []
                for key in group_keys:
                    _group, _condition, grid, key_algorithm, key_seed = key
                    if key_seed == seed:
                        curves.append(
                            make_curve(
                                condition=grid,
                                algorithm=key_algorithm,
                                histories=[by_run[key]],
                                styles=styles,
                            )
                        )
                if len(curves) < 2:
                    continue
                _save_combined_with_policy_diff(
                    curves,
                    group=group,
                    title=_pretty_scene_title(
                        group,
                        descriptor="algorithm and grid comparison",
                        seed=seed,
                    ),
                    out_dir=out_dir,
                    filename=f"{group}_seed-{seed}_combined_learning_curves.png",
                    show_std=False,
                )

            summary_curves = []
            for key in group_keys:
                _group, _condition, grid, algorithm, _seed = key
                histories = [
                    history
                    for (hist_group, _hist_condition, hist_grid, hist_algorithm, _hist_seed), history in by_run.items()
                    if hist_group == group and hist_grid == grid and hist_algorithm == algorithm
                ]
                if histories:
                    candidate = make_curve(
                        condition=grid,
                        algorithm=algorithm,
                        histories=histories,
                        styles=styles,
                    )
                    if candidate["label"] not in {curve["label"] for curve in summary_curves}:
                        summary_curves.append(candidate)
            if len(summary_curves) >= 2:
                _save_combined_with_policy_diff(
                    summary_curves,
                    group=group,
                    title=_pretty_scene_title(
                        group,
                        descriptor="algorithm and grid comparison",
                        all_seeds=True,
                    ),
                    out_dir=out_dir,
                    filename=f"{group}_all-seeds_combined_learning_curves.png",
                    show_std=True,
                )
            continue

        # Hyperparameter groups: include the default baseline alongside each
        # condition so every plot shows "what changed" relative to baseline.
        grids = sorted({grid for _group, _condition, grid, _algorithm, _seed in group_keys})
        seeds = sorted({seed for _group, _condition, _grid, _algorithm, seed in group_keys})
        conditions = ["default"] + [
            condition
            for condition in dict.fromkeys(
                condition for _group, condition, _grid, _algorithm, _seed in group_keys
            )
            if condition != "default"
        ]
        styles = _condition_line_styles(conditions)

        for grid in grids:
            for seed in seeds:
                curves: list[dict[str, Any]] = []
                for algorithm in sorted(LEARNING_ALGORITHMS):
                    default_history = default_histories.get((grid, algorithm, seed))
                    if default_history is not None:
                        curves.append(
                            make_curve(
                                condition="default",
                                algorithm=algorithm,
                                histories=[default_history],
                                styles=styles,
                            )
                        )
                    for key in group_keys:
                        _group, condition, key_grid, key_algorithm, key_seed = key
                        if key_grid == grid and key_algorithm == algorithm and key_seed == seed:
                            curves.append(
                                make_curve(
                                    condition=condition,
                                    algorithm=algorithm,
                                    histories=[by_run[key]],
                                    styles=styles,
                                )
                            )

                if len(curves) < 2:
                    continue

                _save_combined_with_policy_diff(
                    curves,
                    group=group,
                    title=_pretty_scene_title(
                        group,
                        grid,
                        descriptor="algorithm and condition comparison",
                        seed=seed,
                    ),
                    out_dir=out_dir,
                    filename=f"{group}_{grid}_seed-{seed}_combined_learning_curves.png",
                    show_std=False,
                )

            summary_curves: list[dict[str, Any]] = []
            for algorithm in sorted(LEARNING_ALGORITHMS):
                default_seed_histories = [
                    history
                    for (hist_grid, hist_algorithm, _seed), history in default_histories.items()
                    if hist_grid == grid and hist_algorithm == algorithm
                ]
                if default_seed_histories:
                    summary_curves.append(
                        make_curve(
                            condition="default",
                            algorithm=algorithm,
                            histories=default_seed_histories,
                            styles=styles,
                        )
                    )
                for condition in conditions:
                    if condition == "default":
                        continue
                    histories = [
                        history
                        for (hist_group, hist_condition, hist_grid, hist_algorithm, _seed), history in by_run.items()
                        if (
                            hist_group == group
                            and hist_condition == condition
                            and hist_grid == grid
                            and hist_algorithm == algorithm
                        )
                    ]
                    if histories:
                        summary_curves.append(
                            make_curve(
                                condition=condition,
                                algorithm=algorithm,
                                histories=histories,
                                styles=styles,
                            )
                        )

            if len(summary_curves) < 2:
                continue
            _save_combined_with_policy_diff(
                summary_curves,
                group=group,
                title=_pretty_scene_title(
                    group,
                    grid,
                    descriptor="algorithm and condition comparison",
                    all_seeds=True,
                ),
                out_dir=out_dir,
                filename=f"{group}_{grid}_all-seeds_combined_learning_curves.png",
                show_std=True,
            )


def _save_vi_convergence(results: list[RunResult], out_dir: Path) -> None:
    """Save VI Bellman-residual convergence plots for selected groups."""
    # VI convergence is most useful for dynamics/discount changes. Use one
    # representative seed because VI itself is deterministic for a fixed grid
    # and hyperparameter setting.
    selected_groups = {"discount_factor", "stochasticity"}
    first_seed = _first_seed_results(results)
    for group in selected_groups:
        histories: dict[str, dict[str, Any]] = {}
        for result in first_seed:
            if result.row["setup_group"] != group or result.row["algorithm"] != "value_iteration":
                continue
            if result.history and "delta_v" in result.history.get("metrics", {}):
                histories[result.row["condition"]] = result.history
        if len(histories) < 2:
            continue
        try:
            # Plot delta_v on a log scale so convergence-rate differences are
            # visible across many orders of magnitude.
            fig, _ = plot_algorithm_comparison(
                histories,
                metrics=["delta_v"],
                smoothing_window=1,
                title=f"Value Iteration convergence: {group}",
                log_scale=True,
            )
            path = _plot_dir(out_dir, group, "vi_convergence") / f"{group}_vi_convergence.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plt.close("all")


def _save_value_policy_plots(results: list[RunResult], out_dir: Path) -> None:
    """Save spatial value-function and greedy-policy plots."""
    # Spatial plots are saved for the first seed only to avoid producing many
    # near-duplicate images in multi-seed report runs.
    for result in _first_seed_results(results):
        if not result.values or not result.policy:
            continue
        grid = np.load(result.grid_path)
        group = result.row["setup_group"]
        condition = result.row["condition"]
        algorithm = result.row["algorithm"]
        try:
            # Re-load the original grid file so plotting sees walls, obstacles,
            # and targets in their original encoded form.
            fig, _ = plot_value_and_policy(
                grid,
                result.values,
                result.policy,
                title=f"{ALGO_LABELS[algorithm]}: {condition} ({result.row['grid']})",
                agent_start_pos=result.start_pos,
            )
            path = _plot_dir(out_dir, group, "value_policy") / (
                f"{_slug(condition)}_{result.row['grid']}_{algorithm}_value_policy.png"
            )
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plt.close("all")


def _save_policy_disagreement_plots(results: list[RunResult], out_dir: Path) -> None:
    """Save spatial learned-policy-vs-VI disagreement heatmaps."""
    # Only model-free agents have a meaningful disagreement plot; VI is the
    # reference policy itself.
    for result in _first_seed_results(results):
        algorithm = result.row["algorithm"]
        if algorithm == "value_iteration" or not result.optimal_policy or not result.policy:
            continue
        grid = np.load(result.grid_path)
        group = result.row["setup_group"]
        condition = result.row["condition"]
        try:
            # The optimal policy is a set of acceptable optimal actions per
            # state, so tied VI-optimal actions are not counted as mistakes.
            fig, _ = plot_policy_disagreement(
                grid,
                result.optimal_policy,
                result.policy,
                title=f"{ALGO_LABELS[algorithm]} vs VI: {condition} ({result.row['grid']})",
                agent_start_pos=result.start_pos,
            )
            path = _plot_dir(out_dir, group, "policy_disagreement") / (
                f"{_slug(condition)}_{result.row['grid']}_{algorithm}_policy_diff.png"
            )
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plt.close("all")


def save_all(results: list[RunResult], out_dir: Path) -> None:
    """Save all report-oriented plots."""
    if not results:
        return

    # Keep the public entry point small: each helper owns one plot family and
    # writes into its corresponding per-group subdirectory.
    _save_learning_curves(results, out_dir)
    _save_combined_learning_curves(results, out_dir)
    _save_vi_convergence(results, out_dir)
    _save_value_policy_plots(results, out_dir)
    _save_policy_disagreement_plots(results, out_dir)
