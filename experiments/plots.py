"""Report-oriented plots for assignment experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.runner import RunResult
from utils.rl_plots import (
    plot_algorithm_comparison,
    plot_policy_disagreement,
    plot_value_and_policy,
)


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
    "vi_convergence": "vi_convergence",
    "value_policy": "value_policy",
    "policy_disagreement": "policy_disagreement",
}

LEARNING_CURVE_SMOOTHING_WINDOW = 500
LEARNING_CURVE_METRICS = ["undiscounted_return", "delta_q", "policy_diff"]
LEARNING_ALGORITHMS = {"mc", "q_learning"}
LINE_STYLES = ["-", "--", ":", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]


def _slug(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("=", "-")
        .replace(",", "-")
    )


def _plot_dir(out_dir: Path, group: str, plot_type: str) -> Path:
    path = out_dir / group / PLOT_DIRS[plot_type]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _first_seed_results(results: list[RunResult]) -> list[RunResult]:
    if not results:
        return []
    first_seed = min(int(r.row["seed"]) for r in results)
    return [r for r in results if int(r.row["seed"]) == first_seed]


def _history_metrics(history: dict[str, Any], desired: list[str]) -> list[str]:
    available = history.get("metrics", {})
    return [metric for metric in desired if metric in available]


def _smooth(values: list[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(values, dtype=float)
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _history_metric_values(history: dict[str, Any], metric: str) -> tuple[list[int], list[float]] | None:
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
    title: str,
    path: Path,
    show_std: bool,
) -> None:
    available_metrics = [
        metric
        for metric in LEARNING_CURVE_METRICS
        if any(
            metric in (history.get("metrics") or {})
            for curve in curves
            for history in curve["histories"]
        )
    ]
    if not available_metrics:
        return

    fig, axes = plt.subplots(
        len(available_metrics),
        1,
        figsize=(9, 3 * len(available_metrics)),
        constrained_layout=False,
    )
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        for curve in curves:
            matrix_data = _smoothed_metric_matrix(curve["histories"], metric)
            if matrix_data is None:
                continue
            episodes, matrix = matrix_data
            mu = matrix.mean(axis=0)
            color = ALGO_COLORS[curve["algorithm"]]
            linestyle = curve["linestyle"]
            ax.plot(
                episodes,
                mu,
                label=curve["label"],
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

        ax.set_xlabel("Episode")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            fontsize=8,
            frameon=True,
        )
    fig.subplots_adjust(right=0.78, hspace=0.28, top=0.93)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _save_learning_curves(results: list[RunResult], out_dir: Path) -> None:
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

        n_rows = len(available_metrics)
        fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3 * n_rows), constrained_layout=True)
        if n_rows == 1:
            axes = [axes]

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
                label = ALGO_LABELS[algo]
                ax.plot(episodes, mu, color=color, label=label, linewidth=1.4)
                ax.fill_between(episodes, mu - sigma, mu + sigma, color=color, alpha=0.2)

            ax.set_xlabel("Episode")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

        fig.suptitle(f"{group}: {condition} ({grid})", fontsize=10)
        try:
            path = (
                _plot_dir(out_dir, group, "learning_curves")
                / f"{_slug(condition)}_{grid}_learning_curves.png"
            )
            fig.savefig(path, dpi=130, bbox_inches="tight")
        except Exception:
            pass
        plt.close(fig)


def _save_combined_learning_curves(results: list[RunResult], out_dir: Path) -> None:
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
        return {
            "label": f"{ALGO_LABELS[algorithm]} | {condition}",
            "condition": condition,
            "algorithm": algorithm,
            "histories": histories,
            "linestyle": styles[condition],
        }

    for group in group_order:
        if group == "default":
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
                    path = (
                        _plot_dir(out_dir, group, "combined_learning_curves")
                        / f"default_{grid}_seed-{seed}_algorithms_combined_learning_curves.png"
                    )
                    _save_combined_curve_figure(
                        curves,
                        title=f"default: algorithm comparison ({grid}, seed={seed})",
                        path=path,
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
                    path = (
                        _plot_dir(out_dir, group, "combined_learning_curves")
                        / f"default_{grid}_all-seeds_algorithms_combined_learning_curves.png"
                    )
                    _save_combined_curve_figure(
                        summary_curves,
                        title=f"default: algorithm comparison ({grid}, all seeds)",
                        path=path,
                        show_std=True,
                    )
            continue

        group_keys = [key for key in by_run if key[0] == group]
        if group == "grid_comparison":
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
                path = (
                    _plot_dir(out_dir, group, "combined_learning_curves")
                    / f"{group}_seed-{seed}_combined_learning_curves.png"
                )
                _save_combined_curve_figure(
                    curves,
                    title=f"{group}: algorithm and grid comparison (seed={seed})",
                    path=path,
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
                path = (
                    _plot_dir(out_dir, group, "combined_learning_curves")
                    / f"{group}_all-seeds_combined_learning_curves.png"
                )
                _save_combined_curve_figure(
                    summary_curves,
                    title=f"{group}: algorithm and grid comparison (all seeds)",
                    path=path,
                    show_std=True,
                )
            continue

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

                path = (
                    _plot_dir(out_dir, group, "combined_learning_curves")
                    / f"{group}_{grid}_seed-{seed}_combined_learning_curves.png"
                )
                _save_combined_curve_figure(
                    curves,
                    title=f"{group}: algorithm and condition comparison ({grid}, seed={seed})",
                    path=path,
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
            path = (
                _plot_dir(out_dir, group, "combined_learning_curves")
                / f"{group}_{grid}_all-seeds_combined_learning_curves.png"
            )
            _save_combined_curve_figure(
                summary_curves,
                title=f"{group}: algorithm and condition comparison ({grid}, all seeds)",
                path=path,
                show_std=True,
            )


def _save_vi_convergence(results: list[RunResult], out_dir: Path) -> None:
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
    for result in _first_seed_results(results):
        if not result.values or not result.policy:
            continue
        grid = np.load(result.grid_path)
        group = result.row["setup_group"]
        condition = result.row["condition"]
        algorithm = result.row["algorithm"]
        try:
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
    for result in _first_seed_results(results):
        algorithm = result.row["algorithm"]
        if algorithm == "value_iteration" or not result.optimal_policy or not result.policy:
            continue
        grid = np.load(result.grid_path)
        group = result.row["setup_group"]
        condition = result.row["condition"]
        try:
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
    _save_learning_curves(results, out_dir)
    _save_combined_learning_curves(results, out_dir)
    _save_vi_convergence(results, out_dir)
    _save_value_policy_plots(results, out_dir)
    _save_policy_disagreement_plots(results, out_dir)
