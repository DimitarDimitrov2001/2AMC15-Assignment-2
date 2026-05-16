"""Report-oriented plots for assignment experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.runner import RunResult
from experiments.specs import ALGORITHMS, METRIC_FIELDS
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


def _slug(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("=", "-")
        .replace(",", "-")
    )


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_seed_results(results: list[RunResult]) -> list[RunResult]:
    if not results:
        return []
    first_seed = min(int(r.row["seed"]) for r in results)
    return [r for r in results if int(r.row["seed"]) == first_seed]


def _save_metric_bars(results: list[RunResult], out_dir: Path) -> None:
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[result.row["setup_group"]].append(result)

    for group, group_results in grouped.items():
        labels = sorted({f"{r.row['condition']} ({r.row['grid']})" for r in group_results})
        x = np.arange(len(labels))
        width = 0.24
        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        for metric in METRIC_FIELDS:
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 4.8), constrained_layout=True)
            for algo_idx, algorithm in enumerate(ALGORITHMS):
                values_by_label: dict[str, list[float]] = defaultdict(list)
                for result in group_results:
                    if result.row["algorithm"] != algorithm:
                        continue
                    label = f"{result.row['condition']} ({result.row['grid']})"
                    value = _float(result.row[metric])
                    if value is not None:
                        values_by_label[label].append(value)
                heights = [
                    mean(values_by_label[label]) if values_by_label[label] else np.nan
                    for label in labels
                ]
                errors = [
                    stdev(values_by_label[label]) if len(values_by_label[label]) > 1 else 0.0
                    for label in labels
                ]
                offset = (algo_idx - 1) * width
                ax.bar(
                    x + offset,
                    heights,
                    width=width,
                    label=ALGO_LABELS[algorithm],
                    color=ALGO_COLORS[algorithm],
                    yerr=errors,
                    capsize=3,
                    error_kw={"elinewidth": 1.2, "alpha": 0.7},
                )

            ax.set_title(f"{group}: {metric}")
            ax.set_ylabel(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            path = out_dir / group / f"{group}_{metric}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)


def _history_metrics(history: dict[str, Any], desired: list[str]) -> list[str]:
    available = history.get("metrics", {})
    return [metric for metric in desired if metric in available]


def _smooth(values: list[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(values, dtype=float)
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _save_learning_curves(results: list[RunResult], out_dir: Path) -> None:
    curve_results = [
        r for r in results
        if r.row["algorithm"] in {"mc", "q_learning"} and r.history is not None
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
    desired = ["undiscounted_return", "delta_q", "policy_diff"]

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
                smoothing = max(1, min_len // 20)
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
            path = out_dir / group / f"{_slug(condition)}_{grid}_learning_curves.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
        except Exception:
            pass
        plt.close(fig)


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
            path = out_dir / group / f"{group}_vi_convergence.png"
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
            path = out_dir / group / f"{_slug(condition)}_{result.row['grid']}_{algorithm}_value_policy.png"
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
            path = out_dir / group / f"{_slug(condition)}_{result.row['grid']}_{algorithm}_policy_diff.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plt.close("all")


def save_all(results: list[RunResult], out_dir: Path) -> None:
    """Save all report-oriented plots."""
    if not results:
        return
    _save_metric_bars(results, out_dir)
    _save_learning_curves(results, out_dir)
    _save_vi_convergence(results, out_dir)
    _save_value_policy_plots(results, out_dir)
    _save_policy_disagreement_plots(results, out_dir)
