"""Report-oriented plots for assignment experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
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
                offset = (algo_idx - 1) * width
                ax.bar(
                    x + offset,
                    heights,
                    width=width,
                    label=ALGO_LABELS[algorithm],
                    color=ALGO_COLORS[algorithm],
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


def _save_learning_curves(results: list[RunResult], out_dir: Path) -> None:
    curve_results = [
        r for r in _first_seed_results(results)
        if r.row["algorithm"] in {"mc", "q_learning"} and r.history is not None
    ]
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in curve_results:
        key = (result.row["setup_group"], result.row["condition"], result.row["grid"])
        grouped[key][ALGO_LABELS[result.row["algorithm"]]] = result.history

    desired = ["discounted_return", "delta_q", "epsilon", "alpha", "policy_diff"]
    for (group, condition, grid), histories in grouped.items():
        if not histories:
            continue
        metrics: list[str] = []
        for history in histories.values():
            for metric in _history_metrics(history, desired):
                if metric not in metrics:
                    metrics.append(metric)
        if not metrics:
            continue
        n_ep = max(len(history["episodes"]) for history in histories.values())
        smoothing = max(1, n_ep // 20)
        try:
            fig, _ = plot_algorithm_comparison(
                histories,
                metrics=metrics,
                smoothing_window=smoothing,
                title=f"{group}: {condition} ({grid})",
                colors={label: ALGO_COLORS["mc"] if label == ALGO_LABELS["mc"] else ALGO_COLORS["q_learning"] for label in histories},
            )
            path = out_dir / group / f"{_slug(condition)}_{grid}_learning_curves.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plt.close("all")


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
