"""Artifact writers for training and evaluation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.base_agent import BaseAgent
from utils.plotting import TrainingHistory, plot_training_history
from utils.rl_plots import (
    _configure_grid_axes,
    _draw_grid_background,
    plot_policy_disagreement,
    plot_value_and_policy,
)

if TYPE_CHECKING:
    from agents.value_iteration_agent import ValueIterationAgent




def write_json(path: Path, payload: dict) -> None:
    """Write a JSON payload, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)


def save_deep_rl_run_artifacts(
    out_dir: Path,
    run_config: dict[str, Any],
    history: list[dict[str, float]],
    agent: BaseAgent,
    rollout: dict[str, Any] | None = None,
) -> list[Path]:
    """Save training curves, config, metrics, and optional greedy-rollout artifacts.

    Args:
        out_dir: Directory that receives the artifacts.
        run_config: JSON-serializable run configuration.
        history: Per-episode trainer metrics.
        agent: Trained agent used for the evaluation summary.
        rollout: Optional greedy rollout generated from the checkpointed policy.

    Returns:
        Paths written by this function.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "config.json",
        out_dir / "metrics.csv",
        out_dir / "training_curves.png",
        out_dir / "evaluation_summary.txt",
    ]
    write_json(paths[0], run_config)
    _write_metrics_csv(paths[1], history)
    _write_deep_training_curves(paths[2], history)
    _write_deep_evaluation_summary(paths[3], history, agent)

    if rollout is not None:
        rollout_json = out_dir / "policy_rollout.json"
        rollout_png = out_dir / "policy_rollout.png"
        write_json(rollout_json, dict(rollout))
        _write_policy_rollout_plot(rollout_png, rollout)
        _write_policy_rollout_html(rollout_html, rollout)
        paths.extend([rollout_json, rollout_png])

    return paths


def log_wandb_artifact(
    artifact_name: str,
    artifact_type: str,
    paths: list[Path],
    aliases: list[str] | None = None,
) -> None:
    """Log local files as a W&B artifact when a run is active."""
    import wandb

    if wandb.run is None:
        return

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    for path in paths:
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    wandb.log_artifact(artifact, aliases=aliases)


def save_evaluation_summary_artifact(
    out_dir: Path,
    artifact_prefix: str,
    evaluation_metrics: dict,
    policy_difference: float | None = None,
) -> None:
    """Save a short human-readable summary of rollout evaluation metrics.

    ``policy_difference`` is appended as a final line when provided (used by
    callers that compute fraction-of-states-disagreeing-with-VI as an
    optimality proxy). Pass ``None`` when no reference policy is available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{artifact_prefix}_evaluation_summary.txt"
    summary_lines = [
        "Evaluation summary",
        f"episodes: {evaluation_metrics['n_eval_episodes']}",
        f"max_steps_per_episode: {evaluation_metrics['eval_max_steps']}",
        f"success_rate: {evaluation_metrics['success_rate']:.3f}",
        f"mean_discounted_return: {evaluation_metrics['mean_discounted_return']:.3f}",
        f"mean_undiscounted_return: {evaluation_metrics['mean_undiscounted_return']:.3f}",
        f"mean_episode_length: {evaluation_metrics['mean_episode_length']:.3f}",
        f"mean_success_episode_length: {_format_optional_float(evaluation_metrics['mean_success_episode_length'])}",
    ]
    if policy_difference is not None:
        summary_lines.append(f"policy_difference_vs_reference: {policy_difference:.3f}")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def save_value_iteration_artifacts(
    out_dir: Path,
    artifact_prefix: str,
    grid,
    initial_pos: tuple[int, int],
    agent: ValueIterationAgent,
    evaluation_metrics: dict,
    wandb_log: bool = False,
) -> None:
    """Save value-iteration metrics and value/policy diagnostics."""
    history_payload = agent.history.to_dict() if agent.history is not None else {}
    write_json(
        out_dir / f"{artifact_prefix}_metrics.json",
        {
            "training": {
                "converged": agent.converged,
                "iterations": agent.iterations,
                "final_delta_v": agent.final_delta_v,
                "history": history_payload,
            },
            "evaluation": evaluation_metrics,
        },
    )

    fig, _ = plot_value_and_policy(
        grid,
        agent.values,
        agent.policy,
        title=f"Value Iteration - {artifact_prefix}",
        agent_start_pos=initial_pos,
    )
    fig.savefig(out_dir / f"{artifact_prefix}_value_policy.png", dpi=130, bbox_inches="tight")
    if wandb_log:
        import wandb
        if wandb.run is not None:
            wandb.log({"Value and Policy": wandb.Image(fig)})
    plt.close(fig)


def save_training_curves_artifact(
    out_dir: Path,
    artifact_prefix: str,
    history: TrainingHistory,
    smoothing_window: int | None = None,
    wandb_log: bool = False,
    suffix: str = "training_curves",
    title: str | None = None,
) -> None:
    """Save per-episode training curves as ``*_training_curves.png``.

    Plots every metric present in ``history.metrics``, so when the
    trainer was given an ``optimal_policy`` reference the resulting
    figure includes a ``policy_diff`` subplot alongside ``discounted_return``,
    ``epsilon``, etc. Smoothing defaults to ``max(1, n_episodes // 20)``
    — same heuristic the sweep uses.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_episodes = len(history.episodes)
    if n_episodes == 0:
        return
    window = smoothing_window if smoothing_window is not None else max(1, n_episodes // 20)
    fig_title = title if title is not None else f"Training curves - {artifact_prefix}"
    fig, _, _ = plot_training_history(
        history,
        smoothing_window=window,
        title=fig_title,
    )
    fig.savefig(out_dir / f"{artifact_prefix}_{suffix}.png", dpi=130, bbox_inches="tight")
    if wandb_log:
        import wandb
        if wandb.run is not None:
            wandb.log({fig_title: wandb.Image(fig)})
    plt.close(fig)


def save_policy_disagreement_artifact(
    out_dir: Path,
    artifact_prefix: str,
    grid,
    optimal_policy: dict[tuple[int, int], frozenset[int]],
    learned_policy: dict[tuple[int, int], int],
    agent_start_pos: tuple[int, int] | None = None,
    wandb_log: bool = False,
) -> None:
    """Render and save the spatial policy-disagreement heatmap as ``*_policy_diff.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_policy_disagreement(
        grid,
        optimal_policy,
        learned_policy,
        title=f"Policy Disagreement - {artifact_prefix}",
        agent_start_pos=agent_start_pos,
    )
    fig.savefig(out_dir / f"{artifact_prefix}_policy_diff.png", dpi=130, bbox_inches="tight")
    if wandb_log:
        import wandb
        if wandb.run is not None:
            wandb.log({"Policy Disagreement": wandb.Image(fig)})
    plt.close(fig)


def _write_metrics_csv(path: Path, history: list[dict[str, float]]) -> None:
    """Write per-episode metrics to CSV."""
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _write_deep_training_curves(path: Path, history: list[dict[str, float]]) -> None:
    """Write the core deep-RL training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.ravel()
    _plot_history_metric(axes_flat[0], history, "rollout/episode_reward", "Train reward")
    _plot_history_metric(axes_flat[1], history, "eval/mean_reward", "Eval mean reward")
    _plot_history_metric(axes_flat[2], history, "eval/success_rate", "Eval success rate")
    _plot_history_metric(axes_flat[3], history, "losses/td_loss", "TD loss")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_policy_rollout_plot(path: Path, rollout: dict[str, Any]) -> None:
    """Write a static PNG of a greedy policy rollout."""
    grid = np.asarray(rollout["grid"])
    positions = np.asarray(rollout["positions"], dtype=float) - 0.5
    n_cols, n_rows = grid.shape

    fig, ax = plt.subplots(
        figsize=(max(6, n_cols * 0.5), max(5, n_rows * 0.5)),
        constrained_layout=True,
    )
    _draw_grid_background(ax, grid)

    if len(positions) > 0:
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color="#1f77b4",
            linewidth=2,
            marker="o",
            markersize=3,
            zorder=3,
        )
        start_col, start_row = positions[0]
        start_circle = plt.Circle(
            (start_col, start_row),
            0.35,
            facecolor="#E8C13A",
            edgecolor="black",
            linewidth=1.5,
            zorder=4,
        )
        ax.add_patch(start_circle)
        ax.text(
            start_col,
            start_row,
            "S",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="black",
            zorder=5,
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            color="#d62728",
            s=90,
            marker="X",
            edgecolors="black",
            zorder=5,
        )

    ax.set_title(
        "Greedy policy rollout "
        f"(steps={rollout['steps']}, reward={rollout['total_reward']:.3f}, "
        f"success={rollout['success']})"
    )
    _configure_grid_axes(ax, n_cols, n_rows)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_policy_rollout_html(path: Path, rollout: dict[str, Any]) -> None:
    """Write an interactive HTML rollout viewer."""
    payload = _html_rollout_payload(rollout)
    data_json = json.dumps(_json_safe(payload), separators=(",", ":")).replace("</", "<\\/")
    html = Template(_ROLLOUT_HTML_TEMPLATE).substitute(payload=data_json)
    with path.open("w", encoding="utf-8") as f:
        f.write(html)


def _html_rollout_payload(rollout: dict[str, Any]) -> dict[str, Any]:
    """Return the compact payload consumed by the HTML viewer."""
    grid = np.asarray(rollout["grid"])
    positions = np.asarray(rollout["positions"], dtype=float)
    headings = rollout.get("headings")
    if headings is None:
        infos = rollout.get("infos", [])
        headings = [0.0] + [float(info.get("theta", 0.0)) for info in infos]

    return {
        "grid": grid,
        "n_cols": int(grid.shape[0]),
        "n_rows": int(grid.shape[1]),
        "positions": positions,
        "headings": headings,
        "actions": rollout.get("actions", []),
        "rewards": rollout.get("rewards", []),
        "infos": rollout.get("infos", []),
        "total_reward": rollout.get("total_reward", 0.0),
        "steps": rollout.get("steps", 0),
        "success": rollout.get("success", False),
        "terminated": rollout.get("terminated", False),
        "truncated": rollout.get("truncated", False),
        "world_stats": rollout.get("world_stats", {}),
    }


def _plot_history_metric(
    ax: plt.Axes,
    history: list[dict[str, float]],
    key: str,
    title: str,
) -> None:
    """Plot one metric if present in the history."""
    points = [
        (row.get("episode", index + 1), row[key])
        for index, row in enumerate(history)
        if key in row
    ]
    if points:
        xs, ys = zip(*points)
        ax.plot(xs, ys)
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.grid(True, alpha=0.25)


def _write_deep_evaluation_summary(
    path: Path,
    history: list[dict[str, float]],
    agent: BaseAgent,
) -> None:
    """Write a compact text summary for a deep-RL run."""
    final = history[-1] if history else {}
    eval_rows = [row for row in history if "eval/mean_reward" in row]
    best_eval = max(eval_rows, key=lambda row: row["eval/mean_reward"], default={})
    last_eval = eval_rows[-1] if eval_rows else {}
    lines = [
        "Deep-RL evaluation summary",
        f"episodes: {_fmt(final.get('episode'))}",
        f"global_step: {_fmt(final.get('global_step'))}",
        f"final_train_reward: {_fmt(final.get('rollout/episode_reward'))}",
        f"final_train_length: {_fmt(final.get('rollout/episode_length'))}",
        f"best_eval_mean_reward: {_fmt(best_eval.get('eval/mean_reward'))}",
        f"best_eval_success_rate: {_fmt(best_eval.get('eval/success_rate'))}",
        f"last_eval_mean_reward: {_fmt(last_eval.get('eval/mean_reward'))}",
        f"last_eval_success_rate: {_fmt(last_eval.get('eval/success_rate'))}",
        f"agent: {agent.__class__.__name__}",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    """Format optional scalar values."""
    if value is None:
        return "N/A"
    return f"{float(value):.6g}"


def _json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value
