"""Artifact writers for training and evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.value_iteration_agent import ValueIterationAgent
from utils.plotting import TrainingHistory, plot_training_history
from utils.rl_plots import plot_policy_disagreement, plot_value_and_policy


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON payload, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
