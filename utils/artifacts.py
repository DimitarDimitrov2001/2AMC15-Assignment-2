"""Artifact writers for training and evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.value_iteration_agent import ValueIterationAgent
from utils.rl_plots import plot_value_and_policy


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON payload, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_evaluation_summary_artifact(out_dir: Path, artifact_prefix: str, evaluation_metrics: dict) -> None:
    """Save a short human-readable summary of rollout evaluation metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{artifact_prefix}_evaluation_summary.txt"
    summary_lines = [
        "Evaluation summary",
        f"episodes: {evaluation_metrics['n_eval_episodes']}",
        f"max_steps_per_episode: {evaluation_metrics['max_steps']}",
        f"success_rate: {evaluation_metrics['success_rate']:.3f}",
        f"mean_discounted_return: {evaluation_metrics['mean_discounted_return']:.3f}",
        f"mean_undiscounted_return: {evaluation_metrics['mean_undiscounted_return']:.3f}",
        f"mean_episode_length: {evaluation_metrics['mean_episode_length']:.3f}",
        f"mean_success_episode_length: {_format_optional_float(evaluation_metrics['mean_success_episode_length'])}",
    ]
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
    plt.close(fig)
