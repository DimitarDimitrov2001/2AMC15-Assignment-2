from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.dqn_agent import DQNAgent
from utils.rl_plots import _configure_grid_axes, _draw_grid_background




def save_dqn_run_artifacts(
    out_dir: Path,
    run_config: dict[str, Any],
    history: list[dict[str, float]],
    agent: DQNAgent,
    rollout: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / "config.json", run_config)
    _write_json(out_dir / "history.json", history)
    _write_metrics_csv(out_dir / "metrics.csv", history)
    _write_training_curves(out_dir / "training_curves.png", history)
    if rollout is not None:
        _write_json(out_dir / "policy_rollout.json", _rollout_json_payload(rollout))
        _write_policy_rollout_plot(out_dir / "policy_rollout.png", rollout)
        _write_policy_rollout_html(out_dir / "policy_rollout.html", rollout)
    model_path = agent.save(out_dir / "dqn_model.pt")
    _write_evaluation_summary(out_dir / "evaluation_summary.txt", history, agent)
    return model_path


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)


def _write_metrics_csv(path: Path, history: list[dict[str, float]]) -> None:
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _write_training_curves(path: Path, history: list[dict[str, float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.ravel()
    _plot_metric(axes_flat[0], history, "train/episode_reward", "Train reward")
    _plot_metric(axes_flat[1], history, "eval/mean_reward", "Eval mean reward")
    _plot_metric(axes_flat[2], history, "eval/success_rate", "Eval success rate")
    _plot_metric(axes_flat[3], history, "update/dqn/loss", "DQN loss")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_policy_rollout_plot(path: Path, rollout: dict[str, Any]) -> None:
    grid = np.asarray(rollout["grid"])
    positions = np.asarray(rollout["positions"], dtype=float)
    plot_positions = positions - 0.5

    n_cols, n_rows = grid.shape
    fig, ax = plt.subplots(
        figsize=(max(6, n_cols * 0.5), max(5, n_rows * 0.5)),
        constrained_layout=True,
    )
    _draw_grid_background(ax, grid)

    if len(plot_positions) > 0:
        ax.plot(
            plot_positions[:, 0],
            plot_positions[:, 1],
            color="#1f77b4",
            linewidth=2,
            marker="o",
            markersize=3,
            zorder=3,
        )
        start_col, start_row = plot_positions[0]
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
            plot_positions[-1, 0],
            plot_positions[-1, 1],
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
    payload = _html_rollout_payload(rollout)
    data_json = json.dumps(_json_safe(payload), separators=(",", ":")).replace("</", "<\\/")
    html = Template(_ROLLOUT_HTML_TEMPLATE).substitute(payload=data_json)
    with path.open("w", encoding="utf-8") as f:
        f.write(html)


def _html_rollout_payload(rollout: dict[str, Any]) -> dict[str, Any]:
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
        "total_reward": rollout.get("total_reward", 0.0),
        "steps": rollout.get("steps", 0),
        "success": rollout.get("success", False),
        "terminated": rollout.get("terminated", False),
        "truncated": rollout.get("truncated", False),
        "world_stats": rollout.get("world_stats", {}),
    }


def _rollout_json_payload(rollout: dict[str, Any]) -> dict[str, Any]:
    return dict(rollout)


def _plot_metric(
    ax: plt.Axes,
    history: list[dict[str, float]],
    key: str,
    title: str,
) -> None:
    points = [
        (row.get("train/episode", index + 1), row[key])
        for index, row in enumerate(history)
        if key in row
    ]
    if points:
        xs, ys = zip(*points)
        ax.plot(xs, ys)
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.grid(True, alpha=0.25)


def _write_evaluation_summary(path: Path, history: list[dict[str, float]], agent: DQNAgent) -> None:
    final = history[-1] if history else {}
    eval_rows = [row for row in history if "eval/mean_reward" in row]
    last_eval = eval_rows[-1] if eval_rows else {}
    lines = [
        "DQN evaluation summary",
        f"episodes: {_fmt(final.get('train/episode'))}",
        f"global_step: {_fmt(final.get('global_step'))}",
        f"final_train_reward: {_fmt(final.get('train/episode_reward'))}",
        f"final_train_length: {_fmt(final.get('train/episode_length'))}",
        f"final_epsilon: {agent._epsilon():.6g}",
        f"updates: {agent._updates}",
        f"last_eval_mean_reward: {_fmt(last_eval.get('eval/mean_reward'))}",
        f"last_eval_success_rate: {_fmt(last_eval.get('eval/success_rate'))}",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6g}"


def _json_safe(value: Any) -> Any:
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
