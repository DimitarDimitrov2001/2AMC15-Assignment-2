"""Artifact writers for training and evaluation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from string import Template
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


_ROLLOUT_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Policy Rollout</title>
  <style>
    :root {
      color-scheme: light;
      --empty: #f0f0ec;
      --path: #1f77b4;
      --agent: #0b5cad;
      --heading: #d98400;
      --danger: #d62728;
      --ink: #1f2933;
      --muted: #5d6875;
      --panel: #ffffff;
      --border: #d8dde3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #f6f7f9;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 20px auto;
      display: grid;
      grid-template-columns: minmax(360px, 1fr) 280px;
      gap: 16px;
      align-items: start;
    }
    h1 {
      grid-column: 1 / -1;
      margin: 0;
      font-size: 22px;
      font-weight: 650;
    }
    .viewer, .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    }
    .viewer { padding: 12px; }
    svg {
      display: block;
      width: 100%;
      max-height: 78vh;
      aspect-ratio: 1 / 1;
      background: var(--empty);
      border: 1px solid #222;
    }
    .controls {
      display: grid;
      grid-template-columns: auto auto 1fr auto;
      gap: 10px;
      align-items: center;
      margin-top: 12px;
    }
    button {
      min-width: 42px;
      height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }
    button:hover { background: #f0f3f7; }
    input[type="range"] { width: 100%; }
    .step-readout {
      min-width: 76px;
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }
    .panel { padding: 14px; }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 15px;
    }
    dl {
      display: grid;
      grid-template-columns: 100px 1fr;
      gap: 7px 10px;
      margin: 0;
      font-size: 13px;
    }
    dt { color: var(--muted); }
    dd {
      margin: 0;
      font-variant-numeric: tabular-nums;
      overflow-wrap: anywhere;
    }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      h1 { font-size: 19px; }
    }
  </style>
</head>
<body>
  <main>
    <h1>Greedy policy rollout</h1>
    <section class="viewer">
      <svg id="rollout-svg" role="img" aria-label="Greedy policy rollout"></svg>
      <div class="controls">
        <button id="prev" type="button" title="Previous step">&lt;</button>
        <button id="play" type="button" title="Play or pause">Play</button>
        <input id="slider" type="range" min="0" value="0" step="1">
        <output id="step-readout" class="step-readout">0 / 0</output>
      </div>
    </section>
    <aside class="panel">
      <h2>Current Step</h2>
      <dl>
        <dt>Action</dt><dd id="action">start</dd>
        <dt>Reward</dt><dd id="reward">0.000</dd>
        <dt>Total</dt><dd id="total">0.000</dd>
        <dt>Position</dt><dd id="position">-</dd>
        <dt>Heading</dt><dd id="heading">-</dd>
        <dt>Collision</dt><dd id="collision">false</dd>
        <dt>Success</dt><dd id="success">false</dd>
      </dl>
    </aside>
  </main>
  <script id="rollout-data" type="application/json">$payload</script>
  <script>
    const data = JSON.parse(document.getElementById("rollout-data").textContent);
    const svg = document.getElementById("rollout-svg");
    const slider = document.getElementById("slider");
    const playButton = document.getElementById("play");
    const prevButton = document.getElementById("prev");
    const stepReadout = document.getElementById("step-readout");
    const actionText = document.getElementById("action");
    const rewardText = document.getElementById("reward");
    const totalText = document.getElementById("total");
    const positionText = document.getElementById("position");
    const headingText = document.getElementById("heading");
    const collisionText = document.getElementById("collision");
    const successText = document.getElementById("success");
    const NS = "http://www.w3.org/2000/svg";
    const actionNames = ["rotate_left", "rotate_right", "move_forward"];
    const cellColors = {0: "#f0f0ec", 1: "#2b2b2b", 2: "#7a7a7a", 3: "#2f8f2f", 4: "#f0f0ec"};
    const maxStep = Math.max(0, data.positions.length - 1);
    let currentStep = 0;
    let timer = null;
    slider.max = String(maxStep);
    svg.setAttribute("viewBox", "-0.5 -0.5 " + data.n_cols + " " + data.n_rows);
    function svgElement(name, attrs) {
      const node = document.createElementNS(NS, name);
      for (const key in attrs) node.setAttribute(key, attrs[key]);
      return node;
    }
    function plotPoint(step) {
      const position = data.positions[step];
      return [position[0] - 0.5, position[1] - 0.5];
    }
    function drawBackground() {
      for (let col = 0; col < data.n_cols; col += 1) {
        for (let row = 0; row < data.n_rows; row += 1) {
          const code = data.grid[col][row];
          svg.appendChild(svgElement("rect", {
            x: col - 0.5, y: row - 0.5, width: 1, height: 1,
            fill: cellColors[code] || cellColors[0], stroke: "none"
          }));
        }
      }
    }
    drawBackground();
    const pathLine = svgElement("polyline", {
      fill: "none", stroke: "#1f77b4", "stroke-width": "0.075",
      "stroke-linecap": "round", "stroke-linejoin": "round"
    });
    svg.appendChild(pathLine);
    const start = plotPoint(0);
    const startCircle = svgElement("circle", {
      cx: start[0], cy: start[1], r: "0.34", fill: "#e8c13a",
      stroke: "black", "stroke-width": "0.05"
    });
    svg.appendChild(startCircle);
    const startText = svgElement("text", {
      x: start[0], y: start[1] + 0.08, "text-anchor": "middle",
      "font-size": "0.28", "font-weight": "700", fill: "black"
    });
    startText.textContent = "S";
    svg.appendChild(startText);
    const headingLine = svgElement("line", {
      stroke: "#d98400", "stroke-width": "0.09", "stroke-linecap": "round"
    });
    const headingTip = svgElement("circle", {r: "0.07", fill: "#d98400"});
    const agentCircle = svgElement("circle", {
      r: "0.2", fill: "#0b5cad", stroke: "white", "stroke-width": "0.05"
    });
    svg.appendChild(headingLine);
    svg.appendChild(headingTip);
    svg.appendChild(agentCircle);
    function cumulativeReward(step) {
      let total = 0;
      for (let i = 0; i < step; i += 1) total += Number(data.rewards[i] || 0);
      return total;
    }
    function formatNumber(value) { return Number(value).toFixed(3); }
    function render(step) {
      currentStep = Math.max(0, Math.min(maxStep, step));
      slider.value = String(currentStep);
      stepReadout.textContent = currentStep + " / " + maxStep;
      const points = [];
      for (let i = 0; i <= currentStep; i += 1) {
        const point = plotPoint(i);
        points.push(point[0] + "," + point[1]);
      }
      pathLine.setAttribute("points", points.join(" "));
      const current = plotPoint(currentStep);
      const heading = Number(data.headings[currentStep] || 0);
      const radians = heading * Math.PI / 180;
      const hx = current[0] + Math.cos(radians) * 0.45;
      const hy = current[1] + Math.sin(radians) * 0.45;
      agentCircle.setAttribute("cx", current[0]);
      agentCircle.setAttribute("cy", current[1]);
      headingLine.setAttribute("x1", current[0]);
      headingLine.setAttribute("y1", current[1]);
      headingLine.setAttribute("x2", hx);
      headingLine.setAttribute("y2", hy);
      headingTip.setAttribute("cx", hx);
      headingTip.setAttribute("cy", hy);
      const actionIndex = currentStep > 0 ? data.actions[currentStep - 1] : null;
      const reward = currentStep > 0 ? Number(data.rewards[currentStep - 1] || 0) : 0;
      const info = currentStep > 0 ? data.infos?.[currentStep - 1] || {} : {};
      const rawPosition = data.positions[currentStep];
      actionText.textContent = actionIndex === null ? "start" : actionNames[actionIndex] || String(actionIndex);
      rewardText.textContent = formatNumber(reward);
      totalText.textContent = formatNumber(cumulativeReward(currentStep));
      positionText.textContent = "(" + formatNumber(rawPosition[0]) + ", " + formatNumber(rawPosition[1]) + ")";
      headingText.textContent = formatNumber(heading);
      collisionText.textContent = String(Boolean(info.collision));
      successText.textContent = String(Boolean(info.success || (currentStep === maxStep && data.success)));
    }
    function stop() {
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
      playButton.textContent = "Play";
    }
    function play() {
      if (timer !== null) {
        stop();
        return;
      }
      playButton.textContent = "Pause";
      timer = window.setInterval(() => {
        if (currentStep >= maxStep) {
          stop();
          return;
        }
        render(currentStep + 1);
      }, 350);
    }
    playButton.addEventListener("click", play);
    prevButton.addEventListener("click", () => { stop(); render(currentStep - 1); });
    slider.addEventListener("input", () => { stop(); render(Number(slider.value)); });
    document.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") {
        stop();
        render(currentStep - 1);
      } else if (event.key === "ArrowRight") {
        stop();
        render(currentStep + 1);
      } else if (event.key === " ") {
        event.preventDefault();
        play();
      }
    });
    render(0);
  </script>
</body>
</html>
"""


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
        rollout_html = out_dir / "policy_rollout.html"
        write_json(rollout_json, dict(rollout))
        _write_policy_rollout_plot(rollout_png, rollout)
        _write_policy_rollout_html(rollout_html, rollout)
        paths.extend([rollout_json, rollout_png, rollout_html])

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
