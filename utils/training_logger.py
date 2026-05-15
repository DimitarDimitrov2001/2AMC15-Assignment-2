"""General-purpose training loggers for iterative learning algorithms.

This module keeps the same logger call signature used by the existing
Q-learning implementations, but removes environment-specific formatting and
parallel grid-search concerns.  States and actions are displayed as plain
integer indices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import sys

import numpy as np

type Q = np.ndarray


class TrainingLogger(ABC):
    """Interface for training progress loggers.

    Implements the context-manager protocol so callers can use
    ``with logger:`` to guarantee cleanup (e.g. ``wandb.finish()``).

    Implementations are compatible with ``q_learning``, ``double_q_learning``,
    and ``q_learning_vfa`` because those functions call this signature directly
    at each configured logging interval.
    """

    @abstractmethod
    def log_iteration(
        self,
        episode: int,
        q_values: Q,
        q_delta: float,
        converged: bool,
        current_alpha: float | None = None,
        current_epsilon: float | None = None,
        mean_q_delta: float | None = None,
        policy_diff: float | None = None,
        discounted_return: float | None = None,
        env_grid: np.ndarray | None = None,
        optimal_policy: dict | None = None,
        agent_start_pos: tuple[int, int] | None = None,
        agent_values: dict | None = None,
        agent_policy: dict | None = None,
    ) -> None:
        """Log one training iteration."""


class ConsoleTrainingLogger(TrainingLogger):
    """Console logger with a status line and optional integer-indexed Q-table.

    Args:
        show_q_table: Whether to include a Q-table below the status line.
        states: Optional list of integer state rows to display.  When ``None``,
            all rows from ``q_values`` are displayed.
        redraw_mode: ``"frame"`` overwrites the previous render on TTYs;
            ``"scroll"`` appends each log entry.
    """

    _show_q_table: bool
    _states: list[int] | None
    _redraw_mode: str
    _rendered_lines: int

    def __init__(
        self,
        show_q_table: bool = False,
        states: list[int] | None = None,
        redraw_mode: str = "frame",
    ) -> None:
        if redraw_mode not in {"frame", "scroll"}:
            raise ValueError("redraw_mode must be 'frame' or 'scroll'")

        self._show_q_table = show_q_table
        self._states = states
        self._redraw_mode = redraw_mode
        self._rendered_lines = 0

    def log_iteration(
        self,
        episode: int,
        q_values: Q,
        q_delta: float,
        converged: bool,
        current_alpha: float | None = None,
        current_epsilon: float | None = None,
        mean_q_delta: float | None = None,
        policy_diff: float | None = None,
        discounted_return: float | None = None,
        env_grid: np.ndarray | None = None,
        optimal_policy: dict | None = None,
        agent_start_pos: tuple[int, int] | None = None,
        agent_values: dict | None = None,
        agent_policy: dict | None = None,
    ) -> None:
        """Log a training iteration as console text."""
        rendered = self._format_status_line(
            episode=episode,
            q_delta=q_delta,
            converged=converged,
            current_alpha=current_alpha,
            current_epsilon=current_epsilon,
            mean_q_delta=mean_q_delta,
            policy_diff=policy_diff,
            discounted_return=discounted_return,
        )

        if self._show_q_table:
            rendered += "\n" + self._format_q_table(q_values)

        self._write_rendered(rendered)

    def _format_status_line(
        self,
        episode: int,
        q_delta: float,
        converged: bool,
        current_alpha: float | None,
        current_epsilon: float | None,
        mean_q_delta: float | None,
        policy_diff: float | None,
        discounted_return: float | None,
    ) -> str:
        parts = [
            "Episode: %d" % episode,
            "max |dQ|: %.6f" % q_delta,
        ]
        if mean_q_delta is not None:
            parts.append("mean |dQ|: %.6f" % mean_q_delta)
        if current_epsilon is not None:
            parts.append("eps: %.4f" % current_epsilon)
        if current_alpha is not None:
            parts.append("alpha: %.4f" % current_alpha)
        if discounted_return is not None:
            parts.append("G_0: %.4f" % discounted_return)
        if policy_diff is not None:
            parts.append("policy_diff: %.4f" % policy_diff)
        parts.append("converged: %s" % converged)
        return " | ".join(parts)

    def _format_q_table(self, q_values: Q) -> str:
        q_array = np.asarray(q_values, dtype=float)
        if q_array.ndim != 2:
            raise ValueError("q_values must be a 2-D array with shape (num_states, num_actions)")

        num_states, num_actions = q_array.shape
        states = self._states if self._states is not None else list(range(num_states))
        for state in states:
            if state < 0 or state >= num_states:
                raise ValueError("State index %d is outside q_values row range 0..%d" % (state, num_states - 1))

        q_col_width = 10
        q_headers = " ".join("%*s" % (q_col_width, "Q[%d]" % action) for action in range(num_actions))
        header = "%5s | %s | %10s %6s" % ("state", q_headers, "q_max", "argmax")
        separator = "-" * len(header)
        rows = [header, separator]

        for state in states:
            q_row = q_array[state]
            argmax = int(np.argmax(q_row))
            q_max = float(q_row[argmax])
            q_values_text = " ".join("%*.4f" % (q_col_width, q_row[action]) for action in range(num_actions))
            rows.append("%5d | %s | %10.4f %6d" % (state, q_values_text, q_max, argmax))

        return "\n".join(rows)

    def _write_rendered(self, rendered: str) -> None:
        line_count = rendered.count("\n") + 1

        if self._redraw_mode == "frame" and sys.stdout.isatty() and self._rendered_lines > 0:
            sys.stdout.write("\033[%dF\033[J" % self._rendered_lines)

        sys.stdout.write(rendered + "\n")
        sys.stdout.flush()
        self._rendered_lines = line_count


class WandbTrainingLogger(TrainingLogger):
    """Weights & Biases logger for training metrics.

    Expects a W&B run to be active (``wandb.init()`` already called).
    Lifecycle management (init/finish) belongs to the caller.
    """

    def log_iteration(
        self,
        episode: int,
        q_values: Q,
        q_delta: float,
        converged: bool,
        current_alpha: float | None = None,
        current_epsilon: float | None = None,
        mean_q_delta: float | None = None,
        policy_diff: float | None = None,
        discounted_return: float | None = None,
        env_grid: np.ndarray | None = None,
        optimal_policy: dict | None = None,
        agent_start_pos: tuple[int, int] | None = None,
        agent_values: dict | None = None,
        agent_policy: dict | None = None,
    ) -> None:
        import wandb
        from utils.rl_plots import plot_value_and_policy, plot_policy_disagreement
        import matplotlib.pyplot as plt
        
        metrics = {
            "episode": episode,
            "max_dQ": q_delta,
        }
        if mean_q_delta is not None:
            metrics["mean_dQ"] = mean_q_delta
        if current_epsilon is not None:
            metrics["epsilon"] = current_epsilon
        if current_alpha is not None:
            metrics["alpha"] = current_alpha
        if discounted_return is not None:
            metrics["discounted_return"] = discounted_return
        if policy_diff is not None:
            metrics["policy_diff"] = policy_diff
            
        # Generate live plots if we have the necessary data
        if env_grid is not None and agent_values is not None and agent_policy is not None:
            fig_vp, _ = plot_value_and_policy(
                env_grid, agent_values, agent_policy,
                title=f"Live Value and Policy (Ep {episode})",
                agent_start_pos=agent_start_pos,
            )
            metrics["Live Value and Policy"] = wandb.Image(fig_vp)
            plt.close(fig_vp)
            
            if optimal_policy is not None:
                fig_diff, _ = plot_policy_disagreement(
                    env_grid, optimal_policy, agent_policy,
                    title=f"Live Policy Disagreement (Ep {episode})",
                    agent_start_pos=agent_start_pos,
                )
                metrics["Live Policy Disagreement"] = wandb.Image(fig_diff)
                plt.close(fig_diff)
            
        wandb.log(metrics)

