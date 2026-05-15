"""General-purpose plotting utilities for training diagnostics.

Provides a standardized ``TrainingHistory`` container and two flexible plot
functions that visualise arbitrary metrics recorded during training runs.

The module is intentionally **format-agnostic**: callers supply a simple dict
(or ``TrainingHistory`` instance) whose shape is documented in the class
docstring.  No knowledge of the underlying algorithm or environment is needed.

Public API
----------
- ``TrainingHistory``  -- typed wrapper around the standardized dict format.
- ``SubplotConfig``    -- per-metric visual overrides (color, log-scale, …).
- ``plot_training_history``   -- single-run figure with one subplot per metric.
- ``plot_training_histories`` -- multi-run panel grid for comparing runs.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Default colour palette (colour-blind friendly, works on white/light grey)
# ---------------------------------------------------------------------------
_DEFAULT_PALETTE: list[str] = [
    "#1B6CA8",  # blue
    "#C26A1B",  # orange
    "#2F8F2F",  # green
    "#9B2D9B",  # purple
    "#CC3333",  # red
    "#1AADAD",  # teal
    "#8C6D31",  # olive
    "#E05BA0",  # pink
]

_RAW_ALPHA_MULTIPLIER: float = 0.55


# ---------------------------------------------------------------------------
# SubplotConfig
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SubplotConfig:
    """Per-metric visual overrides for a single subplot.

    Attributes:
        label:      Legend label.  Defaults to the metric key name.
        color:      Line colour (hex or named).  Auto-picked when ``None``.
        log_scale:  Use logarithmic y-axis.
        symlog:     Use symmetric-log y-axis (handles negatives gracefully).
        threshold:  Draw a horizontal reference line at this value.
        raw_alpha:  Opacity of the un-smoothed (raw) trace.  Set to 0 to hide.
        y_label:    Custom y-axis label.  Defaults to the metric key name.
    """

    label: str | None = None
    color: str | None = None
    log_scale: bool = False
    symlog: bool = False
    threshold: float | None = None
    raw_alpha: float = 0.4
    y_label: str | None = None


# ---------------------------------------------------------------------------
# TrainingHistory
# ---------------------------------------------------------------------------
class TrainingHistory:
    """Typed wrapper around a standardized training-history dict.

    The canonical dict shape accepted by all plotting functions::

        {
            "episodes": [1, 2, 3, ...],        # x-axis values  (required)
            "metrics": {                        # >=1 entry       (required)
                "discounted_return": [float, ...],
                "delta_q":    [float, ...],
            },
            "hyperparams": {                    # optional
                "epsilon": 0.1,
                "alpha":   0.01,
            },
            "metadata": {                       # optional
                "converged": True,
                "num_episodes": 5000,
            },
        }

    You may pass either a plain ``dict`` **or** a ``TrainingHistory`` instance
    to every public function in this module -- both are accepted.

    Attributes:
        episodes:     1-D array of episode / checkpoint indices (x-axis).
        metrics:      Mapping of metric name -> 1-D array (same length as *episodes*).
        hyperparams:  Arbitrary hyperparameter dict used for auto-generated titles.
        metadata:     Arbitrary metadata dict (convergence flag, wall-time, …).
    """

    episodes: np.ndarray
    metrics: dict[str, np.ndarray]
    hyperparams: dict[str, Any]
    metadata: dict[str, Any]

    def __init__(
        self,
        episodes: np.ndarray | list[int] | list[float],
        metrics: dict[str, np.ndarray | list[float]],
        hyperparams: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.episodes = np.asarray(episodes, dtype=float)
        self.metrics = {k: np.asarray(v, dtype=float) for k, v in metrics.items()}
        self.hyperparams = hyperparams or {}
        self.metadata = metadata or {}
        self._validate()

    # -- dict-like access so callers can treat the object as a dict -----------
    def __getitem__(self, key: str) -> Any:
        return {
            "episodes": self.episodes,
            "metrics": self.metrics,
            "hyperparams": self.hyperparams,
            "metadata": self.metadata,
        }[key]

    def __contains__(self, key: object) -> bool:
        return key in {"episodes", "metrics", "hyperparams", "metadata"}

    # -- construction / serialisation -----------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingHistory:
        """Build a ``TrainingHistory`` from a plain dict.

        Raises ``KeyError`` when required keys (``episodes``, ``metrics``) are
        missing, and ``ValueError`` when array lengths are inconsistent.
        """
        return cls(
            episodes=d["episodes"],
            metrics=d["metrics"],
            hyperparams=d.get("hyperparams", {}),
            metadata=d.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to a plain dict (arrays become lists)."""
        return {
            "episodes": self.episodes.tolist(),
            "metrics": {k: v.tolist() for k, v in self.metrics.items()},
            "hyperparams": dict(self.hyperparams),
            "metadata": dict(self.metadata),
        }

    # -- validation -----------------------------------------------------------
    def _validate(self) -> None:
        if len(self.episodes) == 0:
            raise ValueError("'episodes' must not be empty")
        if len(self.metrics) == 0:
            raise ValueError("'metrics' must contain at least one entry")
        n = len(self.episodes)
        for name, values in self.metrics.items():
            if len(values) != n:
                raise ValueError(
                    "Metric %r has length %d but 'episodes' has length %d"
                    % (name, len(values), n)
                )

    def __repr__(self) -> str:
        metric_names = list(self.metrics.keys())
        return (
            "TrainingHistory(n_episodes=%d, metrics=%r, hyperparams=%r)"
            % (len(self.episodes), metric_names, self.hyperparams)
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average while ignoring NaN values.

    Uses convolution to smooth data, handling NaN values gracefully
    by treating them as zero weight and only averaging valid values.
    """
    if window <= 1:
        return values.copy()

    kernel = np.ones(window, dtype=float)
    valid = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values, nan=0.0)

    summed = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid, kernel, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = summed / counts
    smoothed[counts == 0.0] = np.nan
    return smoothed


def _ensure_history(
    h: TrainingHistory | dict[str, Any],
) -> TrainingHistory:
    """Normalise input to a ``TrainingHistory`` instance."""
    if isinstance(h, TrainingHistory):
        return h
    return TrainingHistory.from_dict(h)


def _auto_title_from_hyperparams(hp: dict[str, Any]) -> str:
    """Format a hyperparams dict into a compact one-line title."""
    if not hp:
        return ""
    parts: list[str] = []
    for key, value in hp.items():
        if isinstance(value, float):
            parts.append("%s=%s" % (key, _format_number(value)))
        else:
            parts.append("%s=%s" % (key, value))
    return ", ".join(parts)


def _format_number(value: float) -> str:
    """Pick a compact numeric representation (fixed or scientific)."""
    if value == 0.0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 1000:
        return "%.2e" % value
    return "%.4g" % value


def _pick_color(index: int, config: SubplotConfig | None) -> str:
    if config is not None and config.color is not None:
        return config.color
    return _DEFAULT_PALETTE[index % len(_DEFAULT_PALETTE)]


def _lighter_variant(hex_color: str) -> str:
    """Return a lighter/washed-out version of *hex_color* for the raw trace."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    factor = 0.45
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return "#%02X%02X%02X" % (r, g, b)


# ---------------------------------------------------------------------------
# plot_training_history  (single run)
# ---------------------------------------------------------------------------
def plot_training_history(
    history: TrainingHistory | dict[str, Any],
    smoothing_window: int = 1,
    subplot_config: dict[str, SubplotConfig] | None = None,
    title: str = "Training Progress",
    figsize: tuple[float, float] = (12, 7),
) -> tuple[plt.Figure, np.ndarray, dict[str, np.ndarray]]:
    """Plot one or more metrics from a single training run.

    Creates one vertically-stacked subplot per metric found in
    ``history.metrics``.  Each subplot shows the raw trace (faded) and
    a smoothed trace (bold) produced by a NaN-safe moving average.

    Args:
        history:          A ``TrainingHistory`` or a plain dict with the
                          standardized shape (see ``TrainingHistory`` docstring).
        smoothing_window: Window size for the moving average (1 = no smoothing).
        subplot_config:   Optional mapping of metric name -> ``SubplotConfig``
                          for per-metric visual overrides.  Metrics without an
                          entry get sensible defaults (auto-colour, linear scale).
        title:            Figure super-title.
        figsize:          ``(width, height)`` in inches.

    Returns:
        ``(fig, axes, smoothed_metrics)`` where *smoothed_metrics* maps each
        metric name to its smoothed array.
    """
    h = _ensure_history(history)
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1")

    episodes = h.episodes
    metric_names = list(h.metrics.keys())
    n_subplots = len(metric_names)
    cfg_map = subplot_config or {}

    fig, axes = plt.subplots(
        n_subplots, 1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat: np.ndarray = axes.ravel()

    smoothed_metrics: dict[str, np.ndarray] = {}

    for idx, name in enumerate(metric_names):
        ax: plt.Axes = axes_flat[idx]
        cfg = cfg_map.get(name)
        raw_values = h.metrics[name]
        smoothed = _moving_average(raw_values, smoothing_window)
        smoothed_metrics[name] = smoothed

        color = _pick_color(idx, cfg)
        raw_color = _lighter_variant(color)
        raw_alpha = cfg.raw_alpha if cfg is not None else 0.4
        label = (cfg.label if cfg is not None and cfg.label else name)
        y_label = (cfg.y_label if cfg is not None and cfg.y_label else name)

        if raw_alpha > 0:
            ax.plot(
                episodes, raw_values,
                color=raw_color,
                linewidth=1.2,
                alpha=raw_alpha,
                label=label,
            )
        ax.plot(
            episodes, smoothed,
            color=color,
            linewidth=2.2,
            label="Smoothed (window=%d)" % smoothing_window,
        )

        if cfg is not None and cfg.threshold is not None:
            ax.axhline(
                cfg.threshold,
                color="#E74C3C",
                linestyle="--",
                linewidth=1.5,
                label="Threshold (%.0e)" % cfg.threshold,
            )

        if cfg is not None and cfg.log_scale:
            ax.set_yscale("log")
        elif cfg is not None and cfg.symlog:
            ax.set_yscale("symlog", linthresh=1.0)

        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")

    axes_flat[-1].set_xlabel("Episode")
    fig.suptitle(title, fontsize=14, fontweight="bold")

    return fig, axes_flat, smoothed_metrics


# ---------------------------------------------------------------------------
# plot_training_histories  (multi-run)
# ---------------------------------------------------------------------------
def plot_training_histories(
    histories: list[TrainingHistory | dict[str, Any]],
    metrics_to_plot: list[str] | None = None,
    smoothing_window: int = 1,
    columns: int = 3,
    common_scale: bool = True,
    subplot_config: dict[str, SubplotConfig] | None = None,
    title: str = "Training Runs",
    figsize_per_panel: tuple[float, float] = (4.5, 3.5),
) -> tuple[plt.Figure, np.ndarray]:
    """Create a panel grid with one column-group per training run.

    Each run is plotted in its own panel (or stack of panels when there are
    multiple *metrics_to_plot*).  Panels are arranged in a grid with the
    given number of *columns*.

    Args:
        histories:        List of ``TrainingHistory`` or plain dicts.
        metrics_to_plot:  Metric names to include.  ``None`` means all metrics
                          found in the first history entry.
        smoothing_window: Moving-average window (1 = no smoothing).
        columns:          Number of columns in the panel grid.
        common_scale:     Normalise y-axes across all panels per metric so
                          magnitudes are directly comparable.
        subplot_config:   Optional per-metric ``SubplotConfig`` overrides.
        title:            Figure super-title.
        figsize_per_panel: ``(width, height)`` per individual panel.

    Returns:
        ``(fig, axes_2d)`` where *axes_2d* has shape
        ``(n_runs * n_metrics_rows, columns)``.
    """
    if len(histories) == 0:
        raise ValueError("histories list is empty")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1")

    resolved: list[TrainingHistory] = [_ensure_history(h) for h in histories]

    if metrics_to_plot is None:
        metrics_to_plot = list(resolved[0].metrics.keys())
    if len(metrics_to_plot) == 0:
        raise ValueError("metrics_to_plot is empty (no metrics to display)")

    n_runs = len(resolved)
    n_metrics = len(metrics_to_plot)
    cfg_map = subplot_config or {}

    cols = min(columns, n_runs)
    rows_per_metric = math.ceil(n_runs / cols)
    total_plot_rows = rows_per_metric * n_metrics

    fig_width = max(8.0, figsize_per_panel[0] * cols)
    fig_height = max(6.0, figsize_per_panel[1] * total_plot_rows)

    fig, axes_grid = plt.subplots(
        total_plot_rows,
        cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )

    smoothed_cache: dict[tuple[int, str], np.ndarray] = {}

    for run_idx, hist in enumerate(resolved):
        col = run_idx % cols
        base_row = (run_idx // cols) * n_metrics

        panel_title = _auto_title_from_hyperparams(hist.hyperparams) or "Run %d" % (run_idx + 1)

        for m_idx, metric_name in enumerate(metrics_to_plot):
            row = base_row + m_idx
            ax: plt.Axes = axes_grid[row, col]

            if metric_name not in hist.metrics:
                ax.set_title("%s | %s (N/A)" % (panel_title, metric_name), fontsize=9)
                ax.text(
                    0.5, 0.5, "metric not available",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="grey",
                )
                continue

            cfg = cfg_map.get(metric_name)
            color = _pick_color(m_idx, cfg)
            raw_values = hist.metrics[metric_name]
            smoothed = _moving_average(raw_values, smoothing_window)
            smoothed_cache[(run_idx, metric_name)] = smoothed

            ax.plot(
                hist.episodes, smoothed,
                color=color,
                linewidth=1.8,
            )

            if cfg is not None and cfg.log_scale:
                ax.set_yscale("log")
            elif cfg is not None and cfg.symlog:
                ax.set_yscale("symlog", linthresh=1.0)

            ax.grid(True, linestyle="--", alpha=0.28)

            if m_idx == 0:
                ax.set_title(panel_title, fontsize=9)
            y_label = (cfg.y_label if cfg is not None and cfg.y_label else metric_name)
            ax.set_ylabel(y_label, fontsize=8)

            if row == total_plot_rows - 1 or (base_row + n_metrics - 1) == total_plot_rows - 1:
                ax.set_xlabel("Episode")

    # Hide unused axes
    for row in range(total_plot_rows):
        for col_idx in range(cols):
            run_idx_for_cell = (row // n_metrics) * cols + col_idx
            if run_idx_for_cell >= n_runs:
                axes_grid[row, col_idx].axis("off")

    # Normalise y-axes across panels for each metric
    if common_scale:
        for m_idx, metric_name in enumerate(metrics_to_plot):
            cfg = cfg_map.get(metric_name)
            is_log = cfg is not None and cfg.log_scale

            series = [
                smoothed_cache[(ri, metric_name)]
                for ri in range(n_runs)
                if (ri, metric_name) in smoothed_cache
            ]
            if len(series) == 0:
                continue

            y_min = float(np.nanmin([np.nanmin(s) for s in series]))
            y_max = float(np.nanmax([np.nanmax(s) for s in series]))
            if y_min == y_max:
                y_min -= 1e-6
                y_max += 1e-6

            if not is_log:
                for run_idx in range(n_runs):
                    col = run_idx % cols
                    row = (run_idx // cols) * n_metrics + m_idx
                    if row < total_plot_rows and col < cols:
                        axes_grid[row, col].set_ylim(y_min, y_max)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig, axes_grid
