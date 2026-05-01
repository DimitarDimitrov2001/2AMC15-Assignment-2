"""
- ``plot_value_function``      -- colour heatmap of V(s) or max_a Q(s,a).
- ``plot_policy``               -- arrow map of the greedy policy.
- ``plot_value_and_policy``     -- side-by-side combination of the above two.
- ``plot_algorithm_comparison`` -- overlaid learning curves for DP / MC / TD.
- ``plot_hyperparameter_comparison`` -- grid of learning curves across multiple hyperparameter settings.
---------------------
The environment stores the grid as ``grid[col, row]`` and represents agent
positions as ``(col, row)`` where col=0 is the left edge and row=0 is the
top edge (row increases downward).  All dicts passed to this module must
follow the same convention:
    values : dict[(col, row) -> float]
    policy : dict[(col, row) -> int]   # action ints: 0=Down 1=Up 2=Left 3=Right

"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import TrainingHistory, _ensure_history, _moving_average, _DEFAULT_PALETTE

# ---------------------------------------------------------------------------
# Grid cell constants  (must match world/grid.py)
# ---------------------------------------------------------------------------
_EMPTY    = 0
_BOUNDARY = 1
_OBSTACLE = 2
_TARGET   = 3

# Action integer -> (delta_col, delta_row)  (matches world/helpers.py)
_ACTION_TO_DELTA: dict[int, tuple[int, int]] = {
    0: ( 0,  1),  # Down
    1: ( 0, -1),  # Up
    2: (-1,  0),  # Left
    3: ( 1,  0),  # Right
}

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_WALL_COLOR     = "#2B2B2B"   # boundary
_OBSTACLE_COLOR = "#7A7A7A"   # internal obstacle
_TARGET_COLOR   = "#2F8F2F"   # delivery target
_EMPTY_COLOR    = "#F0F0EC"   # walkable floor
_START_COLOR    = "#E8C13A"   # agent start marker


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)


def _draw_grid_background(ax: plt.Axes, grid: np.ndarray) -> None:
    """
    Args:
        ax:   Matplotlib axes (must have been created before calling this).
        grid: Grid array with shape ``(n_cols, n_rows)`` and integer cell codes.
    """
    n_cols, n_rows = grid.shape

    # Build (n_rows, n_cols, 4) RGBA image — note the transpose for imshow
    img = np.zeros((n_rows, n_cols, 4), dtype=float)

    cell_rgba = {
        _BOUNDARY: _hex_to_rgba(_WALL_COLOR),
        _OBSTACLE: _hex_to_rgba(_OBSTACLE_COLOR),
        _TARGET:   _hex_to_rgba(_TARGET_COLOR, alpha=0.6),
        _EMPTY:    _hex_to_rgba(_EMPTY_COLOR),
    }
    for cell_type, rgba in cell_rgba.items():
        # grid[col, row] → grid.T[row, col]
        mask = (grid == cell_type).T
        img[mask] = rgba

    ax.imshow(img, aspect="equal", origin="upper", interpolation="nearest", zorder=0)


def _mark_start(ax: plt.Axes, agent_start_pos: tuple[int, int]) -> None:
    """Draw a yellow 'S' marker at the agent start position."""
    col, row = agent_start_pos
    circle = plt.Circle((col, row), 0.35, facecolor=_START_COLOR, edgecolor="black",
                         linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(col, row, "S", ha="center", va="center",
            fontsize=7, fontweight="bold", color="black", zorder=4)


def _configure_grid_axes(ax: plt.Axes, n_cols: int, n_rows: int) -> None:
    """Set tick marks, labels, and grid lines for a grid-world axes."""
    ax.set_xlabel("Column", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.tick_params(labelsize=6)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # keep origin='upper' convention


# ---------------------------------------------------------------------------
# plot_value_function
# ---------------------------------------------------------------------------

def plot_value_function(
    grid: np.ndarray,
    values: dict[tuple[int, int], float],
    title: str = "Value Function",
    agent_start_pos: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "RdYlGn",
) -> tuple[plt.Figure, plt.Axes]:
    """Colour heatmap of the learned value function overlaid on the grid.

    Args:
        grid:            Grid array with shape ``(n_cols, n_rows)``.
        values:          Mapping ``(col, row) -> float``.  Typically V(s) from
                         DP or ``max_a Q(s, a)`` from MC / TD.
        title:           Axes title.
        agent_start_pos: If given, draws a yellow 'S' marker at this position.
                         Follows the ``(col, row)`` convention.
        ax:              Existing axes to draw into.  When ``None`` a new
                         figure is created and returned.
        cmap:            Matplotlib colormap name for the value heatmap.

    Returns:
        ``(fig, ax)``
    """
    import matplotlib.colors as mcolors

    n_cols, n_rows = grid.shape

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.5), max(5, n_rows * 0.5)),
                               constrained_layout=True)
    else:
        fig = ax.get_figure()

    # --- background (walls / obstacles / targets / empty) ---
    _draw_grid_background(ax, grid)

    # --- value heatmap (transparent where no value exists) ---
    val_arr = np.full((n_cols, n_rows), np.nan)
    for (col, row), v in values.items():
        if 0 <= col < n_cols and 0 <= row < n_rows:
            val_arr[col, row] = v

    masked = np.ma.masked_invalid(val_arr.T)   # (n_rows, n_cols) for imshow

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color=(0.0, 0.0, 0.0, 0.0))    # NaN → fully transparent

    # Stretch colormap across the actual data range: min → red, max → green.
    valid_vals = val_arr[np.isfinite(val_arr)]
    norm = None
    if len(valid_vals) > 0:
        norm = mcolors.Normalize(vmin=float(valid_vals.min()),
                                 vmax=float(valid_vals.max()))

    im = ax.imshow(
        masked,
        cmap=cm,
        norm=norm,
        aspect="equal",
        origin="upper",
        interpolation="nearest",
        alpha=0.80,
        zorder=1,
    )

    # --- colorbar ---
    plt.colorbar(im, ax=ax, label="State value", shrink=0.75, pad=0.02)

    # --- start marker ---
    if agent_start_pos is not None:
        _mark_start(ax, agent_start_pos)

    ax.set_title(title, fontsize=11)
    _configure_grid_axes(ax, n_cols, n_rows)

    return fig, ax


# ---------------------------------------------------------------------------
# plot_policy
# ---------------------------------------------------------------------------

def plot_policy(
    grid: np.ndarray,
    policy: dict[tuple[int, int], int],
    title: str = "Policy",
    agent_start_pos: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
    arrow_color: str = "#1B6CA8",
) -> tuple[plt.Figure, plt.Axes]:
    """Arrow map of the greedy policy overlaid on the grid.

    Args:
        grid:            Grid array with shape ``(n_cols, n_rows)``.
        policy:          Mapping ``(col, row) -> action_int``.
                         Action integers: 0=Down 1=Up 2=Left 3=Right.
        title:           Axes title.
        agent_start_pos: If given, draws a yellow 'S' marker.
        ax:              Existing axes to draw into.  ``None`` creates a new figure.
        arrow_color:     Hex or named colour for the arrows.

    Returns:
        ``(fig, ax)``
    """
    n_cols, n_rows = grid.shape

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.5), max(5, n_rows * 0.5)),
                               constrained_layout=True)
    else:
        fig = ax.get_figure()

    _draw_grid_background(ax, grid)

    # --- collect arrow data ---
    x_list, y_list, u_list, v_list = [], [], [], []
    for (col, row), action in policy.items():
        if not (0 <= col < n_cols and 0 <= row < n_rows):
            continue
        if grid[col, row] in (_BOUNDARY, _OBSTACLE):
            continue
        dc, dr = _ACTION_TO_DELTA[action]
        x_list.append(col)
        y_list.append(row)
        u_list.append(dc)
        v_list.append(dr)

    if x_list:
        # scale=2: each unit-vector arrow spans 0.5 cells, leaving breathing room
        ax.quiver(
            x_list, y_list, u_list, v_list,
            angles="xy",
            scale_units="xy",
            scale=2.2,
            color=arrow_color,
            width=0.004,
            headwidth=4,
            headlength=5,
            zorder=2,
        )

    if agent_start_pos is not None:
        _mark_start(ax, agent_start_pos)

    ax.set_title(title, fontsize=11)
    _configure_grid_axes(ax, n_cols, n_rows)

    return fig, ax


# ---------------------------------------------------------------------------
# plot_value_and_policy
# ---------------------------------------------------------------------------

def plot_value_and_policy(
    grid: np.ndarray,
    values: dict[tuple[int, int], float],
    policy: dict[tuple[int, int], int],
    title: str = "",
    agent_start_pos: tuple[int, int] | None = None,
    cmap: str = "RdYlGn",
) -> tuple[plt.Figure, np.ndarray]:
    """Side-by-side value heatmap and policy arrow map.

    Args:
        grid:            Grid array with shape ``(n_cols, n_rows)``.
        values:          Mapping ``(col, row) -> float`` (value function).
        policy:          Mapping ``(col, row) -> int`` (greedy policy).
        title:           Overall figure super-title.  Leave blank to omit.
        agent_start_pos: Start position marker ``(col, row)``.
        cmap:            Colormap for the value heatmap.

    Returns:
        ``(fig, axes)`` where ``axes`` has shape ``(2,)`` —
        ``axes[0]`` is the value plot, ``axes[1]`` is the policy plot.
    """
    n_cols, n_rows = grid.shape
    panel_w = max(5.0, n_cols * 0.45)
    panel_h = max(4.0, n_rows * 0.45)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(panel_w * 2 + 1.0, panel_h),
        constrained_layout=True,
    )

    plot_value_function(grid, values,
                        title="Value Function",
                        agent_start_pos=agent_start_pos,
                        ax=axes[0], cmap=cmap)

    plot_policy(grid, policy,
                title="Policy",
                agent_start_pos=agent_start_pos,
                ax=axes[1])

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    return fig, axes


# ---------------------------------------------------------------------------
# plot_algorithm_comparison
# ---------------------------------------------------------------------------

def plot_algorithm_comparison(
    histories: dict[str, TrainingHistory | dict[str, Any]],
    metrics: list[str] | str | None = None,
    smoothing_window: int = 1,
    title: str = "Algorithm Comparison",
    log_scale: bool = False,
    convergence_threshold: float | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Overlay training curves for multiple algorithms on the same axes.

    Unlike ``plot_training_histories`` (which gives each run its own panel),
    this function draws all algorithms onto the same subplot per metric so
    convergence speed and final performance can be compared directly.

    Args:
        histories:             Mapping of algorithm name -> ``TrainingHistory``
                               or plain dict.  Example::

                                   {
                                       "Value Iteration": h_dp,
                                       "On-policy MC":    h_mc,
                                       "Q-Learning":      h_td,
                                   }

        metrics:               Metric name(s) to plot.  Accepts a single string,
                               a list of strings, or ``None`` (plots all metrics
                               found in the first history).
        smoothing_window:      Moving average window applied to each curve
                               (1 = no smoothing).
        title:                 Figure super-title.
        log_scale:             Apply log y-scale to all subplots.  Useful when
                               plotting ``delta_q`` which decays over many orders
                               of magnitude.
        convergence_threshold: If given, draws a horizontal dashed red reference
                               line at this value on every subplot.  Handy for
                               showing the convergence criterion used during
                               training.
        figsize:               ``(width, height)`` override.  Auto-sized when
                               ``None``.

    Returns:
        ``(fig, axes)`` where ``axes`` has shape ``(n_metrics,)``.
    """
    if not histories:
        raise ValueError("histories dict is empty")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1")

    resolved: dict[str, TrainingHistory] = {
        name: _ensure_history(h) for name, h in histories.items()
    }

    first = next(iter(resolved.values()))
    if metrics is None:
        metric_list = list(first.metrics.keys())
    elif isinstance(metrics, str):
        metric_list = [metrics]
    else:
        metric_list = list(metrics)

    if not metric_list:
        raise ValueError("No metrics to plot.")

    n_metrics = len(metric_list)
    if figsize is None:
        figsize = (10, 3.5 * n_metrics)

    fig, axes = plt.subplots(
        n_metrics, 1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat: np.ndarray = axes.ravel()

    algo_colors = {
        name: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
        for i, name in enumerate(resolved)
    }

    for m_idx, metric_name in enumerate(metric_list):
        ax: plt.Axes = axes_flat[m_idx]
        has_data = False

        for algo_name, hist in resolved.items():
            if metric_name not in hist.metrics:
                continue

            raw = hist.metrics[metric_name]
            smoothed = _moving_average(raw, smoothing_window)
            color = algo_colors[algo_name]

            # faint raw trace
            ax.plot(hist.episodes, raw,
                    color=color, linewidth=0.8, alpha=0.25)
            # bold smoothed trace
            ax.plot(hist.episodes, smoothed,
                    color=color, linewidth=2.0, label=algo_name)
            has_data = True

        if convergence_threshold is not None:
            ax.axhline(
                convergence_threshold,
                color="#E74C3C",
                linestyle="--",
                linewidth=1.4,
                label="Threshold (%.0e)" % convergence_threshold,
            )

        if log_scale:
            ax.set_yscale("log")

        ax.set_ylabel(metric_name, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.35)
        if has_data:
            ax.legend(loc="best", fontsize=8)

    axes_flat[-1].set_xlabel("Episode", fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    return fig, axes_flat


# ---------------------------------------------------------------------------
# plot_hyperparameter_comparison
# ---------------------------------------------------------------------------

def plot_hyperparameter_comparison(
    conditions: dict[str, dict[str, TrainingHistory | dict[str, Any]]],
    metrics: list[str] | str | None = None,
    smoothing_window: int = 1,
    title: str = "Hyperparameter Comparison",
    log_scale: bool = False,
    convergence_threshold: float | None = None,
    common_scale: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare all algorithms across multiple experimental conditions.

    Creates a grid of subplots with one column per condition and one
    row per metric.  All algorithms are overlaid in each cell with the
    same colours across columns, making it easy to see how changing one
    hyperparameter affects each algorithm's learning.

    Typical use — effect of stochasticity::

        conditions = {
            "σ = 0.02": {"Value Iteration": h_dp1, "MC": h_mc1, "Q-Learning": h_td1},
            "σ = 0.50": {"Value Iteration": h_dp2, "MC": h_mc2, "Q-Learning": h_td2},
        }
        fig, axes = plot_hyperparameter_comparison(
            conditions,
            metrics=["avg_reward", "delta_q"],
            title="Effect of Stochasticity  (γ=0.9, A1 grid)",
        )

    Args:
        conditions:            Ordered dict mapping a condition label (e.g.
                               ``"σ=0.02"``) to a ``{algo_name: history}`` dict
                               exactly as accepted by ``plot_algorithm_comparison``.
                               All conditions should contain the same algorithm names.
        metrics:               Metric name(s) to plot.  ``None`` → all metrics
                               from the first history in the first condition.
        smoothing_window:      Moving-average window (1 = no smoothing).
        title:                 Figure super-title.
        log_scale:             Log y-scale on all subplots (useful for delta_q).
        convergence_threshold: Horizontal red dashed reference line on every subplot.
        common_scale:          If ``True``, the y-axis range is shared across all
                               columns for the same metric row so conditions are
                               directly comparable.  Disable if scales differ wildly.

    Returns:
        ``(fig, axes)`` where ``axes`` has shape ``(n_metrics, n_conditions)``.
        ``axes[metric_row, condition_col]`` gives the individual ``Axes`` object.
    """
    if not conditions:
        raise ValueError("conditions dict is empty")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1")

    condition_labels = list(conditions.keys())
    n_conditions = len(condition_labels)

    # Resolve all histories
    resolved_conditions: dict[str, dict[str, TrainingHistory]] = {
        cond: {algo: _ensure_history(h) for algo, h in algo_dict.items()}
        for cond, algo_dict in conditions.items()
    }

    # Determine metrics from first available history
    first_hist = next(iter(next(iter(resolved_conditions.values())).values()))
    if metrics is None:
        metric_list = list(first_hist.metrics.keys())
    elif isinstance(metrics, str):
        metric_list = [metrics]
    else:
        metric_list = list(metrics)

    n_metrics = len(metric_list)

    # Consistent colour per algorithm across all conditions
    all_algo_names: list[str] = list(next(iter(resolved_conditions.values())).keys())
    algo_colors = {
        name: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
        for i, name in enumerate(all_algo_names)
    }

    fig_w = max(5.0 * n_conditions, 8.0)
    fig_h = max(3.5 * n_metrics, 4.0)

    fig, axes_grid = plt.subplots(
        n_metrics, n_conditions,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
        sharey="row" if common_scale else "none",
    )

    for c_idx, cond_label in enumerate(condition_labels):
        algo_dict = resolved_conditions[cond_label]

        for m_idx, metric_name in enumerate(metric_list):
            ax: plt.Axes = axes_grid[m_idx, c_idx]
            has_data = False

            for algo_name, hist in algo_dict.items():
                if metric_name not in hist.metrics:
                    continue

                raw = hist.metrics[metric_name]
                smoothed = _moving_average(raw, smoothing_window)
                color = algo_colors.get(algo_name,
                                        _DEFAULT_PALETTE[0])

                ax.plot(hist.episodes, raw,
                        color=color, linewidth=0.8, alpha=0.20)
                ax.plot(hist.episodes, smoothed,
                        color=color, linewidth=2.0, label=algo_name)
                has_data = True

            if convergence_threshold is not None:
                ax.axhline(
                    convergence_threshold,
                    color="#E74C3C",
                    linestyle="--",
                    linewidth=1.3,
                    label="Threshold",
                )

            if log_scale:
                ax.set_yscale("log")

            # Column header on first row only
            if m_idx == 0:
                ax.set_title(cond_label, fontsize=10, fontweight="bold")

            # Row label on first column only
            if c_idx == 0:
                ax.set_ylabel(metric_name, fontsize=9)

            # x-label on last row only
            if m_idx == n_metrics - 1:
                ax.set_xlabel("Episode", fontsize=9)

            ax.grid(True, linestyle="--", alpha=0.32)
            if has_data and c_idx == n_conditions - 1:
                # Legend only on the rightmost column to avoid repetition
                ax.legend(loc="best", fontsize=7)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig, axes_grid
