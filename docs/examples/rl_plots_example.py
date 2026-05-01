"""Example: using ``utils.rl_plots`` with all three algorithm types.

Run from the project root with uv:

    uv run python docs/examples/rl_plots_example.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import TrainingHistory
from utils.rl_plots import (
    plot_algorithm_comparison,
    plot_hyperparameter_comparison,
    plot_value_and_policy,
    plot_value_function,
    plot_policy,
)


# ---------------------------------------------------------------------------
# Shared synthetic grid  (10 cols x 8 rows)
# grid[col, row]:  0=empty  1=boundary  2=obstacle  3=target
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _make_demo_grid() -> np.ndarray:
    n_cols, n_rows = 10, 8
    g = np.zeros((n_cols, n_rows), dtype=np.int8)
    g[0, :]  = 1;  g[-1, :] = 1   # boundary walls
    g[:, 0]  = 1;  g[:, -1] = 1
    g[3, 2]  = 2;  g[3, 3]  = 2;  g[3, 4] = 2   # obstacles
    g[6, 4]  = 2;  g[6, 5]  = 2
    g[8, 1]  = 3;  g[8, 6]  = 3   # targets
    return g


# ---------------------------------------------------------------------------
# Helpers shared across grid comparisons
# ---------------------------------------------------------------------------

def _synthetic_values_and_policy(
    grid: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[dict, dict]:
    """Build a synthetic value function and greedy policy for any grid.

    Finds targets in the grid, then assigns values based on negative distance
    to the nearest target.  Policy arrows point toward the nearest target.
    This mimics what a converged RL agent would produce.
    """
    n_cols, n_rows = grid.shape
    targets = [(c, r) for c in range(n_cols) for r in range(n_rows)
               if grid[c, r] == 3]

    if not targets:
        return {}, {}

    values: dict[tuple[int, int], float] = {}
    policy: dict[tuple[int, int], int]   = {}

    for col in range(n_cols):
        for row in range(n_rows):
            if grid[col, row] in (1, 2):
                continue
            dist = min(abs(col - tc) + abs(row - tr) for tc, tr in targets)
            values[(col, row)] = -float(dist) + rng.uniform(-0.3, 0.3)

            nearest = min(targets, key=lambda t: abs(col-t[0]) + abs(row-t[1]))
            dc = nearest[0] - col
            dr = nearest[1] - row
            action = 3 if abs(dc) >= abs(dr) and dc > 0 else \
                     2 if abs(dc) >= abs(dr) and dc < 0 else \
                     0 if dr > 0 else 1
            policy[(col, row)] = action

    return values, policy


# ---------------------------------------------------------------------------
# ── ALGORITHM 1: Dynamic Programming ────────────────────────────────────────
#
# DP does NOT run episodes. It sweeps over all states repeatedly until the
# value function converges. delta_q   comes naturally from each Bellman sweep.
# DP typically converges in tens or hundreds of iterations, far fewer than
# the thousands of episodes MC and TD need.  The x-axis scales will differ —
# that is expected and interesting to show in the report.
# ---------------------------------------------------------------------------

def _simulate_dp(
    n_iterations: int,
    gamma: float,
    rng: np.random.RandomState,
) -> tuple[TrainingHistory, dict, dict]:
    """Simulate DP training: Bellman sweeps → converged V and policy."""

    # delta_q: decays exponentially as V converges
    iters = np.arange(1, n_iterations + 1, dtype=float)
    delta_q = np.exp(-iters / (n_iterations * 0.2))
    delta_q += np.abs(rng.normal(scale=0.02, size=n_iterations))
    delta_q = np.clip(delta_q, 1e-6, None)

    # avg_reward: evaluate the policy after convergence.
    # Here we generate one reward value per iteration to keep the history
    # the same length.  In practice you evaluate once at the end.
    avg_reward = -8.0 * np.exp(-iters / (n_iterations * 0.3))
    avg_reward += rng.normal(scale=0.3, size=n_iterations)

    history = TrainingHistory(
        episodes=iters,
        metrics={"avg_reward": avg_reward, "delta_q": delta_q},
        hyperparams={"gamma": gamma, "algorithm": "DP"},
    )

    # Synthetic value function and policy for the grid plot
    grid = _make_demo_grid()
    n_cols, n_rows = grid.shape
    targets = [(8, 1), (8, 6)]

    values = {}
    policy = {}
    for col in range(n_cols):
        for row in range(n_rows):
            if grid[col, row] in (1, 2):
                continue
            dist = min(abs(col - tc) + abs(row - tr) for tc, tr in targets)
            values[(col, row)] = -float(dist) * (1.0 / (1.0 - gamma + 1e-9)) * 0.1
            # greedy toward nearest target
            nearest = min(targets, key=lambda t: abs(col-t[0]) + abs(row-t[1]))
            dc = nearest[0] - col
            dr = nearest[1] - row
            action = 3 if abs(dc) >= abs(dr) and dc > 0 else \
                     2 if abs(dc) >= abs(dr) and dc < 0 else \
                     0 if dr > 0 else 1
            policy[(col, row)] = action

    return history, values, policy


# ---------------------------------------------------------------------------
# ── ALGORITHM 2: Monte Carlo ─────────────────────────────────────────────
#
# MC runs COMPLETE episodes, then updates Q from the returns at episode end.
# This means both avg_reward and delta_q are recorded once per episode.
# ---------------------------------------------------------------------------

def _simulate_mc(
    n_episodes: int,
    gamma: float,
    epsilon: float,
    max_episode_length: int,
    rng: np.random.RandomState,
) -> tuple[TrainingHistory, dict, dict]:
    """Simulate MC training: one reward and delta per episode."""

    eps = np.arange(1, n_episodes + 1, dtype=float)

    # MC is noisy early on (random episodes) and stabilises later
    avg_reward = -15.0 + 12.0 * (1.0 - np.exp(-eps / (n_episodes * 0.4)))
    avg_reward += rng.normal(scale=3.0, size=n_episodes)

    delta_q = np.exp(-eps / (n_episodes * 0.35))
    delta_q *= np.exp(rng.normal(scale=0.5, size=n_episodes))
    delta_q = np.clip(delta_q, 1e-6, None)

    history = TrainingHistory(
        episodes=eps,
        metrics={"avg_reward": avg_reward, "delta_q": delta_q},
        hyperparams={"gamma": gamma, "epsilon": epsilon,
                     "max_ep_len": max_episode_length},
    )

    # Synthetic Q-table → values and policy for grid plot
    grid = _make_demo_grid()
    n_cols, n_rows = grid.shape
    targets = [(8, 1), (8, 6)]
    values, policy = {}, {}
    for col in range(n_cols):
        for row in range(n_rows):
            if grid[col, row] in (1, 2):
                continue
            dist = min(abs(col - tc) + abs(row - tr) for tc, tr in targets)
            values[(col, row)] = -float(dist) * 0.9 + rng.uniform(-0.5, 0.5)
            nearest = min(targets, key=lambda t: abs(col-t[0]) + abs(row-t[1]))
            dc = nearest[0] - col
            dr = nearest[1] - row
            action = 3 if abs(dc) >= abs(dr) and dc > 0 else \
                     2 if abs(dc) >= abs(dr) and dc < 0 else \
                     0 if dr > 0 else 1
            policy[(col, row)] = action

    return history, values, policy


# ---------------------------------------------------------------------------
# ── ALGORITHM 3: Temporal Difference (Q-Learning / SARSA) ───────────────
#
# TD updates Q after EVERY SINGLE STEP (not at episode end like MC).
# avg_reward and delta_q are still recorded once per episode (summed/maxed
# over all steps in that episode).
# ---------------------------------------------------------------------------

def _simulate_td(
    n_episodes: int,
    gamma: float,
    alpha: float,
    epsilon: float,
    rng: np.random.RandomState,
) -> tuple[TrainingHistory, dict, dict]:
    """Simulate TD training: one reward and delta per episode."""

    eps = np.arange(1, n_episodes + 1, dtype=float)

    # TD is faster to converge than MC but still noisy
    avg_reward = -15.0 + 12.5 * (1.0 - np.exp(-eps / (n_episodes * 0.3)))
    avg_reward += rng.normal(scale=2.5, size=n_episodes)

    delta_q = np.exp(-eps / (n_episodes * 0.28))
    delta_q *= np.exp(rng.normal(scale=0.45, size=n_episodes))
    delta_q = np.clip(delta_q, 1e-6, None)

    history = TrainingHistory(
        episodes=eps,
        metrics={"avg_reward": avg_reward, "delta_q": delta_q},
        hyperparams={"gamma": gamma, "alpha": alpha, "epsilon": epsilon},
    )

    grid = _make_demo_grid()
    n_cols, n_rows = grid.shape
    targets = [(8, 1), (8, 6)]
    values, policy = {}, {}
    for col in range(n_cols):
        for row in range(n_rows):
            if grid[col, row] in (1, 2):
                continue
            dist = min(abs(col - tc) + abs(row - tr) for tc, tr in targets)
            values[(col, row)] = -float(dist) * 0.9 + rng.uniform(-0.3, 0.3)
            nearest = min(targets, key=lambda t: abs(col-t[0]) + abs(row-t[1]))
            dc = nearest[0] - col
            dr = nearest[1] - row
            action = 3 if abs(dc) >= abs(dr) and dc > 0 else \
                     2 if abs(dc) >= abs(dr) and dc < 0 else \
                     0 if dr > 0 else 1
            policy[(col, row)] = action

    return history, values, policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.RandomState(42)
    out = Path(tempfile.mkdtemp(prefix="rl_plots_example_"))
    grid    = _make_demo_grid()
    start   = (1, 1)

    # ── Train all three algorithms (synthetic) ──────────────────────────────
    history_dp, values_dp, policy_dp = _simulate_dp(
        n_iterations=200, gamma=0.9, rng=rng,
    )
    history_mc, values_mc, policy_mc = _simulate_mc(
        n_episodes=1000, gamma=0.9, epsilon=0.2,
        max_episode_length=200, rng=rng,
    )
    history_td, values_td, policy_td = _simulate_td(
        n_episodes=1000, gamma=0.9, alpha=0.1, epsilon=0.2, rng=rng,
    )

    # ── Figure 1: Value function + policy for each algorithm ────────────────

    for name, vals, pol in [
        ("dp",  values_dp, policy_dp),
        ("mc",  values_mc, policy_mc),
        ("td",  values_td, policy_td),
    ]:
        fig, _ = plot_value_and_policy(
            grid, vals, pol,
            title=name.upper() + "  —  Value Function & Policy",
            agent_start_pos=start,
        )
        p = out / ("value_policy_%s.png" % name)
        fig.savefig(p, dpi=130, bbox_inches="tight")
        print("%-6s value+policy  →  %s" % (name.upper(), p))
        plt.close(fig)

    # ── Figure 2: All three algorithms on the same axes ─────────────────────

    fig2, _ = plot_algorithm_comparison(
        histories={
            "Value Iteration (DP)": history_dp,
            "On-policy MC":         history_mc,
            "Q-Learning (TD)":      history_td,
        },
        metrics=["avg_reward", "delta_q"],
        smoothing_window=30,
        title="Algorithm Comparison  (γ=0.9, σ=0.1, A1 grid)",
        convergence_threshold=0.01,
    )
    p2 = out / "algorithm_comparison.png"
    fig2.savefig(p2, dpi=130, bbox_inches="tight")
    print("Algorithm comparison       →  %s" % p2)
    plt.close(fig2)

    # ── Figure 3: Effect of a hyperparameter on all three algorithms ─────────

    # Low stochasticity condition
    h_dp_low, _, _ = _simulate_dp(n_iterations=200, gamma=0.9, rng=rng)
    h_mc_low, _, _ = _simulate_mc(n_episodes=1000, gamma=0.9, epsilon=0.2,
                                   max_episode_length=200, rng=rng)
    h_td_low, _, _ = _simulate_td(n_episodes=1000, gamma=0.9,
                                   alpha=0.1, epsilon=0.2, rng=rng)

    # High stochasticity condition — agents struggle more
    h_dp_high, _, _ = _simulate_dp(n_iterations=200, gamma=0.9,
                                    rng=np.random.RandomState(7))
    h_mc_high, _, _ = _simulate_mc(n_episodes=1000, gamma=0.9, epsilon=0.2,
                                    max_episode_length=200,
                                    rng=np.random.RandomState(8))
    h_td_high, _, _ = _simulate_td(n_episodes=1000, gamma=0.9,
                                    alpha=0.1, epsilon=0.2,
                                    rng=np.random.RandomState(9))

    h_dp_high.metrics["avg_reward"] -= 3.0
    h_mc_high.metrics["avg_reward"] -= 6.0
    h_td_high.metrics["avg_reward"] -= 4.5
    h_dp_high.metrics["delta_q"]    *= 2.0
    h_mc_high.metrics["delta_q"]    *= 2.5
    h_td_high.metrics["delta_q"]    *= 2.2

    #the x-axis (episodes) is different for DP vs MC/TD.  This is expected and interesting, maybe to show in the report DP converges much faster than MC/TD.
    fig3, _ = plot_hyperparameter_comparison(
        conditions={
            "σ = 0.02  (low noise)": {
                "Value Iteration (DP)": h_dp_low,
                "On-policy MC":         h_mc_low,
                "Q-Learning (TD)":      h_td_low,
            },
            "σ = 0.50  (high noise)": {
                "Value Iteration (DP)": h_dp_high,
                "On-policy MC":         h_mc_high,
                "Q-Learning (TD)":      h_td_high,
            },
        },
        metrics=["avg_reward", "delta_q"],
        smoothing_window=30,
        title="Experiment 1 — Effect of Stochasticity  (γ=0.9, A1 grid)",
        convergence_threshold=0.01,
        common_scale=True,
    )
    p3 = out / "hyperparameter_comparison.png"
    fig3.savefig(p3, dpi=130, bbox_inches="tight")
    print("Hyperparameter comparison  →  %s" % p3)
    plt.close(fig3)

    # ── Figure 4: Compare results across different grids ─────────────────────

    grids_to_compare = {
        "small_grid  (8×8)":  ("grid_configs/small_grid.npy",  (1, 1)),
        "A1_grid  (15×15)":   ("grid_configs/A1_grid.npy",     (1, 12)),
    }

    # One row per grid, two columns: value function | policy
    fig4, axes4 = plt.subplots(
        len(grids_to_compare), 2,
        figsize=(14, 5 * len(grids_to_compare)),
        constrained_layout=True,
    )
    fig4.suptitle("Grid Comparison — Q-Learning (γ=0.9, σ=0.1)", fontsize=13,
                  fontweight="bold")

    for row_idx, (label, (grid_path, start_pos)) in enumerate(grids_to_compare.items()):
        grid_arr = np.load(PROJECT_ROOT / grid_path)
        values_g, policy_g = _synthetic_values_and_policy(grid_arr, rng)

        plot_value_function(
            grid_arr, values_g,
            title="%s — Value Function" % label,
            agent_start_pos=start_pos,
            ax=axes4[row_idx, 0],
        )
        plot_policy(
            grid_arr, policy_g,
            title="%s — Policy" % label,
            agent_start_pos=start_pos,
            ax=axes4[row_idx, 1],
        )

    p4 = out / "grid_comparison.png"
    fig4.savefig(p4, dpi=130, bbox_inches="tight")
    print("Grid comparison            →  %s" % p4)
    plt.close(fig4)

    print("\nAll figures written to:  %s" % out)


if __name__ == "__main__":
    main()
