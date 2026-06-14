"""Visualize a random agent's path on both environments side by side.

Usage:
    python visualize_random_agent.py
    python visualize_random_agent.py --grid grid_configs/A1_grid.npy
    python visualize_random_agent.py --max_steps 300 --seed 5
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from world import MinimalEnvironment, ContinuousEnvironment, GRID_CONFIGS_FP
from agents.random_agent import RandomAgent


CELL_COLORS = {
    0: "#ffffff",   # empty
    1: "#2b2b2b",   # boundary wall
    2: "#888888",   # obstacle
    3: "#00cc44",   # target
}


def run_episode(env, max_steps: int, num_actions: int):
    """Run one episode with a random agent, return continuous path and stats."""
    agent  = RandomAgent(num_actions=num_actions)
    state  = env.reset()
    initial_grid = np.copy(env.grid)
    path   = [env.pos.copy()]

    for _ in range(max_steps):
        action = agent.select_action(state)
        state, _, done, _ = env.step(action)
        path.append(env.pos.copy())
        if done:
            break

    return np.array(path), env.world_stats.copy(), initial_grid


def draw_grid(ax, grid: np.ndarray):
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(mpatches.Rectangle(
                (i, j), 1, 1,
                color=CELL_COLORS.get(int(grid[i, j]), "#ffffff"),
                ec="#cccccc", linewidth=0.4,
            ))
    ax.set_xlim(0, rows)
    ax.set_ylim(0, cols)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_env(ax, path, stats, initial_grid, title, color):
    draw_grid(ax, initial_grid)

    ax.plot(path[:, 0], path[:, 1],
            color=color, linewidth=1.2, alpha=0.7, zorder=2)
    ax.plot(path[0, 0],  path[0, 1],
            "o", color="green", markersize=9,  zorder=3, label="Start")
    ax.plot(path[-1, 0], path[-1, 1],
            "*", color="red",   markersize=13, zorder=3, label="End")

    reached = stats["targets_reached"] > 0
    ax.set_title(
        f"{title}\n"
        f"steps={stats['total_steps']}  "
        f"collisions={stats['total_collisions']}  "
        f"{'REACHED ✓' if reached else 'not reached'}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")


def main():
    args = parse_args()

    # ---- MinimalEnvironment: (x, y), 4 actions -------------------------
    env1 = MinimalEnvironment(args.grid, step_size=args.step_size, random_seed=args.seed)
    path1, stats1, grid1 = run_episode(env1, args.max_steps, 4)

    print("=== MinimalEnvironment (x, y) ===")
    print(f"  state_dim={env1.state_dim}  n_actions={env1.n_actions}")
    print(f"  steps={stats1['total_steps']}  collisions={stats1['total_collisions']}  reached={stats1['targets_reached'] > 0}")

    # ---- ContinuousEnvironment: (x, y, theta, sensors), 3 actions ------
    env2 = ContinuousEnvironment(args.grid, step_size=args.step_size,
                                 rotation_step=args.rotation_step,
                                 random_seed=args.seed,
                                 action_sigma=0.1,
                                 sensory_sigma=0.1)
    path2, stats2, grid2 = run_episode(env2, args.max_steps, 3)

    print("\n=== ContinuousEnvironment (x, y, theta, d0..d7) ===")
    print(f"  state_dim={env2.state_dim}  n_actions={env2.n_actions}")
    print(f"  steps={stats2['total_steps']}  collisions={stats2['total_collisions']}  reached={stats2['targets_reached'] > 0}")

    # ---- Plot side by side ----------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plot_env(ax1, path1, stats1, grid1,
             "MinimalEnvironment\nstate: (x, y)  |  actions: ↑↓←→",
             color="royalblue")

    plot_env(ax2, path2, stats2, grid2,
             "ContinuousEnvironment\nstate: (x, y, θ, d0..d7)  |  actions: rotate / forward",
             color="crimson")

    legend_patches = [
        mpatches.Patch(color=CELL_COLORS[1], label="Wall"),
        mpatches.Patch(color=CELL_COLORS[2], label="Obstacle"),
        mpatches.Patch(color=CELL_COLORS[3], label="Target"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Random agent — {args.grid.name}  (step_size={args.step_size}  seed={args.seed})",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"\nsaved → {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid",          type=Path,  default=GRID_CONFIGS_FP / "small_grid.npy")
    p.add_argument("--max_steps",     type=int,   default=500)
    p.add_argument("--step_size",     type=float, default=0.5)
    p.add_argument("--rotation_step", type=float, default=30.0)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--out",           type=Path,  default=Path("path_random_agent.png"))
    return p.parse_args()


if __name__ == "__main__":
    main()
