"""Systematic hyperparameter sweep: vary one parameter at a time across agents and grids.

Usage:
    python run_experiments.py [--out_dir results/experiments] [--grid GRID ...]

For each experiment the script trains all three algorithms (value_iteration,
q_learning, mc) on each grid, evaluates them, writes a CSV summary, and saves:
  - metric bar charts (one PNG per grid)
  - training-curve comparison plots (one PNG per hyperparameter group per grid)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from agents.mc_agent import MCAgent
from agents.q_learning_agent import QLearningAgent
from agents.value_iteration_agent import ValueIterationAgent
from utils.evaluation import evaluate_policy_metrics
from utils.rl_plots import plot_hyperparameter_comparison
from world import Environment, build_manhattan_reward_function, find_target_position

# ─── Default hyperparameters ──────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "sigma":              0.1,
    "gamma":              0.9,
    "eval_episodes":      20,
    "max_steps":          500,
    "random_seed":        0,
    "alpha":              0.5,
    "epsilon":            1.0,
    "epsilon_min":        0.05,
    "epsilon_decay":      0.995,
    "alpha_min":          0.05,
    "alpha_decay":        0.999,
    "fixed_epsilon":      False,
    "fixed_alpha":        False,
    "ql_episodes":        3000,
    "mc_episodes":        5000,
    "max_episode_length": 2000,
}

DEFAULT_GRIDS = [
    Path("grid_configs/A1_grid.npy"),
    Path("grid_configs/example_grid.npy"),
]

# ─── Experiment definitions ───────────────────────────────────────────────────

EXPERIMENTS: list[tuple[str, dict[str, Any]]] = [
    ("default",           {}),
    ("sigma=0.0",         {"sigma": 0.0}),
    ("sigma=0.3",         {"sigma": 0.3}),
    ("gamma=0.5",         {"gamma": 0.5}),
    ("gamma=0.99",        {"gamma": 0.99}),
    ("alpha=0.1",         {"alpha": 0.1}),
    ("alpha=0.9",         {"alpha": 0.9}),
    ("epsilon=0.1",       {"epsilon": 0.1}),
    ("epsilon=0.5",       {"epsilon": 0.5}),
    ("decay_epsilon",     {"fixed_epsilon": False}),
    ("fixed_epsilon",     {"fixed_epsilon": True}),
    ("decay_alpha",       {"fixed_alpha": False}),
    ("fixed_alpha",       {"fixed_alpha": True}),
    ("mc_ep_len=200",     {"max_episode_length": 200}),
    ("mc_ep_len=5000",    {"max_episode_length": 5000}),
]

# Groups for training-curve comparison plots (each group = two conditions to compare).
TRAINING_CURVE_GROUPS: list[tuple[str, list[str], list[str], list[str] | None]] = [
    ("sigma",            ["sigma=0.0",    "sigma=0.3"],        ["avg_reward"],            None),
    ("gamma",            ["gamma=0.5",    "gamma=0.99"],       ["avg_reward"],            None),
    ("alpha",            ["alpha=0.1",    "alpha=0.9"],        ["avg_reward"],            None),
    ("epsilon",          ["epsilon=0.1",  "epsilon=0.5"],      ["avg_reward"],            None),
    ("epsilon_schedule", ["decay_epsilon","fixed_epsilon"],     ["avg_reward", "epsilon"], None),
    ("alpha_schedule",   ["decay_alpha",  "fixed_alpha"],      ["avg_reward"],            None),
    ("mc_ep_len",        ["mc_ep_len=200","mc_ep_len=5000"],   ["avg_reward"],            ["mc"]),
]

# Grouping of experiments on the bar-chart x-axis
BAR_CHART_GROUPS: list[tuple[str, list[str]]] = [
    ("baseline",  ["default"]),
    ("sigma",     ["sigma=0.0", "sigma=0.3"]),
    ("gamma",     ["gamma=0.5", "gamma=0.99"]),
    ("alpha",     ["alpha=0.1", "alpha=0.9"]),
    ("epsilon",   ["epsilon=0.1", "epsilon=0.5"]),
    ("ε sched.",  ["decay_epsilon", "fixed_epsilon"]),
    ("α sched.",  ["decay_alpha", "fixed_alpha"]),
    ("mc ep_len", ["mc_ep_len=200", "mc_ep_len=5000"]),
]

EVAL_METRICS = [
    "success_rate",
    "mean_discounted_return",
    "mean_episode_length",
    "policy_difference_from_optimal",
]

ALGO_LABELS = {"value_iteration": "VI", "q_learning": "Q-Learning", "mc": "MC"}
ALGO_COLORS = {"value_iteration": "#1B6CA8", "q_learning": "#C26A1B", "mc": "#2F8F2F"}

CSV_FIELDS = [
    "experiment", "algorithm", "grid",
    "sigma", "gamma", "alpha", "epsilon",
    "fixed_epsilon", "fixed_alpha", "max_episode_length",
    "success_rate", "mean_discounted_return", "mean_undiscounted_return",
    "mean_episode_length", "mean_success_episode_length",
    "policy_difference_from_optimal", "training_time_s",
]

# ─── VI cache ─────────────────────────────────────────────────────────────────

_vi_cache: dict[tuple, ValueIterationAgent] = {}


def _get_vi_agent(grid_path, grid_array, reward_fn, sigma, gamma) -> ValueIterationAgent:
    key = (str(grid_path), sigma, gamma)
    if key not in _vi_cache:
        agent = ValueIterationAgent(grid=grid_array, reward_fn=reward_fn, sigma=sigma, gamma=gamma)
        agent.train()
        _vi_cache[key] = agent
    return _vi_cache[key]


# ─── Policy-difference metric ─────────────────────────────────────────────────

def _policy_difference(vi_policy: dict, agent) -> float:
    states = list(vi_policy.keys())
    if not states:
        return float("nan")
    mismatches = 0
    for state in states:
        vi_action = vi_policy[state]
        if isinstance(agent, QLearningAgent):
            learned = int(np.argmax(agent.q_table[state]))
        elif isinstance(agent, MCAgent):
            learned = agent.greedy_action(state)
        else:
            return float("nan")
        if learned != vi_action:
            mismatches += 1
    return mismatches / len(states)


# ─── Training functions ───────────────────────────────────────────────────────

def _train_value_iteration(
    grid_array: np.ndarray, reward_fn, cfg: dict
) -> tuple[ValueIterationAgent, dict | None]:
    agent = ValueIterationAgent(
        grid=grid_array, reward_fn=reward_fn,
        sigma=cfg["sigma"], gamma=cfg["gamma"],
    )
    agent.train()
    history = agent.history.to_dict() if agent.history is not None else None
    return agent, history


def _train_q_learning(
    grid_path: Path, reward_fn, start_pos: tuple, cfg: dict
) -> tuple[QLearningAgent, dict]:
    env = Environment(
        grid_fp=grid_path, no_gui=True, reward_fn=reward_fn,
        sigma=cfg["sigma"], target_fps=-1,
        agent_start_pos=start_pos, random_seed=cfg["random_seed"],
    )
    agent = QLearningAgent(
        alpha=cfg["alpha"], gamma=cfg["gamma"],
        epsilon=cfg["epsilon"], epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        alpha_min=cfg["alpha_min"], alpha_decay=cfg["alpha_decay"],
        decaying_epsilon=not cfg["fixed_epsilon"],
        decaying_alpha=not cfg["fixed_alpha"],
        n_actions=4,
    )
    episode_rewards: list[float] = []
    for _ in trange(cfg["ql_episodes"], desc="    training", leave=False):
        state = env.reset()
        env.reward_fn = reward_fn
        agent.start_episode()
        ep_reward = 0.0
        for _ in range(cfg["max_steps"]):
            action = agent.take_action(state)
            next_state, reward, terminated, _ = env.step(action)
            agent.update(next_state, reward, action, terminated=terminated)
            state = next_state
            ep_reward += reward
            if terminated:
                break
        agent.end_episode()
        episode_rewards.append(ep_reward)
    agent.set_eval_mode()

    history = {
        "episodes": list(range(1, cfg["ql_episodes"] + 1)),
        "metrics": {"avg_reward": episode_rewards},
        "hyperparams": {
            "alpha": cfg["alpha"], "epsilon": cfg["epsilon"],
            "gamma": cfg["gamma"], "sigma": cfg["sigma"],
        },
        "metadata": {},
    }
    return agent, history


def _train_mc(
    grid_path: Path, reward_fn, start_pos: tuple, cfg: dict
) -> tuple[MCAgent, dict | None]:
    env = Environment(
        grid_fp=grid_path, no_gui=True, reward_fn=reward_fn,
        sigma=cfg["sigma"], target_fps=-1,
        agent_start_pos=start_pos, random_seed=cfg["random_seed"],
    )
    env.reset()
    env.reward_fn = reward_fn

    agent = MCAgent(
        gamma=cfg["gamma"],
        epsilon=cfg["epsilon"], epsilon_min=cfg["epsilon_min"],
        epsilon_decay=1.0 if cfg["fixed_epsilon"] else cfg["epsilon_decay"],
        alpha=cfg["alpha"] if cfg["fixed_alpha"] else None,
        alpha_min=cfg["alpha_min"],
        alpha_decay=1.0 if cfg["fixed_alpha"] else cfg["alpha_decay"],
        max_episode_length=cfg["max_episode_length"],
        random_seed=cfg["random_seed"],
    )
    agent.train(env, n_episodes=cfg["mc_episodes"],
                start_pos=start_pos, verbose=False, reward_fn=reward_fn)
    history = agent.history.to_dict() if agent.history is not None else None
    return agent, history


# ─── Single run ───────────────────────────────────────────────────────────────

def _run_one(
    experiment: str, algorithm: str, grid_path: Path, cfg: dict
) -> tuple[dict, dict | None]:
    grid_array = np.load(grid_path)
    target_pos = find_target_position(grid_array)

    bootstrap_env = Environment(
        grid_fp=grid_path, no_gui=True, reward_fn=lambda _g, _p: 0,
        sigma=cfg["sigma"], target_fps=-1,
        agent_start_pos=None, random_seed=cfg["random_seed"],
    )
    start_pos = bootstrap_env.reset()
    reward_fn = build_manhattan_reward_function(start_pos, target_pos)

    t0 = time.perf_counter()
    if algorithm == "value_iteration":
        agent, history = _train_value_iteration(grid_array, reward_fn, cfg)
    elif algorithm == "q_learning":
        agent, history = _train_q_learning(grid_path, reward_fn, start_pos, cfg)
    elif algorithm == "mc":
        agent, history = _train_mc(grid_path, reward_fn, start_pos, cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    training_time = time.perf_counter() - t0

    metrics = evaluate_policy_metrics(
        grid=grid_path, agent=agent, max_steps=cfg["max_steps"],
        sigma=cfg["sigma"], agent_start_pos=start_pos,
        reward_fn=reward_fn, gamma=cfg["gamma"],
        random_seed=cfg["random_seed"], n_eval_episodes=cfg["eval_episodes"],
    )

    if algorithm == "value_iteration":
        policy_diff = 0.0
    else:
        vi = _get_vi_agent(grid_path, grid_array, reward_fn, cfg["sigma"], cfg["gamma"])
        policy_diff = _policy_difference(vi.policy, agent)

    def _fmt(v):
        return "" if v is None else round(float(v), 5)

    row = {
        "experiment": experiment, "algorithm": algorithm, "grid": grid_path.stem,
        "sigma": cfg["sigma"], "gamma": cfg["gamma"],
        "alpha": cfg["alpha"], "epsilon": cfg["epsilon"],
        "fixed_epsilon": cfg["fixed_epsilon"], "fixed_alpha": cfg["fixed_alpha"],
        "max_episode_length": cfg["max_episode_length"],
        "success_rate":                 _fmt(metrics["success_rate"]),
        "mean_discounted_return":       _fmt(metrics["mean_discounted_return"]),
        "mean_undiscounted_return":     _fmt(metrics["mean_undiscounted_return"]),
        "mean_episode_length":          _fmt(metrics["mean_episode_length"]),
        "mean_success_episode_length":  _fmt(metrics["mean_success_episode_length"]),
        "policy_difference_from_optimal": _fmt(policy_diff),
        "training_time_s":              round(training_time, 2),
    }
    return row, history

# ─── Plotting: training-curve comparison ──────────────────────────────────────

def _save_training_curve_plots(
    all_histories: dict[str, dict[str, dict[str, dict | None]]],
    out_dir: Path,
    algorithms: list[str],
) -> None:
    """For each hyperparameter group and algorithm, save a plot_hyperparameter_comparison.

    Each algorithm gets its own figure so that Q-learning curves (ql_episodes long)
    and MC curves (mc_episodes long) are never overlaid on the same x-axis.
    """
    # VI has no episode-based training curves (only delta_v convergence sweeps)
    curve_algos = [a for a in algorithms if a != "value_iteration"]

    for grid_stem, grid_hists in all_histories.items():
        for group_name, exp_names, metrics, algo_filter in TRAINING_CURVE_GROUPS:
            group_algos = [a for a in curve_algos if algo_filter is None or a in algo_filter]
            for algo in group_algos:
                # Build conditions: one entry per experiment condition, single algorithm
                conditions: dict[str, dict] = {}
                for exp_name in exp_names:
                    hist = grid_hists.get(exp_name, {}).get(algo)
                    if hist is None:
                        continue
                    if any(m in hist.get("metrics", {}) for m in metrics):
                        conditions[exp_name] = {ALGO_LABELS.get(algo, algo): hist}

                if len(conditions) < 2:
                    continue

                # Smoothing window relative to this algorithm's episode count
                n_ep = len(next(iter(conditions.values()))[ALGO_LABELS[algo]]["episodes"])
                smoothing = max(1, n_ep // 20)

                try:
                    fig, _ = plot_hyperparameter_comparison(
                        conditions,
                        metrics=metrics,
                        smoothing_window=smoothing,
                        title=f"{group_name} — {ALGO_LABELS[algo]} ({grid_stem})",
                    )
                    out_path = out_dir / f"{grid_stem}_{group_name}_{algo}_curves.png"
                    fig.savefig(out_path, dpi=130, bbox_inches="tight")
                    plt.close(fig)
                    tqdm.write(f"  Saved {out_path.name}")
                except Exception as exc:
                    tqdm.write(f"  [WARN] curve plot skipped ({group_name}/{algo}/{grid_stem}): {exc}")


# ─── Plotting: VI convergence curves ─────────────────────────────────────────

# Only sigma and gamma affect VI — those are the only meaningful comparisons
_VI_CONVERGENCE_GROUPS = [
    ("sigma", ["sigma=0.0", "sigma=0.3"]),
    ("gamma", ["gamma=0.5", "gamma=0.99"]),
]


def _save_vi_convergence_plots(
    all_histories: dict[str, dict[str, dict[str, dict | None]]],
    out_dir: Path,
) -> None:
    """Plot delta_v over Bellman sweeps for sigma and gamma comparisons."""
    for grid_stem, grid_hists in all_histories.items():
        for group_name, exp_names in _VI_CONVERGENCE_GROUPS:
            conditions: dict[str, dict] = {}
            for exp_name in exp_names:
                hist = grid_hists.get(exp_name, {}).get("value_iteration")
                if hist is None or "delta_v" not in hist.get("metrics", {}):
                    continue
                conditions[exp_name] = {"Value Iteration": hist}

            if len(conditions) < 2:
                continue

            try:
                fig, _ = plot_hyperparameter_comparison(
                    conditions,
                    metrics=["delta_v"],
                    smoothing_window=1,
                    title=f"VI convergence — {group_name} ({grid_stem})",
                    log_scale=True,
                )
                out_path = out_dir / f"{grid_stem}_{group_name}_vi_convergence.png"
                fig.savefig(out_path, dpi=130, bbox_inches="tight")
                plt.close(fig)
                tqdm.write(f"  Saved {out_path.name}")
            except Exception as exc:
                tqdm.write(f"  [WARN] VI convergence plot skipped ({group_name}/{grid_stem}): {exc}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for RL agents.")
    parser.add_argument("--grid", type=Path, nargs="+", default=DEFAULT_GRIDS)
    parser.add_argument("--out_dir", type=Path, default=Path("results/experiments"))
    parser.add_argument(
        "--algorithms", nargs="+",
        choices=["value_iteration", "q_learning", "mc"],
        default=["value_iteration", "q_learning", "mc"],
    )
    parser.add_argument("--ql_episodes",   type=int, default=DEFAULTS["ql_episodes"],
                        help="Q-learning training episodes (default 3000; epsilon bottoms out ~ep 585).")
    parser.add_argument("--mc_episodes",   type=int, default=DEFAULTS["mc_episodes"])
    parser.add_argument("--eval_episodes", type=int, default=DEFAULTS["eval_episodes"])
    parser.add_argument("--max_steps",     type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--no_plots",      action="store_true", help="Skip all plot generation.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"
    grids: list[Path] = args.grid
    algorithms: list[str] = args.algorithms

    total_runs = len(EXPERIMENTS) * len(algorithms) * len(grids)
    print(f"Running {len(EXPERIMENTS)} experiments × {len(algorithms)} algorithms × {len(grids)} grids = {total_runs} runs")
    print(f"Results → {csv_path}\n")

    all_rows: list[dict] = []
    # all_histories[grid_stem][exp_name][algo] = history_dict_or_None
    all_histories: dict[str, dict[str, dict[str, dict | None]]] = {}

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        pbar = tqdm(total=total_runs, unit="run")
        for exp_name, overrides in EXPERIMENTS:
            cfg = {**DEFAULTS, **overrides}
            cfg["ql_episodes"]   = args.ql_episodes
            cfg["mc_episodes"]   = args.mc_episodes
            cfg["eval_episodes"] = args.eval_episodes
            cfg["max_steps"]     = args.max_steps

            for grid_path in grids:
                if not grid_path.exists():
                    tqdm.write(f"  [SKIP] grid not found: {grid_path}")
                    pbar.update(len(algorithms))
                    continue

                g = grid_path.stem
                all_histories.setdefault(g, {}).setdefault(exp_name, {})

                for algorithm in algorithms:
                    pbar.set_description(f"{exp_name} / {algorithm} / {g}")
                    try:
                        row, history = _run_one(exp_name, algorithm, grid_path, cfg)
                        writer.writerow(row)
                        f.flush()
                        all_rows.append(row)
                        all_histories[g][exp_name][algorithm] = history
                    except Exception as exc:
                        tqdm.write(f"  [ERROR] {exp_name}/{algorithm}/{g}: {exc}")
                    pbar.update(1)

        pbar.close()

    if args.no_plots:
        print(f"\nDone. Results in {args.out_dir}/")
    else:
        print(f"\nAll runs complete. Generating plots…")
        _save_training_curve_plots(all_histories, args.out_dir, algorithms)
        if "value_iteration" in algorithms:
            _save_vi_convergence_plots(all_histories, args.out_dir)
        print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
