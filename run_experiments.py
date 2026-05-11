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
from tqdm import tqdm

from agents.trainers import (
    TRAINERS,
    TrainConfig,
    policy_disagreement,
    setup_grid_run,
)
from agents.value_iteration_agent import ValueIterationAgent
from utils.evaluation import evaluate_policy_metrics
from utils.plotting import TrainingHistory
from utils.rl_plots import (
    plot_algorithm_comparison,
    plot_hyperparameter_comparison,
    plot_value_and_policy,
)

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

# ─── Sweep values (only edit here) ───────────────────────────────────────────
# Change these lists to control which values get compared.
# Experiment names, folder structure, and plots all update automatically.

SIGMA_VALUES:     list[float] = [0.0, 0.3]
GAMMA_VALUES:     list[float] = [0.5, 0.99]
ALPHA_VALUES:     list[float] = [0.1, 0.9]
EPSILON_VALUES:   list[float] = [0.1, 0.5]
MC_EP_LEN_VALUES: list[int]   = [200, 5000]

# ─── Experiment definitions (auto-built from sweep values) ────────────────────

def _n(v: float | int) -> str:
    """Format a number for experiment names, e.g. 0.30 → '0.3', 200 → '200'."""
    return f"{v:g}"

EXPERIMENTS: list[tuple[str, dict[str, Any]]] = [
    ("default", {}),
    *[(f"sigma={_n(v)}",      {"sigma": v})              for v in SIGMA_VALUES],
    *[(f"gamma={_n(v)}",      {"gamma": v})              for v in GAMMA_VALUES],
    *[(f"alpha={_n(v)}",      {"alpha": v})              for v in ALPHA_VALUES],
    *[(f"epsilon={_n(v)}",    {"epsilon": v})            for v in EPSILON_VALUES],
    ("decay_epsilon",  {"fixed_epsilon": False}),
    ("fixed_epsilon",  {"fixed_epsilon": True}),
    ("decay_alpha",    {"fixed_alpha": False}),
    ("fixed_alpha",    {"fixed_alpha": True}),
    *[(f"mc_ep_len={_n(v)}", {"max_episode_length": v}) for v in MC_EP_LEN_VALUES],
]

# Groups for training-curve comparison plots (each group = two conditions to compare).
TRAINING_CURVE_GROUPS: list[tuple[str, list[str], list[str], list[str] | None]] = [
    ("sigma",            [f"sigma={_n(v)}"      for v in SIGMA_VALUES],     ["avg_reward", "policy_diff"], None),
    ("gamma",            [f"gamma={_n(v)}"      for v in GAMMA_VALUES],     ["avg_reward", "policy_diff"], None),
    ("alpha",            [f"alpha={_n(v)}"      for v in ALPHA_VALUES],     ["avg_reward", "policy_diff"], None),
    ("epsilon",          [f"epsilon={_n(v)}"    for v in EPSILON_VALUES],   ["avg_reward", "policy_diff"], None),
    ("epsilon_schedule", ["decay_epsilon",        "fixed_epsilon"],          ["avg_reward", "policy_diff"], None),
    ("alpha_schedule",   ["decay_alpha",          "fixed_alpha"],            ["avg_reward", "policy_diff"], None),
    ("mc_ep_len",        [f"mc_ep_len={_n(v)}"  for v in MC_EP_LEN_VALUES], ["avg_reward", "policy_diff"], ["mc"]),
]

# Maps each experiment name to its output subfolder.
# Experiments in the same group share one folder (data + plots together).
EXP_GROUPS: list[tuple[str, list[str]]] = [
    ("baseline",         ["default"]),
    ("sigma",            [f"sigma={_n(v)}"     for v in SIGMA_VALUES]),
    ("gamma",            [f"gamma={_n(v)}"     for v in GAMMA_VALUES]),
    ("alpha",            [f"alpha={_n(v)}"     for v in ALPHA_VALUES]),
    ("epsilon",          [f"epsilon={_n(v)}"   for v in EPSILON_VALUES]),
    ("epsilon_schedule", ["decay_epsilon", "fixed_epsilon"]),
    ("alpha_schedule",   ["decay_alpha",   "fixed_alpha"]),
    ("mc_ep_len",        [f"mc_ep_len={_n(v)}" for v in MC_EP_LEN_VALUES]),
]
EXP_TO_GROUP: dict[str, str] = {
    exp: group for group, exps in EXP_GROUPS for exp in exps
}

EVAL_METRICS = [
    "success_rate",
    "mean_discounted_return",
    "mean_episode_length",
    "policy_difference_from_optimal",
]

ALGO_LABELS = {"value_iteration": "VI", "q_learning": "Q-Learning", "mc": "MC"}
ALGO_COLORS = {"value_iteration": "#1B6CA8", "q_learning": "#C26A1B", "mc": "#2F8F2F"}

# Sequential colormaps used to derive same-hue shades when overlaying
# multiple conditions of a single algorithm on one plot.
_ALGO_CMAPS = {
    "value_iteration": "Blues",
    "q_learning":      "Oranges",
    "mc":              "Greens",
}


def _shade_palette(algo: str, n: int) -> list:
    """Return ``n`` evenly spaced shades of the algorithm's canonical hue.

    Shades run light -> dark so the eye reads later conditions as "stronger".
    Falls back to a neutral grey ramp for unknown algorithms.
    """
    cmap = plt.get_cmap(_ALGO_CMAPS.get(algo, "Greys"))
    if n <= 1:
        return [cmap(0.7)]
    return [cmap(t) for t in np.linspace(0.45, 0.9, n)]

CSV_FIELDS = [
    "experiment", "algorithm", "grid",
    "sigma", "gamma", "alpha", "epsilon",
    "fixed_epsilon", "fixed_alpha", "max_episode_length",
    "success_rate", "mean_discounted_return", "mean_undiscounted_return",
    "mean_episode_length", "mean_success_episode_length",
    "policy_difference_from_optimal", "training_time_s",
]

# ─── VI cache ─────────────────────────────────────────────────────────────────
# VI is deterministic in (grid, sigma, gamma), so we train once per key and
# reuse the policy across experiments that share those values.

_vi_cache: dict[tuple, ValueIterationAgent] = {}


def _get_vi_agent(env, reward_fn, grid_path, sigma, gamma) -> ValueIterationAgent:
    """Return the cached VI agent for (grid, sigma, gamma), training on miss."""
    key = (str(grid_path), sigma, gamma)
    if key not in _vi_cache:
        cfg = TrainConfig(
            sigma=sigma, gamma=gamma, max_steps=0, random_seed=0, eval_episodes=0,
        )
        agent, _ = TRAINERS["value_iteration"](env, reward_fn, cfg)
        _vi_cache[key] = agent
    return _vi_cache[key]


# ─── Single run ───────────────────────────────────────────────────────────────

def _cfg_to_train_config(cfg: dict, start_pos: tuple) -> TrainConfig:
    """Build a TrainConfig from the sweep's per-experiment cfg dict."""
    return TrainConfig(
        sigma=cfg["sigma"],
        gamma=cfg["gamma"],
        max_steps=cfg["max_steps"],
        random_seed=cfg["random_seed"],
        eval_episodes=cfg["eval_episodes"],
        start_pos=start_pos,
        alpha=cfg["alpha"],
        alpha_min=cfg["alpha_min"],
        alpha_decay=cfg["alpha_decay"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        fixed_alpha=cfg["fixed_alpha"],
        fixed_epsilon=cfg["fixed_epsilon"],
        ql_episodes=cfg["ql_episodes"],
        mc_episodes=cfg["mc_episodes"],
        max_episode_length=cfg["max_episode_length"],
    )


def _history_to_dict(history) -> dict | None:
    """Normalise a trainer's history to a plain dict for downstream plotting."""
    if history is None:
        return None
    if isinstance(history, TrainingHistory):
        return history.to_dict()
    return history


def _run_one(
    experiment: str, algorithm: str, grid_path: Path, cfg: dict
) -> tuple[dict, dict | None, dict | None]:
    env, start_pos, reward_fn = setup_grid_run(
        grid_path=grid_path,
        sigma=cfg["sigma"],
        fps=-1,
        no_gui=True,
        start_pos=None,
        random_seed=cfg["random_seed"],
    )
    train_cfg = _cfg_to_train_config(cfg, start_pos)

    trainer = TRAINERS.get(algorithm)
    if trainer is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Pre-fetch the VI reference policy for QL/MC so the trainer can record
    # per-episode policy disagreement. VI uses its own policy, so skip there.
    optimal_policy = None
    if algorithm != "value_iteration":
        vi_agent = _get_vi_agent(env, reward_fn, grid_path, cfg["sigma"], cfg["gamma"])
        optimal_policy = vi_agent.policy

    t0 = time.perf_counter()
    agent, history = trainer(env, reward_fn, train_cfg, optimal_policy=optimal_policy)
    training_time = time.perf_counter() - t0
    history = _history_to_dict(history)

    metrics = evaluate_policy_metrics(
        grid=grid_path, agent=agent, max_steps=cfg["max_steps"],
        sigma=cfg["sigma"], agent_start_pos=start_pos,
        reward_fn=reward_fn, gamma=cfg["gamma"],
        random_seed=cfg["random_seed"], n_eval_episodes=cfg["eval_episodes"],
    )

    if algorithm == "value_iteration":
        policy_diff = 0.0
    else:
        policy_diff = policy_disagreement(optimal_policy, agent)

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

    values = getattr(agent, "values", None)
    policy = getattr(agent, "policy", None)
    vp_payload: dict | None = None
    if values and policy:
        vp_payload = {"values": values, "policy": policy, "initial_pos": start_pos}

    return row, history, vp_payload

# ─── Plotting: training-curve comparison ──────────────────────────────────────

def _save_training_curve_plots(
    all_histories: dict[str, dict[str, dict[str, dict | None]]],
    group_dirs: dict[str, Path],
    algorithms: list[str],
) -> None:
    """For each hyperparameter group and algorithm, save a per-algorithm curve plot.

    Each algorithm gets its own figure so that Q-learning curves (ql_episodes long)
    and MC curves (mc_episodes long) are never overlaid on the same x-axis. When
    the group is restricted to a single algorithm (e.g. ``mc_ep_len``), the
    conditions are rendered as overlaid curves on a single axes via
    ``plot_algorithm_comparison`` instead of a per-condition column grid.
    """
    # VI has no episode-based training curves (only delta_v convergence sweeps)
    curve_algos = [a for a in algorithms if a != "value_iteration"]

    for grid_stem, grid_hists in all_histories.items():
        for group_name, exp_names, metrics, algo_filter in TRAINING_CURVE_GROUPS:
            group_algos = [a for a in curve_algos if algo_filter is None or a in algo_filter]
            single_algo = len(group_algos) == 1
            for algo in group_algos:
                conditions: dict[str, dict] = {}
                histories: dict[str, dict] = {}
                for exp_name in exp_names:
                    hist = grid_hists.get(exp_name, {}).get(algo)
                    if hist is None:
                        continue
                    if any(m in hist.get("metrics", {}) for m in metrics):
                        conditions[exp_name] = {ALGO_LABELS.get(algo, algo): hist}
                        histories[exp_name] = hist

                if len(conditions) < 2:
                    continue

                n_ep = len(next(iter(histories.values()))["episodes"])
                smoothing = max(1, n_ep // 20)
                title = f"{group_name} — {ALGO_LABELS[algo]} ({grid_stem})"
                out_path = group_dirs[group_name] / f"{grid_stem}_{algo}_curves.png"

                try:
                    if single_algo:
                        shade_colors = dict(
                            zip(histories, _shade_palette(algo, len(histories)))
                        )
                        fig, _ = plot_algorithm_comparison(
                            histories,
                            metrics=metrics,
                            smoothing_window=smoothing,
                            title=title,
                            colors=shade_colors,
                        )
                    else:
                        fig, _ = plot_hyperparameter_comparison(
                            conditions,
                            metrics=metrics,
                            smoothing_window=smoothing,
                            title=title,
                        )
                    fig.savefig(out_path, dpi=130, bbox_inches="tight")
                    plt.close(fig)
                    tqdm.write(f"  Saved {out_path.name}")
                except Exception as exc:
                    tqdm.write(f"  [WARN] curve plot skipped ({group_name}/{algo}/{grid_stem}): {exc}")


# ─── Plotting: cross-algorithm overlay ───────────────────────────────────────

def _save_algorithm_overlay_plots(
    all_histories: dict[str, dict[str, dict[str, dict | None]]],
    group_dirs: dict[str, Path],
    algorithms: list[str],
) -> None:
    """For each multi-algo group, save a cross-algorithm overlay per condition.

    Each PNG overlays the available algorithms (QL, MC; VI is excluded
    because it has no per-episode metrics) on one axes per metric for a
    single experimental condition. Algorithms keep their canonical
    ``ALGO_COLORS`` hues so identity is readable across all overlays.
    Single-algo groups are skipped — ``_save_training_curve_plots`` already
    overlays their conditions on a single axes.
    """
    curve_algos = [a for a in algorithms if a != "value_iteration"]

    for grid_stem, grid_hists in all_histories.items():
        for group_name, exp_names, metrics, algo_filter in TRAINING_CURVE_GROUPS:
            group_algos = [a for a in curve_algos if algo_filter is None or a in algo_filter]
            if len(group_algos) < 2:
                continue

            for exp_name in exp_names:
                histories: dict[str, dict] = {}
                colors: dict[str, str] = {}
                for algo in group_algos:
                    hist = grid_hists.get(exp_name, {}).get(algo)
                    if hist is None:
                        continue
                    if any(m in hist.get("metrics", {}) for m in metrics):
                        label = ALGO_LABELS.get(algo, algo)
                        histories[label] = hist
                        colors[label] = ALGO_COLORS[algo]

                if len(histories) < 2:
                    continue

                max_n_ep = max(len(h["episodes"]) for h in histories.values())
                smoothing = max(1, max_n_ep // 20)
                out_path = group_dirs[group_name] / f"{grid_stem}_{exp_name}_algo_overlay.png"

                try:
                    fig, _ = plot_algorithm_comparison(
                        histories,
                        metrics=metrics,
                        smoothing_window=smoothing,
                        title=f"{group_name} / {exp_name} — algorithm overlay ({grid_stem})",
                        colors=colors,
                    )
                    fig.savefig(out_path, dpi=130, bbox_inches="tight")
                    plt.close(fig)
                    tqdm.write(f"  Saved {out_path.name}")
                except Exception as exc:
                    tqdm.write(
                        f"  [WARN] algo overlay skipped ({group_name}/{exp_name}/{grid_stem}): {exc}"
                    )


# ─── Plotting: value + policy per (experiment, algorithm, grid) ──────────────

def _save_value_policy_plots(
    all_value_policy: dict[str, dict[str, dict[str, dict | None]]],
    grid_paths: dict[str, Path],
    group_dirs: dict[str, Path],
) -> None:
    """Save a ``plot_value_and_policy`` PNG for every (experiment, algorithm, grid).

    Each PNG lands in the experiment's group folder under
    ``{grid_stem}_{exp_name}_{algo}_value_policy.png``. Runs with empty
    ``values`` or ``policy`` are skipped cleanly — e.g. an MC agent that
    failed to call ``build_value_and_policy()``.
    """
    for grid_stem, exp_map in all_value_policy.items():
        grid_path = grid_paths.get(grid_stem)
        if grid_path is None or not grid_path.exists():
            continue
        grid_array = np.load(grid_path)

        for exp_name, algo_map in exp_map.items():
            group_name = EXP_TO_GROUP.get(exp_name)
            if group_name is None:
                continue
            out_dir = group_dirs[group_name]

            for algo, vp in algo_map.items():
                if vp is None:
                    continue
                values = vp.get("values")
                policy = vp.get("policy")
                if not values or not policy:
                    continue

                out_path = out_dir / f"{grid_stem}_{exp_name}_{algo}_value_policy.png"
                try:
                    fig, _ = plot_value_and_policy(
                        grid_array,
                        values,
                        policy,
                        title=f"{ALGO_LABELS.get(algo, algo)} — {exp_name} ({grid_stem})",
                        agent_start_pos=vp.get("initial_pos"),
                    )
                    fig.savefig(out_path, dpi=130, bbox_inches="tight")
                    plt.close(fig)
                    tqdm.write(f"  Saved {out_path.name}")
                except Exception as exc:
                    tqdm.write(
                        f"  [WARN] value+policy plot skipped ({exp_name}/{algo}/{grid_stem}): {exc}"
                    )


# ─── Plotting: VI convergence curves ─────────────────────────────────────────

# Only sigma and gamma affect VI — those are the only meaningful comparisons
_VI_CONVERGENCE_GROUPS = [
    ("sigma", ["sigma=0.0", "sigma=0.3"]),
    ("gamma", ["gamma=0.5", "gamma=0.99"]),
]


def _save_vi_convergence_plots(
    all_histories: dict[str, dict[str, dict[str, dict | None]]],
    group_dirs: dict[str, Path],
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
                out_path = group_dirs[group_name] / f"{grid_stem}_vi_convergence.png"
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

    # Create one subfolder per comparison group and open its CSV file.
    group_dirs: dict[str, Path] = {}
    group_files = {}
    group_writers = {}
    for group_name, _ in EXP_GROUPS:
        gdir = args.out_dir / group_name
        gdir.mkdir(exist_ok=True)
        group_dirs[group_name] = gdir
        gf = (gdir / "results.csv").open("w", newline="", encoding="utf-8")
        group_files[group_name] = gf
        gw = csv.DictWriter(gf, fieldnames=CSV_FIELDS)
        gw.writeheader()
        group_writers[group_name] = gw

    total_runs = len(EXPERIMENTS) * len(algorithms) * len(grids)
    print(f"Running {len(EXPERIMENTS)} experiments × {len(algorithms)} algorithms × {len(grids)} grids = {total_runs} runs")
    print(f"Results → {csv_path}\n")

    all_rows: list[dict] = []
    # all_histories[grid_stem][exp_name][algo] = history_dict_or_None
    all_histories: dict[str, dict[str, dict[str, dict | None]]] = {}
    # all_value_policy[grid_stem][exp_name][algo] = {"values", "policy", "initial_pos"} | None
    all_value_policy: dict[str, dict[str, dict[str, dict | None]]] = {}
    # grid_paths[grid_stem] = Path, used to load the grid array for value+policy plots
    grid_paths: dict[str, Path] = {}

    try:
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

                group_name = EXP_TO_GROUP[exp_name]
                group_writer = group_writers[group_name]
                group_file = group_files[group_name]

                for grid_path in grids:
                    if not grid_path.exists():
                        tqdm.write(f"  [SKIP] grid not found: {grid_path}")
                        pbar.update(len(algorithms))
                        continue

                    g = grid_path.stem
                    grid_paths[g] = grid_path
                    all_histories.setdefault(g, {}).setdefault(exp_name, {})
                    all_value_policy.setdefault(g, {}).setdefault(exp_name, {})

                    for algorithm in algorithms:
                        pbar.set_description(f"{exp_name} / {algorithm} / {g}")
                        try:
                            row, history, vp_payload = _run_one(exp_name, algorithm, grid_path, cfg)
                            writer.writerow(row)
                            f.flush()
                            group_writer.writerow(row)
                            group_file.flush()
                            all_rows.append(row)
                            all_histories[g][exp_name][algorithm] = history
                            all_value_policy[g][exp_name][algorithm] = vp_payload
                        except Exception as exc:
                            tqdm.write(f"  [ERROR] {exp_name}/{algorithm}/{g}: {exc}")
                        pbar.update(1)

            pbar.close()
    finally:
        for gf in group_files.values():
            gf.close()

    if args.no_plots:
        print(f"\nDone. Results in {args.out_dir}/")
    else:
        print(f"\nAll runs complete. Generating plots…")
        _save_training_curve_plots(all_histories, group_dirs, algorithms)
        _save_algorithm_overlay_plots(all_histories, group_dirs, algorithms)
        _save_value_policy_plots(all_value_policy, grid_paths, group_dirs)
        if "value_iteration" in algorithms:
            _save_vi_convergence_plots(all_histories, group_dirs)
        print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
