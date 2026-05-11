"""Unified training CLI.

Usage:
    python train.py {value_iteration|q_learning|mc|random} GRID [GRID ...] [--flags]

The first positional argument selects the agent. Each agent has its own
subparser exposing only the flags it needs. Shared flags (sigma, gamma,
max_steps, ...) live on a parent parser used by all subcommands.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from agents.trainers import (
    TRAINERS,
    TrainConfig,
    make_artifact_prefix,
    parse_start_pos,
    policy_disagreement,
    save_run_artifacts,
    setup_grid_run,
)
from utils.evaluation import evaluate_policy_metrics


def _build_shared_parser() -> ArgumentParser:
    """Build the parent parser holding flags common to every agent."""
    parent = ArgumentParser(add_help=False)
    parent.add_argument("GRID", type=Path, nargs="+", help="Paths to one or more grid files.")
    parent.add_argument("--no_gui", action="store_true", help="Disable rendering for faster training.")
    parent.add_argument("--fps", type=int, default=30, help="GUI frame rate (ignored with --no_gui).")
    parent.add_argument("--sigma", type=float, default=0.1, help="Environment stochasticity.")
    parent.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parent.add_argument("--max_steps", type=int, default=500, help="Max env steps per episode/rollout.")
    parent.add_argument("--eval_episodes", type=int, default=20, help="Number of evaluation rollouts.")
    parent.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    parent.add_argument("--start_pos", type=str, default=None, help="Agent start position as col,row.")
    parent.add_argument("--out_dir", type=Path, default=Path("results"), help="Output directory for artifacts.")
    parent.add_argument(
        "--compare_optimal",
        action="store_true",
        help=(
            "Pre-train a Value Iteration agent and use its policy as the optimality "
            "reference: records per-episode policy disagreement (QL/MC only), emits a "
            "spatial *_policy_diff.png heatmap, and adds the scalar to the eval summary."
        ),
    )
    return parent


def _add_alpha_epsilon_args(subparser: ArgumentParser, *, default_epsilon: float, default_epsilon_decay: float,
                            default_alpha_decay: float, default_alpha_min: float, default_epsilon_min: float,
                            default_alpha: float | None) -> None:
    """Attach the shared alpha/epsilon schedule flags to a subparser."""
    subparser.add_argument("--alpha", type=float, default=default_alpha, help="Learning rate.")
    subparser.add_argument("--alpha_min", type=float, default=default_alpha_min, help="Minimum learning rate.")
    subparser.add_argument("--alpha_decay", type=float, default=default_alpha_decay, help="Per-episode decay for alpha.")
    subparser.add_argument("--fixed_alpha", action="store_true", help="Disable alpha decay.")
    subparser.add_argument("--epsilon", type=float, default=default_epsilon, help="Initial exploration rate.")
    subparser.add_argument("--epsilon_min", type=float, default=default_epsilon_min, help="Minimum exploration rate.")
    subparser.add_argument("--epsilon_decay", type=float, default=default_epsilon_decay, help="Per-episode decay for epsilon.")
    subparser.add_argument("--fixed_epsilon", action="store_true", help="Disable epsilon decay.")


def parse_args() -> Namespace:
    """Build the subparser tree and parse the CLI."""
    shared = _build_shared_parser()
    parser = ArgumentParser(description="Unified training entry point for RL agents.")
    subparsers = parser.add_subparsers(dest="agent", required=True, help="Agent to train.")

    vi = subparsers.add_parser("value_iteration", parents=[shared], help="Train a tabular value-iteration agent.")
    vi.add_argument("--theta", type=float, default=1e-6, help="Bellman convergence threshold.")
    vi.add_argument("--vi_max_iter", type=int, default=1000, help="Maximum Bellman sweeps.")

    ql = subparsers.add_parser("q_learning", parents=[shared], help="Train a Q-learning agent.")
    ql.add_argument("--episodes", type=int, default=3000, help="Training episodes.")
    _add_alpha_epsilon_args(
        ql,
        default_alpha=0.5,
        default_alpha_min=0.05,
        default_alpha_decay=0.999,
        default_epsilon=1.0,
        default_epsilon_min=0.05,
        default_epsilon_decay=0.995,
    )

    mc = subparsers.add_parser("mc", parents=[shared], help="Train an on-policy first-visit MC agent.")
    mc.add_argument("--episodes", type=int, default=5000, help="Training episodes.")
    mc.add_argument("--max_episode_length", type=int, default=2000, help="Max steps per training episode.")
    # Matches QL's CLI defaults so MC and QL share a hyperparameter shape.
    # Note: even at these defaults, MC on a 5000-episode budget is high
    # variance across seeds (single-seed runs can swing 0% <-> 100% on A1).
    # See README "MC training notes" — increase --episodes for stability.
    _add_alpha_epsilon_args(
        mc,
        default_alpha=0.5,
        default_alpha_min=0.05,
        default_alpha_decay=0.9995,
        default_epsilon=0.2,
        default_epsilon_min=0.01,
        default_epsilon_decay=0.9995,
    )

    subparsers.add_parser("random", parents=[shared], help="Evaluate a uniform-random baseline.")

    return parser.parse_args()


def _config_from_args(args: Namespace) -> TrainConfig:
    """Materialise a TrainConfig from the parsed CLI args.

    Agent-specific fields are pulled with ``getattr`` so the same builder
    works for every subcommand.
    """
    return TrainConfig(
        sigma=args.sigma,
        gamma=args.gamma,
        max_steps=args.max_steps,
        random_seed=args.random_seed,
        eval_episodes=args.eval_episodes,
        start_pos=parse_start_pos(args.start_pos),
        alpha=getattr(args, "alpha", None),
        alpha_min=getattr(args, "alpha_min", None),
        alpha_decay=getattr(args, "alpha_decay", None),
        epsilon=getattr(args, "epsilon", None),
        epsilon_min=getattr(args, "epsilon_min", None),
        epsilon_decay=getattr(args, "epsilon_decay", None),
        fixed_alpha=getattr(args, "fixed_alpha", False),
        fixed_epsilon=getattr(args, "fixed_epsilon", False),
        ql_episodes=getattr(args, "episodes", None) if args.agent == "q_learning" else None,
        mc_episodes=getattr(args, "episodes", None) if args.agent == "mc" else None,
        max_episode_length=getattr(args, "max_episode_length", None),
        theta=getattr(args, "theta", None),
        vi_max_iter=getattr(args, "vi_max_iter", None),
    )


def main() -> None:
    """CLI entry point: dispatch to the chosen trainer per grid."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config_from_args(args)
    trainer = TRAINERS[args.agent]
    # --compare_optimal is only meaningful for agents that learn a policy
    # mid-training. VI is the reference itself; random has no policy.
    use_reference = bool(getattr(args, "compare_optimal", False)) and args.agent in {"q_learning", "mc"}

    for grid_path in args.GRID:
        env, initial_pos, reward_fn = setup_grid_run(
            grid_path=grid_path,
            sigma=cfg.sigma,
            fps=args.fps,
            no_gui=args.no_gui,
            start_pos=cfg.start_pos,
            random_seed=cfg.random_seed,
        )
        run_cfg = TrainConfig(**{**cfg.__dict__, "start_pos": initial_pos})

        optimal_policy = None
        if use_reference:
            vi_agent, _ = TRAINERS["value_iteration"](env, reward_fn, run_cfg)
            optimal_policy = vi_agent.policy

        agent, history = trainer(env, reward_fn, run_cfg, optimal_policy=optimal_policy)

        metrics = evaluate_policy_metrics(
            grid=grid_path,
            agent=agent,
            max_steps=run_cfg.max_steps,
            sigma=run_cfg.sigma,
            agent_start_pos=initial_pos,
            reward_fn=reward_fn,
            gamma=run_cfg.gamma,
            random_seed=run_cfg.random_seed,
            n_eval_episodes=run_cfg.eval_episodes,
        )

        policy_diff_scalar = (
            policy_disagreement(optimal_policy, agent) if optimal_policy is not None else None
        )

        prefix = make_artifact_prefix(grid_path, args.agent)
        save_run_artifacts(
            out_dir=args.out_dir,
            artifact_prefix=prefix,
            grid_path=grid_path,
            agent=agent,
            env_grid=env.grid,
            initial_pos=initial_pos,
            evaluation_metrics=metrics,
            reward_fn=reward_fn,
            cfg=run_cfg,
            optimal_policy=optimal_policy,
            policy_diff_scalar=policy_diff_scalar,
            history=history,
        )

        diff_part = (
            f"  policy_diff={policy_diff_scalar:.3f}" if policy_diff_scalar is not None else ""
        )
        print(
            f"[{grid_path.stem}] success_rate={metrics['success_rate']:.3f}  "
            f"mean_discounted_return={metrics['mean_discounted_return']:.3f}"
            f"{diff_part}"
        )


if __name__ == "__main__":
    main()
