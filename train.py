"""Unified training CLI.

Usage:
    python train.py {value_iteration|q_learning|mc|off_policy_mc|random} GRID [GRID ...] [--flags]

The first positional argument selects the agent. Each agent has its own
subparser exposing only the flags it needs. Shared flags (sigma, gamma,
max_steps, ...) live on a parent parser used by all subcommands.
"""

from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import datetime
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


def _optional_float(raw: str) -> float | None:
    """Parse a float CLI value, accepting 'none' to disable the setting."""
    if raw.lower() in {"none", "null"}:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ArgumentTypeError(f"expected a float or None, got {raw!r}") from exc


def _build_shared_parser() -> ArgumentParser:
    """Build the parent parser holding flags common to every agent.

    Flags are split into argparse argument groups so ``--help`` output is
    structured per concern (environment, output, evaluation comparisons,
    W&B). Each subparser inherits the entire parent through ``parents=``.
    """
    parent = ArgumentParser(add_help=False)
    parent.add_argument("GRID", type=Path, nargs="+", help="Paths to one or more grid files.")

    env = parent.add_argument_group("environment")
    env.add_argument("--no_gui", action="store_true", help="Disable rendering for faster training.")
    env.add_argument("--fps", type=int, default=30, help="GUI frame rate (ignored with --no_gui).")
    env.add_argument("--sigma", type=float, default=0.1, help="Environment stochasticity.")
    env.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    env.add_argument("--max_steps", type=int, default=500, help="Max env steps per episode/rollout.")
    env.add_argument("--eval_episodes", type=int, default=20, help="Number of evaluation rollouts.")
    env.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    env.add_argument("--start_pos", type=str, default=None, help="Agent start position as col,row.")

    output = parent.add_argument_group("output")
    output.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Output directory for artifacts. Defaults to "
            "results/<agent>_<timestamp>/ when omitted."
        ),
    )
    output.add_argument(
        "--reward",
        type=str,
        choices=("manhattan", "basic"),
        default="manhattan",
        help=(
            "Reward function to use. 'manhattan' is distance-shaped; 'basic' "
            "uses the assignment spec (-1/+10). See README §Reward Function."
        ),
    )

    compare = parent.add_argument_group("evaluation comparison")
    compare.add_argument(
        "--compare_optimal",
        action="store_true",
        help=(
            "Pre-train a Value Iteration agent and use its policy as the optimality "
            "reference: records per-episode policy disagreement (QL/MC/off-policy MC), "
            "emits a spatial *_policy_diff.png heatmap, and adds the scalar to the eval "
            "summary."
        ),
    )

    wandb_group = parent.add_argument_group("Weights & Biases logging")
    wandb_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    wandb_group.add_argument(
        "--wandb_project", type=str, default="rl-in-practice", help="W&B project name.",
    )
    return parent


# ---------------------------------------------------------------------------
# Per-concern flag helpers used to compose the QL / MC / off-policy MC parsers.
#
# Each helper attaches one logical group of CLI flags to a subparser via an
# argparse argument group, so that ``--help`` output groups them together
# under a named heading. Adding a new tabular agent should mean composing
# these helpers, never copy-pasting flag definitions.
# ---------------------------------------------------------------------------


def _add_episodes_args(
    subparser: ArgumentParser, *, default_episodes: int, default_max_episode_length: int | None = None,
) -> None:
    """Attach the episode-budget flags (``--episodes`` plus optionally ``--max_episode_length``)."""
    group = subparser.add_argument_group("episode budget")
    group.add_argument("--episodes", type=int, default=default_episodes, help="Training episodes.")
    if default_max_episode_length is not None:
        group.add_argument(
            "--max_episode_length",
            type=int,
            default=default_max_episode_length,
            help="Max steps per training episode.",
        )


def _add_alpha_args(
    subparser: ArgumentParser,
    *,
    default_alpha: float,
    default_alpha_min: float,
    default_alpha_decay: float,
) -> None:
    """Attach the learning-rate schedule flags shared by tabular Q-table agents."""
    group = subparser.add_argument_group("learning rate (alpha)")
    group.add_argument(
        "--alpha",
        type=float,
        default=default_alpha,
        help=(
            "Learning rate (initial value for exponential, base value for "
            "constant; ignored for visit_count)."
        ),
    )
    group.add_argument(
        "--alpha_min",
        type=float,
        default=default_alpha_min,
        help="Minimum learning rate (exponential schedule only).",
    )
    group.add_argument(
        "--alpha_decay",
        type=float,
        default=default_alpha_decay,
        help="Per-episode decay for alpha (exponential schedule only).",
    )
    group.add_argument(
        "--lr_schedule",
        choices=("exponential", "constant", "visit_count"),
        default="exponential",
        help=(
            "Learning rate schedule. 'exponential' decays alpha per episode "
            "(uses --alpha/--alpha_decay/--alpha_min). 'constant' keeps --alpha "
            "fixed throughout. 'visit_count' uses c / (c + N(s, a)); set c via "
            "--visit_count_c."
        ),
    )
    group.add_argument(
        "--visit_count_c",
        type=float,
        default=1.0,
        help=(
            "Offset c for the visit_count schedule alpha = c / (c + N(s, a)). "
            "Default 1.0 follows the textbook but decays alpha aggressively "
            "(below 0.01 after ~100 visits per (s, a)). Practical values are "
            "typically 5-50 to keep updates substantial for longer."
        ),
    )


def _add_epsilon_args(
    subparser: ArgumentParser,
    *,
    default_epsilon: float,
    default_epsilon_min: float,
    default_epsilon_decay: float,
) -> None:
    """Attach the exploration-rate flags shared by tabular Q-table agents."""
    group = subparser.add_argument_group("exploration (epsilon)")
    group.add_argument("--epsilon", type=float, default=default_epsilon, help="Initial exploration rate.")
    group.add_argument(
        "--epsilon_min", type=float, default=default_epsilon_min, help="Minimum exploration rate.",
    )
    group.add_argument(
        "--epsilon_decay",
        type=float,
        default=default_epsilon_decay,
        help="Per-episode decay for epsilon.",
    )
    group.add_argument("--fixed_epsilon", action="store_true", help="Disable epsilon decay.")


def _add_q_init_args(subparser: ArgumentParser) -> None:
    """Attach the Q-table initialisation flags shared by tabular Q-table agents."""
    group = subparser.add_argument_group("Q-table initialisation")
    group.add_argument(
        "--q_init", type=float, default=0.0, help="Base initial Q-value for new state-action rows.",
    )
    group.add_argument(
        "--q_init_noise",
        type=float,
        default=1e-6,
        help=(
            "Uniform noise radius added to initial Q-values to break action ties; "
            "set 0 for exact init."
        ),
    )


def _add_log_args(subparser: ArgumentParser) -> None:
    """Attach the training-log flags shared by tabular Q-table agents."""
    group = subparser.add_argument_group("training log")
    group.add_argument(
        "--log_interval",
        type=int,
        default=0,
        help="Print training progress every N episodes; 0 disables console logging.",
    )
    group.add_argument(
        "--log_q_table",
        action="store_true",
        help="Include the learned Q-table in each console log entry.",
    )


def _add_stopping_args(subparser: ArgumentParser) -> None:
    """Attach the early-stopping flags shared by tabular Q-table agents."""
    group = subparser.add_argument_group("early stopping")
    group.add_argument(
        "--policy-stable-patience",
        type=int,
        default=50,
        dest="policy_stable_patience",
        help=(
            "Stop training once the tied-greedy policy has been unchanged "
            "for this many consecutive episodes. Pass 0 or a negative "
            "value to disable the criterion and always run the full "
            "episode budget. Default: 50."
        ),
    )


def _add_training_starts_arg(subparser: ArgumentParser) -> None:
    """Attach the ``--exploring_starts`` flag for tabular Q-table agents.

    Sutton & Barto §5.4 exploring-starts: at the *start* of every training
    episode we sample a uniformly random empty cell while evaluation
    rollouts still start from the requested ``--start_pos``. Wired up
    identically for QL, on-policy MC, and off-policy MC via
    :func:`agents.trainers.common.build_episode_start_picker`.
    """
    group = subparser.add_argument_group("training-time exploration")
    group.add_argument(
        "--exploring_starts",
        action="store_true",
        help=(
            "Sample a uniformly random empty cell as the training start of "
            "every episode (Sutton & Barto §5.4). Evaluation still uses "
            "--start_pos."
        ),
    )


def _add_tabular_agent_args(
    subparser: ArgumentParser,
    *,
    default_episodes: int,
    default_max_episode_length: int | None,
    default_alpha: float,
    default_alpha_min: float,
    default_alpha_decay: float,
    default_epsilon: float,
    default_epsilon_min: float,
    default_epsilon_decay: float,
) -> None:
    """Compose the standard tabular-Q-table flag set on *subparser*.

    QL, MC, and off-policy MC all share the same skeleton: episode budget,
    alpha schedule, epsilon schedule, Q-table init, training log. The only
    things that vary per agent are the defaults and the agent-specific
    flags layered on top (e.g. --off_policy_update for off-policy MC).
    """
    _add_episodes_args(
        subparser,
        default_episodes=default_episodes,
        default_max_episode_length=default_max_episode_length,
    )
    _add_alpha_args(
        subparser,
        default_alpha=default_alpha,
        default_alpha_min=default_alpha_min,
        default_alpha_decay=default_alpha_decay,
    )
    _add_epsilon_args(
        subparser,
        default_epsilon=default_epsilon,
        default_epsilon_min=default_epsilon_min,
        default_epsilon_decay=default_epsilon_decay,
    )
    _add_q_init_args(subparser)
    _add_log_args(subparser)
    _add_stopping_args(subparser)


def parse_args() -> Namespace:
    """Build the subparser tree and parse the CLI."""
    shared = _build_shared_parser()
    parser = ArgumentParser(description="Unified training entry point for RL agents.")
    subparsers = parser.add_subparsers(dest="agent", required=True, help="Agent to train.")

    vi = subparsers.add_parser(
        "value_iteration", parents=[shared], help="Train a tabular value-iteration agent.",
    )
    vi_group = vi.add_argument_group("value iteration")
    vi_group.add_argument("--theta", type=float, default=1e-6, help="Bellman convergence threshold.")
    vi_group.add_argument("--vi_max_iter", type=int, default=1000, help="Maximum Bellman sweeps.")

    ql = subparsers.add_parser("q_learning", parents=[shared], help="Train a Q-learning agent.")
    _add_tabular_agent_args(
        ql,
        default_episodes=3000,
        default_max_episode_length=None,
        default_alpha=0.5,
        default_alpha_min=0.05,
        default_alpha_decay=0.999,
        default_epsilon=1.0,
        default_epsilon_min=0.05,
        default_epsilon_decay=0.995,
    )
    _add_training_starts_arg(ql)

    mc = subparsers.add_parser(
        "mc", parents=[shared], help="Train an on-policy first-visit MC agent.",
    )
    _add_tabular_agent_args(
        mc,
        default_episodes=5000,
        default_max_episode_length=2000,
        default_alpha=0.5,
        default_alpha_min=0.05,
        default_alpha_decay=0.9995,
        default_epsilon=0.2,
        default_epsilon_min=0.01,
        default_epsilon_decay=0.9995,
    )
    _add_training_starts_arg(mc)

    off_mc = subparsers.add_parser(
        "off_policy_mc",
        parents=[shared],
        help="Train an off-policy weighted-importance-sampling MC control agent.",
    )
    _add_tabular_agent_args(
        off_mc,
        default_episodes=5000,
        default_max_episode_length=2000,
        default_alpha=0.2,
        default_alpha_min=0.02,
        default_alpha_decay=0.9998,
        default_epsilon=0.3,
        default_epsilon_min=0.02,
        default_epsilon_decay=0.9998,
    )
    _add_training_starts_arg(off_mc)
    off_policy_group = off_mc.add_argument_group("off-policy MC specific")
    off_policy_group.add_argument(
        "--off_policy_update",
        choices=("weighted", "alpha"),
        default="alpha",
        help=(
            "Use constant-alpha importance-weighted updates or textbook cumulative "
            "weighted averaging. The --alpha* and --lr_schedule flags only affect "
            "the 'alpha' update mode; weighted importance sampling has its own "
            "intrinsic step size W / C(s, a)."
        ),
    )
    off_policy_group.add_argument(
        "--importance_weight_clip",
        type=_optional_float,
        default=10.0,
        help=(
            "Maximum importance weight used in alpha mode before multiplying by "
            "alpha; use None to disable."
        ),
    )
    off_policy_group.add_argument(
        "--soft_target_epsilon",
        type=float,
        default=0.0,
        help=(
            "Target-policy exploration rate. 0.0 (default) uses the textbook "
            "deterministic greedy target, which breaks the backward loop at "
            "non-greedy actions. Values > 0 make the target epsilon-soft. Must "
            "be strictly less than --epsilon."
        ),
    )

    subparsers.add_parser("random", parents=[shared], help="Evaluate a uniform-random baseline.")

    return parser.parse_args()


def _resolve_policy_stable_patience(raw: int | None) -> int | None:
    """Map a CLI patience value onto ``TrainConfig.policy_stable_patience``.

    Treat ``None`` and any non-positive integer as "disabled"; positive
    integers pass through unchanged.
    """
    if raw is None or raw <= 0:
        return None
    return raw


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
        lr_schedule=getattr(args, "lr_schedule", "exponential"),
        visit_count_c=getattr(args, "visit_count_c", 1.0),
        epsilon=getattr(args, "epsilon", None),
        epsilon_min=getattr(args, "epsilon_min", None),
        epsilon_decay=getattr(args, "epsilon_decay", None),
        fixed_epsilon=getattr(args, "fixed_epsilon", False),
        ql_episodes=getattr(args, "episodes", None) if args.agent == "q_learning" else None,
        mc_episodes=getattr(args, "episodes", None) if args.agent in {"mc", "off_policy_mc"} else None,
        max_episode_length=getattr(args, "max_episode_length", None),
        log_interval=getattr(args, "log_interval", 100),
        log_q_table=getattr(args, "log_q_table", False),
        q_init=getattr(args, "q_init", 0.0),
        q_init_noise=getattr(args, "q_init_noise", 1e-6),
        exploring_starts=getattr(args, "exploring_starts", False),
        off_policy_update=getattr(args, "off_policy_update", "alpha"),
        importance_weight_clip=getattr(args, "importance_weight_clip", 10.0),
        soft_target_epsilon=getattr(args, "soft_target_epsilon", 0.0),
        theta=getattr(args, "theta", None),
        vi_max_iter=getattr(args, "vi_max_iter", None),
        reward_function=getattr(args, "reward", "manhattan"),
        wandb=getattr(args, "wandb", False),
        wandb_project=getattr(args, "wandb_project", "rl-in-practice"),
        policy_stable_patience=_resolve_policy_stable_patience(
            getattr(args, "policy_stable_patience", None)
        ),
    )


def main() -> None:
    """CLI entry point: dispatch to the chosen trainer per grid."""
    args = parse_args()
    if args.out_dir is None:
        run_timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        args.out_dir = Path("results") / f"{args.agent}_{run_timestamp}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config_from_args(args)
    trainer = TRAINERS[args.agent]
    # --compare_optimal is only meaningful for agents that learn a policy
    # mid-training. VI is the reference itself; random has no policy.
    use_reference = args.compare_optimal and args.agent in {
        "q_learning",
        "mc",
        "off_policy_mc",
    }

    for grid_path in args.GRID:
        env, initial_pos, reward_fn = setup_grid_run(
            grid_path=grid_path,
            sigma=cfg.sigma,
            fps=args.fps,
            no_gui=args.no_gui,
            start_pos=cfg.start_pos,
            random_seed=cfg.random_seed,
            reward_function=cfg.reward_function,
        )
        run_cfg = TrainConfig(**{**cfg.__dict__, "start_pos": initial_pos})

        if run_cfg.wandb:
            import wandb

            wandb.init(
                project=run_cfg.wandb_project,
                name=f"{args.out_dir.name}_{grid_path.stem}",
                config=run_cfg.__dict__,
                reinit="finish_previous",
            )
            # Pin the exact reward function source to this run so the choice
            # (manhattan vs basic) AND the implementation can be reproduced
            # later without checking out the same commit.
            #
            # The reward constants are *also* logged as scalar config keys
            # (under reward_constants.*) so they're filterable and sortable
            # in the W&B sweep/dashboard UI without parsing the source text.
            _rewards_src_path = Path(__file__).resolve().parent / "world" / "rewards.py"
            from world.rewards import (
                MIN_TARGET_REWARD,
                STEP_REWARD,
                TARGET_REWARD,
                WALL_OR_OBSTACLE_REWARD,
            )
            wandb.config.update(
                {
                    "reward_function_source": _rewards_src_path.read_text(encoding="utf-8"),
                    "reward_constants": {
                        "STEP_REWARD": STEP_REWARD,
                        "TARGET_REWARD": TARGET_REWARD,
                        "WALL_OR_OBSTACLE_REWARD": WALL_OR_OBSTACLE_REWARD,
                        "MIN_TARGET_REWARD": MIN_TARGET_REWARD,
                    },
                },
                allow_val_change=True,
            )

        optimal_policy = None
        optimal_values = None
        if use_reference:
            vi_agent, _ = TRAINERS["value_iteration"](env, reward_fn, run_cfg)
            optimal_policy = vi_agent.optimal_action_sets()
            optimal_values = dict(vi_agent.values)

        agent, history = trainer(
            env,
            reward_fn,
            run_cfg,
            optimal_policy=optimal_policy,
            optimal_values=optimal_values,
        )

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
            wandb_log=run_cfg.wandb,
        )

        diff_part = (
            f"  policy_diff={policy_diff_scalar:.3f}" if policy_diff_scalar is not None else ""
        )
        print(
            f"[{grid_path.stem}] success_rate={metrics['success_rate']:.3f}  "
            f"mean_discounted_return={metrics['mean_discounted_return']:.3f}"
            f"{diff_part}"
        )

        if run_cfg.wandb:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
