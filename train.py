"""Unified training CLI.

Usage:
    python train.py {value_iteration|q_learning|mc|random} GRID [GRID ...] [--flags]

The first positional argument selects the agent. Each agent has its own
subparser exposing only the flags it needs. Shared flags (sigma, gamma,
eval_max_steps, ...) live on a parent parser used by all subcommands.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
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


# ---------------------------------------------------------------------------
# Shared CLI arguments
#
# This parent parser holds arguments that every subcommand understands. The
# groups below are organized by what the hyperparameters control:
#   - environment/evaluation dynamics,
#   - output and experiment comparison,
#   - optional external logging.
# Agent-specific learning hyperparameters are added later on each subparser.
# ---------------------------------------------------------------------------


def _build_shared_parser() -> ArgumentParser:
    """Build the parent parser holding flags common to every agent.

    Flags are split into argparse argument groups so ``--help`` output is
    structured per concern (environment, output, evaluation comparisons,
    W&B). Each subparser inherits the entire parent through ``parents=``.
    """
    parent = ArgumentParser(add_help=False)

    # Required input: one or more grid files. Every selected agent is run once
    # per grid path in ``main``.
    parent.add_argument("GRID", type=Path, nargs="+", help="Paths to one or more grid files.")

    # Environment and evaluation hyperparameters:
    # control stochasticity, discounting, rendering, random seed, start state,
    # and how long/how often post-training evaluation rollouts run.
    env = parent.add_argument_group("environment")
    env.add_argument("--no_gui", action="store_true", help="Disable rendering for faster training.")
    env.add_argument("--fps", type=int, default=30, help="GUI frame rate (ignored with --no_gui).")
    env.add_argument("--sigma", type=float, default=0.1, help="Environment stochasticity.")
    env.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    env.add_argument(
        "--eval_max_steps",
        type=int,
        default=500,
        help=(
            "Max env steps per evaluation rollout (post-training). Training "
            "episode length is controlled by --max_episode_length on each "
            "learning subcommand."
        ),
    )
    env.add_argument("--eval_episodes", type=int, default=20, help="Number of evaluation rollouts.")
    env.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    env.add_argument("--start_pos", type=str, default=None, help="Agent start position as col,row.")

    # Output hyperparameters:
    # control where generated metrics, plots, and rollout visualisations go.
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

    # Comparison/evaluation hyperparameters:
    # optionally train a VI reference before Q-learning/MC so model-free
    # policies can be compared against approximately optimal actions.
    compare = parent.add_argument_group("evaluation comparison")
    compare.add_argument(
        "--compare_optimal",
        action="store_true",
        help=(
            "Pre-train a Value Iteration agent and use its policy as the optimality "
            "reference: records per-episode policy disagreement (QL/MC), "
            "emits a spatial *_policy_diff.png heatmap, and adds the scalar to the eval "
            "summary."
        ),
    )

    # External logging hyperparameters:
    # control whether training metrics are mirrored to Weights & Biases.
    wandb_group = parent.add_argument_group("Weights & Biases logging")
    wandb_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    wandb_group.add_argument(
        "--wandb_project", type=str, default="rl-in-practice", help="W&B project name.",
    )
    return parent


# ---------------------------------------------------------------------------
# Per-concern flag helpers used to compose the QL / MC parsers.
#
# Each helper attaches one logical group of CLI flags to a subparser, 
# so that ``--help`` output groups them together
# under a named heading. Adding a new tabular agent should mean composing
# these helpers, never copy-pasting flag definitions.
#
# These helpers are deliberately split by hyperparameter type:
#   - episode budget,
#   - learning rate / alpha,
#   - exploration / epsilon,
#   - Q-table initialization,
#   - training logs,
#   - early stopping,
#   - training start-state sampling.
# ---------------------------------------------------------------------------


def _add_episodes_args(
    subparser: ArgumentParser, *, default_episodes: int, default_max_episode_length: int | None = None,
) -> None:
    """Attach the episode-budget flags (``--episodes`` plus optionally ``--max_episode_length``)."""
    # Episode-budget hyperparameters:
    # set how many training episodes to run and how many environment steps
    # each training episode may consume before being truncated.
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
    # Learning-rate hyperparameters:
    # select the alpha schedule and its parameters. These affect how strongly
    # each Q-value update moves toward its target.
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
    # Exploration hyperparameters:
    # control epsilon-greedy action selection while training.
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
    # Q-table initialization hyperparameters:
    # define the starting value for unseen state-action rows and optional
    # small noise used to avoid deterministic tie bias at initialization.
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
    # Training-log hyperparameters:
    # control how often the trainer emits live diagnostics and whether the
    # current Q-table is included in those logs.
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
    # Early-stopping hyperparameters:
    # stop model-free training once the tied-greedy policy has not changed
    # for enough consecutive episodes.
    group = subparser.add_argument_group("early stopping")
    group.add_argument(
        "--policy-stable-patience",
        type=int,
        default=1000,
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
    identically for QL and on-policy MC via
    :func:`agents.trainers.common.build_episode_start_picker`.
    """
    # Training-start hyperparameters:
    # choose whether every training episode starts from the fixed evaluation
    # start or from a uniformly sampled empty cell.
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

    QL and MC share the same skeleton: episode budget, alpha schedule,
    epsilon schedule, Q-table init, and training log. The only things that
    vary per agent are the defaults.
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
    # ------------------------------------------------------------------
    # Top-level parser and shared parent
    # ------------------------------------------------------------------
    shared = _build_shared_parser()
    parser = ArgumentParser(description="Unified training entry point for RL agents.")
    subparsers = parser.add_subparsers(dest="agent", required=True, help="Agent to train.")

    # ------------------------------------------------------------------
    # Value Iteration subcommand
    # ------------------------------------------------------------------
    # VI has solver-specific hyperparameters instead of episode/epsilon/alpha
    # settings because it performs dynamic programming over the known model.
    vi = subparsers.add_parser(
        "value_iteration", parents=[shared], help="Train a tabular value-iteration agent.",
    )
    vi_group = vi.add_argument_group("value iteration")
    vi_group.add_argument("--theta", type=float, default=1e-6, help="Bellman convergence threshold.")
    vi_group.add_argument("--vi_max_iter", type=int, default=1000, help="Maximum Bellman sweeps.")

    # ------------------------------------------------------------------
    # Q-learning subcommand
    # ------------------------------------------------------------------
    # Q-learning uses the full tabular-Q flag set and updates after every
    # transition, so its default episode length is shorter than MC's.
    ql = subparsers.add_parser("q_learning", parents=[shared], help="Train a Q-learning agent.")
    _add_tabular_agent_args(
        ql,
        default_episodes=3000,
        default_max_episode_length=500,
        default_alpha=0.5,
        default_alpha_min=0.05,
        default_alpha_decay=0.999,
        default_epsilon=1.0,
        default_epsilon_min=0.05,
        default_epsilon_decay=0.995,
    )
    _add_training_starts_arg(ql)

    # ------------------------------------------------------------------
    # On-policy Monte Carlo subcommand
    # ------------------------------------------------------------------
    # MC also uses the tabular-Q flag set, but updates only at episode end,
    # so the default episode cap is larger.
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

    # ------------------------------------------------------------------
    # Random baseline subcommand
    # ------------------------------------------------------------------
    # Random has no training hyperparameters. It shares only environment,
    # output, evaluation, and logging flags from the parent parser.
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
        # Environment/evaluation hyperparameters shared by every agent.
        sigma=args.sigma,
        gamma=args.gamma,
        eval_max_steps=args.eval_max_steps,
        random_seed=args.random_seed,
        eval_episodes=args.eval_episodes,
        start_pos=parse_start_pos(args.start_pos),

        # Learning-rate / alpha hyperparameters for Q-learning and MC.
        # Value Iteration and random ignore these fields.
        alpha=getattr(args, "alpha", None),
        alpha_min=getattr(args, "alpha_min", None),
        alpha_decay=getattr(args, "alpha_decay", None),
        lr_schedule=getattr(args, "lr_schedule", "exponential"),
        visit_count_c=getattr(args, "visit_count_c", 1.0),

        # Exploration / epsilon hyperparameters for Q-learning and MC.
        epsilon=getattr(args, "epsilon", None),
        epsilon_min=getattr(args, "epsilon_min", None),
        epsilon_decay=getattr(args, "epsilon_decay", None),
        fixed_epsilon=getattr(args, "fixed_epsilon", False),

        # Episode-budget hyperparameters. Only the selected model-free agent
        # receives an episode count; the other count remains None.
        ql_episodes=getattr(args, "episodes", None) if args.agent == "q_learning" else None,
        mc_episodes=getattr(args, "episodes", None) if args.agent == "mc" else None,
        max_episode_length=getattr(args, "max_episode_length", None),

        # Training diagnostics and Q-table initialization hyperparameters.
        log_interval=getattr(args, "log_interval", 100),
        log_q_table=getattr(args, "log_q_table", False),
        q_init=getattr(args, "q_init", 0.0),
        q_init_noise=getattr(args, "q_init_noise", 1e-6),
        exploring_starts=getattr(args, "exploring_starts", False),

        # Value Iteration solver hyperparameters.
        theta=getattr(args, "theta", None),
        vi_max_iter=getattr(args, "vi_max_iter", None),

        # External logging hyperparameters.
        wandb=getattr(args, "wandb", False),
        wandb_project=getattr(args, "wandb_project", "rl-in-practice"),

        # Early-stopping hyperparameter for model-free policy learning.
        policy_stable_patience=_resolve_policy_stable_patience(
            getattr(args, "policy_stable_patience", None)
        ),
    )


def main() -> None:
    """CLI entry point: dispatch to the chosen trainer per grid."""
    # ------------------------------------------------------------------
    # Parse CLI and resolve output/config
    # ------------------------------------------------------------------
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
    }

    for grid_path in args.GRID:
        # ------------------------------------------------------------------
        # Environment and reward setup
        # ------------------------------------------------------------------
        # ``setup_grid_run`` loads the grid, resolves the actual start state,
        # installs the reward function, and returns the initial position used
        # by training and evaluation.
        env, initial_pos, reward_fn = setup_grid_run(
            grid_path=grid_path,
            sigma=cfg.sigma,
            fps=args.fps,
            no_gui=args.no_gui,
            start_pos=cfg.start_pos,
            random_seed=cfg.random_seed,
        )
        run_cfg = TrainConfig(**{**cfg.__dict__, "start_pos": initial_pos})

        # ------------------------------------------------------------------
        # Optional W&B run setup
        # ------------------------------------------------------------------
        if run_cfg.wandb:
            import wandb

            wandb.init(
                project=run_cfg.wandb_project,
                name=f"{args.out_dir.name}_{grid_path.stem}",
                config=run_cfg.__dict__,
                reinit="finish_previous",
            )
            # Pin the exact reward function source to this run so the
            # implementation can be reproduced later without checking out
            # the same commit.
            #
            # The reward constants are *also* logged as scalar config keys
            # (under reward_constants.*) so they're filterable and sortable
            # in the W&B sweep/dashboard UI without parsing the source text.
            _rewards_src_path = Path(__file__).resolve().parent / "world" / "rewards.py"
            from world.rewards import (
                STEP_REWARD,
                TARGET_REWARD,
            )
            wandb.config.update(
                {
                    "reward_function_source": _rewards_src_path.read_text(encoding="utf-8"),
                    "reward_constants": {
                        "STEP_REWARD": STEP_REWARD,
                        "TARGET_REWARD": TARGET_REWARD,
                    },
                },
                allow_val_change=True,
            )

        # ------------------------------------------------------------------
        # Optional optimal-policy reference
        # ------------------------------------------------------------------
        # For Q-learning/MC, train a VI agent on the same grid and reward so
        # the trainer can record policy disagreement and optimality gaps.
        optimal_policy = None
        optimal_values = None
        if use_reference:
            vi_agent, _ = TRAINERS["value_iteration"](env, reward_fn, run_cfg)
            optimal_policy = vi_agent.optimal_action_sets()
            optimal_values = dict(vi_agent.values)

        # ------------------------------------------------------------------
        # Train selected agent
        # ------------------------------------------------------------------
        agent, history = trainer(
            env,
            reward_fn,
            run_cfg,
            optimal_policy=optimal_policy,
            optimal_values=optimal_values,
        )

        # ------------------------------------------------------------------
        # Post-training evaluation
        # ------------------------------------------------------------------
        # Evaluation runs fresh rollouts from the resolved start state with
        # the trained/evaluation-mode agent.
        metrics = evaluate_policy_metrics(
            grid=grid_path,
            agent=agent,
            eval_max_steps=run_cfg.eval_max_steps,
            sigma=run_cfg.sigma,
            agent_start_pos=initial_pos,
            reward_fn=reward_fn,
            gamma=run_cfg.gamma,
            random_seed=run_cfg.random_seed,
            n_eval_episodes=run_cfg.eval_episodes,
        )

        # Scalar values for end-of-training policy disagreement 
        policy_diff_scalar = (
            policy_disagreement(optimal_policy, agent) if optimal_policy is not None else None
        )

        # ------------------------------------------------------------------
        # Persist artifacts
        # ------------------------------------------------------------------
        # Save metrics, summaries, path visualisations, value/policy plots,
        # and optional training curves/optimal-policy comparisons.
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

        # ------------------------------------------------------------------
        # Console summary and W&B cleanup
        # ------------------------------------------------------------------
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
