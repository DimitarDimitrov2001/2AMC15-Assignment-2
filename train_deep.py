"""Deep-RL training CLI.

Entry point for the new continuous/minimal environments and the algorithm-
agnostic Trainer. Only the random agent is wired today; learning agents
(DQN, PPO, ...) plug in via the agent factory without touching this script.

Usage:
    uv run python train_deep.py --env minimal --episodes 20000 --visualize
    uv run python train_deep.py --env continuous --grid grid_configs/small_grid.npy
"""

from __future__ import annotations

import functools
from argparse import ArgumentParser, Namespace
from pathlib import Path

# Force a headless backend before pyplot is imported (via visualize_random_agent),
# so periodic in-training rendering never tries to open a GUI window.
import matplotlib

from agents.curiosity import GridCountMotivation, NoMotivation
matplotlib.use("Agg")

import torch

from agents import RandomAgent
from agents.base_agent import BaseAgent
from agents.dqn_agent import DQNAgent
from agents.a3c_agent import A3CAgent
from agents.epsilon_schedules import ConstantEpsilon, LinearEpsilonAnnealing
from agents.defaults import (
    CURIOSITY_RESOLUTION_DEFAULT,
    EPSILON_DEFAULT,
    EPSILON_DEFAULT_MIN,
    EPSILON_ANNEAL_DURATION,
    EPSILON_ANNEAL_START_STEP,
    DQN_DEFAULT_GAMMA,
    DQN_DEFAULT_LEARNING_RATE,
    DQN_DEFAULT_BATCH_SIZE,
    REPLAY_DEFAULT_CAPACITY,
    DQN_DEFAULT_NO_OBS_IN_STATE,
    DQN_DEFAULT_UPDATE_FREQ,
    DQN_DEFAULT_TARGET_UPDATE_FREQ,
    A3C_N_WORKERS,
    A3C_T_MAX,
    A3C_ENTROPY_BETA,
    A3C_VALUE_COEF,
)
from training import Trainer, TrainerConfig
from training.defaults import (
    DEFAULT_TOTAL_EPISODES,
    DEFAULT_MAX_STEPS_PER_EPISODE,
    DEFAULT_SEED,
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_VIZ_MAX_STEPS,
)
from world import GRID_CONFIGS_FP, ContinuousEnvironment, MinimalEnvironment
from visualize_random_agent import visualize_agent


# Environments expose a common subset of the interface the Trainer needs.
EnvType = MinimalEnvironment | ContinuousEnvironment


def _build_env(name: str, grid: Path, seed: int,
               start_pos: tuple[float, float] | None = None,
               use_sensors: bool = True,
               step_size: float | None = None) -> EnvType:
    """Construct the requested environment with sensible defaults.

    ``use_sensors`` only affects the continuous environment (toggles the
    distance-sensor readings in the observation); it is ignored otherwise.
    ``step_size`` overrides the env's default move size when provided.
    """
    # Only override the env's own default step size when explicitly requested.
    step_kwargs = {"step_size": step_size} if step_size is not None else {}
    if name == "minimal":
        return MinimalEnvironment(
            grid_fp=grid,
            agent_start_pos=start_pos,
            random_seed=seed,
            **step_kwargs,
        )
    if name == "continuous":
        return ContinuousEnvironment(
            grid_fp=grid,
            agent_start_pos=start_pos,
            use_sensors=use_sensors,
            random_seed=seed,
            **step_kwargs,
        )
    raise ValueError(f"Unknown environment: {name}")


def _resolve_device(choice: str) -> str:
    """Resolve 'auto' to cuda > mps > cpu; otherwise return the explicit choice."""
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_agent(args: Namespace, env: EnvType, device: str) -> BaseAgent:
    """Construct the requested agent.

    A registry keyed by ``--agent`` so learning agents drop in later without
    changing the training flow.
    """
    builders = {
        "random": lambda: RandomAgent(num_actions=env.n_actions, seed=args.seed),
        "dqn": lambda: DQNAgent(
            env=env,
            seed=args.seed,
            device=device,
            gamma=args.gamma,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            replay_buffer_capacity=args.replay_capacity,
            no_obs_in_state=args.stack_size,
            update_freq=args.update_freq,
            target_update_freq=args.target_update_freq,
            epsilon_scheduler=LinearEpsilonAnnealing(
                duration=args.epsilon_duration,
                start_step=args.epsilon_start_step,
                epsilon_max=args.epsilon,
                epsilon_min=args.epsilon_min,
            ) if args.epsilon_duration > 0 else ConstantEpsilon(args.epsilon),
            intrinsic_motivation=GridCountMotivation(
                max_x=env.observation_high[0],
                max_y=env.observation_high[1],
                resolution=CURIOSITY_RESOLUTION_DEFAULT,
                beta=args.curiosity_beta,
            ) if args.curiosity == "grid_count" else NoMotivation()
        ),
        "a3c": lambda: A3CAgent(
            env=env,
            # Picklable factory (module-level fn + primitive args) so each worker
            # builds its own env under the spawn start method.
            env_fn=functools.partial(
                _build_env,
                args.env,
                args.grid,
                start_pos=(None if args.exploring_starts
                           else (tuple(args.start_pos) if args.start_pos is not None else None)),
                use_sensors=args.use_sensors,
                step_size=args.step_size,
            ),
            seed=args.seed,
            total_steps=(args.a3c_total_steps if args.a3c_total_steps is not None
                         else args.episodes * args.max_steps),
            max_steps_per_episode=args.max_steps,
            n_workers=args.a3c_workers,
            t_max=args.a3c_t_max,
            gamma=args.gamma,
            learning_rate=args.lr,
            entropy_beta=args.a3c_entropy_beta,
            value_coef=args.a3c_value_coef,
            device=device,
        ),
    }
    if args.agent not in builders:
        raise ValueError(f"Unknown agent: {args.agent}")
    return builders[args.agent]()


def parse_args() -> Namespace:
    """Parse the deep-RL training CLI arguments."""
    parser = ArgumentParser(description="Deep-RL training entry point.")

    parser.add_argument("--env", choices=("minimal", "continuous"), default="minimal",
                        help="Environment to train on.")
    parser.add_argument("--no-sensors", action="store_false", dest="use_sensors",
                        help="Continuous env only: drop the 8 distance sensors from the "
                             "observation, leaving the bare (x, y, theta) state.")
    parser.add_argument("--step-size", type=float, default=None, dest="step_size",
                        help="Override the env move/step size (defaults: continuous 0.1, "
                             "minimal 1.0). Larger values mean fewer steps to cross the map, "
                             "so the goal is reachable in a shorter horizon and episodes run faster.")
    parser.add_argument("--agent", choices=("random", "dqn", "a3c"), default="random",
                        help="Agent to train.")
    parser.add_argument("--grid", type=Path, default=GRID_CONFIGS_FP / "small_grid.npy",
                        help="Path to a .npy grid file.")
    parser.add_argument("--start-pos", type=float, nargs=2, default=None, dest="start_pos",
                        metavar=("X", "Y"),
                        help="Fixed continuous (x, y) start for evaluation and visualization "
                             "(and for training unless --exploring-starts is set). Omit to use "
                             "the grid START_CELL or a random empty cell.")
    parser.add_argument("--eval-starting-pos", type=float, nargs=2, default=None, dest="eval_start_pos",
                        metavar=("X", "Y"),
                        help="Fixed continuous (x, y) start used only for evaluation and "
                             "visualization. Overrides --start-pos for the eval/viz env; "
                             "training start is unaffected. Omit to fall back to --start-pos.")
    parser.add_argument("--exploring-starts", action="store_true", dest="exploring_starts",
                        help="Use random start positions during training (exploring starts) "
                             "while evaluation keeps the fixed --start-pos / --eval-starting-pos.")

    parser.add_argument("--episodes", type=int, default=DEFAULT_TOTAL_EPISODES, help="Training episodes.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE, dest="max_steps",
                        help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL, dest="eval_interval",
                        help="Evaluate every N episodes.")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES, dest="eval_episodes",
                        help="Episodes per evaluation.")
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL, dest="log_interval",
                        help="Print/log metrics (and rollout image) every N episodes.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Compute device; 'auto' picks cuda > mps > cpu.")
    parser.add_argument("--epsilon", type=float, default=EPSILON_DEFAULT,
                        help="Start epsilon for annealing (or fixed rate if duration=0).")
    parser.add_argument("--epsilon-min", type=float, default=EPSILON_DEFAULT_MIN, dest="epsilon_min",
                        help="Minimum epsilon after annealing.")
    parser.add_argument("--epsilon-duration", type=int, default=EPSILON_ANNEAL_DURATION, dest="epsilon_duration",
                        help="Number of steps to anneal epsilon over.")
    parser.add_argument("--epsilon-start-step", type=int, default=EPSILON_ANNEAL_START_STEP, dest="epsilon_start_step",
                        help="Steps before epsilon annealing starts.")
    parser.add_argument("--gamma", type=float, default=DQN_DEFAULT_GAMMA, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=DQN_DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=DQN_DEFAULT_BATCH_SIZE, dest="batch_size", help="Batch size.")
    parser.add_argument("--replay-capacity", type=int, default=REPLAY_DEFAULT_CAPACITY, dest="replay_capacity",
                        help="Replay buffer capacity.")
    parser.add_argument("--stack-size", type=int, default=DQN_DEFAULT_NO_OBS_IN_STATE, dest="stack_size",
                        help="Number of observations to stack for state.")
    parser.add_argument("--update-freq", type=int, default=DQN_DEFAULT_UPDATE_FREQ, dest="update_freq",
                        help="Update online network every N steps.")
    parser.add_argument("--target-update-freq", type=int, default=DQN_DEFAULT_TARGET_UPDATE_FREQ, dest="target_update_freq",
                        help="Update target network every N steps.")

    parser.add_argument("--a3c-workers", type=int, default=A3C_N_WORKERS, dest="a3c_workers",
                        help="A3C only: number of asynchronous actor-learner processes.")
    parser.add_argument("--a3c-t-max", type=int, default=A3C_T_MAX, dest="a3c_t_max",
                        help="A3C only: max rollout length between gradient pushes.")
    parser.add_argument("--a3c-entropy-beta", type=float, default=A3C_ENTROPY_BETA, dest="a3c_entropy_beta",
                        help="A3C only: entropy regularization coefficient (uniform across workers).")
    parser.add_argument("--a3c-value-coef", type=float, default=A3C_VALUE_COEF, dest="a3c_value_coef",
                        help="A3C only: weight on the value loss.")
    parser.add_argument("--a3c-total-steps", type=int, default=None, dest="a3c_total_steps",
                        help="A3C only: global env-step budget (defaults to episodes * max-steps).")

    parser.add_argument("--out-dir", type=Path, default=None, dest="out_dir",
                        help="Output dir for checkpoints and history JSON.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-group", type=str, default=None, dest="wandb_group",
                        help="W&B group name to bucket related runs under.")

    parser.add_argument("--visualize", action="store_true",
                        help="Save a post-training rollout path image.")
    parser.add_argument("--viz-out", type=Path, default=None, dest="viz_out",
                        help="Visualization output path (defaults under --out-dir or CWD).")
    parser.add_argument("--viz-max-steps", type=int, default=DEFAULT_VIZ_MAX_STEPS, dest="viz_max_steps",
                        help="Max steps for the visualization rollout.")
    parser.add_argument("--curiosity", type=str, default="no", dest="curiosity",
                        help="What intrinsic motivation to use, currently supported: no, grid_count.")
    parser.add_argument("--curiosity-beta", type=float, default=0.5, dest="curiosity_beta",
                        help="Beta value for the curiosity term.")
    return parser.parse_args()


def _build_config(args: Namespace) -> TrainerConfig:
    """Materialise a TrainerConfig from parsed CLI args."""
    checkpoint_dir = str(args.out_dir) if args.out_dir is not None else None
    history_path = str(args.out_dir / "history.json") if args.out_dir is not None else None

    # Convert Path objects to strings for W&B logging
    full_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    return TrainerConfig(
        total_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        log_interval=args.log_interval,
        checkpoint_dir=checkpoint_dir,
        save_best=checkpoint_dir is not None,
        save_last=checkpoint_dir is not None,
        history_path=history_path,
        use_wandb=args.wandb,
        wandb_group=args.wandb_group,
        run_name=f"{args.agent}_{args.env}",
        full_config=full_config,
    )


def main() -> None:
    """Build env + agent + Trainer, train, and optionally visualize."""
    args = parse_args()
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    if args.agent == "a3c" and device != "cpu":
        # A3C shares the network across processes via CPU shared memory.
        print(f"[a3c] forcing device=cpu (was {device}) for shared-memory multiprocessing")
        device = "cpu"
    base_start = tuple(args.start_pos) if args.start_pos is not None else None
    # --eval-starting-pos overrides the eval/viz start independently of training.
    eval_start = tuple(args.eval_start_pos) if args.eval_start_pos is not None else base_start
    train_start = None if args.exploring_starts else base_start

    env = _build_env(args.env, args.grid, args.seed, train_start, args.use_sensors, args.step_size)
    eval_env = _build_env(args.env, args.grid, args.seed, eval_start, args.use_sensors, args.step_size)
    agent = _build_agent(args, env, device)
    config = _build_config(args)

    print(f"[{args.agent} | {args.env}] grid={args.grid.name} "
          f"state_dim={env.state_dim} n_actions={env.n_actions} device={device} "
          f"train_start={'random' if train_start is None else train_start} "
          f"eval_start={'random' if eval_start is None else eval_start}")

    viz_fn = None
    if args.visualize:
        viz_env = _build_env(args.env, args.grid, args.seed, eval_start, args.use_sensors, args.step_size)
        rollout_dir = (args.out_dir if args.out_dir is not None else Path(".")) / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        def viz_fn(rollout_agent: BaseAgent, episode: int,
                   _env: EnvType = viz_env, _dir: Path = rollout_dir) -> str:
            out = _dir / f"ep_{episode:06d}.png"
            return str(visualize_agent(
                _env, rollout_agent, max_steps=args.viz_max_steps, out=out,
                title=f"{args.agent} | {args.env} | ep {episode}",
            ))

    trainer = Trainer(env=env, agent=agent, config=config, eval_env=eval_env, viz_fn=viz_fn)
    history = trainer.train()

    print(f"Training finished. Logged episodes: {len(history)}")
    if history:
        print(f"Last episode metrics: {history[-1]}")

    if args.visualize:
        viz_out = args.viz_out
        if viz_out is None:
            base = args.out_dir if args.out_dir is not None else Path(".")
            viz_out = base / f"path_{args.agent}_{args.env}.png"
        viz_env = _build_env(args.env, args.grid, args.seed, eval_start, args.use_sensors, args.step_size)
        out = visualize_agent(
            viz_env,
            agent,
            max_steps=args.viz_max_steps,
            out=viz_out,
            title=f"{args.agent} agent on {args.env}\ngrid: {args.grid.name}",
        )
        print(f"saved visualization -> {out}")


if __name__ == "__main__":
    main()
