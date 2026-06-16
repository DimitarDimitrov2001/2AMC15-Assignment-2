"""Deep-RL training CLI.

Entry point for the new continuous/minimal environments and the algorithm-
agnostic Trainer. Only the random agent is wired today; learning agents
(DQN, PPO, ...) plug in via the agent factory without touching this script.

Usage:
    uv run python train_deep.py --env minimal --episodes 20000 --visualize
    uv run python train_deep.py --env continuous --grid grid_configs/small_grid.npy
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

# Force a headless backend before pyplot is imported (via visualize_random_agent),
# so periodic in-training rendering never tries to open a GUI window.
import matplotlib
matplotlib.use("Agg")

import torch

from agents import RandomAgent
from agents.base_agent import BaseAgent
from agents.dqn_agent import DQNAgent
from agents.epsilon_schedules import ConstantEpsilon, _DEFAULT_EPSILON
from training import Trainer, TrainerConfig
from world import GRID_CONFIGS_FP, ContinuousEnvironment, MinimalEnvironment
from visualize_random_agent import visualize_agent


# Environments expose a common subset of the interface the Trainer needs.
EnvType = MinimalEnvironment | ContinuousEnvironment


def _build_env(name: str, grid: Path, seed: int,
               start_pos: tuple[float, float] | None = None) -> EnvType:
    """Construct the requested environment with sensible defaults.
    """
    if name == "minimal":
        return MinimalEnvironment(
            grid_fp=grid,
            step_size=0.5,
            sigma=0.0,
            agent_start_pos=start_pos,
            random_seed=seed,
        )
    if name == "continuous":
        return ContinuousEnvironment(
            grid_fp=grid,
            step_size=0.5,
            rotation_step=30.0,
            max_sensor_range=3.0,
            action_sigma=0.0,
            sensory_sigma=0.0,
            agent_start_pos=start_pos,
            initial_heading=0.0,
            random_seed=seed,
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


def _build_agent(name: str, env: EnvType, seed: int, device: str, epsilon: float) -> BaseAgent:
    """Construct the requested agent.

    A registry keyed by ``--agent`` so learning agents drop in later without
    changing the training flow.
    """
    builders = {
        "random": lambda: RandomAgent(num_actions=env.n_actions, seed=seed),
        "dqn": lambda: DQNAgent(
            env=env,
            seed=seed,
            device=device,
            epsilon_scheduler=ConstantEpsilon(epsilon),
        ),
    }
    if name not in builders:
        raise ValueError(f"Unknown agent: {name}")
    return builders[name]()


def parse_args() -> Namespace:
    """Parse the deep-RL training CLI arguments."""
    parser = ArgumentParser(description="Deep-RL training entry point.")

    parser.add_argument("--env", choices=("minimal", "continuous"), default="minimal",
                        help="Environment to train on.")
    parser.add_argument("--agent", choices=("random", "dqn"), default="random",
                        help="Agent to train.")
    parser.add_argument("--grid", type=Path, default=GRID_CONFIGS_FP / "small_grid.npy",
                        help="Path to a .npy grid file.")
    parser.add_argument("--start-pos", type=float, nargs=2, default=None, dest="start_pos",
                        metavar=("X", "Y"),
                        help="Fixed continuous (x, y) start for evaluation and visualization "
                             "(and for training unless --exploring-starts is set). Omit to use "
                             "the grid START_CELL or a random empty cell.")
    parser.add_argument("--exploring-starts", action="store_true", dest="exploring_starts",
                        help="Use random start positions during training (exploring starts) "
                             "while evaluation keeps the fixed --start-pos.")

    parser.add_argument("--episodes", type=int, default=20_000, help="Training episodes.")
    parser.add_argument("--max-steps", type=int, default=200, dest="max_steps",
                        help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--eval-interval", type=int, default=10, dest="eval_interval",
                        help="Evaluate every N episodes.")
    parser.add_argument("--eval-episodes", type=int, default=5, dest="eval_episodes",
                        help="Episodes per evaluation.")
    parser.add_argument("--log-interval", type=int, default=1, dest="log_interval",
                        help="Print/log metrics (and rollout image) every N episodes.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Compute device; 'auto' picks cuda > mps > cpu.")
    parser.add_argument("--epsilon", type=float, default=_DEFAULT_EPSILON,
                        help="Fixed epsilon-greedy exploration rate for DQN (default 0.1).")

    parser.add_argument("--out-dir", type=Path, default=None, dest="out_dir",
                        help="Output dir for checkpoints and history JSON.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-group", type=str, default=None, dest="wandb_group",
                        help="W&B group name to bucket related runs under.")

    parser.add_argument("--visualize", action="store_true",
                        help="Save a post-training rollout path image.")
    parser.add_argument("--viz-out", type=Path, default=None, dest="viz_out",
                        help="Visualization output path (defaults under --out-dir or CWD).")
    parser.add_argument("--viz-max-steps", type=int, default=500, dest="viz_max_steps",
                        help="Max steps for the visualization rollout.")
    return parser.parse_args()


def _build_config(args: Namespace) -> TrainerConfig:
    """Materialise a TrainerConfig from parsed CLI args."""
    checkpoint_dir = str(args.out_dir) if args.out_dir is not None else None
    history_path = str(args.out_dir / "history.json") if args.out_dir is not None else None

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
    )


def main() -> None:
    """Build env + agent + Trainer, train, and optionally visualize."""
    args = parse_args()
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    eval_start = tuple(args.start_pos) if args.start_pos is not None else None
    train_start = None if args.exploring_starts else eval_start

    env = _build_env(args.env, args.grid, args.seed, train_start)
    eval_env = _build_env(args.env, args.grid, args.seed, eval_start)
    agent = _build_agent(args.agent, env, args.seed, device, args.epsilon)
    config = _build_config(args)

    print(f"[{args.agent} | {args.env}] grid={args.grid.name} "
          f"state_dim={env.state_dim} n_actions={env.n_actions} device={device} "
          f"train_start={'random' if train_start is None else train_start} "
          f"eval_start={'random' if eval_start is None else eval_start}")

    viz_fn = None
    if args.visualize:
        viz_env = _build_env(args.env, args.grid, args.seed, eval_start)
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
        viz_env = _build_env(args.env, args.grid, args.seed, eval_start)
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
