from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from agents import DQNAgent, DQNConfig
from agents.dqn.normalization import StateNormalizer
from training import Trainer, TrainerConfig
from utils.dqn_artifacts import save_dqn_run_artifacts
from world.continuous_environment import ContinuousEnvironment


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    out_dir = args.out_dir
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / f"dqn_{timestamp}"

    grid_shape = tuple(np.load(args.grid).shape)
    normalizer = StateNormalizer(
        grid_shape=grid_shape,
        max_sensor_range=args.max_sensor_range,
    )

    train_env = _build_env(args, random_seed=args.seed)
    eval_env = _build_env(args, random_seed=args.seed + 10_000)

    dqn_config = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        optimizer=args.optimizer,
        hidden_sizes=tuple(args.hidden_sizes),
        buffer_capacity=args.buffer_capacity,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        train_frequency=args.train_frequency,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_steps=args.epsilon_decay_steps,
        reward_clip=args.reward_clip if args.reward_clip > 0 else None,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        device=args.device,
    )
    agent = DQNAgent(
        state_dim=train_env.state_dim,
        n_actions=train_env.n_actions,
        normalizer=normalizer,
        config=dqn_config,
        seed=args.seed,
    )

    trainer_config = TrainerConfig(
        total_episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        log_interval=args.log_interval,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_group="dqn",
        run_name=args.run_name,
    )
    trainer = Trainer(
        env=train_env,
        eval_env=eval_env,
        agent=agent,
        config=trainer_config,
    )

    history = trainer.train()
    run_config = _run_config(args, out_dir, grid_shape, trainer_config, dqn_config)
    rollout = _run_policy_rollout(
        args=args,
        agent=agent,
        max_steps=args.rollout_steps,
        random_seed=args.seed + 20_000,
    )
    model_path = save_dqn_run_artifacts(out_dir, run_config, history, agent, rollout=rollout)
    _print_summary(out_dir, model_path, history)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN baseline on ContinuousEnvironment.")
    parser.add_argument("grid", type=Path, help="Path to a .npy grid file.")

    core = parser.add_argument_group("core")
    core.add_argument("--out_dir", type=Path, default=None)
    core.add_argument("--episodes", type=int, default=500)
    core.add_argument("--max_steps_per_episode", type=int, default=500)
    core.add_argument("--eval_interval", type=int, default=25)
    core.add_argument("--eval_episodes", type=int, default=10)
    core.add_argument("--log_interval", type=int, default=1)
    core.add_argument("--seed", type=int, default=0)
    core.add_argument("--wandb", action="store_true")
    core.add_argument("--wandb_project", type=str, default="rl-in-practice-assignment-2")
    core.add_argument("--run_name", type=str, default=None)
    core.add_argument(
        "--rollout_steps",
        type=int,
        default=None,
        help="Max steps for the saved greedy policy rollout. Defaults to max_steps_per_episode.",
    )

    env = parser.add_argument_group("environment")
    env.add_argument("--step_size", type=float, default=0.5)
    env.add_argument("--rotation_step", type=float, default=30.0)
    env.add_argument("--max_sensor_range", type=float, default=3.0)
    env.add_argument("--action_sigma", type=float, default=0.0)
    env.add_argument("--sensory_sigma", type=float, default=0.0)
    env.add_argument("--start_pos", type=_parse_start_pos, default=None)
    env.add_argument("--initial_heading", type=float, default=0.0)

    dqn = parser.add_argument_group("DQN")
    dqn.add_argument("--gamma", type=float, default=0.99)
    dqn.add_argument("--lr", type=float, default=1e-3)
    dqn.add_argument("--optimizer", choices=("adam", "rmsprop"), default="adam")
    dqn.add_argument("--hidden_sizes", type=int, nargs="+", default=[128, 128])
    dqn.add_argument("--buffer_capacity", type=int, default=100_000)
    dqn.add_argument("--learning_starts", type=int, default=10_000)
    dqn.add_argument("--batch_size", type=int, default=64)
    dqn.add_argument("--target_update_interval", type=int, default=1_000)
    dqn.add_argument("--train_frequency", type=int, default=1)
    dqn.add_argument("--epsilon_start", type=float, default=1.0)
    dqn.add_argument("--epsilon_final", type=float, default=0.1)
    dqn.add_argument("--epsilon_decay_steps", type=int, default=100_000)
    dqn.add_argument("--reward_clip", type=float, default=1.0)
    dqn.add_argument("--grad_clip_norm", type=float, default=10.0)
    dqn.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.grid.exists():
        raise FileNotFoundError(f"Grid file not found: {args.grid}")
    positive_ints = {
        "episodes": args.episodes,
        "max_steps_per_episode": args.max_steps_per_episode,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "log_interval": args.log_interval,
        "batch_size": args.batch_size,
        "buffer_capacity": args.buffer_capacity,
        "target_update_interval": args.target_update_interval,
        "train_frequency": args.train_frequency,
        "epsilon_decay_steps": args.epsilon_decay_steps,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.learning_starts < 0:
        raise ValueError("learning_starts must be non-negative")
    if args.max_sensor_range <= 0:
        raise ValueError("max_sensor_range must be positive")
    if args.epsilon_final > args.epsilon_start:
        raise ValueError("epsilon_final must be <= epsilon_start")
    if args.rollout_steps is None:
        args.rollout_steps = args.max_steps_per_episode
    if args.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")


def _build_env(args: argparse.Namespace, random_seed: int) -> ContinuousEnvironment:
    return ContinuousEnvironment(
        grid_fp=args.grid,
        step_size=args.step_size,
        rotation_step=args.rotation_step,
        max_sensor_range=args.max_sensor_range,
        action_sigma=args.action_sigma,
        sensory_sigma=args.sensory_sigma,
        agent_start_pos=args.start_pos,
        initial_heading=args.initial_heading,
        random_seed=random_seed,
    )


def _parse_start_pos(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("start_pos must be formatted as x,y")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("start_pos must contain numeric x,y values") from exc


def _run_config(
    args: argparse.Namespace,
    out_dir: Path,
    grid_shape: tuple[int, int],
    trainer_config: TrainerConfig,
    dqn_config: DQNConfig,
) -> dict[str, Any]:
    return {
        "grid": str(args.grid),
        "grid_shape": grid_shape,
        "out_dir": str(out_dir),
        "environment": {
            "step_size": args.step_size,
            "rotation_step": args.rotation_step,
            "max_sensor_range": args.max_sensor_range,
            "action_sigma": args.action_sigma,
            "sensory_sigma": args.sensory_sigma,
            "start_pos": args.start_pos,
            "initial_heading": args.initial_heading,
        },
        "trainer": asdict(trainer_config),
        "dqn": asdict(dqn_config),
        "visualization": {
            "rollout_steps": args.rollout_steps,
        },
    }


def _run_policy_rollout(
    args: argparse.Namespace,
    agent: DQNAgent,
    max_steps: int,
    random_seed: int,
) -> dict[str, Any]:
    env = _build_env(args, random_seed=random_seed)
    state = env.reset()
    initial_grid = np.copy(env.grid)
    positions = [env.pos.copy()]
    headings = [float(env.theta)]
    actions: list[int] = []
    rewards: list[float] = []
    infos: list[dict[str, Any]] = []
    total_reward = 0.0
    terminated = False
    truncated = False

    for step_idx in range(max_steps):
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, info = env.step(action)
        if step_idx == max_steps - 1 and not terminated:
            truncated = True
            info["time_limit"] = True

        actions.append(int(action))
        rewards.append(float(reward))
        infos.append(_json_safe_info(info))
        total_reward += float(reward)
        positions.append(env.pos.copy())
        headings.append(float(env.theta))
        state = next_state

        if terminated or truncated:
            break

    return {
        "grid": initial_grid,
        "positions": np.asarray(positions, dtype=float),
        "headings": headings,
        "actions": actions,
        "rewards": rewards,
        "infos": infos,
        "total_reward": float(total_reward),
        "steps": len(actions),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(terminated),
        "world_stats": dict(env.world_stats),
    }


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def _print_summary(out_dir: Path, model_path: Path, history: list[dict[str, float]]) -> None:
    final = history[-1] if history else {}
    eval_rows = [row for row in history if "eval/mean_reward" in row]
    last_eval = eval_rows[-1] if eval_rows else {}
    print()
    print("DQN training finished.")
    print(f"output_dir: {out_dir}")
    print(f"model_path: {model_path}")
    print(f"final_train_reward: {_fmt(final.get('train/episode_reward'))}")
    print(f"last_eval_mean_reward: {_fmt(last_eval.get('eval/mean_reward'))}")
    print(f"last_eval_success_rate: {_fmt(last_eval.get('eval/success_rate'))}")


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
