"""Train and evaluate RL agents for the delivery grid world."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from tqdm import trange

from agents.random_agent import RandomAgent
from agents.value_iteration_agent import ValueIterationAgent
from utils.artifacts import (
    save_evaluation_summary_artifact,
    save_value_iteration_artifacts,
    write_json,
)
from utils.evaluation import evaluate_policy_metrics
from world import Environment, build_manhattan_reward_function, find_target_position


def parse_args() -> Namespace:
    """Parse command line arguments for the training script."""

    parser = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    parser.add_argument("GRID", type=Path, nargs="+", help="Paths to one or more grid files.")
    parser.add_argument(
        "--agent",
        choices=("value_iteration", "random"),
        default="value_iteration",
        help="Agent to train/evaluate.",
    )
    parser.add_argument("--no_gui", action="store_true", help="Disables rendering to train faster.")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Sigma value for the stochasticity of the environment.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second to render at. Only used if no_gui is not set.",
    )
    parser.add_argument("--iter", type=int, default=1000, help="Max environment steps for rollouts/evaluation.")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Discount factor for value iteration and evaluation.",
    )
    parser.add_argument("--theta", type=float, default=1e-6, help="Value-iteration convergence threshold.")
    parser.add_argument("--vi_max_iter", type=int, default=1000, help="Maximum Bellman sweeps for value iteration.")
    parser.add_argument("--out_dir", type=Path, default=Path("results"), help="Directory for metrics and plots.")
    parser.add_argument("--eval_episodes", type=int, default=20, help="Number of evaluation rollouts for metrics.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed value for the environment.")
    parser.add_argument(
        "--start_pos",
        type=str,
        default=None,
        help="Agent start position as col,row (e.g. 2,3). "
        "If not set, the GUI lets you click to place it. "
        "In no_gui mode, defaults to random placement.",
    )
    return parser.parse_args()


def _uninitialized_reward_function(_grid: object, _agent_pos: tuple[int, int]) -> int:
    raise RuntimeError("Reward function must be initialized after the environment reset.")


def _parse_start_pos(raw_start_pos: str | None) -> tuple[int, int] | None:
    if raw_start_pos is None:
        return None
    parts = raw_start_pos.split(",")
    if len(parts) != 2:
        raise ValueError("--start_pos must be formatted as col,row")
    return int(parts[0]), int(parts[1])


def main(
    grid_paths: list[Path],
    agent_name: str,
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    gamma: float,
    theta: float,
    vi_max_iter: int,
    out_dir: Path,
    eval_episodes: int,
    random_seed: int,
    start_pos: tuple[int, int] | None,
) -> None:
    """Main training and evaluation loop."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for grid_path in grid_paths:
        env = Environment(
            grid_fp=grid_path,
            no_gui=no_gui,
            reward_fn=_uninitialized_reward_function,
            sigma=sigma,
            target_fps=fps,
            agent_start_pos=start_pos,
            random_seed=random_seed,
        )

        initial_pos = env.reset()
        target_pos = find_target_position(env.grid)
        reward_fn = build_manhattan_reward_function(initial_pos, target_pos)
        env.reward_fn = reward_fn
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        artifact_prefix = f"{grid_path.stem}_{agent_name}_{timestamp}"

        if agent_name == "value_iteration":
            agent = ValueIterationAgent(
                grid=env.grid,
                reward_fn=reward_fn,
                sigma=sigma,
                gamma=gamma,
                theta=theta,
                max_iterations=vi_max_iter,
            )
            agent.train()
        elif agent_name == "random":
            agent = RandomAgent()
            state = initial_pos
            for _ in trange(iters, desc=f"Training random agent on {grid_path.name}"):
                action = agent.take_action(state)
                state, reward, terminated, info = env.step(action)
                if terminated:
                    break
                agent.update(state, reward, info["actual_action"])
        else:
            raise ValueError(f"Unsupported agent: {agent_name}")

        evaluation_metrics = evaluate_policy_metrics(
            grid=grid_path,
            agent=agent,
            max_steps=iters,
            sigma=sigma,
            agent_start_pos=initial_pos,
            reward_fn=reward_fn,
            gamma=gamma,
            random_seed=random_seed,
            n_eval_episodes=eval_episodes,
        )

        if isinstance(agent, ValueIterationAgent):
            save_value_iteration_artifacts(
                out_dir=out_dir,
                artifact_prefix=artifact_prefix,
                grid=env.grid,
                initial_pos=initial_pos,
                agent=agent,
                evaluation_metrics=evaluation_metrics,
            )
        else:
            write_json(out_dir / f"{artifact_prefix}_metrics.json", {"evaluation": evaluation_metrics})
        save_evaluation_summary_artifact(out_dir, artifact_prefix, evaluation_metrics)

        Environment.evaluate_agent(
            grid_fp=grid_path,
            agent=agent,
            max_steps=iters,
            sigma=sigma,
            agent_start_pos=initial_pos,
            reward_fn=reward_fn,
            random_seed=random_seed,
            out_dir=out_dir,
            file_name=f"{artifact_prefix}_path",
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.GRID,
        args.agent,
        args.no_gui,
        args.iter,
        args.fps,
        args.sigma,
        args.gamma,
        args.theta,
        args.vi_max_iter,
        args.out_dir,
        args.eval_episodes,
        args.random_seed,
        _parse_start_pos(args.start_pos),
    )
