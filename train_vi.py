"""Train and evaluate a Value Iteration agent."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from agents.value_iteration_agent import ValueIterationAgent
from utils.artifacts import save_evaluation_summary_artifact, save_value_iteration_artifacts
from world import Environment, build_manhattan_reward_function, find_target_position


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Value Iteration trainer.")
    parser.add_argument("GRID", type=Path, nargs="+", help="Paths to one or more grid files.")
    parser.add_argument("--no_gui",      action="store_true")
    parser.add_argument("--fps",         type=int,   default=30)
    parser.add_argument("--sigma",       type=float, default=0.1,  help="Environment stochasticity.")
    parser.add_argument("--gamma",       type=float, default=0.9,  help="Discount factor.")
    parser.add_argument("--theta",       type=float, default=1e-6, help="Convergence threshold.")
    parser.add_argument("--vi_max_iter", type=int,   default=1000, help="Maximum Bellman sweeps.")
    parser.add_argument("--max_steps",   type=int,   default=500,  help="Max steps per eval rollout.")
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--random_seed", type=int,   default=0)
    parser.add_argument("--start_pos",  type=str,   default=None,  help="col,row (e.g. 2,3).")
    parser.add_argument("--out_dir",    type=Path,  default=Path("results"))
    return parser.parse_args()


def _parse_start_pos(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    col, row = raw.split(",")
    return int(col), int(row)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_pos = _parse_start_pos(args.start_pos)

    for grid_path in args.GRID:
        env = Environment(
            grid_fp=grid_path, no_gui=args.no_gui,
            reward_fn=lambda _g, _p: 0,
            sigma=args.sigma, target_fps=args.fps,
            agent_start_pos=start_pos, random_seed=args.random_seed,
        )
        initial_pos = env.reset()
        env.agent_start_pos = initial_pos
        target_pos = find_target_position(env.grid)
        reward_fn = build_manhattan_reward_function(initial_pos, target_pos)
        env.reward_fn = reward_fn

        agent = ValueIterationAgent(
            grid=env.grid, reward_fn=reward_fn,
            sigma=args.sigma, gamma=args.gamma,
            theta=args.theta, max_iterations=args.vi_max_iter,
        )
        agent.train()

        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        prefix = f"{grid_path.stem}_value_iteration_{timestamp}"

        from utils.evaluation import evaluate_policy_metrics
        metrics = evaluate_policy_metrics(
            grid=grid_path, agent=agent, max_steps=args.max_steps,
            sigma=args.sigma, agent_start_pos=initial_pos,
            reward_fn=reward_fn, gamma=args.gamma,
            random_seed=args.random_seed, n_eval_episodes=args.eval_episodes,
        )
        save_value_iteration_artifacts(
            out_dir=args.out_dir, artifact_prefix=prefix,
            grid=env.grid, initial_pos=initial_pos,
            agent=agent, evaluation_metrics=metrics,
        )
        save_evaluation_summary_artifact(args.out_dir, prefix, metrics)
        Environment.evaluate_agent(
            grid_fp=grid_path, agent=agent, max_steps=args.max_steps,
            sigma=args.sigma, agent_start_pos=initial_pos,
            reward_fn=reward_fn, random_seed=args.random_seed,
            out_dir=args.out_dir, file_name=f"{prefix}_path",
        )
        print(f"[{grid_path.stem}] converged={agent.converged}  iterations={agent.iterations}  "
              f"success_rate={metrics['success_rate']:.3f}")


if __name__ == "__main__":
    main()
