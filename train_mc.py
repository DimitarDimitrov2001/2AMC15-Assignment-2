"""Train and evaluate a Monte Carlo control agent."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from agents.mc_agent import MCAgent
from utils.artifacts import save_evaluation_summary_artifact, write_json
from utils.evaluation import evaluate_policy_metrics
from world import Environment, build_manhattan_reward_function, find_target_position


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Monte Carlo control trainer.")
    parser.add_argument("GRID", type=Path, nargs="+", help="Paths to one or more grid files.")
    parser.add_argument("--no_gui",        action="store_true")
    parser.add_argument("--fps",           type=int,   default=30)
    # Shared
    parser.add_argument("--sigma",         type=float, default=0.1,   help="Environment stochasticity.")
    parser.add_argument("--gamma",         type=float, default=0.9,   help="Discount factor.")
    parser.add_argument("--max_steps",     type=int,   default=500,   help="Max steps per eval rollout.")
    parser.add_argument("--eval_episodes", type=int,   default=20)
    parser.add_argument("--random_seed",   type=int,   default=0)
    parser.add_argument("--start_pos",     type=str,   default=None,  help="col,row (e.g. 2,3).")
    parser.add_argument("--out_dir",       type=Path,  default=Path("results"))
    # MC specific
    parser.add_argument("--episodes",          type=int,   default=5000,  help="Training episodes.")
    parser.add_argument("--max_episode_length",type=int,   default=2000,  help="Max steps per training episode.")
    parser.add_argument("--alpha",             type=float, default=None,  help="Fixed learning rate. Omit to use incremental 1/N mean.")
    parser.add_argument("--alpha_min",         type=float, default=0.001)
    parser.add_argument("--alpha_decay",       type=float, default=0.9995)
    parser.add_argument("--fixed_alpha",       action="store_true",       help="Disable alpha decay (requires --alpha).")
    parser.add_argument("--epsilon",           type=float, default=0.2,   help="Initial exploration rate.")
    parser.add_argument("--epsilon_min",       type=float, default=0.01)
    parser.add_argument("--epsilon_decay",     type=float, default=0.9995)
    parser.add_argument("--fixed_epsilon",     action="store_true",       help="Disable epsilon decay.")
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

        agent = MCAgent(
            gamma=args.gamma,
            epsilon=args.epsilon, epsilon_min=args.epsilon_min,
            epsilon_decay=1.0 if args.fixed_epsilon else args.epsilon_decay,
            alpha=args.alpha if (args.alpha is not None) else None,
            alpha_min=args.alpha_min,
            alpha_decay=1.0 if args.fixed_alpha else args.alpha_decay,
            max_episode_length=args.max_episode_length,
            random_seed=args.random_seed,
        )
        agent.train(
            env, n_episodes=args.episodes,
            start_pos=initial_pos, verbose=True, reward_fn=reward_fn,
        )

        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        prefix = f"{grid_path.stem}_mc_{timestamp}"

        metrics = evaluate_policy_metrics(
            grid=grid_path, agent=agent, max_steps=args.max_steps,
            sigma=args.sigma, agent_start_pos=initial_pos,
            reward_fn=reward_fn, gamma=args.gamma,
            random_seed=args.random_seed, n_eval_episodes=args.eval_episodes,
        )
        write_json(args.out_dir / f"{prefix}_metrics.json", {"evaluation": metrics})
        save_evaluation_summary_artifact(args.out_dir, prefix, metrics)
        Environment.evaluate_agent(
            grid_fp=grid_path, agent=agent, max_steps=args.max_steps,
            sigma=args.sigma, agent_start_pos=initial_pos,
            reward_fn=reward_fn, random_seed=args.random_seed,
            out_dir=args.out_dir, file_name=f"{prefix}_path",
        )
        print(f"[{grid_path.stem}] success_rate={metrics['success_rate']:.3f}  "
              f"mean_discounted_return={metrics['mean_discounted_return']:.3f}")


if __name__ == "__main__":
    main()
