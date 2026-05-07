"""Train and evaluate a Q-Learning agent."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from tqdm import trange

from agents.q_learning_agent import QLearningAgent
from utils.artifacts import save_evaluation_summary_artifact, write_json
from utils.evaluation import evaluate_policy_metrics
from world import Environment, build_manhattan_reward_function, find_target_position


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Q-Learning trainer.")
    parser.add_argument("GRID", type=Path, nargs="+", help="Paths to one or more grid files.")
    parser.add_argument("--no_gui",        action="store_true")
    parser.add_argument("--fps",           type=int,   default=30)
    # Shared
    parser.add_argument("--sigma",         type=float, default=0.1,   help="Environment stochasticity.")
    parser.add_argument("--gamma",         type=float, default=0.9,   help="Discount factor.")
    parser.add_argument("--max_steps",     type=int,   default=500,   help="Max env steps per episode/rollout.")
    parser.add_argument("--eval_episodes", type=int,   default=20)
    parser.add_argument("--random_seed",   type=int,   default=0)
    parser.add_argument("--start_pos",     type=str,   default=None,  help="col,row (e.g. 2,3).")
    parser.add_argument("--out_dir",       type=Path,  default=Path("results"))
    # Q-Learning specific
    parser.add_argument("--episodes",      type=int,   default=1000,  help="Training episodes.")
    parser.add_argument("--alpha",         type=float, default=0.5,   help="Initial learning rate.")
    parser.add_argument("--alpha_min",     type=float, default=0.05,  help="Minimum learning rate.")
    parser.add_argument("--alpha_decay",   type=float, default=0.999, help="Decay rate for alpha.")
    parser.add_argument("--fixed_alpha",   action="store_true",       help="Disable alpha decay.")
    parser.add_argument("--epsilon",       type=float, default=1.0,   help="Initial exploration rate.")
    parser.add_argument("--epsilon_min",   type=float, default=0.05,  help="Minimum exploration rate.")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Decay rate for epsilon.")
    parser.add_argument("--fixed_epsilon", action="store_true",       help="Disable epsilon decay.")
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

        agent = QLearningAgent(
            alpha=args.alpha, gamma=args.gamma,
            epsilon=args.epsilon, epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            alpha_min=args.alpha_min, alpha_decay=args.alpha_decay,
            decaying_epsilon=not args.fixed_epsilon,
            decaying_alpha=not args.fixed_alpha,
            n_actions=4,
        )

        for _ in trange(args.episodes, desc=f"Training Q-learning on {grid_path.name}"):
            state = env.reset()
            env.reward_fn = reward_fn
            agent.start_episode()
            for _ in range(args.max_steps):
                action = agent.take_action(state)
                next_state, reward, terminated, _ = env.step(action)
                agent.update(next_state, reward, action, terminated=terminated)
                state = next_state
                if terminated:
                    break
            agent.end_episode()

        agent.set_eval_mode()

        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        prefix = f"{grid_path.stem}_q_learning_{timestamp}"

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
