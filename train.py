"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import trange

from world import Environment, build_manhattan_reward_function, find_target_position
from agents.random_agent import RandomAgent
from agents.q_learning_agent import QLearningAgent


def parse_args() -> Namespace:
    """Parse command line arguments for the training script."""

    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file to use. There can be more than " "one.")

    # Added to help choose an agent
    p.add_argument("--agent", choices=("random", "q_learning"), default="q_learning", help="Agent to run.")

    p.add_argument("--no_gui", action="store_true", help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1, help="Sigma value for the stochasticity of the environment.")
    p.add_argument(
        "--fps", type=int, default=30, help="Frames per second to render at. Only used if " "no_gui is not set."
    )
    p.add_argument("--iter", type=int, default=1000, help="Number of steps per episode.")
    p.add_argument("--episodes", type=int, default=1000, help="Number of Q-learning training episodes.")
    p.add_argument("--alpha", type=float, default=0.5, help="Q-learning learning rate.")
    p.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    p.add_argument("--epsilon", type=float, default=1.0, help="Q-learning initial exploration rate.")
    p.add_argument("--epsilon_min", type=float, default=0.05, help="Q-learning minimum exploration rate.")
    p.add_argument("--epsilon_decay", type=float, default=0.995, help="Decay rate of epsilon after every episode.")
    p.add_argument("--alpha_min", type=float, default=0.05, help="Q-learning minimum learning rate.")
    p.add_argument("--alpha_decay", type=float, default=0.999, help="Decay rate of alpha after every episode.")
    p.add_argument("--random_seed", type=int, default=0, help="Random seed value for the environment.")
    p.add_argument("--fixed_epsilon", action="store_true", help="Use fixed epsilon instead of decaying epsilon.")
    p.add_argument("--fixed_alpha", action="store_true", help="Use fixed alpha instead of decaying alpha.")
    p.add_argument(
        "--start_pos",
        type=str,
        default=None,
        help="Agent start position as col,row (e.g. 2,3). "
        "If not set, the GUI lets you click to place it. "
        "In no_gui mode, defaults to random placement.",
    )
    return p.parse_args()


def _uninitialized_reward_function(_grid: object, _agent_pos: tuple[int, int]) -> int:
    raise RuntimeError("Reward function must be initialized after the environment reset.")

def parse_start_pos(raw_start_pos: str | None) -> tuple[int, int] | None:
    if raw_start_pos is None:
        return None
    parts = raw_start_pos.split(",")
    if len(parts) != 2:
        raise ValueError("--start_pos must be formatted as col,row")
    return int(parts[0]), int(parts[1])

def train_q_learning_agent(
    env: Environment,
    agent: QLearningAgent,
    episodes: int,
    max_steps_per_episode: int,
) -> None:

    for _episode in trange(episodes, desc="Training Q-learning agent"):
        state = env.reset()
        agent.start_episode()

        for _step in range(max_steps_per_episode):
            action = agent.take_action(state)

            next_state, reward, terminated, _info = env.step(action)

            agent.update(
                next_state,
                reward,
                action,
                terminated=terminated,
            )

            state = next_state

            if terminated:
                break

        agent.end_episode()

# Main program loop
def main(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    episodes: int,
    fps: int,
    sigma: float,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    alpha_min: float,
    alpha_decay: float,
    fixed_epsilon: bool,
    fixed_alpha: bool,
    random_seed: int,
    start_pos: tuple[int, int] | None,
) -> None:

    for grid in grid_paths:
        # Set up the environment
        env = Environment(
            grid,
            no_gui,
            reward_fn=_uninitialized_reward_function,
            sigma=sigma,
            target_fps=fps,
            agent_start_pos=start_pos,
            random_seed=random_seed,
        )

        # Initialize agent
        # agent = RandomAgent()

        # Reset the environment once to obtain initial state
        initial_pos = env.reset()
        # Use the same start position for every episode
        env.agent_start_pos = initial_pos
        target_pos = find_target_position(env.grid)
        env.reward_fn = build_manhattan_reward_function(initial_pos, target_pos)
        
        
        agent = QLearningAgent(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            alpha_min=alpha_min,
            alpha_decay=alpha_decay,
            decaying_epsilon=not fixed_epsilon,
            decaying_alpha=not fixed_alpha,
            n_actions=4,
        )

        train_q_learning_agent(
            env=env,
            agent=agent,
            episodes=episodes,
            max_steps_per_episode=iters,
        )

        # Evaluate the agent
        agent.set_eval_mode()
        Environment.evaluate_agent(grid, agent, iters, sigma, agent_start_pos=initial_pos, random_seed=random_seed)


if __name__ == "__main__":
    args = parse_args()

    main(
        grid_paths=args.GRID,
        no_gui=args.no_gui,
        iters=args.iter,
        episodes=args.episodes,
        fps=args.fps,
        sigma=args.sigma,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        alpha_min=args.alpha_min,
        alpha_decay=args.alpha_decay,
        fixed_epsilon=args.fixed_epsilon,
        fixed_alpha=args.fixed_alpha,
        random_seed=args.random_seed,
        start_pos=parse_start_pos(args.start_pos),
    )
