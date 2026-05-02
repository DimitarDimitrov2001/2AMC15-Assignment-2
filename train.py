"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import trange

from world import Environment, build_manhattan_reward_function, find_target_position
from agents.random_agent import RandomAgent


def parse_args() -> Namespace:
    """Parse command line arguments for the training script."""

    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file to use. There can be more than " "one.")
    p.add_argument("--no_gui", action="store_true", help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1, help="Sigma value for the stochasticity of the environment.")
    p.add_argument(
        "--fps", type=int, default=30, help="Frames per second to render at. Only used if " "no_gui is not set."
    )
    p.add_argument("--iter", type=int, default=1000, help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0, help="Random seed value for the environment.")
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


def main(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    random_seed: int,
    start_pos: tuple[int, int] | None,
) -> None:
    """Main loop of the program."""

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
        agent = RandomAgent()

        # Always reset the environment to initial state
        initial_pos = env.reset()
        target_pos = find_target_position(env.grid)
        env.reward_fn = build_manhattan_reward_function(initial_pos, target_pos)
        state = initial_pos
        for _ in trange(iters):

            # Agent takes an action based on the latest observation and info.
            action = agent.take_action(state)

            # The action is performed in the environment
            state, reward, terminated, info = env.step(action)

            # If the final state is reached, stop.
            if terminated:
                break

            agent.update(state, reward, info["actual_action"])

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, agent_start_pos=initial_pos, random_seed=random_seed)


if __name__ == "__main__":
    args = parse_args()
    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(",")
        start_pos = (int(parts[0]), int(parts[1]))
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, start_pos)
