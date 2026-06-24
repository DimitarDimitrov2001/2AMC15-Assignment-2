"""Deep-RL training CLI.

Entry point for the new continuous/minimal environments and the algorithm-
agnostic Trainer. Learning agents (DQN, Dueling-DQN, A3C, ...) plug in via
the agent factory without touching this script.

Usage:
    uv run python train_deep.py --env minimal
    uv run python train_deep.py --env continuous --grid grid_configs/small_grid.npy
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import functools
from argparse import ArgumentParser, Namespace
from pathlib import Path
import re
import tempfile
from typing import Any

# Force a headless backend before pyplot is imported (via visualize_random_agent),
# so periodic in-training rendering never tries to open a GUI window.
import matplotlib
import numpy as np

from agents.curiosity import GridCountMotivation, NoMotivation
from world.defaults import COLLISION_PENALTY, GOAL_REWARD, LIVING_PENALTY
from world.environment_base import BaseGridEnvironment, RewardFn, cell_index
from world.grid_codes import TARGET_CELL
matplotlib.use("Agg")

import torch

from agents import RandomAgent
from agents.base_agent import BaseAgent
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DuelingDQNAgent
from agents.a3c_agent import A3CAgent
from agents.epsilon_schedules import ConstantEpsilon, EpsilonSchedule, ExponentialEpsilonDecay, LinearEpsilonAnnealing
from agents.defaults import (
    A3C_DEFAULT_TOTAL_STEPS,
    A3C_LEARNING_RATE,
    CURIOSITY_RESOLUTION_DEFAULT,
    EPSILON_DEFAULT_DECAY,
    EPSILON_DEFAULT_MAX,
    EPSILON_DEFAULT_MIN,
    EPSILON_ANNEAL_DURATION,
    EPSILON_ANNEAL_START_STEP,
    DQN_DEFAULT_GAMMA,
    DQN_DEFAULT_LEARNING_RATE,
    DQN_DEFAULT_BATCH_SIZE,
    DQN_DEFAULT_GRAD_CLIP_NORM,
    EPSILON_SCHEDULER_DEFAULT,
    REPLAY_DEFAULT_CAPACITY,
    DQN_DEFAULT_NO_OBS_IN_STATE,
    DQN_DEFAULT_UPDATE_FREQ,
    DQN_DEFAULT_TARGET_UPDATE_FREQ,
    A3C_N_WORKERS,
    A3C_PROGRESS_REWARD_SCALE,
    A3C_RANDOM_ACTION_DECAY_STEPS,
    A3C_RANDOM_ACTION_FINAL,
    A3C_RANDOM_ACTION_START,
    A3C_T_MAX,
    A3C_ENTROPY_BETA,
    A3C_VALUE_COEF,
    BETA_DEFAULT,
)
from training import Trainer, TrainerConfig
from training.defaults import (
    DEFAULT_ENV_NAME,
    DEFAULT_GRID_FILENAME,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOTAL_EPISODES,
    DEFAULT_MAX_STEPS_PER_EPISODE,
    DEFAULT_SEED,
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_FINAL_EVAL_RUNS,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_VIZ_MAX_STEPS,
    DEFAULT_WANDB_VIZ_INTERVAL,
)
from utils.artifacts import (
    aggregate_rollout_metrics,
    log_wandb_artifact,
    save_deep_rl_run_artifacts,
)
from world import GRID_CONFIGS_FP, ContinuousEnvironment, MinimalEnvironment
from visualize_random_agent import visualize_agent

def _build_reward_fn(
        target_reward: float,
        living_penalty: float,
        collision_penalty: float
    ) -> RewardFn:
    def reward_fn(
        grid: np.ndarray,
        pos: np.ndarray,
        new_pos: np.ndarray,
        collision: bool,
    ) -> float:
        """Return the default step reward.
        """
        if collision:
            return collision_penalty
        i, j = cell_index(new_pos)
        if grid[i, j] == TARGET_CELL:
            return target_reward
        return living_penalty
    return reward_fn

def _build_env(
    name: str,
    grid: Path,
    seed: int,
    start_pos: tuple[float, float] | None = None,
    use_sensors: bool = True,
    step_size: float | None = None,
    reward_fn: RewardFn | None = None,
    sigma: float = 0.0,
) -> BaseGridEnvironment:
    """Construct the requested environment with sensible defaults.

    ``use_sensors`` only affects the continuous environment (toggles the
    distance-sensor readings in the observation); it is ignored otherwise.
    ``step_size`` overrides the env's default move size when provided.
    ``sigma`` sets continuous-env action noise std-dev; ignored for minimal.
    """
    # Only override the env's own default step size when explicitly requested.
    step_kwargs = {"step_size": step_size} if step_size is not None else {}
    if name == "minimal":
        return MinimalEnvironment(
            grid_fp=grid,
            agent_start_pos=start_pos,
            random_seed=seed,
            reward_fn=reward_fn,
            **step_kwargs,
        )
    if name == "continuous":
        return ContinuousEnvironment(
            grid_fp=grid,
            agent_start_pos=start_pos,
            use_sensors=use_sensors,
            random_seed=seed,
            reward_fn=reward_fn,
            sigma=sigma,
            **step_kwargs,
        )
    raise ValueError(f"Unknown environment: {name}")

def _build_epsilon_schedule(
        choice: str,
        epsilon_duration: int,
        epsilon_start_step: int,
        epsilon_max: float,
        epsilon_min: float,
        decay: float
    ) -> EpsilonSchedule:
    """Resolves epsilon schedule method and returns the scheduler"""
    if choice == "linear_annealing":
        return LinearEpsilonAnnealing(
            duration=epsilon_duration,
            start_step=epsilon_start_step,
            epsilon_max=epsilon_max,
            epsilon_min=epsilon_min,
        ) 
    elif choice == "exponential_decay":
        return ExponentialEpsilonDecay(
            epsilon_max=epsilon_max,
            epsilon_min=epsilon_min,
            decay=decay
        )
    elif choice == "constant":
        return ConstantEpsilon(epsilon=epsilon_max)


def _resolve_device(choice: str) -> str:
    """Resolve 'auto' to cuda > mps > cpu; otherwise return the explicit choice."""
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_intrinsic_motivation(args: Namespace, env: BaseGridEnvironment) -> NoMotivation | GridCountMotivation:
    """Return the curiosity module selected on the CLI."""
    if args.curiosity == "grid_count":
        return GridCountMotivation(
            max_x=env.observation_high[0],
            max_y=env.observation_high[1],
            resolution=CURIOSITY_RESOLUTION_DEFAULT,
            beta=args.curiosity_beta,
        )
    return NoMotivation()


def _build_dqn_agent(args: Namespace, env: BaseGridEnvironment, device: str, agent_cls: type[DQNAgent]) -> DQNAgent:
    """Construct a DQN-family agent with shared deep-RL hyperparameters."""
    return agent_cls(
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
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        epsilon_scheduler=_build_epsilon_schedule(
            choice=args.epsilon_schedule,
            epsilon_max=args.epsilon_max,
            epsilon_min=args.epsilon_min,
            decay=args.decay,
            epsilon_start_step=args.epsilon_start_step,
            epsilon_duration=args.epsilon_duration,
        ),
        intrinsic_motivation=_build_intrinsic_motivation(args, env),
    )


def _build_agent(args: Namespace, env: BaseGridEnvironment, device: str) -> BaseAgent:
    """Construct the requested agent.

    A registry keyed by ``--agent`` so learning agents drop in later without
    changing the training flow.
    """
    builders = {
        "random": lambda: RandomAgent(num_actions=env.n_actions, seed=args.seed),
        "dqn": lambda: _build_dqn_agent(args, env, device, DQNAgent),
        "dueling-dqn": lambda: _build_dqn_agent(args, env, device, DuelingDQNAgent),
        "ddqn": lambda: _build_dqn_agent(args, env, device, DuelingDQNAgent),
        "a3c": lambda: A3CAgent(
            env=env,
            # Picklable factory (module-level fn + primitive args) so each worker
            # builds its own env under the spawn start method.
            env_fn=functools.partial(
                _build_env,
                args.env,
                args.grid,
                start_pos=(None if args.exploring_starts
                           else (tuple(args.start_pos) if args.start_pos is not None else 
                                 (tuple(env.agent_start_pos) if env.agent_start_pos is not None else None))),
                use_sensors=args.use_sensors,
                step_size=args.step_size,
                sigma=args.sigma,
            ),
            seed=args.seed,
            total_steps=(args.a3c_total_steps if args.a3c_total_steps is not None
                         else args.episodes * args.max_steps),
            max_steps_per_episode=args.max_steps,
            n_workers=args.a3c_workers,
            t_max=args.a3c_t_max,
            gamma=args.gamma,
            learning_rate=args.a3c_lr,
            entropy_beta=args.a3c_entropy_beta,
            value_coef=args.a3c_value_coef,
            random_action_start=args.a3c_random_action_start,
            random_action_final=args.a3c_random_action_final,
            random_action_decay_steps=args.a3c_random_action_decay_steps,
            progress_reward_scale=args.a3c_progress_reward_scale,
            curiosity_beta=(args.curiosity_beta if args.curiosity == "grid_count" else 0.0),
            curiosity_resolution=CURIOSITY_RESOLUTION_DEFAULT,
            device=device,
        ),
    }
    if args.agent not in builders:
        raise ValueError(f"Unknown agent: {args.agent}")
    return builders[args.agent]()


def parse_args() -> Namespace:
    """Parse the deep-RL training CLI arguments."""
    parser = ArgumentParser(description="Deep-RL training entry point.")

    parser.add_argument("--env", choices=("minimal", "continuous"), default=DEFAULT_ENV_NAME,
                        help="Environment to train on.")
    parser.add_argument("--no-sensors", action="store_false", dest="use_sensors",
                        help="Continuous env only: drop the 8 distance sensors from the "
                             "observation, leaving the bare (x, y, theta) state.")
    parser.add_argument("--step-size", type=float, default=0.5, dest="step_size",
                        help="Override the env move/step size (defaults: continuous 0.5, "
                             "minimal 0.5). Larger values mean fewer steps to cross the map, "
                             "so the goal is reachable in a shorter horizon and episodes run faster.")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Continuous env only: action noise std-dev (action_sigma); "
                             "0 keeps actions deterministic.")
    parser.add_argument(
        "--agent",
        choices=("random", "dqn", "dueling-dqn", "ddqn", "a3c"),
        default="dqn",
        help="Agent to train (`ddqn` is an alias for `dueling-dqn`).",
    )
    parser.add_argument("--grid", type=Path, default=GRID_CONFIGS_FP / DEFAULT_GRID_FILENAME,
                        help="Path to a .npy grid file.")
    parser.add_argument("--start-pos", type=float, nargs=2, default=None, dest="start_pos",
                        metavar=("X", "Y"),
                        help="Fixed continuous (x, y) start for evaluation and visualization "
                             "(and for training unless --exploring-starts is set). Omit to use "
                             "the grid START_CELL or a random empty cell.")
    parser.add_argument("--eval-start-pos", type=float, nargs=2, default=None, dest="eval_start_pos",
                        metavar=("X", "Y"),
                        help="Fixed continuous (x, y) start used only for evaluation and "
                             "visualization. Overrides --start-pos for the eval/viz env; "
                             "training start is unaffected. Omit to fall back to --start-pos.")
    parser.add_argument("--exploring-starts", action="store_true", dest="exploring_starts",
                        help="Use random start positions during training (exploring starts) "
                             "while evaluation keeps the fixed --start-pos / --eval-start-pos.")

    parser.add_argument("--episodes", type=int, default=DEFAULT_TOTAL_EPISODES, help="Training episodes.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE, dest="max_steps",
                        help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL, dest="eval_interval",
                        help="Evaluate every N episodes.")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES, dest="eval_episodes",
                        help="Episodes per evaluation.")
    parser.add_argument("--final-eval-runs", type=int, default=DEFAULT_FINAL_EVAL_RUNS, dest="final_eval_runs",
                        help="Greedy evaluation rollouts from the best checkpoint after training, "
                             "each with a distinct seed (distinct from --eval-episodes).")
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL, dest="log_interval",
                        help="Print/log metrics every N episodes (default: same as --eval-interval).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Compute device; 'auto' picks cuda > mps > cpu.")
    
    parser.add_argument("--epsilon-schedule", type=str, default=EPSILON_SCHEDULER_DEFAULT,
                        choices=("linear_annealing","exponential_decay","constant"),
                        help="Epsilon scheduler method to control exploration.")
    parser.add_argument("--epsilon-max", type=float, default=EPSILON_DEFAULT_MAX, dest="epsilon_max",
                        help="Start epsilon for annealing (or fixed rate if duration=0).")
    parser.add_argument("--epsilon-min", type=float, default=EPSILON_DEFAULT_MIN, dest="epsilon_min",
                        help="Minimum epsilon after annealing.")
    parser.add_argument("--epsilon-duration", type=int, default=EPSILON_ANNEAL_DURATION, dest="epsilon_duration",
                        help="Number of steps to anneal epsilon over.")
    parser.add_argument("--epsilon-start-step", type=int, default=EPSILON_ANNEAL_START_STEP, dest="epsilon_start_step",
                        help="Steps before epsilon annealing starts.")
    parser.add_argument("--epsilon-decay", type=float, default=EPSILON_DEFAULT_DECAY, dest="decay",
                        help="Decay in case of exponential epsilon scheduling.")
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
    parser.add_argument("--grad-clip-norm", type=float, default=(DQN_DEFAULT_GRAD_CLIP_NORM or 0.0), dest="grad_clip_norm",
                        help="Max global gradient norm for clipping; <=0 only measures the norm without clipping.")

    parser.add_argument("--a3c-workers", type=int, default=A3C_N_WORKERS, dest="a3c_workers",
                        help="A3C only: number of asynchronous actor-learner processes.")
    parser.add_argument("--a3c-lr", type=float, default=A3C_LEARNING_RATE, dest="a3c_lr",
                        help="A3C only: optimizer learning rate.")
    parser.add_argument("--a3c-t-max", type=int, default=A3C_T_MAX, dest="a3c_t_max",
                        help="A3C only: max rollout length between gradient pushes.")
    parser.add_argument("--a3c-entropy-beta", type=float, default=A3C_ENTROPY_BETA, dest="a3c_entropy_beta",
                        help="A3C only: entropy regularization coefficient (uniform across workers).")
    parser.add_argument("--a3c-random-action-start", type=float, default=A3C_RANDOM_ACTION_START,
                        dest="a3c_random_action_start",
                        help="A3C only: initial probability of forcing a uniform-random action.")
    parser.add_argument("--a3c-random-action-final", type=float, default=A3C_RANDOM_ACTION_FINAL,
                        dest="a3c_random_action_final",
                        help="A3C only: final random-action probability after decay.")
    parser.add_argument("--a3c-random-action-decay-steps", type=int, default=A3C_RANDOM_ACTION_DECAY_STEPS,
                        dest="a3c_random_action_decay_steps",
                        help="A3C only: env steps over which random-action probability decays.")
    parser.add_argument("--a3c-progress-reward-scale", type=float, default=A3C_PROGRESS_REWARD_SCALE,
                        dest="a3c_progress_reward_scale",
                        help="A3C only: training-only reward for reducing distance to the target; <=0 disables.")
    parser.add_argument("--a3c-value-coef", type=float, default=A3C_VALUE_COEF, dest="a3c_value_coef",
                        help="A3C only: weight on the value loss.")
    parser.add_argument("--a3c-total-steps", type=int, default=A3C_DEFAULT_TOTAL_STEPS, dest="a3c_total_steps",
                        help="A3C only: global env-step budget (defaults to episodes * max-steps).")

    parser.add_argument("--out-dir", type=Path, default=None, dest="out_dir",
                        help="Output dir for checkpoints and history JSON "
                             f"(default: {DEFAULT_OUTPUT_ROOT}/<agent>_<timestamp>).")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-group", type=str, default=None, dest="wandb_group",
                        help="W&B group name to bucket related runs under.")

    parser.add_argument("--no-visualize", action="store_true", dest="no_visualize",
                        help="Disable in-training W&B rollout images (requires --wandb).")
    parser.add_argument("--wandb-visualisations", type=int, default=DEFAULT_WANDB_VIZ_INTERVAL,
                        dest="wandb_viz_interval",
                        help="When --wandb is set, log a greedy rollout of the best-so-far "
                             "policy to W&B every N episodes (default: 100). "
                             "No local PNG is kept.")
    parser.add_argument("--viz-max-steps", type=int, default=DEFAULT_VIZ_MAX_STEPS, dest="viz_max_steps",
                        help="Max steps for the visualization rollout.")
    parser.add_argument("--curiosity", type=str, default="grid_count", dest="curiosity",
                        choices=("no", "grid_count", "grid-count"),
                        help="DQN/Dueling-DQN/A3C only: intrinsic motivation to use. Supports no, grid_count, and grid-count.")
    parser.add_argument("--curiosity-beta", type=float, default=BETA_DEFAULT, dest="curiosity_beta",
                        help="Beta value for the curiosity term.")
    parser.add_argument("--target-reward", type=float, default=GOAL_REWARD, dest="target_reward",
                        help="Reward for reaching the target.")
    parser.add_argument("--living-penalty", type=float, default=LIVING_PENALTY, dest="living_penalty",
                        help="Penalty for taking an action.")
    parser.add_argument("--collision-penalty", type=float, default=COLLISION_PENALTY, dest="collision_penalty",
                        help="Penalty for colliding with a wall or obstacle.")
    args = parser.parse_args()
    args.curiosity = args.curiosity.replace("-", "_")
    return args


def _build_config(args: Namespace) -> TrainerConfig:
    """Materialise a TrainerConfig from parsed CLI args."""
    checkpoint_dir = str(args.out_dir)
    history_path = str(args.out_dir / "history.json")

    # Convert Path objects to strings for W&B logging
    full_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    grid_stem = Path(args.grid).stem
    time = datetime.now().isoformat()
        
    return TrainerConfig(
        total_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        log_interval=args.log_interval,
        wandb_viz_interval=args.wandb_viz_interval,
        checkpoint_dir=checkpoint_dir,
        save_best=True,
        save_last=True,
        history_path=history_path,
        use_wandb=args.wandb,
        wandb_group=args.wandb_group,
        run_name=f"{args.agent}_{args.env}_{grid_stem}_{time}",
        full_config=full_config,
        finish_wandb_on_train_end=not args.wandb,
    )


def _default_out_dir(agent: str) -> Path:
    """Return the default timestamped directory for run artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(DEFAULT_OUTPUT_ROOT) / f"{agent}_{timestamp}"


def _run_config(args: Namespace, config: TrainerConfig, device: str) -> dict[str, Any]:
    """Return the JSON configuration saved with run artifacts."""
    payload = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    return {
        "cli": payload,
        "trainer": asdict(config),
        "resolved_device": device,
    }


def _reset_env_for_rollout(env: BaseGridEnvironment, seed: int) -> np.ndarray:
    """Reset an environment for artifact rollout generation."""
    try:
        state = env.reset(seed=seed)
    except TypeError:
        state = env.reset()
    return np.asarray(state, dtype=np.float32)


def _step_env_for_rollout(
    env: BaseGridEnvironment,
    action: int,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Step an env while supporting both 4-value and 5-value step APIs."""
    result = env.step(action)
    if len(result) == 4:
        next_state, reward, terminated, info = result
        truncated = False
    elif len(result) == 5:
        next_state, reward, terminated, truncated, info = result
    else:
        raise ValueError("env.step(action) must have either 4 or 5 values.")
    return (
        np.asarray(next_state, dtype=np.float32),
        float(reward),
        bool(terminated),
        bool(truncated),
        dict(info),
    )


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    """Convert environment info values into JSON-safe objects."""
    safe: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def _run_policy_rollout(
    args: Namespace,
    agent: BaseAgent,
    eval_start: tuple[float, float] | None,
    max_steps: int,
    rollout_seed: int,
) -> dict[str, Any]:
    """Run one greedy rollout for artifact generation."""
    env = _build_env(
        args.env,
        args.grid,
        rollout_seed,
        eval_start,
        args.use_sensors,
        args.step_size,
        sigma=args.sigma,
    )
    agent.on_episode_start(0)
    state = _reset_env_for_rollout(env, seed=rollout_seed)
    initial_grid = np.copy(env.grid)
    positions = [np.asarray(env.pos, dtype=float).copy()]
    headings = [float(getattr(env, "theta", 0.0))]
    actions: list[int] = []
    rewards: list[float] = []
    infos: list[dict[str, Any]] = []
    total_reward = 0.0
    terminated = False
    truncated = False

    for step_idx in range(max_steps):
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, info = _step_env_for_rollout(env, action)
        if step_idx == max_steps - 1 and not terminated and not truncated:
            truncated = True
            info["time_limit"] = True

        actions.append(int(action))
        rewards.append(float(reward))
        infos.append(_json_safe_info(info))
        total_reward += float(reward)
        positions.append(np.asarray(env.pos, dtype=float).copy())
        headings.append(float(getattr(env, "theta", 0.0)))
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
        "rollout_seed": rollout_seed,
    }


def _best_checkpoint_path(config: TrainerConfig) -> Path | None:
    """Return the checkpoint path to use for final rollout rendering."""
    if config.checkpoint_dir is None:
        return None
    checkpoint_dir = Path(config.checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        return best_path
    last_path = checkpoint_dir / "last.pt"
    if last_path.exists():
        return last_path
    return None


def _wandb_artifact_name(name: str) -> str:
    """Return a W&B-safe artifact name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def main() -> None:
    """Build env + agent + Trainer, train, and optionally visualize."""
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = _default_out_dir(args.agent)
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
    reward_fn = _build_reward_fn(args.target_reward, args.living_penalty, args.collision_penalty)
    env = _build_env(
        args.env,
        args.grid,
        args.seed,
        train_start,
        args.use_sensors,
        args.step_size,
        reward_fn,
        sigma=args.sigma,
    )
    eval_env = _build_env(
        args.env,
        args.grid,
        args.seed,
        eval_start,
        args.use_sensors,
        args.step_size,
        sigma=args.sigma,
    )
    agent = _build_agent(args, env, device)
    config = _build_config(args)

    print(f"[{args.agent} | {args.env}] grid={args.grid.name} "
          f"state_dim={env.state_dim} n_actions={env.n_actions} device={device} "
          f"train_start={'random' if train_start is None else train_start} "
          f"eval_start={'random' if eval_start is None else eval_start}")

    viz_fn = None
    if args.wandb and not args.no_visualize and args.wandb_viz_interval > 0:
        viz_env = _build_env(
            args.env,
            args.grid,
            args.seed,
            eval_start,
            args.use_sensors,
            args.step_size,
            sigma=args.sigma,
        )
        viz_agent = _build_agent(args, viz_env, device)
        checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir is not None else None

        def viz_fn(
            train_agent: BaseAgent,
            episode: int,
            _viz_agent: BaseAgent = viz_agent,
            _viz_env: BaseGridEnvironment = viz_env,
            _checkpoint_dir: Path | None = checkpoint_dir,
        ) -> str:
            """Render the best-so-far policy to a temporary PNG path."""
            best_path = _checkpoint_dir / "best.pt" if _checkpoint_dir is not None else None
            if best_path is not None and best_path.exists():
                _viz_agent.load_checkpoint(str(best_path))
            else:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as checkpoint_tmp:
                    checkpoint_tmp_path = Path(checkpoint_tmp.name)
                try:
                    train_agent.save_checkpoint(str(checkpoint_tmp_path))
                    _viz_agent.load_checkpoint(str(checkpoint_tmp_path))
                finally:
                    checkpoint_tmp_path.unlink(missing_ok=True)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_tmp:
                image_path = Path(image_tmp.name)
            visualize_agent(
                _viz_env,
                _viz_agent,
                max_steps=args.viz_max_steps,
                out=image_path,
                title=f"{args.agent} | {args.env} | ep {episode} (best-so-far)",
            )
            return str(image_path)

    trainer = Trainer(env=env, agent=agent, config=config, eval_env=eval_env, viz_fn=viz_fn)
    try:
        history = trainer.train()

        print(f"Training finished. Logged episodes: {len(history)}")
        if history:
            print(f"Last episode metrics: {history[-1]}")

        rollouts: list[dict[str, Any]] | None = None
        checkpoint_path = _best_checkpoint_path(config)
        if checkpoint_path is not None:
            agent.load_checkpoint(str(checkpoint_path))
            print(f"loaded checkpoint for final rollout -> {checkpoint_path}")
            rollouts = []
            for run_idx in range(args.final_eval_runs):
                rollout_seed = args.seed + 20_000 + run_idx
                rollouts.append(
                    _run_policy_rollout(
                        args=args,
                        agent=agent,
                        eval_start=eval_start,
                        max_steps=args.viz_max_steps,
                        rollout_seed=rollout_seed,
                    )
                )
            if args.final_eval_runs > 1:
                agg = aggregate_rollout_metrics(rollouts)
                print(
                    f"final eval ({args.final_eval_runs} runs): "
                    f"mean_reward={agg['mean_reward']:.3f} "
                    f"success_rate={agg['success_rate']:.3f} "
                    f"mean_steps={agg['mean_steps']:.1f}"
                )

        rollout_payload: dict[str, Any] | list[dict[str, Any]] | None = None
        if rollouts is not None:
            rollout_payload = rollouts[0] if len(rollouts) == 1 else rollouts

        artifact_paths = save_deep_rl_run_artifacts(
            out_dir=args.out_dir,
            run_config=_run_config(args, config, device),
            history=history,
            agent=agent,
            rollout=rollout_payload,
        )
        artifact_paths.extend(
            path
            for path in (args.out_dir / "best.pt", args.out_dir / "last.pt", args.out_dir / "history.json")
            if path.exists()
        )

        if args.wandb:
            log_wandb_artifact(
                artifact_name=_wandb_artifact_name(f"{config.run_name or args.agent}_artifacts"),
                artifact_type="training-run",
                paths=artifact_paths,
                aliases=["latest", "best-policy"],
            )

        if rollouts is not None:
            print(f"saved rollout html -> {args.out_dir / 'policy_rollout.html'}")
            if args.wandb:
                import wandb

                if wandb.run is not None:
                    agg = aggregate_rollout_metrics(rollouts)
                    wandb.run.summary.update(
                        {
                            "final_eval/n_runs": agg["n_runs"],
                            "final_eval/mean_reward": agg["mean_reward"],
                            "final_eval/std_reward": agg["std_reward"],
                            "final_eval/success_rate": agg["success_rate"],
                            "final_eval/mean_steps": agg["mean_steps"],
                        }
                    )
    finally:
        if args.wandb:
            trainer.finish_wandb()


if __name__ == "__main__":
    main()
