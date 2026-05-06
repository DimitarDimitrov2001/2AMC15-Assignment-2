"""On-policy first-visit Monte Carlo control agent."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from utils.plotting import plot_training_history, SubplotConfig
from utils.rl_plots import plot_value_and_policy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.base_agent import BaseAgent

try:
    from utils.plotting import TrainingHistory
    _HAS_TRAINING_HISTORY = True
except ImportError:
    _HAS_TRAINING_HISTORY = False


class MCAgent(BaseAgent):

    def __init__(
        self,
        n_actions: int = 4,
        gamma: float = 0.99,
        # --- epsilon ---
        epsilon: float = 0.2,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        # --- alpha ---
        alpha: float | None = None,   # if none = use 1/N incremental mean
        alpha_decay: float = 1.0,
        alpha_min: float = 0.001,
        # --- stopping ---
        max_episode_length: int = 2000,
        convergence_threshold: float = 1e-3,
        patience: int = 200,
        random_seed: int | None = 0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.alpha = alpha
        self._alpha_current = alpha       
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min

        self.max_episode_length = max_episode_length
        self.convergence_threshold = convergence_threshold
        self.patience = patience

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.Q: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))
        self._N: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions, dtype=int))
        self._episode: list[tuple] = []

        self._episode_rewards: list[float] = []
        self._episode_delta_q: list[float] = []
        self._epsilon_history: list[float] = []
        self._alpha_history: list[float] = []

        self.values: dict[tuple, float] = {}
        self.policy: dict[tuple, int] = {}
        self.history = None

    def take_action(self, state: tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def update(self, state: tuple[int, int], reward: float, action: int) -> None:
        pass

    def _record_step(self, state: tuple, action: int, reward: float) -> None:
        self._episode.append((state, action, reward))

    def _get_step_size(self, state: tuple, action: int) -> float:
        """Return the step size for this (s,a) update.

        If alpha is None, use incremental mean 1/N.
        Otherwise use the current (possibly decayed) alpha.
        """
        if self._alpha_current is None:
            return 1.0 / self._N[state][action]
        return self._alpha_current

    def end_episode(self) -> float:
        if not self._episode:
            return 0.0

        G = 0.0
        returns = []
        for state, action, reward in reversed(self._episode):
            G = reward + self.gamma * G
            returns.append((state, action, G))
        returns.reverse()

        visited: set = set()
        max_delta = 0.0
        for state, action, G_t in returns:
            if (state, action) in visited:
                continue
            visited.add((state, action))
            self._N[state][action] += 1
            old_q = self.Q[state][action]
            step = self._get_step_size(state, action)
            self.Q[state][action] += step * (G_t - old_q)
            max_delta = max(max_delta, abs(self.Q[state][action] - old_q))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self._alpha_current is not None:
            self._alpha_current = max(self.alpha_min, self._alpha_current * self.alpha_decay)

        self._episode_rewards.append(sum(r for _, _, r in self._episode))
        self._episode_delta_q.append(max_delta)
        self._epsilon_history.append(self.epsilon)
        self._alpha_history.append(self._alpha_current if self._alpha_current is not None else np.nan)

        self._episode.clear()
        return max_delta

    def train(self, env, n_episodes: int, start_pos: tuple | None = None,
              verbose: bool = True, reward_fn=None) -> None:
        consecutive_converged = 0

        for ep in range(1, n_episodes + 1):
            reset_kwargs = {"agent_start_pos": start_pos} if start_pos is not None else {}
            state = env.reset(**reset_kwargs)
            if reward_fn is not None:          # reapply after reset — reset() wipes reward_fn
                env.reward_fn = reward_fn

            for _ in range(self.max_episode_length):
                action = self.take_action(state)
                next_state, reward, terminated, info = env.step(action)
                self._record_step(state, info["actual_action"], reward)
                state = next_state
                if terminated:
                    break

            max_delta = self.end_episode()

            if verbose and ep % 100 == 0:
                alpha_str = f"{self._alpha_current:.4f}" if self._alpha_current is not None else "1/N"
                avg_r = np.mean(self._episode_rewards[-100:])
                print(
                    f"Ep {ep:>5} | avg_reward: {avg_r:>8.2f} | "
                    f"max|ΔQ|: {max_delta:.6f} | "
                    f"ε: {self.epsilon:.4f} | α: {alpha_str}"
                )

            # convergence check on reward stability, more meaningful than delta_q for MC
            if len(self._episode_rewards) >= 100:
                recent = np.mean(self._episode_rewards[-50:])
                older  = np.mean(self._episode_rewards[-100:-50])
                if abs(recent - older) < self.convergence_threshold:
                    consecutive_converged += 1
                    if consecutive_converged >= self.patience:
                        if verbose:
                            print(f"Converged at episode {ep}.")
                        break
                else:
                    consecutive_converged = 0

        self._build_value_and_policy()
        self._build_history()

    def _build_value_and_policy(self) -> None:
        for state, q_vals in self.Q.items():
            self.values[state] = float(np.max(q_vals))
            self.policy[state] = int(np.argmax(q_vals))

    def _build_history(self) -> None:
        episodes = np.arange(1, len(self._episode_rewards) + 1, dtype=float)
        metrics = {
            "avg_reward": np.array(self._episode_rewards),
            "delta_q":    np.array(self._episode_delta_q),
            "epsilon":    np.array(self._epsilon_history),
            "alpha":      np.array(self._alpha_history),
        }
        hyperparams = {
            "gamma":              self.gamma,
            "epsilon":            self.epsilon,
            "epsilon_decay":      self.epsilon_decay,
            "alpha":              self.alpha if self.alpha is not None else "1/N",
            "alpha_decay":        self.alpha_decay,
            "max_episode_length": self.max_episode_length,
        }
        if _HAS_TRAINING_HISTORY:
            self.history = TrainingHistory(episodes=episodes, metrics=metrics, hyperparams=hyperparams)
        else:
            self.history = {"episodes": episodes, "metrics": metrics, "hyperparams": hyperparams, "metadata": {}}

    def greedy_action(self, state: tuple[int, int]) -> int:
        return int(np.argmax(self.Q[state]))


def train_mc_agent(
    grid_fp: Path,
    n_episodes: int = 10000,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    epsilon_decay: float = 0.9995,
    epsilon_min: float = 0.01,
    alpha: float | None = 0.1,
    alpha_decay: float = 0.9995,
    alpha_min: float = 0.001,
    max_episode_length: int = 2000,
    sigma: float = 0.1,
    start_pos: tuple | None = None,
    convergence_threshold: float = 1e-3,
    patience: int = 200,
    random_seed: int = 0,
    verbose: bool = True,
) -> MCAgent:
    from world import Environment, find_target_position, build_manhattan_reward_function

    #WE FIND ACTUAL TARGET FOR EVAL
    grid = np.load(grid_fp)
    target_pos = find_target_position(grid)
    print(f"Target at: {target_pos}")

    def simple_reward(grid, new_pos):
        if new_pos == target_pos:
            return 10.0
        return -1.0

    env = Environment(grid_fp=grid_fp, no_gui=True, sigma=sigma,
                      agent_start_pos=start_pos, random_seed=random_seed,
                      reward_fn=simple_reward)
    env.reset()

    agent = MCAgent(
        gamma=gamma,
        epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        alpha=alpha, alpha_decay=alpha_decay, alpha_min=alpha_min,
        max_episode_length=max_episode_length,
        convergence_threshold=convergence_threshold,
        patience=patience,
        random_seed=random_seed,
    )
    agent.train(env, n_episodes=n_episodes, start_pos=start_pos,
                verbose=verbose, reward_fn=simple_reward)
    return agent

if __name__ == "__main__":
    from world import Environment, find_target_position, build_manhattan_reward_function

    GRID = Path("grid_configs/A1_grid.npy")

    grid = np.load(GRID)
    target_pos = find_target_position(grid)
    print(f"Target found at: {target_pos}")
    
    start_pos = (1, 2)  
    print(f"Start position: {start_pos}")
    
    agent = train_mc_agent(
        grid_fp=GRID,
        n_episodes=20000,
        epsilon=0.3,  
        epsilon_decay=0.9998,  
        epsilon_min=0.05,  
        max_episode_length=2000,
        convergence_threshold=1e-3,  
        patience=100,  
        gamma=0.99,
        alpha=None,  
        alpha_decay=1.0,
        alpha_min=0.001,
        start_pos=start_pos,
        verbose=True,
    )

    fig, _, _ = plot_training_history(
        history=agent.history.to_dict(),
        smoothing_window=50,
        subplot_config={
            "avg_reward": SubplotConfig(y_label="Avg Reward", symlog=True),
            "delta_q":    SubplotConfig(y_label="Max |ΔQ|", log_scale=True, threshold=1e-3),
            "epsilon":    SubplotConfig(y_label="ε"),
            "alpha":      SubplotConfig(y_label="α"),
        },
        title="MC — Single Run",
    )
    fig.savefig("mc_single_run.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    grid = np.load(GRID)
    fig2, _ = plot_value_and_policy(
        grid, agent.values, agent.policy,
        title="MC — Value & Policy",
        agent_start_pos=start_pos,
    )
    fig2.savefig("mc_value_policy.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    Environment.evaluate_agent(
        grid_fp=GRID,
        agent=agent,
        max_steps=200,
        sigma=0.1,
        agent_start_pos=start_pos,
        random_seed=0,
    )

