"""On-policy first-visit Monte Carlo control agent."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np

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
        gamma: float = 0.9,
        epsilon: float = 0.2,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        max_episode_length: int = 500,
        convergence_threshold: float = 1e-4,
        patience: int = 20,
        random_seed: int | None = 0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
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

        self.values: dict[tuple, float] = {}
        self.policy: dict[tuple, int] = {}
        self.history = None

    def take_action(self, state: tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def update(self, state: tuple[int, int], reward: float, action: int) -> None:
        # No-op: MC updates at episode end, not per step. Use _record_step().
        pass

    def _record_step(self, state: tuple, action: int, reward: float) -> None:
        self._episode.append((state, action, reward))

    def end_episode(self) -> float:
        if not self._episode:
            return 0.0

        # Backward pass: compute discounted returns
        G = 0.0
        returns = []
        for state, action, reward in reversed(self._episode):
            G = reward + self.gamma * G
            returns.append((state, action, G))
        returns.reverse()

        # First-visit MC update via incremental mean
        visited: set = set()
        max_delta = 0.0
        for state, action, G_t in returns:
            if (state, action) in visited:
                continue
            visited.add((state, action))
            self._N[state][action] += 1
            old_q = self.Q[state][action]
            self.Q[state][action] += (G_t - old_q) / self._N[state][action]
            max_delta = max(max_delta, abs(self.Q[state][action] - old_q))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episode_rewards.append(sum(r for _, _, r in self._episode))
        self._episode_delta_q.append(max_delta)
        self._episode.clear()
        return max_delta

    def train(self, env, n_episodes: int, start_pos: tuple | None = None, verbose: bool = True) -> None:
        consecutive_converged = 0

        for ep in range(1, n_episodes + 1):
            reset_kwargs = {"agent_start_pos": start_pos} if start_pos is not None else {}
            state = env.reset(**reset_kwargs)

            for _ in range(self.max_episode_length):
                action = self.take_action(state)
                next_state, reward, terminated, info = env.step(action)
                self._record_step(state, info["actual_action"], reward)
                state = next_state
                if terminated:
                    break

            max_delta = self.end_episode()

            if verbose and ep % 100 == 0:
                avg_r = np.mean(self._episode_rewards[-100:])
                print(f"Ep {ep:>5} | avg_reward: {avg_r:>8.2f} | max|ΔQ|: {max_delta:.6f} | ε: {self.epsilon:.4f}")

            if max_delta < self.convergence_threshold:
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
        }
        hyperparams = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
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
    n_episodes: int = 2000,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    epsilon_decay: float = 0.999,
    epsilon_min: float = 0.01,
    max_episode_length: int = 500,
    sigma: float = 0.1,
    start_pos: tuple | None = None,
    convergence_threshold: float = 1e-4,
    patience: int = 20,
    random_seed: int = 0,
    verbose: bool = True,
) -> MCAgent:
    from world import Environment, build_manhattan_reward_function, find_target_position

    env = Environment(grid_fp=grid_fp, no_gui=True, sigma=sigma,
                      agent_start_pos=start_pos, random_seed=random_seed)
    initial_pos = env.reset()
    env.reward_fn = build_manhattan_reward_function(initial_pos, find_target_position(env.grid))

    agent = MCAgent(
        gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min, max_episode_length=max_episode_length,
        convergence_threshold=convergence_threshold, patience=patience,
        random_seed=random_seed,
    )
    agent.train(env, n_episodes=n_episodes, start_pos=start_pos, verbose=verbose)
    return agent