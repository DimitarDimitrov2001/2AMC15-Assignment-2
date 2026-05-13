"""Q-learning agent for the grid-world delivery task

The agent learns by trial-and-error by interacting with the grid-world
environment (in episodes). The action values are stored in a Q-table and
updated after every transition (using received reward and the expected
value of next state). The training uses epsilon-greedy policy to balance
exploration and explotation. After training, for each cell we choose the
action with the highest q-value.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from agents import BaseAgent


class QLearningAgent(BaseAgent):

    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    alpha_min: float
    alpha_decay: float
    decaying_epsilon: bool
    decaying_alpha: bool
    n_actions: int
    q_init: float
    q_init_noise: float
    training: bool
    q_table: dict[tuple[int, int], np.ndarray]
    values: dict[tuple[int, int], float]
    policy: dict[tuple[int, int], int]

    _last_state: tuple[int, int] | None
    _rng: random.Random

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        alpha_min: float = 0.05,
        alpha_decay: float = 0.999,
        decaying_epsilon: bool = True,
        decaying_alpha: bool = True,
        n_actions: int = 4,
        q_init: float = 0.0,
        q_init_noise: float = 1e-6,
        random_seed: int = 0,
    ):
        super().__init__()
        if q_init_noise < 0.0:
            raise ValueError("q_init_noise must be >= 0")

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.decaying_epsilon = decaying_epsilon
        self.decaying_alpha = decaying_alpha
        self.n_actions = n_actions
        self.q_init = q_init
        self.q_init_noise = q_init_noise

        self.training = True
        self._rng = random.Random(random_seed)

        self.q_table = defaultdict(self._initial_q_values)

        self.values = {}
        self.policy = {}

        self._last_state = None

    def _initial_q_values(self) -> np.ndarray:
        """Factory for new Q-table rows, used by the ``defaultdict``."""
        if self.q_init_noise == 0.0:
            return np.full(self.n_actions, self.q_init, dtype=float)
        return np.array(
            [
                self.q_init + self._rng.uniform(-self.q_init_noise, self.q_init_noise)
                for _ in range(self.n_actions)
            ],
            dtype=float,
        )

    def start_episode(self) -> None:
        # Reset memory at the start of each episode
        self._last_state = None

    def end_episode(self) -> None:
        # If decaying epsilon and/or alpha is set to True (default), then we decay exploration and learning rate
        if self.training:
            if self.decaying_epsilon:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if self.decaying_alpha:
                self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

        # Reset memory at the end of the episode
        self._last_state = None

    def take_action(self, state: tuple[int, int]) -> int:
        """Choose an action via epsilon-greedy exploration."""
        self._last_state = state

        if self.training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        q_values = self.q_table[state]
        best_value = np.max(q_values)

        # Equally good actions decided randomly
        best_actions = np.flatnonzero(q_values == best_value)
        return int(random.choice(best_actions.tolist()))

    def update(self, state: tuple[int, int], reward: float, action: int, terminated: bool = False) -> None:
        """Update Q-value for the previous state and the chosen action."""
        if self._last_state is None:
            return

        previous_state = self._last_state

        old_q_value = self.q_table[previous_state][action]

        if terminated:
            target = reward
        else:
            best_next_q_value = np.max(self.q_table[state])
            target = reward + self.gamma * best_next_q_value

        self.q_table[previous_state][action] = old_q_value + self.alpha * (
            target - old_q_value
        )

    def set_eval_mode(self) -> None:
        """Switch to greedy evaluation and freeze ``values``/``policy`` from the Q-table."""
        self.training = False
        self.epsilon = 0.0
        self._last_state = None

        # Match VI's shape: expose values/policy as attributes, not methods.
        self.values = {state: float(np.max(action_values)) for state, action_values in self.q_table.items()}
        self.policy = {state: int(np.argmax(action_values)) for state, action_values in self.q_table.items()}
