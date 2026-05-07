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

    def __init__(self, alpha: float = 0.5, gamma: float = 0.95, epsilon: float = 1.0, epsilon_min = 0.05, epsilon_decay = 0.995, alpha_min = 0.05, alpha_decay = 0.999, decaying_epsilon: bool = True, decaying_alpha: bool = True, n_actions: int = 4):
        super().__init__()

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

        self.training = True

        self.q_table: dict[tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

        self._last_state: tuple[int, int] | None = None

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

    # Choose action based on epsilon-greedy exploration
    def take_action(self, state: tuple[int, int]) -> int:
        self._last_state = state

        # Choose random action (Explore)
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        # Choose action with highest Q-value (Exploit)
        q_values = self.q_table[state]
        best_value = np.max(q_values)

        # Equally good actions decided randomly
        best_actions = np.flatnonzero(q_values == best_value)
        return int(random.choice(best_actions.tolist()))

    # Update Q-value for the previous state and the chosen action
    def update(self, state: tuple[int, int], reward: float, action: int, terminated: bool = False) -> None:

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
        # After training there is no exploration
        self.training = False
        self.epsilon = 0.0
        self._last_state = None

    def values(self) -> dict[tuple[int, int], float]:
        # Return V(s) = max_a Q(s, a) 
        return {
            state: float(np.max(action_values))
            for state, action_values in self.q_table.items()
        }

    def policy(self) -> dict[tuple[int, int], int]:
        # Return the greedy action for every state
        return {
            state: int(np.argmax(action_values))
            for state, action_values in self.q_table.items()
        }
