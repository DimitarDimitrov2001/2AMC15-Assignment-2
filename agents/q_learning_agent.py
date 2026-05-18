"""Q-learning agent for the grid-world delivery task.

The agent learns by trial-and-error by interacting with the grid-world
environment (in episodes). The action values are stored in a Q-table and
updated after every transition (using received reward and the expected
value of next state). The training uses epsilon-greedy policy to balance
exploration and exploitation. After training, for each cell we choose the
action with the highest Q-value.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from agents import BaseAgent
from agents.learning_rates import ExponentialDecaySchedule, LearningRateSchedule


class QLearningAgent(BaseAgent):

    # Public training hyperparameters and learned artifacts. The trainer
    # reads ``values`` and ``policy`` after calling ``set_eval_mode()``.
    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    decaying_epsilon: bool
    n_actions: int
    q_init: float
    q_init_noise: float
    training: bool
    q_table: dict[tuple[int, int], np.ndarray]
    values: dict[tuple[int, int], float]
    policy: dict[tuple[int, int], int]
    lr_schedule: LearningRateSchedule
    last_episode_mean_alpha: float | None
    last_episode_alpha_min: float | None
    last_episode_alpha_max: float | None

    # Private transition/episode state. Q-learning updates one transition at
    # a time, so it only needs to remember the previous state.
    _last_state: tuple[int, int] | None
    _rng: random.Random
    _episode_alphas: list[float]

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
        lr_schedule: LearningRateSchedule | None = None,
    ):
        super().__init__()
        if q_init_noise < 0.0:
            raise ValueError("q_init_noise must be >= 0")

        # ------------------------------------------------------------------
        # Store scalar hyperparameters
        # ------------------------------------------------------------------
        # ``gamma`` discounts the bootstrap value of the next state.
        # ``epsilon`` controls exploration while training.
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decaying_epsilon = decaying_epsilon
        self.n_actions = n_actions
        self.q_init = q_init
        self.q_init_noise = q_init_noise

        # ------------------------------------------------------------------
        # Learning-rate schedule
        # ------------------------------------------------------------------
        # Trainers pass an explicit schedule; legacy callers (tests, manual
        # construction) keep the old alpha/alpha_decay/alpha_min/decaying_alpha
        # surface, which we collapse onto an ExponentialDecaySchedule here so
        # there's a single update path internally.
        if lr_schedule is None:
            effective_decay = alpha_decay if decaying_alpha else 1.0
            effective_min = alpha_min if decaying_alpha else alpha
            lr_schedule = ExponentialDecaySchedule(
                alpha=alpha, decay=effective_decay, minimum=effective_min,
            )
        self.lr_schedule = lr_schedule

        # ``alpha`` is kept as a scalar compatibility/debug field. For
        # visit-count schedules there is no single global alpha, so NaN marks
        # that the real rates are state-action specific.
        global_rate = self.lr_schedule.get_global_rate()
        self.alpha = global_rate if global_rate is not None else float("nan")
        self.last_episode_mean_alpha = None
        self.last_episode_alpha_min = None
        self.last_episode_alpha_max = None

        # ------------------------------------------------------------------
        # Learned and episode state
        # ------------------------------------------------------------------
        self.training = True
        self._rng = random.Random(random_seed)
        self._episode_alphas = []

        # ``defaultdict`` creates Q-value rows lazily when a state is first
        # encountered. ``values`` and ``policy`` are derived snapshots used
        # by evaluation/plotting after training.
        self.q_table = defaultdict(self._initial_q_values)

        self.values = {}
        self.policy = {}

        self._last_state = None

    def _initial_q_values(self) -> np.ndarray:
        """Factory for new Q-table rows, used by the ``defaultdict``."""
        # A tiny seeded perturbation avoids always choosing action 0 when all
        # actions are initially tied. Setting ``q_init_noise=0`` restores exact
        # identical initialization.
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
        """Reset per-episode state before the trainer starts stepping."""
        # ``_last_state`` is set by ``take_action`` and consumed by ``update``.
        # Alpha diagnostics are collected over all updates in this episode.
        self._last_state = None
        self._episode_alphas = []

    def end_episode(self) -> None:
        """Apply episode-level decay and summarize alpha diagnostics."""
        if self.training:
            # Q-values were already updated step-by-step. Only schedules that
            # advance per episode are updated here.
            if self.decaying_epsilon:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.lr_schedule.update_episode()

        # Trainer plots use a single alpha trace. For state-action schedules,
        # report the episode mean plus min/max spread.
        if self._episode_alphas:
            self.last_episode_mean_alpha = float(
                sum(self._episode_alphas) / len(self._episode_alphas)
            )
            self.last_episode_alpha_min = float(min(self._episode_alphas))
            self.last_episode_alpha_max = float(max(self._episode_alphas))
        else:
            # No updates this episode (e.g. agent terminated before its first
            # transition). Fall back to the schedule's global rate when one
            # exists so logging still has a sensible scalar.
            global_rate = self.lr_schedule.get_global_rate()
            self.last_episode_mean_alpha = global_rate
            self.last_episode_alpha_min = global_rate
            self.last_episode_alpha_max = global_rate

        global_rate = self.lr_schedule.get_global_rate()
        self.alpha = (
            global_rate if global_rate is not None else (self.last_episode_mean_alpha or float("nan"))
        )

        self._last_state = None

    def take_action(self, state: tuple[int, int]) -> int:
        """Choose an action via epsilon-greedy exploration."""
        # Remember the state paired with this chosen action. The environment
        # returns the next state later, and ``update`` uses this saved state
        # as the Q-learning update's ``s``.
        self._last_state = state

        # Training mode explores with probability epsilon. Evaluation mode
        # has ``training=False`` and ``epsilon=0``, so this branch is skipped.
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        q_values = self.q_table[state]
        best_value = np.max(q_values)

        # Random tie-breaking avoids a fixed bias toward the lowest-index
        # action when multiple actions currently have the same Q-value.
        best_actions = np.flatnonzero(q_values == best_value)
        return int(random.choice(best_actions.tolist()))

    def update(self, state: tuple[int, int], reward: float, action: int, terminated: bool = False) -> None:
        """Update Q-value for the previous state and the chosen action."""
        # The first update after a malformed call sequence has no previous
        # state to update. The trainer normally calls ``take_action`` first.
        if self._last_state is None:
            return

        previous_state = self._last_state

        old_q_value = self.q_table[previous_state][action]

        # ------------------------------------------------------------------
        # Bellman target
        # ------------------------------------------------------------------
        # Terminal transitions have no future value. Otherwise bootstrap from
        # the best action value in the observed next state.
        if terminated:
            target = reward
        else:
            best_next_q_value = np.max(self.q_table[state])
            target = reward + self.gamma * best_next_q_value

        # ------------------------------------------------------------------
        # Q-learning update
        # ------------------------------------------------------------------
        # Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)].
        applied_alpha = self.lr_schedule.get_rate(previous_state, action)
        self._episode_alphas.append(applied_alpha)
        self.q_table[previous_state][action] = old_q_value + applied_alpha * (
            target - old_q_value
        )

    def set_eval_mode(self) -> None:
        """Switch to greedy evaluation and freeze ``values``/``policy`` from the Q-table."""
        # Disable exploration and expose the value/policy dictionaries used by
        # evaluation and plotting. These are snapshots of the final Q-table.
        self.training = False
        self.epsilon = 0.0
        self._last_state = None

        # Match VI's shape: expose values/policy as attributes, not methods.
        self.values = {state: float(np.max(action_values)) for state, action_values in self.q_table.items()}
        self.policy = {state: int(np.argmax(action_values)) for state, action_values in self.q_table.items()}
