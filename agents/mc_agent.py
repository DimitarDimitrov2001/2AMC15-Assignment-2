"""On-policy first-visit Monte Carlo control agent.

This class only knows how to choose actions, record transitions, and finalise
an episode. The driving loop (episode iteration, env interaction, history
construction) lives in ``agents/trainers/mc.py`` -- same pattern as Q-learning.

Step-size update is driven by a ``LearningRateSchedule``. The default is the
constant-alpha (Sutton & Barto §6.1) variant via ``ExponentialDecaySchedule``
with ``decay=1.0``; trainers can swap in exponential decay or a state-action
visit-count schedule by passing ``lr_schedule`` explicitly. The classical
sample-mean / 1/N variant is *not* a special case the agent provides --
expressing it as ``VisitCountSchedule(c=0)`` would be slightly different (the
visit-count schedule uses ``c / (c + N)`` with ``c > 0`` for the Robbins-Monro
guarantee).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from agents.base_agent import BaseAgent
from agents.learning_rates import ExponentialDecaySchedule, LearningRateSchedule


@dataclass(frozen=True)
class MCEpisodeResult:
    """Per-episode metrics returned by ``MCAgent.end_episode()``.

    The trainer accumulates these into the ``TrainingHistory`` after the
    loop, the same way the Q-learning trainer accumulates episode rewards.

    ``alpha`` is the mean of the alphas actually applied during this
    episode's first-visit updates. For schedules with a fixed global rate
    (constant or exponential) every applied alpha equals the global rate,
    so the mean is that rate; for visit-count it is a genuine per-episode
    average. ``alpha_min``/``alpha_max`` capture the spread within the
    episode (always equal for fixed-rate schedules; informative for
    visit-count).
    """

    delta_q: float
    total_reward: float
    epsilon: float
    alpha: float | None
    alpha_min: float | None
    alpha_max: float | None


class MCAgent(BaseAgent):
    """On-policy first-visit Monte Carlo control with epsilon-greedy behaviour."""

    n_actions: int
    gamma: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    alpha: float
    q_init: float
    q_init_noise: float
    q_table: dict[tuple[int, int], np.ndarray]
    values: dict[tuple[int, int], float]
    policy: dict[tuple[int, int], int]
    lr_schedule: LearningRateSchedule
    last_episode_mean_alpha: float | None
    last_episode_alpha_min: float | None
    last_episode_alpha_max: float | None

    _episode: list[tuple[tuple[int, int], int, float]]
    _rng: random.Random
    _training: bool

    def __init__(
        self,
        n_actions: int = 4,
        gamma: float = 0.99,
        # --- epsilon ---
        epsilon: float = 0.2,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        # --- alpha (legacy; ignored when lr_schedule is supplied) ---
        alpha: float = 0.1,
        alpha_decay: float = 1.0,
        alpha_min: float = 0.01,
        # --- q-init ---
        q_init: float = 0.0,
        q_init_noise: float = 1e-6,
        random_seed: int | None = 0,
        lr_schedule: LearningRateSchedule | None = None,
    ):
        super().__init__()
        if q_init_noise < 0.0:
            raise ValueError("q_init_noise must be >= 0")

        self.n_actions = n_actions
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_init = q_init
        self.q_init_noise = q_init_noise

        # Local RNG instance so each agent has its own independent stream and
        # does not mutate the global random/np.random state. The environment
        # and other agents are unaffected by construction order.
        self._rng = random.Random(random_seed)

        # Trainers pass an explicit schedule; legacy callers keep the old
        # alpha/alpha_decay/alpha_min surface, which we collapse onto an
        # ExponentialDecaySchedule here. Validation now lives inside the
        # schedule constructor.
        if lr_schedule is None:
            lr_schedule = ExponentialDecaySchedule(
                alpha=alpha, decay=alpha_decay, minimum=alpha_min,
            )
        self.lr_schedule = lr_schedule

        global_rate = self.lr_schedule.get_global_rate()
        self.alpha = global_rate if global_rate is not None else float("nan")
        self.last_episode_mean_alpha = None
        self.last_episode_alpha_min = None
        self.last_episode_alpha_max = None

        self.q_table = defaultdict(self._initial_q_values)
        self._episode = []
        self._training = True

        self.values = {}
        self.policy = {}

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

    def take_action(self, state: tuple[int, int]) -> int:
        """Epsilon-greedy action selection with random tie-breaking on greedy ties."""
        if not self._training:
            if state in self.policy:
                return self.policy[state]
            return int(np.argmax(self.q_table[state]))

        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.n_actions - 1)
        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return int(self._rng.choice(best_actions.tolist()))

    # ``update()`` is inherited from BaseAgent as a no-op. MC does not learn
    # on a per-step basis; the trainer drives ``record_step`` and ``end_episode``.

    def start_episode(self) -> None:
        """Reset per-episode state. Called by the trainer at the top of each episode."""
        self._episode.clear()

    def record_step(self, state: tuple[int, int], action: int, reward: float) -> None:
        """Record one transition (``state``, ``action``, ``reward``) for the current episode.

        ``action`` MUST be the action selected by the policy (i.e. the value
        returned from :meth:`take_action`), not the action the environment
        actually executed under stochastic transitions. On-policy first-visit
        MC control estimates ``Q(s, a)`` for the policy's chosen action; the
        env-side action noise is part of the transition dynamics, not the
        thing being learned.
        """
        self._episode.append((state, action, reward))

    def end_episode(self) -> MCEpisodeResult:
        """Apply first-visit MC updates from the current episode trajectory."""
        if not self._episode:
            self.lr_schedule.update_episode()
            return MCEpisodeResult(
                delta_q=0.0,
                total_reward=0.0,
                epsilon=self.epsilon,
                alpha=self.lr_schedule.get_global_rate(),
                alpha_min=None,
                alpha_max=None,
            )

        returns: list[tuple[tuple[int, int], int, float]] = []
        g_return = 0.0
        for state, action, reward in reversed(self._episode):
            g_return = reward + self.gamma * g_return
            returns.append((state, action, g_return))
        returns.reverse()

        visited: set[tuple[tuple[int, int], int]] = set()
        max_delta = 0.0
        episode_alphas: list[float] = []
        for state, action, g_t in returns:
            if (state, action) in visited:
                continue
            visited.add((state, action))
            applied_alpha = self.lr_schedule.get_rate(state, action)
            episode_alphas.append(applied_alpha)
            old_q = self.q_table[state][action]
            self.q_table[state][action] += applied_alpha * (g_t - old_q)
            max_delta = max(max_delta, abs(self.q_table[state][action] - old_q))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.lr_schedule.update_episode()

        if episode_alphas:
            self.last_episode_mean_alpha = float(sum(episode_alphas) / len(episode_alphas))
            self.last_episode_alpha_min = float(min(episode_alphas))
            self.last_episode_alpha_max = float(max(episode_alphas))
        else:
            global_rate = self.lr_schedule.get_global_rate()
            self.last_episode_mean_alpha = global_rate
            self.last_episode_alpha_min = global_rate
            self.last_episode_alpha_max = global_rate

        global_rate = self.lr_schedule.get_global_rate()
        self.alpha = (
            global_rate if global_rate is not None else (self.last_episode_mean_alpha or float("nan"))
        )

        total_reward = float(sum(r for _, _, r in self._episode))
        self._episode.clear()
        return MCEpisodeResult(
            delta_q=float(max_delta),
            total_reward=total_reward,
            epsilon=float(self.epsilon),
            alpha=self.last_episode_mean_alpha,
            alpha_min=self.last_episode_alpha_min,
            alpha_max=self.last_episode_alpha_max,
        )

    def build_value_and_policy(self) -> None:
        """Populate ``self.values`` and ``self.policy`` from the trained Q-table."""
        self.values = {state: float(np.max(q_vals)) for state, q_vals in self.q_table.items()}
        self.policy = {state: int(np.argmax(q_vals)) for state, q_vals in self.q_table.items()}

    def set_eval_mode(self) -> None:
        """Switch to the learned greedy policy for fixed-policy evaluation."""
        self.build_value_and_policy()
        self.epsilon = 0.0
        self._training = False
