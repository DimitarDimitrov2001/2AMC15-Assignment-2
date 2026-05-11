"""On-policy first-visit Monte Carlo control agent.

This class only knows how to choose actions, record transitions, and finalise
an episode. The driving loop (episode iteration, env interaction, history
construction) lives in ``agents/trainers/mc.py`` — same pattern as Q-learning.

Step-size update is constant-alpha (Sutton & Barto §6.1). The classical
sample-mean / 1/N variant has been removed: it stalls quickly on grids where
state-action pairs are revisited often, making the agent unusable on A1 within
a reasonable episode budget. If you need 1/N for a comparison, reintroduce
it locally rather than as the agent's default.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from agents.base_agent import BaseAgent


@dataclass(frozen=True)
class MCEpisodeResult:
    """Per-episode metrics returned by ``MCAgent.end_episode()``.

    The trainer accumulates these into the ``TrainingHistory`` after the
    loop, the same way the Q-learning trainer accumulates episode rewards.
    """

    delta_q: float
    total_reward: float
    epsilon: float
    alpha: float


class MCAgent(BaseAgent):
    """On-policy first-visit Monte Carlo control with epsilon-greedy behaviour."""

    n_actions: int
    gamma: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    alpha: float
    alpha_decay: float
    alpha_min: float
    q_table: dict[tuple[int, int], np.ndarray]
    values: dict[tuple[int, int], float]
    policy: dict[tuple[int, int], int]

    _episode: list[tuple[tuple[int, int], int, float]]
    _rng: random.Random

    def __init__(
        self,
        n_actions: int = 4,
        gamma: float = 0.99,
        # --- epsilon ---
        epsilon: float = 0.2,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        # --- alpha ---
        alpha: float = 0.1,
        alpha_decay: float = 1.0,
        alpha_min: float = 0.01,
        random_seed: int | None = 0,
    ):
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= alpha_min <= alpha:
            raise ValueError("alpha_min must be in [0, alpha]")
        if not 0.0 < alpha_decay <= 1.0:
            raise ValueError("alpha_decay must be in (0, 1]")

        self.n_actions = n_actions
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min

        # Local RNG instance so each agent has its own independent stream and
        # does not mutate the global random/np.random state. The environment
        # and other agents are unaffected by construction order.
        self._rng = random.Random(random_seed)

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self._episode = []

        self.values = {}
        self.policy = {}

    def take_action(self, state: tuple[int, int]) -> int:
        """Epsilon-greedy action selection with random tie-breaking on greedy ties."""
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
            return MCEpisodeResult(
                delta_q=0.0,
                total_reward=0.0,
                epsilon=self.epsilon,
                alpha=self.alpha,
            )

        returns: list[tuple[tuple[int, int], int, float]] = []
        g_return = 0.0
        for state, action, reward in reversed(self._episode):
            g_return = reward + self.gamma * g_return
            returns.append((state, action, g_return))
        returns.reverse()

        visited: set[tuple[tuple[int, int], int]] = set()
        max_delta = 0.0
        for state, action, g_t in returns:
            if (state, action) in visited:
                continue
            visited.add((state, action))
            old_q = self.q_table[state][action]
            self.q_table[state][action] += self.alpha * (g_t - old_q)
            max_delta = max(max_delta, abs(self.q_table[state][action] - old_q))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

        total_reward = float(sum(r for _, _, r in self._episode))
        self._episode.clear()
        return MCEpisodeResult(
            delta_q=float(max_delta),
            total_reward=total_reward,
            epsilon=float(self.epsilon),
            alpha=float(self.alpha),
        )

    def build_value_and_policy(self) -> None:
        """Populate ``self.values`` and ``self.policy`` from the trained Q-table."""
        self.values = {state: float(np.max(q_vals)) for state, q_vals in self.q_table.items()}
        self.policy = {state: int(np.argmax(q_vals)) for state, q_vals in self.q_table.items()}
