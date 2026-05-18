"""Value-iteration agent for the grid-world delivery task.

The agent does not learn from trial-and-error episodes. Instead, it uses the
known grid, reward function, and environment stochasticity to compute a value
for every reachable cell. After convergence, it turns those values into a
greedy policy: for each cell, choose the action with the best expected return.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from agents.base_agent import BaseAgent
from utils.plotting import TrainingHistory
from world.grid_codes import (
    BOUNDARY_WALL_CELL,
    EMPTY_CELL,
    OBSTACLE_CELL,
    START_CELL,
    TARGET_CELL,
)
from world.helpers import ACTIONS_TO_DIRECTIONS
from world.rewards import STEP_REWARD


RewardFunction = Callable[[np.ndarray, tuple[int, int]], float]
Position = tuple[int, int]
ValueTable = dict[Position, float]
Policy = dict[Position, int]


@dataclass(frozen=True)
class TransitionOutcome:
    """One possible result of choosing an action in a stochastic environment.

    Example: if the robot chooses "right" and sigma > 0, it may actually move
    right, up, down, or left. Each of those possible results is represented by
    one TransitionOutcome.
    """

    probability: float
    next_state: Position
    reward: float
    terminated: bool
    actual_action: int


class ValueIterationAgent(BaseAgent):
    """Tabular dynamic-programming agent using known grid transitions.

    State representation:
        A state is only the robot position, stored as (col, row).

    Important attributes after ``train()``:
        ``values`` maps each state to V(s).
        ``policy`` maps each state to the best action.
        ``history`` stores convergence metrics for plotting/reporting.
    """

    def __init__(
        self,
        grid: np.ndarray,
        reward_fn: RewardFunction,
        sigma: float,
        gamma: float = 0.9,
        theta: float = 1e-6,
        max_iterations: int = 1000,
    ) -> None:
        if not 0.0 <= sigma <= 1.0:
            raise ValueError("sigma must be between 0 and 1")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if theta <= 0.0:
            raise ValueError("theta must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        # ------------------------------------------------------------------
        # Store model and solver hyperparameters
        # ------------------------------------------------------------------
        # The grid is copied because VI mutates START cells into ordinary
        # empty cells: after reset, the robot position is the state and the
        # start marker should not behave like a separate cell type.
        self.grid = np.array(grid, copy=True)
        self.grid[self.grid == START_CELL] = EMPTY_CELL
        self.reward_fn = reward_fn
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.max_iterations = int(max_iterations)

        # ------------------------------------------------------------------
        # Discover state space
        # ------------------------------------------------------------------
        # All empty cells and target cells are valid states. Walls/obstacles
        # cannot be occupied, so they are not states.
        self.states = self._discover_states()
        self.target_states = {state for state in self.states if int(self.grid[state]) == TARGET_CELL}

        # Values start at zero for every valid state. The greedy policy is
        # derived only after the value function has converged or hit the
        # iteration limit.
        self.values = {state: 0.0 for state in self.states}
        self.policy: Policy = {}
        self.history: TrainingHistory | None = None
        self.converged = False
        self.iterations = 0
        self.final_delta_v = float("inf")

    # ------------------------------------------------------------------
    # Grid/state helpers
    # ------------------------------------------------------------------

    def _discover_states(self) -> list[Position]:
        """Return all grid positions the robot can legally occupy."""
        states: list[Position] = []
        for col in range(self.grid.shape[0]):
            for row in range(self.grid.shape[1]):
                if int(self.grid[col, row]) in (EMPTY_CELL, TARGET_CELL):
                    states.append((col, row))
        return states

    def _is_inside_grid(self, pos: Position) -> bool:
        """Whether ``pos`` is a valid array index in the grid."""
        return 0 <= pos[0] < self.grid.shape[0] and 0 <= pos[1] < self.grid.shape[1]

    def _next_position(self, state: Position, action: int) -> Position:
        """Apply an action direction without checking walls or bounds."""
        direction = ACTIONS_TO_DIRECTIONS[action]
        return state[0] + direction[0], state[1] + direction[1]

    def _is_blocked_cell(self, pos: Position) -> bool:
        """Whether ``pos`` is an in-grid wall/obstacle cell."""
        return int(self.grid[pos]) in (BOUNDARY_WALL_CELL, OBSTACLE_CELL)

    # ------------------------------------------------------------------
    # Transition model
    # ------------------------------------------------------------------

    def _actual_action_probability(self, intended_action: int, actual_action: int) -> float:
        """Probability that an intended action becomes a specific actual action."""
        # The environment executes the intended action with probability
        # 1 - sigma, then adds uniformly distributed action noise with total
        # mass sigma. The intended action also receives its share of that
        # uniform noise.
        probability = self.sigma / len(ACTIONS_TO_DIRECTIONS)
        if actual_action == intended_action:
            probability += 1.0 - self.sigma
        return probability

    def transition_outcomes(self, state: Position, intended_action: int) -> list[TransitionOutcome]:
        """Return all possible outcomes for ``state`` and ``intended_action``.

        This mirrors ``Environment.step``:
        - with probability 1 - sigma, the intended action is executed;
        - with probability sigma, a random action is chosen uniformly;
        - wall/obstacle moves keep the robot in the same state;
        - target moves terminate the episode.
        """
        if state not in self.values:
            raise ValueError(f"Unknown state: {state}")
        if intended_action not in ACTIONS_TO_DIRECTIONS:
            raise ValueError(f"Unknown action: {intended_action}")
        if state in self.target_states:
            # Terminal states have no outgoing reward-bearing transitions in
            # the Bellman update; their value is fixed at zero.
            return [TransitionOutcome(1.0, state, 0.0, True, intended_action)]

        outcomes: list[TransitionOutcome] = []
        for actual_action in ACTIONS_TO_DIRECTIONS:
            probability = self._actual_action_probability(intended_action, actual_action)
            candidate = self._next_position(state, actual_action)

            if not self._is_inside_grid(candidate):
                # Moving outside the array is equivalent to bumping into a
                # wall: the agent stays in the same state and receives the
                # step penalty.
                outcomes.append(
                    TransitionOutcome(probability, state, float(STEP_REWARD), False, actual_action)
                )
                continue

            # In-grid candidates use the same reward function as sampled
            # environment rollouts. ``_resolve_move`` then decides whether
            # the attempted move changes state or terminates.
            reward = float(self.reward_fn(self.grid, candidate))
            next_state, terminated = self._resolve_move(state, candidate)
            outcomes.append(TransitionOutcome(probability, next_state, reward, terminated, actual_action))

        return outcomes

    def _resolve_move(self, current_state: Position, candidate: Position) -> tuple[Position, bool]:
        """Translate an attempted next cell into the actual next state."""
        cell_value = int(self.grid[candidate])
        if cell_value in (BOUNDARY_WALL_CELL, OBSTACLE_CELL):
            # Wall/obstacle bumps leave the robot in place.
            return current_state, False
        if cell_value == TARGET_CELL:
            # Reaching the delivery target ends the episode.
            return candidate, True
        if cell_value == EMPTY_CELL:
            return candidate, False
        raise ValueError(f"Unsupported grid cell value {cell_value} at {candidate}")

    # ------------------------------------------------------------------
    # Bellman updates
    # ------------------------------------------------------------------

    def action_value(self, state: Position, action: int, values: ValueTable | None = None) -> float:
        """Compute Q(s, a) from the current value table.

        Q(s, a) = sum over outcomes p(outcome) * [reward + gamma * V(next_state)]
        """
        table = self.values if values is None else values
        expected_return = 0.0

        # Sum over every stochastic outcome of the intended action. Terminal
        # outcomes have no gamma * V(next_state) continuation term.
        for outcome in self.transition_outcomes(state, action):
            continuation = 0.0 if outcome.terminated else self.gamma * table[outcome.next_state]
            expected_return += outcome.probability * (outcome.reward + continuation)
        return expected_return

    def _best_state_value(self, state: Position, old_values: ValueTable) -> float:
        """Compute V(s) = max_a Q(s, a)."""
        # Terminal target states are defined to have zero continuation value.
        if state in self.target_states:
            return 0.0
        return max(self.action_value(state, action, old_values) for action in ACTIONS_TO_DIRECTIONS)

    def train(self) -> TrainingHistory:
        """Run Bellman optimality sweeps until convergence or iteration limit."""
        # ------------------------------------------------------------------
        # Convergence traces
        # ------------------------------------------------------------------
        # The trainer/plotting code treats each Bellman sweep like an episode
        # index so VI can share the same TrainingHistory structure as agents
        # that learn from sampled episodes.
        episodes: list[int] = []
        max_deltas: list[float] = []
        mean_deltas: list[float] = []

        for iteration in range(1, self.max_iterations + 1):
            # Synchronous value iteration: every state's new value is computed
            # from a frozen copy of the previous sweep's values.
            old_values = self.values.copy()
            new_values: ValueTable = {}
            state_deltas: list[float] = []

            for state in self.states:
                new_value = self._best_state_value(state, old_values)
                new_values[state] = new_value
                state_deltas.append(abs(new_value - old_values[state]))

            self.values = new_values
            delta_v = max(state_deltas) if state_deltas else 0.0
            mean_delta_v = float(np.mean(state_deltas)) if state_deltas else 0.0
            episodes.append(iteration)
            max_deltas.append(delta_v)
            mean_deltas.append(mean_delta_v)

            # Stop when the largest single-state value change is below the
            # requested Bellman residual threshold.
            if delta_v < self.theta:
                self.converged = True
                break

        # ------------------------------------------------------------------
        # Final policy and history
        # ------------------------------------------------------------------
        # Once values are fixed, derive a greedy action for every non-terminal
        # state and package convergence diagnostics for plotting/reporting.
        self.iterations = episodes[-1]
        self.final_delta_v = max_deltas[-1]
        self.policy = self._derive_policy()
        self.history = TrainingHistory(
            episodes=episodes,
            metrics={"delta_v": max_deltas, "mean_delta_v": mean_deltas},
            hyperparams={
                "algorithm": "value_iteration",
                "gamma": self.gamma,
                "sigma": self.sigma,
                "theta": self.theta,
                "max_iterations": self.max_iterations,
            },
            metadata={
                "converged": self.converged,
                "iterations": self.iterations,
                "final_delta_v": self.final_delta_v,
            },
        )
        return self.history

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def _derive_policy(self) -> Policy:
        """Choose the best action in every non-terminal state."""
        policy: Policy = {}
        for state in self.states:
            if state in self.target_states:
                continue
            # Use the final value table to evaluate each action, then keep the
            # action with the largest expected return. Ties are resolved by
            # Python's ``max`` insertion order, which follows ACTIONS order.
            action_values = {action: self.action_value(state, action, self.values) for action in ACTIONS_TO_DIRECTIONS}
            policy[state] = max(action_values, key=action_values.get)
        return policy

    def optimal_action_sets(self, tol: float | None = None) -> dict[Position, frozenset[int]]:
        """Per-state set of (approximately) optimal actions.

        Two actions are considered tied when their Q-values differ by no
        more than ``tol``. Default ``tol = 10 * self.theta`` absorbs VI's
        Bellman-residual convergence error while staying tight enough to
        keep genuinely different actions (with Q-gaps on the order of
        ``gamma^d`` for path-length differences ``d``) separated.
        """
        if tol is None:
            tol = 10 * self.theta
        result: dict[Position, frozenset[int]] = {}
        for state in self.states:
            if state in self.target_states:
                continue
            # This is used as the reference for Q-learning/MC policy
            # disagreement. Returning all near-tied optimal actions avoids
            # penalising a learner for choosing an equally good action.
            action_values = {
                action: self.action_value(state, action, self.values)
                for action in ACTIONS_TO_DIRECTIONS
            }
            v_max = max(action_values.values())
            result[state] = frozenset(
                action for action, value in action_values.items() if v_max - value <= tol
            )
        return result

    def take_action(self, state: Position) -> int:
        """Return the greedy action from the trained policy."""
        # Evaluation code calls this just like it calls model-free agents.
        # VI has no exploration mode: once trained, the policy is fixed.
        if not self.policy:
            raise RuntimeError("ValueIterationAgent must be trained before taking actions.")
        if state not in self.policy:
            return 0
        return self.policy[state]

    # ``update()`` is inherited from BaseAgent as a no-op. VI trains before
    # any rollout via the model-based ``train()`` method.
