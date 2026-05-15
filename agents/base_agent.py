"""Agent Base.

We define the base class for all agents in this file.
"""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self):
        """Base agent. All other agents should build on this class.

        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """

    @abstractmethod
    def take_action(self, state: tuple[int, int]) -> int:
        """Any code that does the action should be included here.

        Args:
            state: The updated position of the agent.
        """
        raise NotImplementedError

    def update(self, state: tuple[int, int], reward: float, action: int) -> None:
        """Process a reward and update the agent's policy/value estimates.

        Default implementation is a no-op. Agents that learn from per-step
        transitions (e.g. temporal-difference methods like Q-learning) should
        override this. Monte Carlo accumulates updates per episode and Value
        Iteration trains before any rollout, so both inherit the no-op.

        Args:
            state: The updated position of the agent.
            reward: The value returned by the environment as a reward.
            action: The action that was taken by the agent.
        """
        return
