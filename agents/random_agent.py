"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint

import numpy as np

from agents import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, num_actions: int, seed: int | None = None) -> None:
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        return int(self.rng.integers(self.num_actions))