"""Learning rate schedules for tabular RL agents.

Defines an interface for learning-rate schedules used by Q-learning,
Monte Carlo, and off-policy MC agents, plus two concrete schedules:

- ``ExponentialDecaySchedule`` -- per-episode exponential decay with a floor
  (the long-standing default behaviour). Setting ``decay=1.0`` yields a
  constant learning rate, which is why we do *not* maintain a separate
  ``ConstantSchedule`` class.
- ``VisitCountSchedule`` -- state-action visit-count schedule using the
  ``c / (c + N(s, a))`` formula. This satisfies the Robbins-Monro
  conditions (sum of step sizes diverges, sum of squares converges),
  guaranteeing tabular-method convergence to the optimal value function
  in the limit. That is the entire theoretical motivation for using it
  over plain exponential decay.

Schedules expose a stateful ``get_rate(state, action)`` call (it may
mutate internal counts) plus an episode-level ``update_episode()`` hook.
For logging and hyperparameter dumps they also expose the schedule's
intrinsic global rate via ``get_global_rate()`` (or ``None`` when no
single global rate exists, e.g. visit count) and a self-describing dict
via ``describe()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

State = tuple[int, int]


class LearningRateSchedule(ABC):
    """Interface every learning-rate schedule must implement."""

    @abstractmethod
    def get_rate(self, state: State, action: int) -> float:
        """Return the learning rate to apply for the (state, action) update.

        May have side effects (e.g. incrementing a visit counter for
        ``VisitCountSchedule``). Trainers must call this exactly once per
        applied update so visit counts stay in sync with applied steps.
        """

    @abstractmethod
    def update_episode(self) -> None:
        """End-of-episode hook for global decay (no-op for stateless schedules)."""

    @abstractmethod
    def get_global_rate(self) -> float | None:
        """Return the schedule's intrinsic global rate, or ``None`` if there is none.

        Used purely for logging and the ``hyperparams`` dump in the
        training history. Returning ``None`` signals to the trainer that
        the per-episode mean of applied rates is the only meaningful
        scalar to log.
        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Self-describing dict for ``TrainingHistory.hyperparams``.

        For example::

            {"type": "exponential", "alpha": 0.5, "decay": 0.999, "min": 0.05}
            {"type": "visit_count", "c": 5.0}
        """


class ExponentialCommitter:
    """Internal helper that keeps the exponential-decay arithmetic in one place."""

    # Implemented as a regular class rather than module-level functions because
    # `ExponentialDecaySchedule` keeps mutable state (current alpha) and
    # encapsulating the math here documents the invariant that ``current >= min``
    # always holds after construction.

    _current: float
    _decay: float
    _min: float

    def __init__(self, initial: float, decay: float, minimum: float) -> None:
        if not 0.0 < initial <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= minimum <= initial:
            raise ValueError("alpha_min must be in [0, alpha]")
        if not 0.0 < decay <= 1.0:
            raise ValueError("alpha_decay must be in (0, 1]")
        self._current = max(initial, minimum)
        self._decay = decay
        self._min = minimum

    @property
    def current(self) -> float:
        """Return the current global alpha value."""
        return self._current

    def step(self) -> None:
        """Apply one episode of decay, respecting the floor."""
        self._current = max(self._min, self._current * self._decay)


class ExponentialDecaySchedule(LearningRateSchedule):
    """Per-episode exponential decay of a single global alpha.

    Use ``decay=1.0`` to disable decay entirely; that is the canonical
    way to express a constant learning rate in this module.
    """

    _committer: ExponentialCommitter
    _initial: float
    _decay: float
    _min: float

    def __init__(self, alpha: float, decay: float, minimum: float) -> None:
        self._committer = ExponentialCommitter(alpha, decay, minimum)
        self._initial = alpha
        self._decay = decay
        self._min = minimum

    def get_rate(self, state: State, action: int) -> float:
        """Return the current global alpha (state and action are unused)."""
        return self._committer.current

    def update_episode(self) -> None:
        """Decay the global alpha by ``decay``, respecting ``min``."""
        self._committer.step()

    def get_global_rate(self) -> float | None:
        """Return the current global alpha; never ``None`` for this schedule."""
        return self._committer.current

    def describe(self) -> dict[str, Any]:
        """Self-describing dict capturing the exponential-decay configuration."""
        return {
            "type": "exponential",
            "alpha": self._initial,
            "decay": self._decay,
            "min": self._min,
        }


class VisitCountSchedule(LearningRateSchedule):
    """State-action visit-count learning rate ``c / (c + N(s, a))``.

    Satisfies the Robbins-Monro conditions for convergence of tabular
    stochastic approximation. ``c`` is a positive offset that controls
    how aggressively the rate decays with visits: small ``c`` (e.g. 1)
    drops alpha very fast, larger ``c`` (5-50) keeps updates substantial
    for longer.

    The schedule has no global rate; ``get_global_rate()`` returns
    ``None`` and trainers should log the per-episode mean of applied
    rates instead.
    """

    _c: float
    _visits: dict[tuple[State, int], int]

    def __init__(self, c: float = 1.0) -> None:
        if c <= 0.0:
            raise ValueError("visit_count_c must be positive")
        self._c = c
        self._visits = defaultdict(int)

    def get_rate(self, state: State, action: int) -> float:
        """Increment ``N(s, a)`` and return ``c / (c + N(s, a))``."""
        key = (state, action)
        self._visits[key] += 1
        return self._c / (self._c + self._visits[key])

    def update_episode(self) -> None:
        """No-op: visit-count schedule has no episode-level state."""

    def get_global_rate(self) -> float | None:
        """Visit-count schedule has no global rate; always returns ``None``."""
        return None

    def describe(self) -> dict[str, Any]:
        """Self-describing dict capturing the visit-count configuration."""
        return {"type": "visit_count", "c": self._c}


VALID_SCHEDULE_TYPES: tuple[str, ...] = ("exponential", "constant", "visit_count")


def build_lr_schedule(
    schedule_type: str,
    *,
    alpha: float,
    alpha_decay: float,
    alpha_min: float,
    visit_count_c: float,
) -> LearningRateSchedule:
    """Factory that maps a CLI-style schedule name to a concrete schedule.

    ``constant`` resolves to ``ExponentialDecaySchedule(decay=1.0, min=alpha)``
    rather than a separate class because a constant rate is mathematically
    the no-decay limit of the exponential schedule.
    """
    if schedule_type == "exponential":
        return ExponentialDecaySchedule(alpha=alpha, decay=alpha_decay, minimum=alpha_min)
    if schedule_type == "constant":
        return ExponentialDecaySchedule(alpha=alpha, decay=1.0, minimum=alpha)
    if schedule_type == "visit_count":
        return VisitCountSchedule(c=visit_count_c)
    raise ValueError(
        f"Unknown lr_schedule {schedule_type!r}; choose from {VALID_SCHEDULE_TYPES}"
    )


__all__ = [
    "ExponentialDecaySchedule",
    "LearningRateSchedule",
    "VALID_SCHEDULE_TYPES",
    "VisitCountSchedule",
    "build_lr_schedule",
]
