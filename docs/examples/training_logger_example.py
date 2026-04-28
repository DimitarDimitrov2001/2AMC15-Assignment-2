"""Example: using ``ConsoleTrainingLogger`` with synthetic Q-values.

Run from the project root:

    uv run python docs/examples/training_logger_example.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.training_logger import ConsoleTrainingLogger


def _simulate_logs(logger: ConsoleTrainingLogger, label: str) -> None:
    """Simulate a short converging training loop."""
    print("\n%s" % label)
    rng = np.random.RandomState(42)
    q_values = rng.normal(loc=0.0, scale=0.1, size=(6, 3))

    for step in range(1, 6):
        episode = step * 100
        update_scale = 0.5 / step
        q_values += rng.normal(loc=0.0, scale=update_scale, size=q_values.shape)
        q_delta = float(update_scale)
        mean_q_delta = float(update_scale / 2.0)
        converged = step == 5

        logger.log_iteration(
            episode=episode,
            q_values=q_values,
            q_delta=q_delta,
            converged=converged,
            current_alpha=0.01 / step,
            current_epsilon=max(0.01, 0.2 / step),
            mean_q_delta=mean_q_delta,
        )
        time.sleep(0.35)


def main() -> None:
    scroll_logger = ConsoleTrainingLogger(
        show_q_table=True,
        states=[0, 2, 4],
        redraw_mode="scroll",
    )
    _simulate_logs(scroll_logger, "Scroll mode example")

    frame_logger = ConsoleTrainingLogger(
        show_q_table=True,
        states=[0, 1],
        redraw_mode="frame",
    )
    _simulate_logs(frame_logger, "Frame mode example")


if __name__ == "__main__":
    main()
