"""Example: using ``utils.plotting`` to visualise training runs.

Generates synthetic data that mimics a typical RL training curve (reward
climbing, delta-Q decaying) and produces two figures:

1. **Single-run plot** -- ``plot_training_history`` with log-scale on the
   delta-Q subplot and a convergence threshold line (similar to the
   task_3 best-config figure).
2. **Multi-run panel grid** -- ``plot_training_histories`` comparing four
   fictional hyperparameter configurations side-by-side.

Run from the project root:

    uv run python docs/examples/plotting_example.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# # Allow importing from the project root when running as a script
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.plotting import (
    SubplotConfig,
    TrainingHistory,
    plot_training_histories,
    plot_training_history,
)


def _synthetic_history(
    n_episodes: int,
    reward_plateau: float,
    dq_floor: float,
    noise_scale: float,
    rng: np.random.RandomState,
    hyperparams: dict[str, float] | None = None,
) -> TrainingHistory:
    """Build a TrainingHistory with realistic-looking synthetic curves."""
    episodes = np.arange(1, n_episodes + 1, dtype=float)

    tau_reward = n_episodes * 0.15
    reward_raw = reward_plateau * (1.0 - np.exp(-episodes / tau_reward))
    reward_raw += rng.normal(scale=noise_scale * abs(reward_plateau) * 0.3, size=n_episodes)

    tau_dq = n_episodes * 0.20
    dq_raw = (1.0 - dq_floor) * np.exp(-episodes / tau_dq) + dq_floor
    dq_raw *= np.exp(rng.normal(scale=0.6, size=n_episodes))

    return TrainingHistory(
        episodes=episodes,
        metrics={
            "avg_reward": reward_raw,
            "delta_q": dq_raw,
        },
        hyperparams=hyperparams or {},
        metadata={"synthetic": True},
    )


def main() -> None:
    rng = np.random.RandomState(42)
    output_dir = Path(tempfile.mkdtemp(prefix="plotting_example_"))

    # ── 1. Single-run plot (dict-based input) ───────────────────────────
    single = _synthetic_history(
        n_episodes=2000,
        reward_plateau=-0.3,
        dq_floor=1e-4,
        noise_scale=0.6,
        rng=rng,
        hyperparams={"epsilon": 0.5, "alpha": 0.001},
    )

    fig_single, _, _ = plot_training_history(
        history=single.to_dict(),
        smoothing_window=50,
        subplot_config={
            "avg_reward": SubplotConfig(
                label="Avg Reward",
                y_label="Average Reward",
                symlog=True,
            ),
            "delta_q": SubplotConfig(
                label="Avg \u0394Q",
                y_label="Average \u0394Q",
                log_scale=True,
                threshold=1e-4,
            ),
        },
        title="Single Run (eps=0.5, alpha=0.001)",
    )
    path_single = output_dir / "single_run.png"
    fig_single.savefig(path_single, dpi=150, bbox_inches="tight")
    print("Single-run plot saved to %s" % path_single)

    # ── 2. Multi-run panel grid (class-based input) ─────────────────────
    configs = [
        {"epsilon": 0.1, "alpha": 0.01},
        {"epsilon": 0.3, "alpha": 0.005},
        {"epsilon": 0.5, "alpha": 0.001},
        {"epsilon": 0.8, "alpha": 0.05},
    ]
    histories = [
        _synthetic_history(
            n_episodes=1500,
            reward_plateau=-0.3 + rng.uniform(-0.1, 0.1),
            dq_floor=10 ** rng.uniform(-5, -3),
            noise_scale=0.5,
            rng=rng,
            hyperparams=cfg,
        )
        for cfg in configs
    ]

    fig_multi, _ = plot_training_histories(
        histories=histories,
        metrics_to_plot=["avg_reward", "delta_q"],
        smoothing_window=30,
        columns=2,
        common_scale=True,
        subplot_config={
            "avg_reward": SubplotConfig(y_label="Avg Reward"),
            "delta_q": SubplotConfig(y_label="\u0394Q", log_scale=True),
        },
        title="Hyperparameter Comparison",
    )
    path_multi = output_dir / "multi_run.png"
    fig_multi.savefig(path_multi, dpi=150, bbox_inches="tight")
    print("Multi-run plot saved to %s" % path_multi)


if __name__ == "__main__":
    main()
