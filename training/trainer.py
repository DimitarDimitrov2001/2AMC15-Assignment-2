from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agents.base_agent import BaseAgent, Transition
from training.config import TrainerConfig
from world import BaseGridEnvironment


class Trainer:
    """
    General trainer for Deep RL agents

    The trainer is algorithm-agnostic. 
    It trains any agent that implements the BaseAgent interface (DQN, PPO etc.)
    Responsibility of the trainer includes:
    - Running episodes
    - Asking agent for actions
    - Stepping the environment
    - Passing transitions back to the agent
    - call agent.update()
    - Evaluate the current policy
    - logging to W&B
    """

    # Create the trainer
    def __init__(
        self,
        env: BaseGridEnvironment,
        agent: BaseAgent,
        config: TrainerConfig, # Includes all the settings from the config
        eval_env: BaseGridEnvironment | None = None, # Separate evaluation environment? (Optional)
        # Optional hook: render a greedy rollout for ``agent`` at ``episode`` and
        # return the saved image path. Logged to W&B at the logging cadence.
        viz_fn: Callable[[BaseAgent, int], str] | None = None,
    ) -> None:
        self.env = env
        # Use separate evaluation environment, otherwise use same environment for training and evaluation by default
        self.eval_env = eval_env if eval_env is not None else env
        self.agent = agent
        self.config = config
        self._viz_fn = viz_fn

        # Counts all steps across all episodes
        self.global_step = 0
        # Store all episode metrics as a dictionary
        self.history: list[dict[str, float]] = []

        # Best value seen so far for config.best_metric (used for checkpointing)
        self._best_metric_value: float | None = None

        # Stores the W&B module (if enabled)
        self._wandb: Any = None

    def train(self) -> list[dict[str, float]]:
        """
        Runs the full training loop.
        For each episode:
        - Reset the environment
            For each step:
            - Ask the agent for action / step in environment
            - Create transition, inform agent
            - Update agent
            - Stop if the episode ends
        - Save episode metrics (or evaluate)
        Returns dict of metrics for each episode
        """
        # Random behavior can be reproducible
        self._set_seed(self.config.seed)
        # W&B only if its enabled in the config
        self._setup_wandb()

        # Agents that own their training loop (e.g. A3C) drive their own
        # environments; the Trainer only consumes their per-episode reports.
        if getattr(self.agent, "trains_externally", False):
            return self._train_external()

        start_time = time.perf_counter()

        # Loop over training episodes (First episode is 1)
        for episode in range(1, self.config.total_episodes + 1):
            # Option to do something at the start of each episode
            self.agent.on_episode_start(episode)

            # Reset the environment and get the initial state (add seed to each episode)
            state = self._reset_env(self.env, seed=self.config.seed + episode)

            # Episode statistics
            episode_reward = 0.0
            episode_length = 0

            # Becomes true if environment terminates
            terminated = False
            truncated = False

            # Accumulate agent.update() metrics across the whole episode so the
            # logged value is a per-episode mean instead of only the last step.
            update_sums: dict[str, float] = {}
            update_counts: dict[str, int] = {}

            # Set when the env-step budget is reached mid-episode.
            stop_training = False

            # Run one episode
            for _step in range(self.config.max_steps_per_episode):
                # Asks agent to choose action
                action = self.agent.select_action(state, training=True) # training=True means its allowed to explore / No evaluation

                # Do the selected action
                next_state, reward, terminated, truncated, info = self._step_env(
                    self.env,
                    action,
                )

                # Treat the step-budget timeout as a truncation so the agent can
                # ground its TD targets (done=True)
                if (
                    _step == self.config.max_steps_per_episode - 1
                    and not terminated
                    and not truncated
                ):
                    truncated = True
                    info["time_limit"] = True

                # Make an interaction to be a Transition object
                transition = Transition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )

                # Agent sees the transition
                self.agent.observe(transition)
                # Agent does a learning update
                update_metrics = self.agent.update()
                for key, value in update_metrics.items():
                    update_sums[key] = update_sums.get(key, 0.0) + float(value)
                    update_counts[key] = update_counts.get(key, 0) + 1

                # Episode counters
                episode_reward += float(reward)
                episode_length += 1
                self.global_step += 1 # Global across all episodes

                # Move to next state
                state = next_state

                # Stop episode if environment says so (terminated = reach goal, truncated = max episode length reached)
                if terminated or truncated or stop_training:
                    break

            # Collect episode metrics
            episode_metrics: dict[str, float] = {
                "episode": float(episode),
                "global_step": float(self.global_step),
                "rollout/episode_reward": float(episode_reward),
                "rollout/episode_length": float(episode_length),
                "rollout/success": float(terminated),
                "charts/SPS": float(self.global_step / max(time.perf_counter() - start_time, 1e-9)),
            }

            # Per-episode mean of the agent.update() metrics
            for key, total in update_sums.items():
                episode_metrics[key] = total / update_counts[key]

            end_metrics = self.agent.on_episode_end(episode, episode_metrics)
            for key, value in end_metrics.items():
                episode_metrics[key] = float(value)

            # Evaluate the current policy (Separate from training)
            if episode % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                episode_metrics.update(eval_metrics)
                # Checkpoint the best policy seen so far (if enabled)
                self._maybe_save_best(eval_metrics)

            self.history.append(episode_metrics)

            # Print metrics and pass to W&B
            if episode % self.config.log_interval == 0:
                self._log(episode_metrics, episode)
                self._maybe_log_rollout(episode)

            # Honor the optional env-step budget
            if stop_training:
                break

        # Persist the final agent state and history (if enabled)
        self._maybe_save_last()
        if self.config.history_path is not None:
            self.save_history(self.config.history_path)

        # Some entry points generate and log post-training artifacts after
        # train() returns, so they keep the run open and finish it themselves.
        if self.config.finish_wandb_on_train_end:
            self._finish_wandb()
        return self.history

    def _train_external(self) -> list[dict[str, float]]:
        """Consume episode reports from a self-training agent (e.g. A3C).

        The agent's ``train_iter`` generator owns the parallel environment
        rollouts and yields one metrics dict per finished episode. The Trainer
        retains ownership of evaluation, logging, checkpointing and W&B, all run
        on the agent's shared global network on the main process.
        """
        start_time = time.perf_counter()

        for episode, report in enumerate(self.agent.train_iter(), start=1):
            self.global_step = int(report.get("global_step", self.global_step))

            episode_metrics: dict[str, float] = dict(report)
            episode_metrics["episode"] = float(episode)
            episode_metrics["global_step"] = float(self.global_step)
            episode_metrics["charts/SPS"] = float(
                self.global_step / max(time.perf_counter() - start_time, 1e-9)
            )

            # Evaluate the shared global policy on a cadence (no learning here).
            if episode % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                episode_metrics.update(eval_metrics)
                self._maybe_save_best(eval_metrics)

            self.history.append(episode_metrics)

            if episode % self.config.log_interval == 0:
                self._log(episode_metrics, episode)
                self._maybe_log_rollout(episode)

        self._maybe_save_last()
        if self.config.history_path is not None:
            self.save_history(self.config.history_path)

        if self.config.finish_wandb_on_train_end:
            self._finish_wandb()
        return self.history

    def _maybe_log_rollout(self, episode: int) -> None:
        """Render a greedy rollout and log it to W&B as an image (no-op if disabled).

        The image path is logged directly to W&B and intentionally kept out of
        ``episode_metrics`` (which is JSON-serialised by ``save_history``).
        """
        if self._wandb is None or self._viz_fn is None:
            return
        image_path = self._viz_fn(self.agent, episode)
        image = self._wandb.Image(image_path, caption=f"episode {episode}")
        self._wandb.log({"viz/rollout": image}, step=self.global_step)

    def save_history(self, path: str) -> None:
        """Write the per-episode metric history to ``path`` as JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

    def _maybe_save_best(self, eval_metrics: dict[str, float]) -> None:
        """Save a 'best' checkpoint when config.best_metric improves."""
        if not self.config.save_best or self.config.checkpoint_dir is None:
            return

        value = eval_metrics.get(self.config.best_metric)
        if value is None:
            return

        if self._best_metric_value is None or value > self._best_metric_value:
            self._best_metric_value = value
            path = Path(self.config.checkpoint_dir) / "best.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.agent.save_checkpoint(str(path))

    def _maybe_save_last(self) -> None:
        """Save a 'last' checkpoint at the end of training when enabled."""
        if not self.config.save_last or self.config.checkpoint_dir is None:
            return

        path = Path(self.config.checkpoint_dir) / "last.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_checkpoint(str(path))

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the current policy.
        This means no learning. It is possible to run multiple episodes and take the mean.
        It is called with training=False, the agent does not explore.
        """
        rewards: list[float] = []
        lengths: list[int] = []
        successes: list[float] = []

        # Run evaluation episodes (can be multiple so randomness plays less of a role)
        for eval_episode in range(self.config.eval_episodes):
            self.agent.on_episode_start(eval_episode)

            # Use different seed than for training
            state = self._reset_env(
                self.eval_env,
                seed=self.config.seed + 10_000 + eval_episode,
            )

            episode_reward = 0.0
            episode_length = 0

            terminated = False
            truncated = False
            # Stores if the goal was reached
            final_info: dict[str, Any] = {} 

            for _step in range(self.config.max_steps_per_episode):
                # Select action (evaluation mode)
                action = self.agent.select_action(state, training=False)

                # Step in the environment
                next_state, reward, terminated, truncated, info = self._step_env(
                    self.eval_env,
                    action,
                )

                episode_reward += float(reward)
                episode_length += 1
                state = next_state
                final_info = info

                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)

            # Terminated is considered a success
            success = final_info.get("success", terminated)
            successes.append(float(success))

        # Return the averages of evaluation metrics
        return {
            "eval/mean_reward": float(np.mean(rewards)),
            "eval/std_reward": float(np.std(rewards)),
            "eval/mean_length": float(np.mean(lengths)),
            "eval/success_rate": float(np.mean(successes)),
        }

    def _reset_env(self, env: Any, seed: int | None = None) -> np.ndarray:
        """
        Reset the environment and return to initial state.
        Helper function to support both state = env.reset() and state, info = env.reset(seed=seed).
        """
        try:
            result = env.reset(seed=seed)
        except TypeError:
            result = env.reset()

        if isinstance(result, tuple):
            state, _info = result
        else:
            state = result

        return np.asarray(state, dtype=np.float32)

    def _step_env(self, env: Any, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        One step in the environment.
        Helper function to support both env.step(action) =
        next_state, reward, done, info
        next_state, reward, terminated, truncated, info
        """
        result = env.step(action)

        # next_state, reward, done, info
        if len(result) == 4:
            next_state, reward, done, info = result
            terminated = bool(done)
            truncated = False
        # next_state, reward, terminated, truncated, info
        elif len(result) == 5:
            next_state, reward, terminated, truncated, info = result
        else:
            raise ValueError(
                "env.step(action) must have either 4 or 5 values."
            )

        return (
            np.asarray(next_state, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    def _log(self, metrics: dict[str, float], episode: int) -> None:
        """Print window-averaged stats to the terminal and raw metrics to W&B.

        Terminal values are means over the last ``log_interval`` episodes to
        smooth per-episode noise (``term_rate`` is the goal-reach fraction);
        W&B still receives the raw current-episode metrics for its own charts.
        """
        # Window = the episodes since the previous log (current one included).
        window = self.history[-self.config.log_interval:]

        reward = self._window_mean(window, "rollout/episode_reward")
        length = self._window_mean(window, "rollout/episode_length")
        term_rate = self._window_mean(window, "rollout/success")
        # Fall back to the policy loss for agents without a TD loss (e.g. A3C).
        loss = self._window_mean(window, "losses/td_loss")
        if not np.isfinite(loss):
            loss = self._window_mean(window, "losses/policy_loss")
        q_value = self._window_mean(window, "qvals/q_taken")
        if not np.isfinite(q_value):
            q_value = self._window_mean(window, "qvals/state_value")
        if not np.isfinite(q_value):
            q_value = self._window_mean(window, "qvals/returns")
        # DQN reports epsilon; A3C reports explicit random-action probability
        # plus entropy. Prefer the action probability for terminal readability.
        exploration = metrics.get(
            "charts/epsilon",
            metrics.get("charts/random_action_prob", metrics.get("charts/entropy", float("nan"))),
        )

        log_parts = [
            f"Episode {episode:5d}",
            f"reward={reward:8.3f}",
            f"len={length:5.1f}",
            f"term_rate={term_rate:4.2f}",
            f"loss={loss:8.4f}",
            f"value={q_value:7.3f}",
            f"explore={exploration:4.2f}",
            f"steps={self.global_step:7d}",
        ]

        optional_fields = [
            ("dqn_eps", "dqn/epsilon", ".3f"),
            ("dqn_loss", "dqn/loss", ".4g"),
            ("dqn_td", "dqn/td_error_mean", ".4g"),
            ("dqn_qmax", "dqn/q_max", ".3f"),
            ("buffer", "dqn/buffer_size", ".0f"),
            ("updates", "dqn/updates", ".0f"),
            ("success", "success", ".0f"),
            ("collisions", "collisions", ".0f"),
            ("eval_reward", "eval/mean_reward", ".3f"),
            ("eval_success", "eval/success_rate", ".3f"),
        ]
        for label, key, fmt in optional_fields:
            value = metrics.get(key)
            if value is not None:
                log_parts.append(f"{label}={float(value):{fmt}}")

        print(" | ".join(log_parts))

        # Send to W&B if enabled
        if self._wandb is not None:
            self._wandb.log(metrics, step=self.global_step)

    @staticmethod
    def _window_mean(window: list[dict[str, float]], key: str) -> float:
        """Mean of ``key`` over window episodes where it is present and finite."""
        values = [m[key] for m in window if key in m and np.isfinite(m[key])]
        return float(np.mean(values)) if values else float("nan")

    def _setup_wandb(self) -> None:
        """
        Start W&B run
        """

        # Do nothing if disabled
        if not self.config.use_wandb:
            return

        # Use W&B only when needed
        import wandb

        self._wandb = wandb
        
        # Use full_config if provided, otherwise fallback to asdict(self.config)
        config_to_log = asdict(self.config)
        if self.config.full_config is not None:
            config_to_log = self.config.full_config

        # start_method="thread" runs the W&B backend in-process instead of a
        # separate spawned service. On macOS the spawned-service transport
        # frequently deadlocks during wandb.finish() (network flush hangs),
        # which does not happen on Linux/Windows; threads avoid that stall.
        self._wandb.init(
            project=self.config.wandb_project,
            group=self.config.wandb_group,
            name=self.config.run_name,
            config=config_to_log,
            settings=self._wandb.Settings(start_method="thread"),
        )

    # Finish W&B run (if started)
    def _finish_wandb(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

    def finish_wandb(self) -> None:
        """Finish the active W&B run, if one was started."""
        self._finish_wandb()

    @staticmethod
    def _set_seed(seed: int) -> None:
        """
        Helper function to set random seeds.
        Important to compare experiments.
        """
        random.seed(seed)
        np.random.seed(seed)

        # Optional use of pytorch
        try:
            import torch

            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
