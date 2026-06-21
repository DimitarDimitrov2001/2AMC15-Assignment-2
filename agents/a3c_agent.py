"""Asynchronous Advantage Actor-Critic (A3C) agent.

Hogwild!-style multi-process A3C (Mnih et al., 2016, arXiv:1602.01783). A
single shared Actor-Critic network lives in CPU shared memory. N worker
processes each maintain their own environment instance, collect t_max-step
rollouts, and push gradients to the shared parameters asynchronously without
locks. The Trainer delegates the full training loop to ``train_iter`` and only
handles evaluation, logging, and checkpointing on the main process.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agents.base_agent import BaseAgent
from agents.defaults import (
    A3C_DEFAULT_CHECKPOINT_PATH,
    A3C_ENTROPY_BETA,
    A3C_GAMMA,
    A3C_LEARNING_RATE,
    A3C_MAX_GRAD_NORM,
    A3C_N_HIDDEN_NODES,
    A3C_N_WORKERS,
    A3C_PROGRESS_REWARD_SCALE,
    A3C_RANDOM_ACTION_DECAY_STEPS,
    A3C_RANDOM_ACTION_FINAL,
    A3C_RANDOM_ACTION_START,
    A3C_T_MAX,
    A3C_VALUE_COEF,
    A3C_VALUE_TARGET_CLIP,
    CURIOSITY_RESOLUTION_DEFAULT,
)
from world import BaseGridEnvironment
from world.grid_codes import TARGET_CELL


class _ActorCriticNet(nn.Module):
    """Two-headed MLP: outputs action logits and a scalar state value."""

    _body: nn.Sequential
    _policy_head: nn.Linear
    _value_head: nn.Linear

    def __init__(self, state_dim: int, n_actions: int, n_hidden: int) -> None:
        super().__init__()
        self._body = nn.Sequential(
            nn.Linear(state_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self._policy_head = nn.Linear(n_hidden, n_actions)
        self._value_head = nn.Linear(n_hidden, 1)
        # Small-gain orthogonal init keeps all actions equally unbiased before
        # the first gradient step and avoids value-head magnitude blow-up.
        nn.init.orthogonal_(self._policy_head.weight, gain=0.01)
        nn.init.zeros_(self._policy_head.bias)
        nn.init.orthogonal_(self._value_head.weight, gain=1.0)
        nn.init.zeros_(self._value_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action_logits, state_value)`` for input ``x``."""
        h = self._body(x)
        return self._policy_head(h), self._value_head(h).squeeze(-1)


class _SharedAdam(torch.optim.Adam):
    """Adam whose moment buffers live in shared CPU memory.

    Each spawned worker needs read/write access to the optimizer state without
    serializing it via pickling on every gradient step. Initialising the moment
    tensors here and calling ``share_memory_`` achieves this once at
    construction time.
    """

    def __init__(self, params: list, lr: float = 1e-3) -> None:
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def _linear_decay(step: int, start: float, final: float, decay_steps: int) -> float:
    """Linearly decay a scalar schedule value."""
    if decay_steps <= 0:
        return final
    fraction = min(1.0, max(0.0, step / float(decay_steps)))
    return start + fraction * (final - start)


def _target_position(env: BaseGridEnvironment) -> np.ndarray | None:
    """Return the centre point of the first target cell, if one exists."""
    target_cells = np.argwhere(env.grid == TARGET_CELL)
    if target_cells.size == 0:
        return None
    row, col = target_cells[0]
    # Environment positions follow the same axis order as grid indexing:
    # pos[0] -> row axis, pos[1] -> column axis.
    return np.asarray([float(row) + 0.5, float(col) + 0.5], dtype=np.float32)


def _distance_to_target(state: np.ndarray, target_pos: np.ndarray | None) -> float | None:
    """Return Euclidean distance from a raw observation to the target."""
    if target_pos is None:
        return None
    return float(np.linalg.norm(state[:2] - target_pos))


def _progress_shaped_reward(
    reward: float,
    previous_state: np.ndarray,
    next_state: np.ndarray,
    target_pos: np.ndarray | None,
    progress_reward_scale: float,
) -> float:
    """Add a dense training-only reward for moving closer to the target."""
    if progress_reward_scale <= 0.0:
        return reward
    previous_distance = _distance_to_target(previous_state, target_pos)
    next_distance = _distance_to_target(next_state, target_pos)
    if previous_distance is None or next_distance is None:
        return reward
    return reward + progress_reward_scale * (previous_distance - next_distance)


def _grid_count_bonus(
    state: np.ndarray,
    visit_counts: np.ndarray | None,
    resolution: float,
    beta: float,
) -> float:
    """Return a target-free novelty bonus and update the local visit table."""
    if visit_counts is None or beta <= 0.0:
        return 0.0

    idx_x = int(state[0] / resolution)
    idx_y = int(state[1] / resolution)
    idx_x = max(0, min(idx_x, visit_counts.shape[0] - 1))
    idx_y = max(0, min(idx_y, visit_counts.shape[1] - 1))

    count = int(visit_counts[idx_x, idx_y])
    visit_counts[idx_x, idx_y] += 1
    if count == 0:
        return beta * 2.0
    return beta / float(np.sqrt(count))


def _preprocess_observation(
    state: np.ndarray,
    obs_scale: np.ndarray,
    angular_indices: np.ndarray,
    angular_periods: np.ndarray,
) -> np.ndarray:
    """Match DQN preprocessing: wrap angles, scale observations, then clip."""
    phi = np.asarray(state, dtype=np.float32)
    if angular_indices.size:
        phi = phi.copy()
        phi[..., angular_indices] = np.mod(phi[..., angular_indices], angular_periods)
    return np.clip(phi * obs_scale, 0.0, 1.0)


def _worker_process(
    worker_id: int,
    env_fn: Callable[[int], BaseGridEnvironment],
    shared_net: _ActorCriticNet,
    shared_optimizer: _SharedAdam,
    obs_scale: np.ndarray,
    angular_indices: np.ndarray,
    angular_periods: np.ndarray,
    t_max: int,
    gamma: float,
    entropy_beta: float,
    value_coef: float,
    max_grad_norm: float,
    random_action_start: float,
    random_action_final: float,
    random_action_decay_steps: int,
    progress_reward_scale: float,
    curiosity_beta: float,
    curiosity_resolution: float,
    curiosity_shape: tuple[int, int],
    value_target_clip: float,
    total_steps: int,
    max_steps_per_episode: int,
    seed: int,
    report_queue: "mp.Queue[dict | None]",
    global_step: "mp.Value",  # type: ignore[type-arg]
) -> None:
    """Run episodes and push gradient updates until the step budget is spent.

    Each finished episode puts one metrics dict into ``report_queue``. A
    sentinel ``None`` is placed when the worker exits so the main process can
    count terminations.
    """
    rng = np.random.default_rng(seed + worker_id * 1000)
    env = env_fn(seed)
    visit_counts = (
        np.zeros(curiosity_shape, dtype=np.int32)
        if curiosity_beta > 0.0
        else None
    )

    while True:
        with global_step.get_lock():
            current_step = global_step.value
        if current_step >= total_steps:
            break

        # ----- episode reset -----
        raw_state = np.asarray(env.reset(seed=int(rng.integers(1 << 30))), dtype=np.float32)
        target_pos = _target_position(env) if progress_reward_scale > 0.0 else None

        state = _preprocess_observation(raw_state, obs_scale, angular_indices, angular_periods)
        episode_reward = 0.0
        shaped_episode_reward = 0.0
        intrinsic_episode_reward = 0.0
        episode_steps = 0
        terminated = False
        truncated = False

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        random_action_prob_sum = 0.0
        policy_action_sum = 0.0
        return_sum = 0.0
        advantage_sum = 0.0
        value_sum = 0.0
        update_count = 0

        # ----- inner episode loop -----
        while not (terminated or truncated) and episode_steps < max_steps_per_episode:
            with global_step.get_lock():
                if global_step.value >= total_steps:
                    truncated = True
                    break

            # --- collect up to t_max steps ---
            states_buf: list[np.ndarray] = []
            actions_buf: list[int] = []
            rewards_buf: list[float] = []
            random_action_probs_buf: list[float] = []

            for _ in range(t_max):
                states_buf.append(state)

                with global_step.get_lock():
                    step_for_schedule = global_step.value
                random_action_prob = _linear_decay(
                    step=step_for_schedule,
                    start=random_action_start,
                    final=random_action_final,
                    decay_steps=random_action_decay_steps,
                )
                with torch.no_grad():
                    x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                    logits, _ = shared_net(x)
                    probs = F.softmax(logits, dim=-1)
                    n_actions = probs.shape[-1]
                    behavior_probs = (1.0 - random_action_prob) * probs
                    behavior_probs = behavior_probs + random_action_prob / float(n_actions)
                    action = int(torch.multinomial(behavior_probs, 1).item())
                actions_buf.append(action)
                random_action_probs_buf.append(random_action_prob)
                random_action_prob_sum += random_action_prob
                policy_action_sum += 1.0 - random_action_prob

                previous_raw_state = raw_state
                step_result = env.step(action)
                if len(step_result) == 4:
                    ns, r, done, _info = step_result
                    terminated = bool(done)
                    truncated = False
                else:
                    ns, r, terminated, truncated, _info = step_result

                raw_state = np.asarray(ns, dtype=np.float32)
                shaped_reward = _progress_shaped_reward(
                    reward=float(r),
                    previous_state=previous_raw_state,
                    next_state=raw_state,
                    target_pos=target_pos,
                    progress_reward_scale=progress_reward_scale,
                )
                intrinsic_reward = _grid_count_bonus(
                    state=raw_state,
                    visit_counts=visit_counts,
                    resolution=curiosity_resolution,
                    beta=curiosity_beta,
                )
                training_reward = shaped_reward + intrinsic_reward
                state = _preprocess_observation(raw_state, obs_scale, angular_indices, angular_periods)
                rewards_buf.append(training_reward)
                episode_reward += float(r)
                shaped_episode_reward += training_reward
                intrinsic_episode_reward += intrinsic_reward
                episode_steps += 1

                with global_step.get_lock():
                    global_step.value += 1

                if episode_steps >= max_steps_per_episode and not terminated and not truncated:
                    truncated = True
                if terminated or truncated:
                    break

            if not rewards_buf:
                break

            # --- bootstrap value ---
            if terminated or truncated:
                R = 0.0
            else:
                with torch.no_grad():
                    x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                    _, v = shared_net(x)
                    R = float(np.clip(v.item(), -value_target_clip, value_target_clip))

            # --- build discounted returns ---
            returns: list[float] = []
            for r in reversed(rewards_buf):
                R = r + gamma * R
                R = float(np.clip(R, -value_target_clip, value_target_clip))
                returns.insert(0, R)

            # --- gradient update on shared parameters ---
            states_t = torch.as_tensor(np.stack(states_buf), dtype=torch.float32)
            actions_t = torch.as_tensor(actions_buf, dtype=torch.int64)
            returns_t = torch.as_tensor(returns, dtype=torch.float32)
            random_action_probs_t = torch.as_tensor(random_action_probs_buf, dtype=torch.float32)

            logits, values = shared_net(states_t)
            log_probs = F.log_softmax(logits, dim=-1)
            probs_s = F.softmax(logits, dim=-1)
            n_actions = logits.shape[-1]
            behavior_probs_s = (1.0 - random_action_probs_t.unsqueeze(1)) * probs_s
            behavior_probs_s = behavior_probs_s + random_action_probs_t.unsqueeze(1) / float(n_actions)
            behavior_log_probs = torch.log(behavior_probs_s.clamp_min(1e-8))

            log_probs_taken = behavior_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            entropy = -(probs_s * log_probs).sum(dim=-1).mean()
            raw_advantages = returns_t - values.detach()
            if raw_advantages.numel() > 1:
                advantages = (raw_advantages - raw_advantages.mean()) / (
                    raw_advantages.std(unbiased=False) + 1e-8
                )
            else:
                advantages = raw_advantages
            p_loss = -(log_probs_taken * advantages).mean()
            v_loss = F.smooth_l1_loss(values, returns_t)
            loss = p_loss + value_coef * v_loss - entropy_beta * entropy

            shared_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shared_net.parameters(), max_grad_norm)
            shared_optimizer.step()

            policy_loss_sum += float(p_loss.item())
            value_loss_sum += float(v_loss.item())
            entropy_sum += float(entropy.item())
            return_sum += float(returns_t.mean().item())
            advantage_sum += float(raw_advantages.mean().item())
            value_sum += float(values.detach().mean().item())
            update_count += 1

        if episode_steps == 0:
            continue

        with global_step.get_lock():
            step_snapshot = global_step.value

        report_queue.put({
            "global_step": step_snapshot,
            "rollout/episode_reward": episode_reward,
            "rollout/shaped_reward": shaped_episode_reward,
            "rollout/intrinsic_reward": intrinsic_episode_reward,
            "rollout/episode_length": float(episode_steps),
            "rollout/success": float(terminated),
            "losses/policy_loss": policy_loss_sum / max(update_count, 1),
            "losses/value_loss": value_loss_sum / max(update_count, 1),
            "qvals/returns": return_sum / max(update_count, 1),
            "qvals/advantages": advantage_sum / max(update_count, 1),
            "qvals/state_value": value_sum / max(update_count, 1),
            "charts/entropy": entropy_sum / max(update_count, 1),
            "charts/random_action_prob": random_action_prob_sum / max(episode_steps, 1),
            "charts/policy_action_rate": policy_action_sum / max(episode_steps, 1),
        })

    report_queue.put(None)


class A3CAgent(BaseAgent):
    """A3C agent: async actor-learners share one network via CPU shared memory.

    The agent owns its training loop (``trains_externally = True``). The
    Trainer calls :meth:`train_iter` and consumes per-episode dicts from it,
    retaining responsibility for evaluation, logging, and checkpointing.
    """

    # Class-level fields with types (defined before __init__)
    trains_externally: bool = True

    _net: _ActorCriticNet
    _optimizer: _SharedAdam
    _obs_scale: np.ndarray
    _angular_indices: np.ndarray
    _angular_periods: np.ndarray
    _n_workers: int
    _t_max: int
    _gamma: float
    _entropy_beta: float
    _value_coef: float
    _max_grad_norm: float
    _random_action_start: float
    _random_action_final: float
    _random_action_decay_steps: int
    _progress_reward_scale: float
    _curiosity_beta: float
    _curiosity_resolution: float
    _curiosity_shape: tuple[int, int]
    _value_target_clip: float
    _total_steps: int
    _max_steps_per_episode: int
    _seed: int
    _checkpoint_path: str
    _env_fn: Callable[[], BaseGridEnvironment]

    def __init__(
        self,
        env: BaseGridEnvironment,
        env_fn: Callable[[], BaseGridEnvironment],
        seed: int,
        total_steps: int,
        max_steps_per_episode: int,
        n_workers: int = A3C_N_WORKERS,
        t_max: int = A3C_T_MAX,
        gamma: float = A3C_GAMMA,
        learning_rate: float = A3C_LEARNING_RATE,
        entropy_beta: float = A3C_ENTROPY_BETA,
        value_coef: float = A3C_VALUE_COEF,
        max_grad_norm: float = A3C_MAX_GRAD_NORM,
        random_action_start: float = A3C_RANDOM_ACTION_START,
        random_action_final: float = A3C_RANDOM_ACTION_FINAL,
        random_action_decay_steps: int = A3C_RANDOM_ACTION_DECAY_STEPS,
        progress_reward_scale: float = A3C_PROGRESS_REWARD_SCALE,
        curiosity_beta: float = 0.0,
        curiosity_resolution: float = CURIOSITY_RESOLUTION_DEFAULT,
        value_target_clip: float = A3C_VALUE_TARGET_CLIP,
        n_hidden: int = A3C_N_HIDDEN_NODES,
        checkpoint_path: str = A3C_DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
    ) -> None:
        # A3C requires CPU for shared-memory multiprocessing; warn if overridden.
        if device != "cpu":
            import logging
            logging.getLogger(__name__).warning(
                "A3C forces device=cpu for shared-memory multiprocessing; "
                "ignoring requested device=%s",
                device,
            )

        obs_high = np.asarray(env.observation_high, dtype=np.float32)
        obs_high = np.where(obs_high == 0.0, 1.0, obs_high)
        self._obs_scale = (1.0 / obs_high).astype(np.float32)
        self._angular_indices = np.asarray(env.angular_dims, dtype=np.int64)
        self._angular_periods = (
            obs_high[self._angular_indices]
            if self._angular_indices.size
            else self._angular_indices.astype(np.float32)
        )

        state_dim = env.state_dim
        n_actions = env.n_actions

        self._net = _ActorCriticNet(state_dim, n_actions, n_hidden)
        # Must share memory before spawning workers so all processes access the
        # same physical tensors without copying.
        self._net.share_memory()

        self._optimizer = _SharedAdam(list(self._net.parameters()), lr=learning_rate)

        self._env_fn = env_fn
        self._n_workers = n_workers
        self._t_max = t_max
        self._gamma = gamma
        self._entropy_beta = entropy_beta
        self._value_coef = value_coef
        self._max_grad_norm = max_grad_norm
        self._random_action_start = random_action_start
        self._random_action_final = random_action_final
        self._random_action_decay_steps = random_action_decay_steps
        self._progress_reward_scale = progress_reward_scale
        self._curiosity_beta = curiosity_beta
        self._curiosity_resolution = curiosity_resolution
        self._curiosity_shape = (
            int(obs_high[0] / curiosity_resolution) + 1,
            int(obs_high[1] / curiosity_resolution) + 1,
        )
        self._value_target_clip = value_target_clip
        self._total_steps = total_steps
        self._max_steps_per_episode = max_steps_per_episode
        self._seed = seed
        self._checkpoint_path = checkpoint_path

    def train_iter(self) -> Generator[dict, None, None]:
        """Spawn workers and yield one metrics dict per completed episode.

        The generator exits once all workers have stopped (i.e. the global step
        budget is exhausted). The Trainer consumes this generator from the main
        process and handles evaluation / checkpointing between yields.
        """
        ctx = mp.get_context("spawn")
        report_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
        global_step = ctx.Value("i", 0)

        workers = [
            ctx.Process(
                target=_worker_process,
                args=(
                    worker_id,
                    self._env_fn,
                    self._net,
                    self._optimizer,
                    self._obs_scale,
                    self._angular_indices,
                    self._angular_periods,
                    self._t_max,
                    self._gamma,
                    self._entropy_beta,
                    self._value_coef,
                    self._max_grad_norm,
                    self._random_action_start,
                    self._random_action_final,
                    self._random_action_decay_steps,
                    self._progress_reward_scale,
                    self._curiosity_beta,
                    self._curiosity_resolution,
                    self._curiosity_shape,
                    self._value_target_clip,
                    self._total_steps,
                    self._max_steps_per_episode,
                    self._seed + worker_id,  # Have different seed for each subagent.
                    report_queue,
                    global_step,
                ),
                daemon=True,
            )
            for worker_id in range(self._n_workers)
        ]

        for w in workers:
            w.start()

        finished_workers = 0
        try:
            while finished_workers < self._n_workers:
                item = report_queue.get()
                if item is None:
                    finished_workers += 1
                else:
                    yield item
        finally:
            for w in workers:
                w.terminate()
                w.join(timeout=5)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action from the global network.

        During evaluation (``training=False``) the greedy (argmax) action is
        returned. Stochastic sampling is used only when ``training=True``
        (called by eval inside the Trainer — A3C workers never call this).

        Args:
            state: Raw environment observation.
            training: If ``True`` samples stochastically; if ``False`` greedy.

        Returns:
            Discrete action index.
        """
        scaled = _preprocess_observation(
            np.asarray(state, dtype=np.float32),
            self._obs_scale,
            self._angular_indices,
            self._angular_periods,
        )
        x = torch.as_tensor(scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._net(x)
        if training:
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())
        return int(logits.argmax(dim=-1).item())

    def save_checkpoint(self, path: str) -> None:
        """Persist the shared network and optimizer state to ``path``.

        Args:
            path: Destination file. Parent directories are created as needed.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self._net.state_dict(),
                "optimizer": self._optimizer.state_dict(),
            },
            out_path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore the shared network and optimizer state from ``path``.

        Args:
            path: Source file written by :meth:`save_checkpoint`.
        """
        checkpoint = torch.load(path, map_location="cpu")
        self._net.load_state_dict(checkpoint["net"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
