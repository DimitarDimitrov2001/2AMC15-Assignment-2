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
    A3C_T_MAX,
    A3C_VALUE_COEF,
)
from world import BaseGridEnvironment


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


def _worker_process(
    worker_id: int,
    env_fn: Callable[[int], BaseGridEnvironment],
    shared_net: _ActorCriticNet,
    shared_optimizer: _SharedAdam,
    obs_scale: np.ndarray,
    t_max: int,
    gamma: float,
    entropy_beta: float,
    value_coef: float,
    max_grad_norm: float,
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

    while True:
        with global_step.get_lock():
            current_step = global_step.value
        if current_step >= total_steps:
            break

        # ----- episode reset -----
        state = env.reset(seed=int(rng.integers(1 << 30)))

        state = np.asarray(state, dtype=np.float32) * obs_scale
        episode_reward = 0.0
        episode_steps = 0
        terminated = False
        truncated = False

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
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

            for _ in range(t_max):
                states_buf.append(state)

                with torch.no_grad():
                    x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                    logits, _ = shared_net(x)
                    probs = F.softmax(logits, dim=-1)
                    action = int(torch.multinomial(probs, 1).item())
                actions_buf.append(action)

                step_result = env.step(action)
                if len(step_result) == 4:
                    ns, r, done, _info = step_result
                    terminated = bool(done)
                    truncated = False
                else:
                    ns, r, terminated, truncated, _info = step_result

                state = np.asarray(ns, dtype=np.float32) * obs_scale
                rewards_buf.append(float(r))
                episode_reward += float(r)
                episode_steps += 1

                with global_step.get_lock():
                    global_step.value += 1

                if terminated or truncated or episode_steps >= max_steps_per_episode:
                    break

            if not rewards_buf:
                break

            # --- bootstrap value ---
            if terminated:
                R = 0.0
            else:
                with torch.no_grad():
                    x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                    _, v = shared_net(x)
                    R = float(v.item())

            # --- build discounted returns ---
            returns: list[float] = []
            for r in reversed(rewards_buf):
                R = r + gamma * R
                returns.insert(0, R)

            # --- gradient update on shared parameters ---
            states_t = torch.as_tensor(np.stack(states_buf), dtype=torch.float32)
            actions_t = torch.as_tensor(actions_buf, dtype=torch.int64)
            returns_t = torch.as_tensor(returns, dtype=torch.float32)

            logits, values = shared_net(states_t)
            log_probs = F.log_softmax(logits, dim=-1)
            probs_s = F.softmax(logits, dim=-1)

            log_probs_taken = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            entropy = -(probs_s * log_probs).sum(dim=-1).mean()
            advantages = returns_t - values.detach()
            p_loss = -(log_probs_taken * advantages).mean()
            v_loss = F.mse_loss(values, returns_t)
            loss = p_loss + value_coef * v_loss - entropy_beta * entropy

            shared_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shared_net.parameters(), max_grad_norm)
            shared_optimizer.step()

            policy_loss_sum += float(p_loss.item())
            value_loss_sum += float(v_loss.item())
            update_count += 1

        with global_step.get_lock():
            step_snapshot = global_step.value

        report_queue.put({
            "global_step": step_snapshot,
            "rollout/episode_reward": episode_reward,
            "rollout/episode_length": float(episode_steps),
            "rollout/success": float(terminated),
            "losses/policy_loss": policy_loss_sum / max(update_count, 1),
            "losses/value_loss": value_loss_sum / max(update_count, 1),
            "qvals/returns": returns_t / max(update_count, 1),
            "qvals/advantages": advantages  / max(update_count, 1)
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
    _n_workers: int
    _t_max: int
    _gamma: float
    _entropy_beta: float
    _value_coef: float
    _max_grad_norm: float
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
                    self._t_max,
                    self._gamma,
                    self._entropy_beta,
                    self._value_coef,
                    self._max_grad_norm,
                    self._total_steps,
                    self._max_steps_per_episode,
                    self._seed + worker_id, # Have different seed for each subagent
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
        scaled = np.asarray(state, dtype=np.float32) * self._obs_scale
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
