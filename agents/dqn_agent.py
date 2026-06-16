from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

import numpy as np

from agents import BaseAgent, Transition, ReplayBuffer, Batch
from agents.epsilon_schedules import EpsilonSchedule, ConstantEpsilon, _DEFAULT_EPSILON
from world import BaseGridEnvironment

_N_HIDDEN_NODES = 64
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_LEARNING_RATE = 2.5e-4
_DEFAULT_GAMMA = 0.99
_DEFAULT_TARGET_UPDATE_FREQ = 1000
_DEFAULT_CHECKPOINT_PATH = "models/dqn/best_model.pt"

class DQNAgent(BaseAgent):
    _rng: np.random.Generator
    n_actions: np.ndarray
    state_dim: int
    replay_buffer: ReplayBuffer
    batch_size: int
    _learning_rate: float
    _update_network: nn.Sequential
    _target_network: nn.Sequential
    _checkpoint_path: str
    _target_update_freq: int
    _learn_steps: int
    _device: torch.device
    _obs_scale: np.ndarray

    def __init__(
        self,
        env: BaseGridEnvironment,
        seed: int,
        gamma: float = _DEFAULT_GAMMA,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        replay_buffer_capacity: int | None = None,
        epsilon_scheduler: EpsilonSchedule | None = None,
        target_update_freq: int = _DEFAULT_TARGET_UPDATE_FREQ,
        checkpoint_path: str = _DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
    ):
        self.env = env
        self.n_actions = env.n_actions
        self.state_dim = env.state_dim
        self.gamma = gamma
        self._learning_rate = learning_rate
        self._device = torch.device(device)
        obs_high = np.asarray(env.observation_high, dtype=np.float32)
        obs_high = np.where(obs_high == 0.0, 1.0, obs_high)
        self._obs_scale = (1.0 / obs_high).astype(np.float32)
        self._update_network = self._build_q_network().to(self._device)
        self._target_network = self._build_q_network().to(self._device)
        self._target_network.load_state_dict(self._update_network.state_dict())
        self._optimizer = torch.optim.Adam(self._update_network.parameters(), learning_rate)
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(obs_dim=self.state_dim, capacity=replay_buffer_capacity, seed=seed)
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._checkpoint_path = checkpoint_path
        self._target_update_freq = target_update_freq
        self._learn_steps = 0
        self.epsilon_scheduler = epsilon_scheduler if epsilon_scheduler is not None else ConstantEpsilon()

    def _build_q_network(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.state_dim, _N_HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(_N_HIDDEN_NODES, _N_HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(_N_HIDDEN_NODES, self.n_actions),
        )

    def _preprocess(self, state: np.ndarray) -> np.ndarray:
        """Map a raw (optionally batched) state to the normalized observation phi.

        Broadcasts over the trailing observation dimension, so it handles both a
        single state of shape ``(state_dim,)`` and a batch ``(B, state_dim)``.
        """
        return np.asarray(state, dtype=np.float32) * self._obs_scale

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Pick an action greedily from the online net, exploring while training."""
        if training and self._rng.random() < self.epsilon_scheduler.epsilon(state):
            return int(self._rng.integers(self.n_actions))
        with torch.no_grad():
            phi = torch.as_tensor(self._preprocess(state), dtype=torch.float32, device=self._device)
            q_values = self._update_network(phi)
        return int(q_values.argmax().item())

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.add(
            state=self._preprocess(transition.state),
            action=transition.action,
            reward=transition.reward,
            next_state=self._preprocess(transition.next_state),
            done=transition.terminated,
        )
        return None

    def update(self) -> dict[str, float]:
        """Run one DQN gradient step and return training diagnostics.

        Returns an empty dict until the buffer holds a full batch; otherwise a
        dict of scalars (loss, Q/target statistics, gradient norm, learning
        rate, buffer fill) that the Trainer averages per episode and logs.
        """
        if not self.replay_buffer.can_sample(self.batch_size):
            return {}
        batch: Batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self._device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self._device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self._device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self._device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self._device)

        # Q(φ_j, ·; θ) for all actions, then the Q of the action actually taken.
        q_all = self._update_network(states)
        q_pred = q_all.gather(1, actions).squeeze(1)

        # y_j: no gradient flows through the target.
        with torch.no_grad():
            q_next = self._target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * q_next * (1.0 - dones)

        loss = torch.nn.functional.mse_loss(q_pred, targets)

        self._optimizer.zero_grad()
        loss.backward()
        # max_norm=inf measures the gradient norm without actually clipping it.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._update_network.parameters(), max_norm=float("inf")
        )
        self._optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self._target_update_freq == 0:
            self._sync_target_network()

        return {
            "loss": float(loss.item()),
            "q_value_mean": float(q_pred.mean().item()),
            "state_value_mean": float(q_all.max(dim=1).values.mean().item()),
            "target_mean": float(targets.mean().item()),
            "td_error_abs": float((targets - q_pred).abs().mean().item()),
            "grad_norm": float(grad_norm),
            "learning_rate": float(self._optimizer.param_groups[0]["lr"]),
            "buffer_size": float(len(self.replay_buffer)),
            "learn_steps": float(self._learn_steps),
            "epsilon": float(self.epsilon_scheduler.epsilon(None)),
        }

    def _sync_target_network(self) -> None:
        """Copy the online network weights into the target network."""
        self._target_network.load_state_dict(self._update_network.state_dict())


    def on_episode_start(self, episode: int) -> None:
        
        return None

    def on_episode_end(self, episode: int, episode_metrics: dict[str, float]) -> dict[str, float]:
        # Advance the epsilon schedule once per episode; nothing else does this.
        self.epsilon_scheduler.on_episode_end()
        return {"epsilon": float(self.epsilon_scheduler.epsilon(None))}

    def save_checkpoint(self, path: str) -> None:
        """Persist the online/target networks and optimizer state to ``path``.

        Args:
            path: Destination file. Parent directories are created as needed.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "update_network": self._update_network.state_dict(),
                "target_network": self._target_network.state_dict(),
                "optimizer": self._optimizer.state_dict(),
            },
            out_path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore the online/target networks and optimizer state from ``path``.

        Args:
            path: Source file written by :meth:`save_checkpoint`.
        """
        # map_location keeps loading portable across CPU/GPU machines.
        checkpoint = torch.load(path, map_location=self._device)
        self._update_network.load_state_dict(checkpoint["update_network"])
        self._target_network.load_state_dict(checkpoint["target_network"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])