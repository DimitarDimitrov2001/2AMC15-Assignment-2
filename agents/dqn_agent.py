from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agents.base_agent import BaseAgent, Transition
from agents.curiosity import IntrinsicMotivation, NoMotivation
from agents.defaults import (
    DQN_DEFAULT_BATCH_SIZE,
    DQN_DEFAULT_CHECKPOINT_PATH,
    DQN_DEFAULT_GAMMA,
    DQN_DEFAULT_GRAD_CLIP_NORM,
    DQN_DEFAULT_LEARNING_RATE,
    DQN_DEFAULT_NO_OBS_IN_STATE,
    DQN_DEFAULT_TARGET_UPDATE_FREQ,
    DQN_DEFAULT_UPDATE_FREQ,
    DQN_N_HIDDEN_NODES,
)
from agents.epsilon_schedules import EpsilonSchedule, LinearEpsilonAnnealing
from agents.replay_buffer import Batch, ReplayBuffer
from world import BaseGridEnvironment


class DQNAgent(BaseAgent):
    """Baseline DQN implementation"""

    _rng: np.random.Generator
    n_actions: int
    state_dim: int
    _single_obs_dim: int
    replay_buffer: ReplayBuffer
    batch_size: int
    _learning_rate: float
    _update_network: nn.Module
    _target_network: nn.Module
    _checkpoint_path: str
    _no_obs_in_state: int
    _update_freq: int
    _target_update_freq: int
    _learn_steps: int
    _total_steps: int
    _device: torch.device
    _obs_scale: np.ndarray
    _angular_indices: np.ndarray
    _angular_periods: np.ndarray
    _obs_buffer: collections.deque[np.ndarray]
    _loss_fn: nn.SmoothL1Loss
    _grad_clip_norm: float | None
    intrinsic_motivation: IntrinsicMotivation
    _episode_intrinsic_reward: float

    def __init__(
        self,
        env: BaseGridEnvironment,
        seed: int,
        gamma: float = DQN_DEFAULT_GAMMA,
        learning_rate: float = DQN_DEFAULT_LEARNING_RATE,
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int = DQN_DEFAULT_BATCH_SIZE,
        replay_buffer_capacity: int | None = None,
        epsilon_scheduler: EpsilonSchedule | None = None,
        intrinsic_motivation: IntrinsicMotivation | None = None,
        no_obs_in_state: int = DQN_DEFAULT_NO_OBS_IN_STATE,
        update_freq: int = DQN_DEFAULT_UPDATE_FREQ,
        target_update_freq: int = DQN_DEFAULT_TARGET_UPDATE_FREQ,
        checkpoint_path: str = DQN_DEFAULT_CHECKPOINT_PATH,
        grad_clip_norm: float | None = DQN_DEFAULT_GRAD_CLIP_NORM,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.n_actions = env.n_actions
        self._single_obs_dim = env.state_dim
        self._no_obs_in_state = no_obs_in_state
        self.state_dim = self._single_obs_dim * self._no_obs_in_state
        self.gamma = gamma
        self._learning_rate = learning_rate
        self._grad_clip_norm = grad_clip_norm
        self._device = self._resolve_device(device)
        torch.manual_seed(seed)
        if self._device.type == "cuda" and hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(seed)
        elif self._device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)

        # Keep feature scales predictable for the MLP. Angle dimensions are
        # wrapped in _preprocess before this scale is applied.
        obs_high = np.asarray(env.observation_high, dtype=np.float32)
        obs_high = np.where(obs_high == 0.0, 1.0, obs_high)
        single_obs_scale = (1.0 / obs_high).astype(np.float32)
        self._obs_scale = np.tile(single_obs_scale, self._no_obs_in_state)
        single_angular = np.asarray(env.angular_dims, dtype=np.int64)
        self._angular_indices = (
            np.concatenate(
                [
                    single_angular + frame * self._single_obs_dim
                    for frame in range(self._no_obs_in_state)
                ]
            ).astype(np.int64)
            if single_angular.size
            else single_angular
        )
        self._angular_periods = (
            obs_high[single_angular]
            if single_angular.size
            else single_angular.astype(np.float32)
        )
        self._angular_periods = np.tile(self._angular_periods, self._no_obs_in_state)
        self._update_network = self._build_q_network().to(self._device)
        self._target_network = self._build_q_network().to(self._device)
        # The target network starts as a copy and is only refreshed every
        # _target_update_freq learning steps.
        self._target_network.load_state_dict(self._update_network.state_dict())
        self._target_network.eval()
        self._optimizer = torch.optim.Adam(self._update_network.parameters(), learning_rate)
        self._loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = (
            replay_buffer
            if replay_buffer is not None
            else ReplayBuffer(
                obs_dim=self.state_dim,
                capacity=replay_buffer_capacity,
                seed=seed,
                device=self._device,
            )
        )
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._checkpoint_path = checkpoint_path
        self._update_freq = update_freq
        self._target_update_freq = target_update_freq
        self._learn_steps = 0
        self._total_steps = 0
        self._obs_buffer = collections.deque(maxlen=self._no_obs_in_state)
        self.epsilon_scheduler = (
            epsilon_scheduler if epsilon_scheduler is not None else LinearEpsilonAnnealing()
        )
        self.intrinsic_motivation = (
            intrinsic_motivation if intrinsic_motivation is not None else NoMotivation()
        )
        self._episode_intrinsic_reward = 0.0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve a device string, supporting cuda, mps, and cpu."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return torch.device(device)

    def _build_q_network(self) -> nn.Module:
        """Build the online/target Q-network architecture."""
        return nn.Sequential(
            nn.Linear(self.state_dim, DQN_N_HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(DQN_N_HIDDEN_NODES, DQN_N_HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(DQN_N_HIDDEN_NODES, self.n_actions),
        )

    def _preprocess(self, state: np.ndarray) -> np.ndarray:
        """Map a raw state to normalized DQN input features."""
        phi = np.asarray(state, dtype=np.float32)
        if self._angular_indices.size:
            phi = phi.copy()
            phi[..., self._angular_indices] = np.mod(
                phi[..., self._angular_indices],
                self._angular_periods,
            )
        return np.clip(phi * self._obs_scale, 0.0, 1.0)

    def _get_phi(self) -> np.ndarray:
        """Return the current stacked observation."""
        return np.concatenate(list(self._obs_buffer), axis=-1)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Pick an action greedily from the online net, exploring while training."""
        if len(self._obs_buffer) == 0:
            # At episode start there is no history yet, so repeat the first
            # observation to fill the stack.
            for _ in range(self._no_obs_in_state):
                self._obs_buffer.append(state)
        elif not training:
            self._obs_buffer.append(state)

        phi_state = self._get_phi()
        if training and self._rng.random() < self.epsilon_scheduler.epsilon(phi_state):
            return int(self._rng.integers(self.n_actions))

        with torch.no_grad():
            phi = torch.as_tensor(
                self._preprocess(phi_state),
                dtype=torch.float32,
                device=self._device,
            )
            q_values = self._update_network(phi)
        return int(q_values.argmax().item())

    def observe(self, transition: Transition) -> None:
        """Store one trainer transition with normalized states in replay memory."""
        # Replay stores the stacked state before and after the environment step.
        phi_t = self._get_phi()
        self._obs_buffer.append(transition.next_state)
        phi_tp1 = self._get_phi()

        # Curiosity, when enabled, is added on top of the environment reward.
        extrinsic = transition.reward
        bonus = self.intrinsic_motivation.get_bonus_and_update(transition.next_state)
        reward = extrinsic + bonus
        self._episode_intrinsic_reward += bonus

        self.replay_buffer.add(
            state=self._preprocess(phi_t),
            action=transition.action,
            reward=reward,
            next_state=self._preprocess(phi_tp1),
            done=bool(transition.terminated or transition.truncated),
        )
        self._total_steps += 1
        self.epsilon_scheduler.step()

    def update(self) -> dict[str, float]:
        """Run one DQN gradient step and return training diagnostics."""
        # Learning is delayed by both the update frequency and replay warmup.
        if self._total_steps % self._update_freq != 0:
            return {}
        if not self.replay_buffer.can_sample(self.batch_size):
            return {}
        batch: Batch = self.replay_buffer.sample(self.batch_size)
        actions = batch.actions.unsqueeze(1)

        q_all = self._update_network(batch.states)
        q_pred = q_all.gather(1, actions).squeeze(1)

        with torch.no_grad():
            # Standard DQN target
            q_next = self._target_network(batch.next_states).max(dim=1).values
            targets = batch.rewards + self.gamma * q_next * (1.0 - batch.dones)

        loss = self._loss_fn(q_pred, targets)

        self._optimizer.zero_grad()
        loss.backward()
        max_norm = self._grad_clip_norm if self._grad_clip_norm is not None else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._update_network.parameters(),
            max_norm=max_norm,
        )
        self._optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self._target_update_freq == 0:
            self._sync_target_network()

        return {
            "losses/td_loss": float(loss.item()),
            "losses/td_error_abs": float((targets - q_pred).abs().mean().item()),
            "losses/grad_norm": float(grad_norm),
            "qvals/q_taken": float(q_pred.mean().item()),
            "qvals/q_max": float(q_all.max(dim=1).values.mean().item()),
        }

    def _sync_target_network(self) -> None:
        """Copy the online network weights into the target network."""
        self._target_network.load_state_dict(self._update_network.state_dict())

    def on_episode_start(self, episode: int) -> None:
        """Reset per-episode DQN state."""
        self._obs_buffer.clear()
        self._episode_intrinsic_reward = 0.0

    def on_episode_end(self, episode: int, episode_metrics: dict[str, float]) -> dict[str, float]:
        """Return point-in-time DQN metrics for the completed episode."""
        self.epsilon_scheduler.on_episode_end()
        return {
            "charts/epsilon": float(self.epsilon_scheduler.epsilon(None)),
            "charts/buffer_size": float(len(self.replay_buffer)),
            "rollout/intrinsic_reward": float(self._episode_intrinsic_reward),
        }

    def save_checkpoint(self, path: str) -> None:
        """Persist the online/target networks and optimizer state to ``path``."""
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
        """Restore the online/target networks and optimizer state from ``path``."""
        checkpoint = torch.load(path, map_location=self._device)
        self._update_network.load_state_dict(checkpoint["update_network"])
        self._target_network.load_state_dict(checkpoint["target_network"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        