from __future__ import annotations

from pathlib import Path
import collections
import torch
from torch import nn

import numpy as np

from agents.base_agent import BaseAgent, Transition
from agents.curiosity import IntrinsicMotivation, NoMotivation
from agents.replay_buffer import ReplayBuffer, Batch
from agents.epsilon_schedules import EpsilonSchedule, LinearEpsilonAnnealing
from agents.defaults import (
    DQN_N_HIDDEN_NODES,
    DQN_DEFAULT_BATCH_SIZE,
    DQN_DEFAULT_LEARNING_RATE,
    DQN_DEFAULT_GAMMA,
    DQN_DEFAULT_NO_OBS_IN_STATE,
    DQN_DEFAULT_UPDATE_FREQ,
    DQN_DEFAULT_TARGET_UPDATE_FREQ,
    DQN_DEFAULT_CHECKPOINT_PATH,
    DQN_DEFAULT_REWARD_CLIP,
    DQN_DEFAULT_GRAD_CLIP_NORM,
)
from world import BaseGridEnvironment

class DQNAgent(BaseAgent):
    """Based on Mnih et al (https://www.nature.com/articles/nature14236) method, but different architecture."""
    _rng: np.random.Generator
    n_actions: int
    state_dim: int
    _single_obs_dim: int
    replay_buffer: ReplayBuffer
    batch_size: int
    _learning_rate: float
    _update_network: nn.Sequential
    _target_network: nn.Sequential
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
    _reward_clip: float | None
    _grad_clip_norm: float | None
    intrinsic_motivation: IntrinsicMotivation
    # Per-episode intrinsic-bonus accounting (extrinsic return is tracked by the Trainer).
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
        reward_clip: float | None = DQN_DEFAULT_REWARD_CLIP,
        grad_clip_norm: float | None = DQN_DEFAULT_GRAD_CLIP_NORM,
        device: str = "cpu",
    ):
        self.env = env
        self.n_actions = env.n_actions
        self._single_obs_dim = env.state_dim
        self._no_obs_in_state = no_obs_in_state
        self.state_dim = self._single_obs_dim * self._no_obs_in_state
        self.gamma = gamma
        self._learning_rate = learning_rate
        self._reward_clip = reward_clip
        self._grad_clip_norm = grad_clip_norm
        self._device = self._resolve_device(device)
        # Seed torch so network init (which happens before the Trainer seeds
        # torch) is reproducible across runs.
        torch.manual_seed(seed)
        if self._device.type == "cuda" and hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(seed)
        elif self._device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)
        obs_high = np.asarray(env.observation_high, dtype=np.float32)
        obs_high = np.where(obs_high == 0.0, 1.0, obs_high)
        # Tile the observation scale to match the stacked state dimension
        single_obs_scale = (1.0 / obs_high).astype(np.float32)
        self._obs_scale = np.tile(single_obs_scale, self._no_obs_in_state)
        # Angular (periodic) dims, wrapped per stacked frame before scaling.
        single_angular = np.asarray(env.angular_dims, dtype=np.int64)
        self._angular_indices = np.concatenate(
            [single_angular + frame * self._single_obs_dim for frame in range(self._no_obs_in_state)]
        ).astype(np.int64) if single_angular.size else single_angular
        self._angular_periods = obs_high[single_angular] if single_angular.size else single_angular.astype(np.float32)
        self._angular_periods = np.tile(self._angular_periods, self._no_obs_in_state)
        self._update_network = self._build_q_network().to(self._device)
        self._target_network = self._build_q_network().to(self._device)
        self._target_network.load_state_dict(self._update_network.state_dict())
        # Target net is only used for inference; eval() guards future train/eval
        # sensitive layers (BatchNorm/Dropout).
        self._target_network.eval()
        self._optimizer = torch.optim.Adam(self._update_network.parameters(), learning_rate)
        self._loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(obs_dim=self.state_dim, capacity=replay_buffer_capacity, seed=seed, device=self._device)
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._checkpoint_path = checkpoint_path
        self._update_freq = update_freq
        self._target_update_freq = target_update_freq
        self._learn_steps = 0
        self._total_steps = 0
        self._obs_buffer = collections.deque(maxlen=self._no_obs_in_state)
        self.epsilon_scheduler = epsilon_scheduler if epsilon_scheduler is not None else LinearEpsilonAnnealing()
        self.intrinsic_motivation = intrinsic_motivation if intrinsic_motivation is not None else NoMotivation()
        self._episode_intrinsic_reward = 0.0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve a device string, supporting cuda, mps, and cpu.

        ``auto`` picks cuda > mps > cpu; an explicit ``cuda``/``mps`` request
        raises if that backend is unavailable.
        """
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

    def _build_q_network(self) -> nn.Sequential:
        # Plain MLP with PyTorch default (Kaiming) init, matching the converged
        # baseline. A small-gain output init was tried but suppressed initial Q
        # magnitudes and stalled learning on the sparse-goal grids.
        return nn.Sequential(
            nn.Linear(self.state_dim, DQN_N_HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(DQN_N_HIDDEN_NODES, DQN_N_HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(DQN_N_HIDDEN_NODES, self.n_actions),
        )

    def _preprocess(self, state: np.ndarray) -> np.ndarray:
        """Map a raw (optionally batched) state to the normalized observation phi.

        Wraps angular dims by their period, scales every dim to roughly [0, 1],
        and clips to [0, 1]. Broadcasts over the trailing observation dimension,
        so it handles both a single state ``(state_dim,)`` and a batch
        ``(B, state_dim)``.
        """
        phi = np.asarray(state, dtype=np.float32)
        if self._angular_indices.size:
            phi = phi.copy()
            phi[..., self._angular_indices] = np.mod(
                phi[..., self._angular_indices], self._angular_periods
            )
        return np.clip(phi * self._obs_scale, 0.0, 1.0)

    def _get_phi(self) -> np.ndarray:
        """Get the current stacked state (phi) from the observation buffer."""
        return np.concatenate(list(self._obs_buffer), axis=-1)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Pick an action greedily from the online net, exploring while training."""
        if len(self._obs_buffer) == 0:
            for _ in range(self._no_obs_in_state):
                self._obs_buffer.append(state)
        elif not training:
            # During evaluation, observe() is not called, so we update the buffer here
            self._obs_buffer.append(state)

        phi_state = self._get_phi()
        if training and self._rng.random() < self.epsilon_scheduler.epsilon(phi_state):
            return int(self._rng.integers(self.n_actions))

        with torch.no_grad():
            phi = torch.as_tensor(self._preprocess(phi_state), dtype=torch.float32, device=self._device)
            q_values = self._update_network(phi)
        return int(q_values.argmax().item())

    def observe(self, transition: Transition) -> None:
        phi_t = self._get_phi()
        self._obs_buffer.append(transition.next_state)
        phi_tp1 = self._get_phi()
        # Clip the extrinsic reward (baseline-style stability) before adding the
        # intrinsic bonus, so curiosity stays unbounded.
        extrinsic = transition.reward
        if self._reward_clip is not None and self._reward_clip > 0:
            extrinsic = float(np.clip(extrinsic, -self._reward_clip, self._reward_clip))
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
        return None

    def update(self) -> dict[str, float]:
        """Run one DQN gradient step and return training diagnostics.

        Returns an empty dict until the buffer holds a full batch; otherwise a
        dict of per-step scalars (loss, Q estimates, TD error, gradient norm)
        under W&B group prefixes that the Trainer averages per episode. Only
        quantities whose per-episode mean is meaningful live here; point-in-time
        values (epsilon, buffer fill) are reported from ``on_episode_end``.
        """
        if self._total_steps % self._update_freq != 0:
            return {}
        if not self.replay_buffer.can_sample(self.batch_size):
            return {}
        batch: Batch = self.replay_buffer.sample(self.batch_size)
        actions = batch.actions.unsqueeze(1)

        # Q(φ_j, ·; θ) for all actions, then the Q of the action actually taken.
        q_all = self._update_network(batch.states)
        q_pred = q_all.gather(1, actions).squeeze(1)

        # y_j: no gradient flows through the target.
        with torch.no_grad():
            q_next = self._target_network(batch.next_states).max(dim=1).values
            targets = batch.rewards + self.gamma * q_next * (1.0 - batch.dones)

        loss = self._loss_fn(q_pred, targets)

        self._optimizer.zero_grad()
        loss.backward()
        # When grad_clip_norm is None, max_norm=inf measures the gradient norm
        # without clipping; otherwise it clips to the configured value.
        max_norm = self._grad_clip_norm if self._grad_clip_norm is not None else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._update_network.parameters(), max_norm=max_norm
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
        self._obs_buffer.clear()
        self._episode_intrinsic_reward = 0.0
        return None

    def on_episode_end(self, episode: int, episode_metrics: dict[str, float]) -> dict[str, float]:
        # Advance the epsilon schedule once per episode (if it uses per-episode logic).
        self.epsilon_scheduler.on_episode_end()
        # Point-in-time values reported once per episode
        return {
            "charts/epsilon": float(self.epsilon_scheduler.epsilon(None)),
            "charts/buffer_size": float(len(self.replay_buffer)),
            "rollout/intrinsic_reward": float(self._episode_intrinsic_reward),
        }

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