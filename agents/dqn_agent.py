from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agents.base_agent import BaseAgent, Transition
from agents.curiosity import IntrinsicMotivation, NoMotivation
from agents.dqn.networks import QNetwork
from agents.dqn.normalization import StateNormalizer
from agents.dqn.replay_buffer import ReplayBuffer as ConfigReplayBuffer
from agents.defaults import (
    DQN_DEFAULT_BATCH_SIZE,
    DQN_DEFAULT_CHECKPOINT_PATH,
    DQN_DEFAULT_GAMMA,
    DQN_DEFAULT_GRAD_CLIP_NORM,
    DQN_DEFAULT_LEARNING_RATE,
    DQN_DEFAULT_NO_OBS_IN_STATE,
    DQN_DEFAULT_REWARD_CLIP,
    DQN_DEFAULT_TARGET_UPDATE_FREQ,
    DQN_DEFAULT_UPDATE_FREQ,
    DQN_N_HIDDEN_NODES,
)
from agents.epsilon_schedules import EpsilonSchedule, LinearEpsilonAnnealing
from agents.replay_buffer import Batch, ReplayBuffer
from world import BaseGridEnvironment


@dataclass
class DQNConfig:
    """Configuration container used by standalone DQN entry points."""

    gamma: float = DQN_DEFAULT_GAMMA
    lr: float = DQN_DEFAULT_LEARNING_RATE
    optimizer: str = "adam"
    hidden_sizes: tuple[int, ...] = (DQN_N_HIDDEN_NODES, DQN_N_HIDDEN_NODES)
    buffer_capacity: int = 100_000
    learning_starts: int = 10_000
    batch_size: int = DQN_DEFAULT_BATCH_SIZE
    target_update_interval: int = DQN_DEFAULT_TARGET_UPDATE_FREQ
    train_frequency: int = DQN_DEFAULT_UPDATE_FREQ
    epsilon_start: float = 1.0
    epsilon_final: float = 0.1
    epsilon_decay_steps: int = 100_000
    reward_clip: float | None = DQN_DEFAULT_REWARD_CLIP
    grad_clip_norm: float | None = DQN_DEFAULT_GRAD_CLIP_NORM
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.optimizer not in {"adam", "rmsprop"}:
            raise ValueError("optimizer must be 'adam' or 'rmsprop'")
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        for name in (
            "buffer_capacity",
            "batch_size",
            "target_update_interval",
            "train_frequency",
            "epsilon_decay_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be non-negative")
        if self.epsilon_start < 0 or self.epsilon_final < 0:
            raise ValueError("epsilon values must be non-negative")
        if self.epsilon_final > self.epsilon_start:
            raise ValueError("epsilon_final must be <= epsilon_start")
        if self.reward_clip is not None and self.reward_clip < 0:
            raise ValueError("reward_clip must be non-negative or None")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive or None")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")


class DQNAgent(BaseAgent):
    """Based on Mnih et al. DQN with project-specific preprocessing and replay."""

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
    _episode_intrinsic_reward: float

    def __init__(
        self,
        env: BaseGridEnvironment | None = None,
        seed: int | None = None,
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
        state_dim: int | None = None,
        n_actions: int | None = None,
        normalizer: StateNormalizer | None = None,
        config: DQNConfig | None = None,
    ) -> None:
        if env is None:
            if state_dim is None or n_actions is None or normalizer is None:
                raise ValueError("state_dim, n_actions, and normalizer are required without env")
            self._init_config_agent(
                state_dim=state_dim,
                n_actions=n_actions,
                normalizer=normalizer,
                config=config,
                seed=seed,
            )
            return
        if seed is None:
            raise ValueError("seed is required when env is provided")

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
        torch.manual_seed(seed)
        if self._device.type == "cuda" and hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(seed)
        elif self._device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)
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

    def _init_config_agent(
        self,
        state_dim: int,
        n_actions: int,
        normalizer: StateNormalizer,
        config: DQNConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the standalone config-style DQN used by train_DQN.py."""
        self._config_mode = True
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.normalizer = normalizer
        self.config = config if config is not None else DQNConfig()
        self.seed = seed

        self.device = self._resolve_device(self.config.device)
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            elif self.device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.manual_seed(seed)

        self.online_network = QNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_network = QNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = self._build_optimizer()
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ConfigReplayBuffer(
            capacity=self.config.buffer_capacity,
            state_dim=self.state_dim,
            device=self.device,
            seed=seed,
        )

        self._env_steps = 0
        self._updates = 0
        self._episode_collisions = 0
        self._episode_success = False

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

    def _build_q_network(self) -> nn.Sequential:
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
        if getattr(self, "_config_mode", False):
            return self._select_config_action(state, training=training)

        if len(self._obs_buffer) == 0:
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
        """Store one trainer transition in replay memory."""
        if getattr(self, "_config_mode", False):
            self._observe_config_transition(transition)
            return

        phi_t = self._get_phi()
        self._obs_buffer.append(transition.next_state)
        phi_tp1 = self._get_phi()
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

    def update(self) -> dict[str, float]:
        """Run one DQN gradient step and return training diagnostics."""
        if getattr(self, "_config_mode", False):
            return self._update_config_agent()

        if self._total_steps % self._update_freq != 0:
            return {}
        if not self.replay_buffer.can_sample(self.batch_size):
            return {}
        batch: Batch = self.replay_buffer.sample(self.batch_size)
        actions = batch.actions.unsqueeze(1)

        q_all = self._update_network(batch.states)
        q_pred = q_all.gather(1, actions).squeeze(1)

        with torch.no_grad():
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

    def _select_config_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action for the config-style DQN path."""
        if training and self.rng.random() < self._epsilon():
            return int(self.rng.integers(self.n_actions))

        normalized_state = self.normalizer.normalize(state)
        state_tensor = torch.as_tensor(
            normalized_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def _observe_config_transition(self, transition: Transition) -> None:
        """Store one transition for the config-style DQN path."""
        clipped_reward = float(transition.reward)
        if self.config.reward_clip is not None and self.config.reward_clip > 0:
            clip_value = float(self.config.reward_clip)
            clipped_reward = float(np.clip(clipped_reward, -clip_value, clip_value))

        done = bool(transition.terminated or transition.truncated)
        self.replay_buffer.add(
            state=self.normalizer.normalize(transition.state),
            action=transition.action,
            reward=clipped_reward,
            next_state=self.normalizer.normalize(transition.next_state),
            done=done,
        )
        self._env_steps += 1

        if transition.info.get("collision", False):
            self._episode_collisions += 1
        if transition.info.get("success", False) or transition.terminated:
            self._episode_success = True

    def _update_config_agent(self) -> dict[str, float]:
        """Run one update for the config-style DQN path."""
        metrics = self._base_config_metrics()
        if len(self.replay_buffer) < self.config.learning_starts:
            return metrics
        if len(self.replay_buffer) < self.config.batch_size:
            return metrics
        if self._env_steps % self.config.train_frequency != 0:
            return metrics

        batch = self.replay_buffer.sample(self.config.batch_size)

        q_values = self.online_network(batch.states)
        q_sa = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(batch.next_states).max(dim=1).values
            target = batch.rewards + self.config.gamma * (1.0 - batch.dones) * next_q

        loss = self.loss_fn(q_sa, target)
        td_error = target - q_sa

        self.optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.online_network.parameters(),
                self.config.grad_clip_norm,
            )
            grad_norm_value = float(grad_norm.item())
        else:
            grad_norm_value = self._config_grad_norm()

        self.optimizer.step()
        self._updates += 1

        if self._updates % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        metrics.update(
            {
                "dqn/loss": float(loss.detach().cpu().item()),
                "dqn/td_error_mean": float(td_error.detach().abs().mean().cpu().item()),
                "dqn/td_error_max": float(td_error.detach().abs().max().cpu().item()),
                "dqn/q_mean": float(q_values.detach().mean().cpu().item()),
                "dqn/q_max": float(q_values.detach().max().cpu().item()),
                "dqn/grad_norm": grad_norm_value,
            }
        )
        metrics.update(self._base_config_metrics())
        return metrics

    def _sync_target_network(self) -> None:
        """Copy the online network weights into the target network."""
        self._target_network.load_state_dict(self._update_network.state_dict())

    def on_episode_start(self, episode: int) -> None:
        """Reset per-episode DQN state."""
        if getattr(self, "_config_mode", False):
            self._episode_collisions = 0
            self._episode_success = False
            return

        self._obs_buffer.clear()
        self._episode_intrinsic_reward = 0.0

    def on_episode_end(self, episode: int, episode_metrics: dict[str, float]) -> dict[str, float]:
        """Return point-in-time DQN metrics for the completed episode."""
        if getattr(self, "_config_mode", False):
            return {
                "collisions": float(self._episode_collisions),
                "success": float(self._episode_success),
            }

        self.epsilon_scheduler.on_episode_end()
        return {
            "charts/epsilon": float(self.epsilon_scheduler.epsilon(None)),
            "charts/buffer_size": float(len(self.replay_buffer)),
            "rollout/intrinsic_reward": float(self._episode_intrinsic_reward),
        }

    def save_checkpoint(self, path: str) -> None:
        """Persist the online/target networks and optimizer state to ``path``."""
        if getattr(self, "_config_mode", False):
            self.save(path)
            return

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
        if getattr(self, "_config_mode", False):
            checkpoint = torch.load(Path(path), map_location=self.device)
            self.online_network.load_state_dict(checkpoint["online_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._env_steps = int(checkpoint.get("env_steps", 0))
            self._updates = int(checkpoint.get("updates", 0))
            return

        checkpoint = torch.load(path, map_location=self._device)
        self._update_network.load_state_dict(checkpoint["update_network"])
        self._target_network.load_state_dict(checkpoint["target_network"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])

    def save(self, path: Path | str) -> Path:
        """Persist a config-style checkpoint and return its path."""
        if not getattr(self, "_config_mode", False):
            self.save_checkpoint(str(path))
            return Path(path)

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "config": asdict(self.config),
                "normalizer": self.normalizer.to_dict(),
                "online_state_dict": self.online_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "env_steps": self._env_steps,
                "updates": self._updates,
                "seed": self.seed,
                "resolved_device": str(self.device),
            },
            checkpoint_path,
        )
        return checkpoint_path

    @classmethod
    def load(
        cls,
        path: Path | str,
        normalizer: StateNormalizer | None = None,
        device: str | None = None,
    ) -> DQNAgent:
        """Load a config-style DQN checkpoint."""
        checkpoint = torch.load(Path(path), map_location="cpu")
        config = DQNConfig(**checkpoint["config"])
        if device is not None:
            config = replace(config, device=device)
        loaded_normalizer = normalizer or StateNormalizer.from_dict(checkpoint["normalizer"])
        agent = cls(
            state_dim=int(checkpoint["state_dim"]),
            n_actions=int(checkpoint["n_actions"]),
            normalizer=loaded_normalizer,
            config=config,
            seed=checkpoint.get("seed"),
        )
        agent.online_network.load_state_dict(checkpoint["online_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent._env_steps = int(checkpoint.get("env_steps", 0))
        agent._updates = int(checkpoint.get("updates", 0))
        return agent

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the config-style optimizer."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(self.online_network.parameters(), lr=self.config.lr)
        if self.config.optimizer == "rmsprop":
            return torch.optim.RMSprop(self.online_network.parameters(), lr=self.config.lr)
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _epsilon(self) -> float:
        """Return the current config-style epsilon value."""
        fraction = min(1.0, self._env_steps / float(self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_final - self.config.epsilon_start
        )

    def _base_config_metrics(self) -> dict[str, float]:
        """Return config-style point-in-time DQN metrics."""
        return {
            "dqn/epsilon": float(self._epsilon()),
            "dqn/buffer_size": float(len(self.replay_buffer)),
            "dqn/updates": float(self._updates),
        }

    def _config_grad_norm(self) -> float:
        """Return the global gradient norm for the config-style network."""
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.online_network.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return 0.0
        return float(torch.norm(torch.stack(norms), 2).cpu().item())
