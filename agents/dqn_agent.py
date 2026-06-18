from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from agents.base_agent import BaseAgent, Transition
from agents.dqn.networks import QNetwork
from agents.dqn.normalization import StateNormalizer
from agents.dqn.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    optimizer: str = "adam"
    hidden_sizes: tuple[int, ...] = (128, 128)
    buffer_capacity: int = 100_000
    learning_starts: int = 10_000
    batch_size: int = 64
    target_update_interval: int = 1_000
    train_frequency: int = 1
    epsilon_start: float = 1.0
    epsilon_final: float = 0.1
    epsilon_decay_steps: int = 100_000
    reward_clip: float | None = 1.0
    grad_clip_norm: float | None = 10.0
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
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


class DQNAgent(BaseAgent):
    """DQN baseline agent for the continuous-state, discrete-action environment."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        normalizer: StateNormalizer,
        config: DQNConfig | None = None,
        seed: int | None = None,
    ) -> None:
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
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            state_dim=self.state_dim,
            device=self.device,
            seed=seed,
        )

        self._env_steps = 0
        self._updates = 0
        self._episode_collisions = 0
        self._episode_success = False

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
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

    def observe(self, transition: Transition) -> None:
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

    def update(self) -> dict[str, float]:
        metrics = self._base_metrics()
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
            grad_norm_value = self._grad_norm()

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
        metrics.update(self._base_metrics())
        return metrics

    def on_episode_start(self, episode: int) -> None:
        self._episode_collisions = 0
        self._episode_success = False

    def on_episode_end(self, episode: int, episode_metrics: dict[str, float]) -> dict[str, float]:
        return {
            "collisions": float(self._episode_collisions),
            "success": float(self._episode_success),
        }

    def save(self, path: Path | str) -> Path:
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
    ) -> "DQNAgent":
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
        if self.config.optimizer == "adam":
            return torch.optim.Adam(self.online_network.parameters(), lr=self.config.lr)
        if self.config.optimizer == "rmsprop":
            return torch.optim.RMSprop(self.online_network.parameters(), lr=self.config.lr)
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _epsilon(self) -> float:
        fraction = min(1.0, self._env_steps / float(self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_final - self.config.epsilon_start
        )

    def _base_metrics(self) -> dict[str, float]:
        return {
            "dqn/epsilon": float(self._epsilon()),
            "dqn/buffer_size": float(len(self.replay_buffer)),
            "dqn/updates": float(self._updates),
        }

    def _grad_norm(self) -> float:
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.online_network.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return 0.0
        return float(torch.norm(torch.stack(norms), 2).cpu().item())

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device(device)
