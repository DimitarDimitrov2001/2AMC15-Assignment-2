"""
Dueling Double DQN agent for the env-coupled training stack.

Built on top of DQNAgent: inherits replay, epsilon scheduling, reward clipping,
and env-driven state normalization in observe(). Swaps the MLP for a dueling
architecture and uses Double DQN targets in update().
"""

from __future__ import annotations
import torch
from torch import nn

from agents.defaults import DQN_N_HIDDEN_NODES
from agents.dqn_agent import DQNAgent
from agents.dueling_dqn.dueling_network import DuelingQNetwork
from agents.replay_buffer import Batch


class DuelingDQNAgent(DQNAgent):
    """Dueling architecture with Double DQN bootstrapping."""

    def _build_q_network(self) -> nn.Module:
        """Return a dueling Q-network with the same depth as the vanilla DQN MLP."""
        return DuelingQNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_sizes=(DQN_N_HIDDEN_NODES, DQN_N_HIDDEN_NODES),
        )

    def update(self) -> dict[str, float]:
        """Run one Double DQN gradient step and return training diagnostics."""
        if self._total_steps % self._update_freq != 0:
            return {}
        if not self.replay_buffer.can_sample(self.batch_size):
            return {}
        batch: Batch = self.replay_buffer.sample(self.batch_size)
        actions = batch.actions.unsqueeze(1)

        q_all = self._update_network(batch.states)
        q_pred = q_all.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self._update_network(batch.next_states).argmax(dim=1)
            q_next = self._target_network(batch.next_states).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
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
