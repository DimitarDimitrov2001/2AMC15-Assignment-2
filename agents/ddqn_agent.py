"""
Dueling DQN agent: a standalone subclass of DQNAgent.

Does NOT modify dqn_agent.py, networks.py, replay_buffer.py, or
normalization.py in any way. DQNAgent.__init__ hardcodes QNetwork for
both the online and target networks, so this subclass lets the parent
build itself fully (replay buffer, optimizer, epsilon schedule, etc.),
then swaps both networks for DuelingQNetwork and rebuilds the optimizer
to point at the new parameters.

Dueling DQN (Wang et al., 2016) splits the Q-network into two streams
after a shared trunk:

    V(s)      -- a scalar estimate of how good the state itself is
    A(s, a)   -- a per-action estimate of how much better/worse action a
                 is relative to the other actions available in that state

combined as:

    Q(s, a) = V(s) + ( A(s, a) - mean_a' A(s, a') )

Why this can help: in plenty of states the choice of action barely
matters (e.g. open space, several headings are roughly equally fine), so
there's little point learning a separate accurate value for every
action in that state. The value stream learns V(s) from every visited
state regardless of which action was taken there, while the advantage
stream only has to capture *relative* differences between actions.

Everything else -- replay buffer, epsilon-greedy, reward clipping,
gradient clipping, target network sync interval, save/load -- is
inherited unchanged from DQNAgent.
"""

from __future__ import annotations

from agents.dqn_agent import DQNAgent, DQNConfig
from agents.dqn.normalization import StateNormalizer
from agents.dueling_dqn.dueling_network import DuelingQNetwork
import torch


class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN agent built entirely by post-processing a regular DQNAgent.

    No changes to DQNAgent itself. After the parent __init__ finishes
    (which builds a vanilla QNetwork online/target pair, an optimizer
    bound to those parameters, and the replay buffer), this subclass
    swaps in DuelingQNetwork instances and rebuilds the optimizer so it
    tracks the new parameters instead of the discarded ones.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        normalizer: StateNormalizer,
        config: DQNConfig | None = None,
        seed: int | None = None,
    ) -> None:
        # Let DQNAgent build everything as normal -- vanilla QNetwork
        # online/target pair, optimizer, replay buffer, RNG, device, etc.
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            normalizer=normalizer,
            config=config,
            seed=seed,
        )

        # Replace both networks with the dueling architecture.
        self.online_network = DuelingQNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_network = DuelingQNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        # The optimizer built by DQNAgent.__init__ is bound to the
        # parameters of the (now-discarded) vanilla QNetwork. Rebuild it
        # so it tracks the dueling network's parameters instead.
        self.optimizer = self._build_optimizer()

    def update(self) -> dict[str, float]:
        """
        Identical to DQNAgent.update(), except the bootstrapped target uses
        Double DQN action selection: the ONLINE network picks the greedy
        next action, the TARGET network evaluates it. This decouples
        action-selection from value-estimation and reduces the
        overestimation bias that DQN's plain max-target is prone to --
        a bias that compounds badly with the dueling architecture, since
        a biased V(s) estimate gets shared across every action uniformly
        rather than diluted across independently-estimated Q-values.
        (Wang et al., 2016 pair Dueling DQN with Double DQN for this reason.)
        """
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
            # Double DQN target: select with online, evaluate with target.
            next_actions = self.online_network(batch.next_states).argmax(dim=1)
            next_q = self.target_network(batch.next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
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