"""PPO agent — Proximal Policy Optimization with clipped surrogate."""

from typing import Any, Dict

import numpy as np

from core.agent import BaseAgent
from core.types import Transition
from components.rollout_buffer import RolloutBuffer

try:
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    torch = None


class PPOAgent(BaseAgent):
    """PPO with clipped surrogate objective.

    Collects rollout_length steps into a buffer (across episode boundaries),
    then performs ppo_epochs of minibatch SGD when the buffer is full.
    Works with the standard EpisodicTrainer — training triggers automatically.
    """

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        if torch is None:
            raise ImportError("PyTorch required for PPO. Install: pip install torch")

        from components.networks import ActorCriticNetwork, get_device

        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.gamma = kwargs.get("discount_factor", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_epsilon = kwargs.get("clip_epsilon", 0.2)
        self.epochs = kwargs.get("ppo_epochs", 10)
        self.batch_size = kwargs.get("batch_size", 64)
        self.value_coeff = kwargs.get("value_coeff", 0.5)
        self.entropy_coeff = kwargs.get("entropy_coeff", 0.01)
        self.grad_clip = kwargs.get("gradient_clip", 0.5)
        self.rollout_length = kwargs.get("rollout_length", 2048)

        device_str = kwargs.get("device", "auto")
        self.device = get_device(device_str)

        hidden = kwargs.get("hidden_dims", [128, 128])
        self.network = ActorCriticNetwork(
            self.state_dim, self.action_dim, hidden
        ).to(self.device)

        lr = kwargs.get("learning_rate", 3e-4)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.buffer = RolloutBuffer(
            self.rollout_length,
            self.state_dim,
            self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            discrete=True,
        )

        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_loss = 0.0

    @property
    def name(self) -> str:
        return "ppo"

    def _to_tensor(self, state):
        return (
            torch.FloatTensor(np.asarray(state, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )

    def select_action(self, state, training: bool = True) -> int:
        state_t = self._to_tensor(state)
        with torch.no_grad():
            logits, value = self.network(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        if training:
            action = dist.sample()
            self._last_log_prob = dist.log_prob(action).item()
            self._last_value = value.item()
        else:
            action = torch.argmax(logits, dim=-1)
        return action.item()

    def update(self, transition: Transition) -> Dict[str, float]:
        self.buffer.store(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            done=transition.done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )

        if self.buffer.is_full:
            return self._train_on_rollout(transition.next_state)
        return {}

    def _train_on_rollout(self, last_state) -> Dict[str, float]:
        """Run PPO update: compute advantages, then K epochs of clipped SGD."""
        # Bootstrap value for the last state
        with torch.no_grad():
            _, last_val = self.network(self._to_tensor(last_state))
        self.buffer.compute_advantages(last_val.item())

        total_loss = 0.0
        total_pg = 0.0
        total_vl = 0.0
        total_ent = 0.0
        num_updates = 0

        for _epoch in range(self.epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                (
                    states,
                    actions,
                    old_log_probs,
                    returns,
                    advantages,
                    _old_values,
                ) = batch

                states_t = torch.FloatTensor(states).to(self.device)
                actions_t = torch.LongTensor(actions).to(self.device)
                old_lp_t = torch.FloatTensor(old_log_probs).to(self.device)
                returns_t = torch.FloatTensor(returns).to(self.device)
                advantages_t = torch.FloatTensor(advantages).to(self.device)

                # Normalize advantages
                if advantages_t.std() > 1e-8:
                    advantages_t = (advantages_t - advantages_t.mean()) / (
                        advantages_t.std() + 1e-8
                    )

                logits, values = self.network(states_t)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()

                # Clipped surrogate
                ratio = (new_log_probs - old_lp_t).exp()
                clip_ratio = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )
                policy_loss = -torch.min(
                    ratio * advantages_t, clip_ratio * advantages_t
                ).mean()
                value_loss = F.mse_loss(values, returns_t)

                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.grad_clip
                    )
                self.optimizer.step()

                total_loss += loss.item()
                total_pg += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                num_updates += 1

        self.buffer.reset()
        n = max(num_updates, 1)
        self._last_loss = total_loss / n
        return {
            "loss": total_loss / n,
            "policy_loss": total_pg / n,
            "value_loss": total_vl / n,
            "entropy": total_ent / n,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.network.load_state_dict(state_dict["network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
