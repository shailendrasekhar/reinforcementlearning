"""A2C agent — Advantage Actor-Critic (synchronous, single-env)."""

from typing import Any, Dict, List

import numpy as np

from core.agent import BaseAgent
from core.types import Transition

try:
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    torch = None


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic with shared backbone."""

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        if torch is None:
            raise ImportError("PyTorch required for A2C. Install: pip install torch")

        from components.networks import ActorCriticNetwork, get_device

        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.gamma = kwargs.get("discount_factor", 0.99)
        self.value_coeff = kwargs.get("value_coeff", 0.5)
        self.entropy_coeff = kwargs.get("entropy_coeff", 0.01)
        self.grad_clip = kwargs.get("gradient_clip", 0.5)

        device_str = kwargs.get("device", "auto")
        self.device = get_device(device_str)

        hidden = kwargs.get("hidden_dims", [128, 128])
        self.network = ActorCriticNetwork(
            self.state_dim, self.action_dim, hidden
        ).to(self.device)

        lr = kwargs.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self._trajectory: List[Dict] = []

    @property
    def name(self) -> str:
        return "a2c"

    def _to_tensor(self, state):
        return (
            torch.FloatTensor(np.asarray(state, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )

    def select_action(self, state, training: bool = True) -> int:
        state_t = self._to_tensor(state)
        logits, value = self.network(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        if training:
            action = dist.sample()
            self._trajectory.append({
                "log_prob": dist.log_prob(action),
                "value": value.squeeze(),
                "entropy": dist.entropy().squeeze(),
            })
        else:
            action = torch.argmax(logits, dim=-1)
        return action.item()

    def update(self, transition: Transition) -> Dict[str, float]:
        if self._trajectory:
            self._trajectory[-1]["reward"] = transition.reward
            self._trajectory[-1]["done"] = transition.done
        return {}

    def on_episode_start(self) -> None:
        self._trajectory = []

    def on_episode_end(self, episode_reward: float) -> Dict[str, float]:
        if not self._trajectory:
            return {}

        # Compute returns
        returns = []
        G = 0.0
        for t in reversed(self._trajectory):
            G = t["reward"] + self.gamma * G * (1 - float(t["done"]))
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns).to(self.device)
        values = torch.stack([t["value"] for t in self._trajectory])
        log_probs = torch.stack([t["log_prob"] for t in self._trajectory])
        entropies = torch.stack([t["entropy"] for t in self._trajectory])

        advantages = returns_t - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns_t)
        entropy_loss = -entropies.mean()

        loss = (
            policy_loss
            + self.value_coeff * value_loss
            + self.entropy_coeff * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()

        self._trajectory = []
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
        }

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.network.load_state_dict(state_dict["network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
