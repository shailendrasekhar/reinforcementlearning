"""A3C agent — Asynchronous Advantage Actor-Critic (single-threaded variant).

This implementation uses the A2C architecture but with an n-step return
bootstrapping approach, making it compatible with the episodic trainer.
For true asynchronous multi-worker training, a separate process-based
launcher would be needed, but this captures the core A3C algorithm logic
in a single-threaded context.
"""

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


class A3CAgent(BaseAgent):
    """A3C-style agent (single-threaded) with n-step returns.

    Uses shared actor-critic network, n-step bootstrapping,
    and entropy regularization.
    """

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        if torch is None:
            raise ImportError("PyTorch required for A3C. Install: pip install torch")

        from components.networks import ActorCriticNetwork, get_device

        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.gamma = kwargs.get("discount_factor", 0.99)
        self.value_coeff = kwargs.get("value_coeff", 0.5)
        self.entropy_coeff = kwargs.get("entropy_coeff", 0.01)
        self.grad_clip = kwargs.get("gradient_clip", 0.5)
        self.n_steps = kwargs.get("n_steps", 5)

        device_str = kwargs.get("device", "auto")
        self.device = get_device(device_str)

        hidden = kwargs.get("hidden_dims", [128, 128])
        self.network = ActorCriticNetwork(
            self.state_dim, self.action_dim, hidden
        ).to(self.device)

        lr = kwargs.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self._trajectory: List[Dict] = []
        self._step_count = 0

    @property
    def name(self) -> str:
        return "a3c"

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
                "state": state,
            })
        else:
            action = torch.argmax(logits, dim=-1)
        return action.item()

    def update(self, transition: Transition) -> Dict[str, float]:
        if self._trajectory:
            self._trajectory[-1]["reward"] = transition.reward
            self._trajectory[-1]["done"] = transition.done
            self._trajectory[-1]["next_state"] = transition.next_state

        self._step_count += 1

        # Perform n-step update when we have enough steps or episode ends
        if len(self._trajectory) >= self.n_steps or transition.done:
            return self._n_step_update(transition.done, transition.next_state)

        return {}

    def _n_step_update(self, done: bool, last_state) -> Dict[str, float]:
        """Perform n-step return update."""
        if not self._trajectory:
            return {}

        # Bootstrap value for the last state
        if done:
            R = 0.0
        else:
            with torch.no_grad():
                _, last_val = self.network(self._to_tensor(last_state))
            R = last_val.item()

        # Compute n-step returns
        returns = []
        for t in reversed(self._trajectory):
            R = t["reward"] + self.gamma * R * (1 - float(t["done"]))
            returns.insert(0, R)

        returns_t = torch.FloatTensor(returns).to(self.device)
        values = torch.stack([t["value"] for t in self._trajectory])
        log_probs = torch.stack([t["log_prob"] for t in self._trajectory])
        entropies = torch.stack([t["entropy"] for t in self._trajectory])

        advantages = returns_t - values.detach()
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

    def on_episode_start(self) -> None:
        self._trajectory = []

    def on_episode_end(self, episode_reward: float) -> Dict[str, float]:
        # Flush any remaining trajectory data
        if self._trajectory:
            return self._n_step_update(done=True, last_state=None)
        return {}

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.network.load_state_dict(state_dict["network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
