"""REINFORCE agent — Monte Carlo policy gradient."""

from typing import Any, Dict, List

import numpy as np

from core.agent import BaseAgent
from core.types import Transition

try:
    import torch
    import torch.optim as optim
except ImportError:
    torch = None


class ReinforceAgent(BaseAgent):
    """REINFORCE with optional baseline (running mean of returns)."""

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        if torch is None:
            raise ImportError("PyTorch required for REINFORCE. Install: pip install torch")

        from components.networks import PolicyNetwork, get_device

        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.gamma = kwargs.get("discount_factor", 0.99)
        self.use_baseline = kwargs.get("use_baseline", True)

        device_str = kwargs.get("device", "auto")
        self.device = get_device(device_str)

        hidden = kwargs.get("hidden_dims", [128, 128])
        self.policy = PolicyNetwork(self.state_dim, self.action_dim, hidden).to(
            self.device
        )

        lr = kwargs.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self._trajectory: List[Dict] = []
        self._baseline_returns: List[float] = []

    @property
    def name(self) -> str:
        return "reinforce"

    def _to_tensor(self, state):
        return (
            torch.FloatTensor(np.asarray(state, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )

    def select_action(self, state, training: bool = True) -> int:
        probs = self.policy(self._to_tensor(state))
        dist = torch.distributions.Categorical(probs)
        if training:
            action = dist.sample()
            self._trajectory.append({"log_prob": dist.log_prob(action)})
        else:
            action = torch.argmax(probs, dim=-1)
        return action.item()

    def update(self, transition: Transition) -> Dict[str, float]:
        # Just record reward; actual training happens in on_episode_end
        if self._trajectory:
            self._trajectory[-1]["reward"] = transition.reward
        return {}

    def on_episode_start(self) -> None:
        self._trajectory = []

    def on_episode_end(self, episode_reward: float) -> Dict[str, float]:
        if not self._trajectory:
            return {}

        rewards = [t["reward"] for t in self._trajectory]
        log_probs = [t["log_prob"] for t in self._trajectory]

        # Discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Baseline subtraction
        if self.use_baseline and len(self._baseline_returns) > 0:
            baseline = np.mean(self._baseline_returns[-100:])
            returns_t = returns_t - baseline
        self._baseline_returns.append(sum(rewards))

        # Normalize
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy gradient loss
        loss = torch.stack(
            [-lp * G for lp, G in zip(log_probs, returns_t)]
        ).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._trajectory = []
        return {"loss": loss.item()}

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
