"""DQN agent — deep, off-policy, value-based with target network."""

from typing import Any, Dict, Optional
import copy

import numpy as np

from core.agent import BaseAgent
from core.types import Transition
from components.replay_buffer import UniformReplayBuffer
from components.exploration import EpsilonGreedy

try:
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    torch = None


class DQNAgent(BaseAgent):
    """Deep Q-Network with experience replay and target network."""

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        if torch is None:
            raise ImportError("PyTorch required for DQN. Install: pip install torch")

        from components.networks import QNetwork, get_device

        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.gamma = kwargs.get("discount_factor", 0.99)
        self.batch_size = kwargs.get("batch_size", 64)
        self.warmup_steps = kwargs.get("warmup_steps", 1000)
        self.train_frequency = kwargs.get("train_frequency", 4)
        self.target_update_freq = kwargs.get("target_update_frequency", 1000)
        self.grad_clip = kwargs.get("gradient_clip", None)

        device_str = kwargs.get("device", "auto")
        self.device = get_device(device_str)

        hidden = kwargs.get("hidden_dims", [128, 128])
        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        lr = kwargs.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        buffer_size = kwargs.get("buffer_size", 100_000)
        self.buffer = UniformReplayBuffer(buffer_size)

        exp_cfg = kwargs.get("exploration", {})
        params = exp_cfg.get("params", {})
        self.exploration = EpsilonGreedy(**params)

        self.step_count = 0
        self._last_loss = 0.0

        # Detect discrete observation space (e.g. FrozenLake) → one-hot encode
        import gymnasium as gym
        obs_space = env_info.get("observation_space")
        self._discrete_obs = isinstance(obs_space, gym.spaces.Discrete)
        self._obs_n = int(obs_space.n) if self._discrete_obs else 0

    @property
    def name(self) -> str:
        return "dqn"

    @property
    def epsilon(self) -> Optional[float]:
        return self.exploration.current_value

    def _encode_state(self, state) -> np.ndarray:
        """Convert state to float vector. One-hot encodes discrete obs."""
        if self._discrete_obs:
            vec = np.zeros(self._obs_n, dtype=np.float32)
            vec[int(state)] = 1.0
            return vec
        return np.asarray(state, dtype=np.float32)

    def _to_tensor(self, state):
        return (
            torch.FloatTensor(self._encode_state(state))
            .unsqueeze(0)
            .to(self.device)
        )

    def select_action(self, state, training: bool = True) -> int:
        if training and self.step_count < self.warmup_steps:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            q_values = self.q_net(self._to_tensor(state)).cpu().numpy().flatten()

        if training:
            return self.exploration.select(q_values)
        return int(np.argmax(q_values))

    def update(self, transition: Transition) -> Dict[str, float]:
        self.step_count += 1
        self.buffer.push(transition)
        self.exploration.decay()

        if self.step_count < self.warmup_steps or len(self.buffer) < self.batch_size:
            return {}

        if self.step_count % self.train_frequency != 0:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample_tensors(
            self.batch_size
        )

        # One-hot encode if discrete observation space
        if self._discrete_obs:
            states = np.array([self._encode_state(s) for s in states])
            next_states = np.array([self._encode_state(s) for s in next_states])

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Hard target update
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self._last_loss = loss.item()
        return {"loss": loss.item()}

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.q_net.load_state_dict(state_dict["q_net"])
        self.target_net.load_state_dict(state_dict["target_net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.step_count = state_dict["step_count"]
