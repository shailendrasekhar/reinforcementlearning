"""DDPG agent — Deep Deterministic Policy Gradient for continuous control."""

from typing import Any, Dict
import copy

import numpy as np

from core.agent import BaseAgent
from core.types import Transition
from components.replay_buffer import UniformReplayBuffer
from components.noise import OrnsteinUhlenbeckNoise, GaussianNoise

try:
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    torch = None


class DDPGAgent(BaseAgent):
    """DDPG for continuous action spaces."""

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        if torch is None:
            raise ImportError("PyTorch required for DDPG. Install: pip install torch")

        from components.networks import DeterministicActor, QCritic, get_device

        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.max_action = float(kwargs.get("max_action", 1.0))
        self.gamma = kwargs.get("discount_factor", 0.99)
        self.tau = kwargs.get("tau", 0.005)
        self.batch_size = kwargs.get("batch_size", 256)
        self.warmup_steps = kwargs.get("warmup_steps", 1000)
        self.grad_clip = kwargs.get("gradient_clip", None)

        device_str = kwargs.get("device", "auto")
        self.device = get_device(device_str)

        hidden = kwargs.get("hidden_dims", [256, 256])
        self.actor = DeterministicActor(
            self.state_dim, self.action_dim, hidden, self.max_action
        ).to(self.device)
        self.critic = QCritic(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        actor_lr = kwargs.get("actor_lr", kwargs.get("learning_rate", 1e-3))
        critic_lr = kwargs.get("critic_lr", kwargs.get("learning_rate", 1e-3))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        buffer_size = kwargs.get("buffer_size", 100_000)
        self.buffer = UniformReplayBuffer(buffer_size)

        # Exploration noise
        noise_cfg = kwargs.get("noise", {})
        noise_type = noise_cfg.get("type", "ou")
        noise_params = {k: v for k, v in noise_cfg.items() if k != "type"}
        if noise_type == "gaussian":
            self.noise = GaussianNoise(self.action_dim, **noise_params)
        else:
            self.noise = OrnsteinUhlenbeckNoise(self.action_dim, **noise_params)

        self.step_count = 0

    @property
    def name(self) -> str:
        return "ddpg"

    def _to_tensor(self, x):
        return torch.FloatTensor(np.asarray(x, dtype=np.float32)).to(self.device)

    def select_action(self, state, training: bool = True):
        if training and self.step_count < self.warmup_steps:
            return np.random.uniform(
                -self.max_action, self.max_action, self.action_dim
            )

        with torch.no_grad():
            action = (
                self.actor(self._to_tensor(state).unsqueeze(0)).cpu().numpy().flatten()
            )

        if training:
            action = action + self.noise.sample()
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def update(self, transition: Transition) -> Dict[str, float]:
        self.step_count += 1
        self.buffer.push(transition)

        if self.step_count < self.warmup_steps or len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample_tensors(
            self.batch_size
        )

        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions)
        rewards_t = self._to_tensor(rewards)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones)

        # --- Update critic ---
        with torch.no_grad():
            target_actions = self.target_actor(next_states_t)
            target_q = self.target_critic(next_states_t, target_actions)
            target_value = rewards_t + self.gamma * target_q * (1 - dones_t)

        current_q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # --- Update actor ---
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # --- Soft update targets ---
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

        return {
            "loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def on_episode_start(self) -> None:
        self.noise.reset()

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_opt"])
        self.critic_optimizer.load_state_dict(state_dict["critic_opt"])
        self.step_count = state_dict["step_count"]
