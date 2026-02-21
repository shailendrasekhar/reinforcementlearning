"""TD3 agent — Twin Delayed Deep Deterministic Policy Gradient."""

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


class TD3Agent(BaseAgent):
    """TD3: Twin Delayed DDPG for continuous action spaces.

    Key improvements over DDPG:
    - Twin Q-networks to reduce overestimation bias
    - Delayed policy updates (every d steps)
    - Target policy smoothing (add noise to target actions)
    """

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        if torch is None:
            raise ImportError("PyTorch required for TD3. Install: pip install torch")

        from components.networks import DeterministicActor, QCritic, get_device

        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.max_action = float(kwargs.get("max_action", 1.0))
        self.gamma = kwargs.get("discount_factor", 0.99)
        self.tau = kwargs.get("tau", 0.005)
        self.batch_size = kwargs.get("batch_size", 256)
        self.warmup_steps = kwargs.get("warmup_steps", 1000)
        self.grad_clip = kwargs.get("gradient_clip", None)

        # TD3-specific
        self.policy_delay = kwargs.get("policy_delay", 2)
        self.target_noise_std = kwargs.get("target_noise_std", 0.2)
        self.target_noise_clip = kwargs.get("target_noise_clip", 0.5)

        device_str = kwargs.get("device", "auto")
        self.device = get_device(device_str)

        hidden = kwargs.get("hidden_dims", [256, 256])

        # Actor
        self.actor = DeterministicActor(
            self.state_dim, self.action_dim, hidden, self.max_action
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)

        # Twin critics
        self.critic1 = QCritic(self.state_dim, self.action_dim, hidden).to(self.device)
        self.critic2 = QCritic(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        actor_lr = kwargs.get("actor_lr", kwargs.get("learning_rate", 1e-3))
        critic_lr = kwargs.get("critic_lr", kwargs.get("learning_rate", 1e-3))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        buffer_size = kwargs.get("buffer_size", 100_000)
        self.buffer = UniformReplayBuffer(buffer_size)

        # Exploration noise
        noise_cfg = kwargs.get("noise", {})
        noise_type = noise_cfg.get("type", "gaussian")
        noise_params = {k: v for k, v in noise_cfg.items() if k != "type"}
        if noise_type == "ou":
            self.noise = OrnsteinUhlenbeckNoise(self.action_dim, **noise_params)
        else:
            self.noise = GaussianNoise(self.action_dim, **noise_params)

        self.step_count = 0
        self._update_count = 0

    @property
    def name(self) -> str:
        return "td3"

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

        # --- Target policy smoothing ---
        with torch.no_grad():
            noise = (
                torch.randn_like(actions_t) * self.target_noise_std
            ).clamp(-self.target_noise_clip, self.target_noise_clip)
            target_actions = (
                self.target_actor(next_states_t) + noise
            ).clamp(-self.max_action, self.max_action)

            # Twin Q targets
            target_q1 = self.target_critic1(next_states_t, target_actions)
            target_q2 = self.target_critic2(next_states_t, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards_t + self.gamma * target_q * (1 - dones_t)

        # --- Update critics ---
        current_q1 = self.critic1(states_t, actions_t)
        current_q2 = self.critic2(states_t, actions_t)
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip)
        self.critic2_optimizer.step()

        self._update_count += 1
        metrics = {
            "loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
        }

        # --- Delayed policy update ---
        if self._update_count % self.policy_delay == 0:
            actor_loss = -self.critic1(states_t, self.actor(states_t)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Soft update targets
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def on_episode_start(self) -> None:
        if hasattr(self.noise, 'reset'):
            self.noise.reset()

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic1_opt": self.critic1_optimizer.state_dict(),
            "critic2_opt": self.critic2_optimizer.state_dict(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.target_critic1.load_state_dict(state_dict["target_critic1"])
        self.target_critic2.load_state_dict(state_dict["target_critic2"])
        self.actor_optimizer.load_state_dict(state_dict["actor_opt"])
        self.critic1_optimizer.load_state_dict(state_dict["critic1_opt"])
        self.critic2_optimizer.load_state_dict(state_dict["critic2_opt"])
        self.step_count = state_dict["step_count"]
