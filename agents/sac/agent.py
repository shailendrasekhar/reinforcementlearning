"""SAC agent — Soft Actor-Critic with automatic entropy tuning."""

from typing import Any, Dict
import copy

import numpy as np

from core.agent import BaseAgent
from core.types import Transition
from components.replay_buffer import UniformReplayBuffer

try:
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    torch = None


class SACAgent(BaseAgent):
    """Soft Actor-Critic with twin Q-networks and entropy auto-tuning."""

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        if torch is None:
            raise ImportError("PyTorch required for SAC. Install: pip install torch")

        from components.networks import GaussianActor, QCritic, get_device

        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.gamma = kwargs.get("discount_factor", 0.99)
        self.tau = kwargs.get("tau", 0.005)
        self.batch_size = kwargs.get("batch_size", 256)
        self.warmup_steps = kwargs.get("warmup_steps", 1000)
        self.auto_entropy = kwargs.get("auto_entropy", True)

        device_str = kwargs.get("device", "auto")
        self.device = get_device(device_str)

        hidden = kwargs.get("hidden_dims", [256, 256])
        self.actor = GaussianActor(self.state_dim, self.action_dim, hidden).to(
            self.device
        )
        self.critic1 = QCritic(self.state_dim, self.action_dim, hidden).to(self.device)
        self.critic2 = QCritic(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        actor_lr = kwargs.get("actor_lr", kwargs.get("learning_rate", 3e-4))
        critic_lr = kwargs.get("critic_lr", kwargs.get("learning_rate", 3e-4))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Entropy tuning
        init_alpha = kwargs.get("init_alpha", 0.2)
        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.tensor(
            np.log(init_alpha),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        alpha_lr = kwargs.get("alpha_lr", 3e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        buffer_size = kwargs.get("buffer_size", 100_000)
        self.buffer = UniformReplayBuffer(buffer_size)

        self.step_count = 0

    @property
    def name(self) -> str:
        return "sac"

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _to_tensor(self, x):
        return torch.FloatTensor(np.asarray(x, dtype=np.float32)).to(self.device)

    def select_action(self, state, training: bool = True):
        if training and self.step_count < self.warmup_steps:
            return np.random.uniform(-1.0, 1.0, self.action_dim)

        state_t = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            if training:
                action, _, _ = self.actor.sample(state_t)
            else:
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean)
        return action.cpu().numpy().flatten()

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

        alpha = self.alpha.detach()

        # --- Update critics ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states_t)
            target_q1 = self.target_critic1(next_states_t, next_actions)
            target_q2 = self.target_critic2(next_states_t, next_actions)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
            target_value = rewards_t + self.gamma * (1 - dones_t) * target_q

        q1 = self.critic1(states_t, actions_t)
        q2 = self.critic2(states_t, actions_t)
        critic1_loss = F.mse_loss(q1, target_value)
        critic2_loss = F.mse_loss(q2, target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Update actor ---
        new_actions, log_probs, _ = self.actor.sample(states_t)
        q1_new = self.critic1(states_t, new_actions)
        q2_new = self.critic2(states_t, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update alpha ---
        alpha_loss_val = 0.0
        if self.auto_entropy:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        # --- Soft update targets ---
        for sp, tp in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        for sp, tp in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return {
            "loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "actor_loss": actor_loss.item(),
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "alpha": self.alpha.item(),
        }

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self.target_critic1.load_state_dict(state_dict["target_critic1"])
        self.target_critic2.load_state_dict(state_dict["target_critic2"])
        self.log_alpha = torch.tensor(
            state_dict["log_alpha"],
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.step_count = state_dict["step_count"]
