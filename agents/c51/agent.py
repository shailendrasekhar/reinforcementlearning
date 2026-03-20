"""C51 — Categorical DQN (Distributional RL).

Instead of learning E[G_t], C51 learns the full return distribution P(G_t)
represented as a categorical distribution over N atoms spanning [V_min, V_max].

The distributional Bellman update projects the next-state distribution onto
the current atom support using L2 projection.

Reference:
  Bellemare, Dabney, Munos (2017) — "A Distributional Perspective on
  Reinforcement Learning"  https://arxiv.org/abs/1707.06887
"""

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


class C51Agent(BaseAgent):
    """Categorical DQN (C51) — distributional value learning."""

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        if torch is None:
            raise ImportError("PyTorch required for C51.")

        from components.networks import CategoricalQNetwork, get_device

        self.state_dim  = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.gamma      = kwargs.get("discount_factor", 0.99)
        self.batch_size = kwargs.get("batch_size", 64)
        self.warmup_steps       = kwargs.get("warmup_steps", 1000)
        self.train_frequency    = kwargs.get("train_frequency", 4)
        self.target_update_freq = kwargs.get("target_update_frequency", 1000)
        self.grad_clip          = kwargs.get("gradient_clip", 10.0)

        # Distribution parameters
        self.n_atoms = kwargs.get("n_atoms", 51)
        self.v_min   = kwargs.get("v_min", -10.0)
        self.v_max   = kwargs.get("v_max",  10.0)

        device_str  = kwargs.get("device", "auto")
        self.device = get_device(device_str)

        # Atom support: z_i = v_min + i * delta_z
        self.support   = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        self.delta_z   = (self.v_max - self.v_min) / (self.n_atoms - 1)

        hidden = kwargs.get("hidden_dims", [128, 128])
        self.q_net      = CategoricalQNetwork(self.state_dim, self.action_dim,
                                              self.n_atoms, hidden).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        lr = kwargs.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        buffer_size = kwargs.get("buffer_size", 100_000)
        self.buffer = UniformReplayBuffer(buffer_size)

        exp_cfg = kwargs.get("exploration", {})
        params  = exp_cfg.get("params", {})
        self.exploration = EpsilonGreedy(**params)

        self.step_count = 0
        self._last_loss = 0.0

        import gymnasium as gym
        obs_space = env_info.get("observation_space")
        self._discrete_obs = isinstance(obs_space, gym.spaces.Discrete)
        self._obs_n = int(obs_space.n) if self._discrete_obs else 0

    @property
    def name(self) -> str:
        return "c51"

    @property
    def epsilon(self) -> Optional[float]:
        return self.exploration.current_value

    def _encode_state(self, state) -> np.ndarray:
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

    def _q_values_from_log_probs(self, log_probs: "torch.Tensor") -> "torch.Tensor":
        """Expected Q-values: sum_i p_i * z_i per action."""
        probs = log_probs.exp()                    # (B, A, N)
        return (probs * self.support).sum(dim=2)   # (B, A)

    def select_action(self, state, training: bool = True) -> int:
        if training and self.step_count < self.warmup_steps:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            log_probs = self.q_net(self._to_tensor(state))        # (1, A, N)
            q_vals    = self._q_values_from_log_probs(log_probs)  # (1, A)
            q_arr     = q_vals.cpu().numpy().flatten()
        if training:
            return self.exploration.select(q_arr)
        return int(np.argmax(q_arr))

    def _project_distribution(self, rewards_t, next_log_probs, dones_t):
        """Distributional Bellman projection onto support atoms."""
        B = rewards_t.shape[0]
        N = self.n_atoms

        with torch.no_grad():
            # Greedy action selection from target distribution
            next_q     = self._q_values_from_log_probs(next_log_probs)  # (B, A)
            next_acts  = next_q.argmax(1)                                # (B,)
            # Gather target distributions for greedy actions
            idx        = next_acts.unsqueeze(1).unsqueeze(2).expand(B, 1, N)
            next_probs = next_log_probs.exp().gather(1, idx).squeeze(1)  # (B, N)

            # Compute projected support
            rewards_exp = rewards_t.unsqueeze(1).expand_as(next_probs)
            dones_exp   = dones_t.unsqueeze(1).expand_as(next_probs)
            Tz          = (rewards_exp + self.gamma * self.support * (1 - dones_exp))
            Tz          = Tz.clamp(self.v_min, self.v_max)

            # L2 projection
            b   = (Tz - self.v_min) / self.delta_z            # (B, N)
            l   = b.floor().long().clamp(0, N - 1)
            u   = b.ceil().long().clamp(0, N - 1)

            proj = torch.zeros_like(next_probs)
            proj.scatter_add_(1, l, next_probs * (u.float() - b))
            proj.scatter_add_(1, u, next_probs * (b - l.float()))

        return proj  # (B, N)  — target probability distribution

    def update(self, transition: Transition) -> Dict[str, float]:
        self.step_count += 1
        self.buffer.push(transition)
        self.exploration.decay()

        if self.step_count < self.warmup_steps or len(self.buffer) < self.batch_size:
            return {}
        if self.step_count % self.train_frequency != 0:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample_tensors(self.batch_size)

        if self._discrete_obs:
            states      = np.array([self._encode_state(s) for s in states])
            next_states = np.array([self._encode_state(s) for s in next_states])

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current distribution for taken actions
        log_probs = self.q_net(states_t)                               # (B, A, N)
        B, _, N   = log_probs.shape
        idx       = actions_t.unsqueeze(1).unsqueeze(2).expand(B, 1, N)
        log_probs_taken = log_probs.gather(1, idx).squeeze(1)          # (B, N)

        # Target distribution (projected Bellman update)
        with torch.no_grad():
            next_log_probs = self.target_net(next_states_t)            # (B, A, N)
        target_probs = self._project_distribution(rewards_t, next_log_probs, dones_t)

        # Cross-entropy loss: -sum_i target_i * log_pred_i
        loss = -(target_probs * log_probs_taken).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self._last_loss = loss.item()
        return {"loss": loss.item()}

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.q_net.load_state_dict(state_dict["q_net"])
        self.target_net.load_state_dict(state_dict["target_net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.step_count = state_dict["step_count"]
