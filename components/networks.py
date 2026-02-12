"""Neural network architectures for deep RL.

All networks require PyTorch. A helpful error is raised if it's missing.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep RL agents. Install with: pip install torch"
        )


def get_device(device_str: str = "auto") -> "torch.device":
    """Resolve device string to torch.device."""
    _check_torch()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Generic building block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128),
                 activation=None, output_activation=None):
        _check_torch()
        super().__init__()
        if activation is None:
            activation = nn.ReLU
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        if output_activation:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Discrete-action networks
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Q-value network: state -> Q(s, a) for all discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 128)):
        _check_torch()
        super().__init__()
        self.net = MLP(state_dim, action_dim, hidden_dims)

    def forward(self, state):
        return self.net(state)


class PolicyNetwork(nn.Module):
    """Discrete policy: state -> action probabilities."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 128)):
        _check_torch()
        super().__init__()
        self.net = MLP(state_dim, action_dim, hidden_dims)

    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)


class ActorCriticNetwork(nn.Module):
    """Shared-backbone actor-critic for discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 128)):
        _check_torch()
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.actor_head = nn.Linear(prev, action_dim)
        self.critic_head = nn.Linear(prev, 1)

    def forward(self, state):
        features = self.backbone(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value.squeeze(-1)

    def get_action_and_value(self, state):
        """Return policy distribution, sampled action, log_prob, entropy, and value."""
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ---------------------------------------------------------------------------
# Continuous-action networks
# ---------------------------------------------------------------------------

class DeterministicActor(nn.Module):
    """Deterministic policy for continuous actions (DDPG)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256),
                 max_action: float = 1.0):
        _check_torch()
        super().__init__()
        self.net = MLP(state_dim, action_dim, hidden_dims, output_activation=nn.Tanh)
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action


class QCritic(nn.Module):
    """Q-value critic for (state, action) pairs (DDPG, SAC)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256)):
        _check_torch()
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class GaussianActor(nn.Module):
    """Stochastic Gaussian policy for continuous actions (SAC)."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256)):
        _check_torch()
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(self, state):
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """Sample action with reparameterization trick + tanh squashing."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        # Log-prob with tanh correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob, mean
