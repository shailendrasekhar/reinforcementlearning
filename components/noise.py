"""Action noise for continuous control."""

import numpy as np


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.action_dim
        )
        self.state += dx
        return self.state.copy()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu


class GaussianNoise:
    """Simple Gaussian noise with optional decay."""

    def __init__(
        self,
        action_dim: int,
        sigma: float = 0.1,
        sigma_min: float = 0.01,
        decay_rate: float = 1.0,
    ):
        self.action_dim = action_dim
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay_rate = decay_rate

    def sample(self) -> np.ndarray:
        return np.random.randn(self.action_dim) * self.sigma

    def decay(self):
        self.sigma = max(self.sigma_min, self.sigma * self.decay_rate)

    def reset(self):
        pass
