"""Rollout buffer for on-policy algorithms (PPO, A2C)."""

import numpy as np
from typing import Generator


class RolloutBuffer:
    """Stores rollout data and computes advantages using GAE."""

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        discrete: bool = True,
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.discrete = discrete

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        if discrete:
            self.actions = np.zeros(buffer_size, dtype=np.int64)
        else:
            self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def store(self, state, action, reward, done, log_prob, value):
        """Store a single step."""
        self.states[self.ptr] = np.asarray(state, dtype=np.float32).flatten()
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_advantages(self, last_value: float = 0.0):
        """Compute GAE advantages and returns."""
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_batches(self, batch_size: int) -> Generator:
        """Yield shuffled minibatches."""
        indices = np.random.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                self.states[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.returns[batch_idx],
                self.advantages[batch_idx],
                self.values[batch_idx],
            )

    def reset(self):
        self.ptr = 0
        self.full = False

    @property
    def is_full(self) -> bool:
        return self.full
