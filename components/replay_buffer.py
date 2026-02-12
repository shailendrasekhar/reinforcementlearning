"""Replay buffers for off-policy algorithms."""

from collections import deque
from typing import List
import random

import numpy as np

from core.types import Transition


class UniformReplayBuffer:
    """Fixed-size buffer with uniform random sampling."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        return len(self.buffer) > 0

    def sample_tensors(self, batch_size: int):
        """Sample and return as numpy arrays (convenience for deep RL)."""
        batch = self.sample(batch_size)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones
