"""Prioritized Experience Replay (PER) buffer.

Implements the proportional prioritization variant from:
    Schaul et al. (2016) — "Prioritized Experience Replay"
    https://arxiv.org/abs/1511.05952

Key ideas:
  - Transitions with higher TD-error are sampled more frequently.
  - Importance-sampling (IS) weights correct the resulting bias.
  - A sum-tree data structure gives O(log N) push and sample.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from core.types import Transition


# ---------------------------------------------------------------------------
# Sum-tree
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum-tree for O(log N) priority updates and sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Leaf nodes live in positions [capacity, 2*capacity)
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: List = [None] * capacity
        self.write = 0          # circular write pointer
        self.n_entries = 0

    # ---- helpers ----

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    # ---- public API ----

    def add(self, priority: float, data) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def sample(self, s: float) -> Tuple[int, float, object]:
        """Return (tree_idx, priority, data) for cumulative sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries


# ---------------------------------------------------------------------------
# PER buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Experience replay with proportional prioritization.

    Parameters
    ----------
    capacity   : Maximum number of transitions stored.
    alpha      : Priority exponent — 0 = uniform, 1 = full prioritization.
    beta_start : Initial IS-weight exponent (annealed towards 1.0).
    beta_steps : Steps over which beta is linearly annealed to 1.0.
    epsilon    : Small constant added to priorities to prevent zero-prob.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 100_000,
        epsilon: float = 1e-6,
    ) -> None:
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        self._step = 0
        self._max_priority: float = 1.0

    # ---- internal ----

    def _priority(self, td_error: float) -> float:
        return (abs(td_error) + self.epsilon) ** self.alpha

    def _anneal_beta(self) -> None:
        self._step += 1
        self.beta = min(
            1.0,
            self.beta_start + self._step * (1.0 - self.beta_start) / self.beta_steps,
        )

    # ---- public API ----

    def push(self, transition: Transition, td_error: float = None) -> None:
        """Store a transition.  New entries get max priority by default."""
        priority = self._priority(td_error) if td_error is not None else self._max_priority
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample a batch.

        Returns
        -------
        transitions : List[Transition]
        indices     : np.ndarray — tree indices for priority updates.
        weights     : np.ndarray — IS correction weights (max-normalised).
        """
        self._anneal_beta()

        segment = self.tree.total / batch_size
        indices, priorities, transitions = [], [], []

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, p, t = self.tree.sample(s)
            indices.append(idx)
            priorities.append(p)
            transitions.append(t)

        # IS weights
        n = len(self.tree)
        probs = np.array(priorities, dtype=np.float64) / (self.tree.total + 1e-12)
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()           # normalise so max weight = 1

        return transitions, np.array(indices, dtype=np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled transitions after computing new TD errors."""
        for idx, err in zip(indices, td_errors):
            p = self._priority(float(err))
            self._max_priority = max(self._max_priority, p)
            self.tree.update(int(idx), p)

    def sample_tensors(self, batch_size: int):
        """Convenience: returns numpy arrays + indices + IS weights."""
        transitions, indices, weights = self.sample(batch_size)

        states      = np.array([t.state      for t in transitions], dtype=np.float32)
        actions     = np.array([t.action     for t in transitions])
        rewards     = np.array([t.reward     for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones       = np.array([t.done       for t in transitions], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def __len__(self) -> int:
        return len(self.tree)
