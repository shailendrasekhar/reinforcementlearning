"""Shared types and data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

# Type aliases
State = Any
Action = Any
Reward = float


@dataclass
class Transition:
    """Single environment transition."""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    total_reward: float
    length: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Accumulated training metrics."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    eval_episodes: List[int] = field(default_factory=list)
    extra_metrics: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def num_episodes(self) -> int:
        return len(self.episode_rewards)

    def moving_average(self, window: int = 100) -> List[float]:
        rewards = self.episode_rewards
        if len(rewards) < window:
            return list(np.cumsum(rewards) / np.arange(1, len(rewards) + 1))
        kernel = np.ones(window) / window
        return list(np.convolve(rewards, kernel, mode="valid"))


@dataclass
class EvaluationResult:
    """Evaluation metrics."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    successes: List[bool] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes)) * 100 if self.successes else 0.0

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
