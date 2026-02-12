"""Exploration strategies for discrete-action agents."""

from abc import ABC, abstractmethod
import numpy as np


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    @abstractmethod
    def select(self, q_values: np.ndarray) -> int:
        ...

    @abstractmethod
    def decay(self) -> None:
        ...

    @property
    @abstractmethod
    def current_value(self) -> float:
        ...


class EpsilonGreedy(ExplorationStrategy):
    """Epsilon-greedy with configurable decay."""

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        decay_type: str = "exponential",
        decay_rate: float = 0.995,
        decay_steps: int = 10000,
    ):
        self._epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.step = 0

    def select(self, q_values: np.ndarray) -> int:
        if np.random.random() < self._epsilon:
            return np.random.randint(len(q_values))
        return int(np.argmax(q_values))

    def decay(self) -> None:
        self.step += 1
        if self.decay_type == "exponential":
            self._epsilon = max(self.epsilon_end, self._epsilon * self.decay_rate)
        elif self.decay_type == "linear":
            self._epsilon = max(
                self.epsilon_end,
                1.0 - self.step / self.decay_steps * (1.0 - self.epsilon_end),
            )

    @property
    def current_value(self) -> float:
        return self._epsilon


class Boltzmann(ExplorationStrategy):
    """Softmax / Boltzmann exploration."""

    def __init__(
        self,
        temperature: float = 1.0,
        temp_min: float = 0.01,
        decay_rate: float = 0.995,
    ):
        self.temperature = temperature
        self.temp_min = temp_min
        self.decay_rate = decay_rate

    def select(self, q_values: np.ndarray) -> int:
        exp_q = np.exp((q_values - np.max(q_values)) / max(self.temperature, 1e-8))
        probs = exp_q / exp_q.sum()
        return int(np.random.choice(len(q_values), p=probs))

    def decay(self) -> None:
        self.temperature = max(self.temp_min, self.temperature * self.decay_rate)

    @property
    def current_value(self) -> float:
        return self.temperature
