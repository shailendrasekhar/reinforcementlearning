"""Environment wrapper interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from core.types import State, Action


class EnvWrapper(ABC):
    """Abstract environment wrapper."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        ...

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        ...

    @property
    def is_discrete(self) -> bool:
        return isinstance(self.action_space, gym.spaces.Discrete)

    @property
    def state_dim(self) -> int:
        space = self.observation_space
        if isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        return int(np.prod(space.shape))

    @property
    def action_dim(self) -> int:
        space = self.action_space
        if isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        return int(np.prod(space.shape))

    @property
    def action_low(self) -> Optional[np.ndarray]:
        if isinstance(self.action_space, gym.spaces.Box):
            return self.action_space.low
        return None

    @property
    def action_high(self) -> Optional[np.ndarray]:
        if isinstance(self.action_space, gym.spaces.Box):
            return self.action_space.high
        return None

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> State:
        ...

    @abstractmethod
    def step(self, action: Action) -> Tuple[State, float, bool, bool, Dict]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def render(self) -> None:
        pass


class GymEnvWrapper(EnvWrapper):
    """Concrete wrapper around a gymnasium environment."""

    def __init__(self, env_id: str, **kwargs):
        self._env = gym.make(env_id, **kwargs)
        self._name = env_id
        self._discrete_obs = isinstance(self._env.observation_space, gym.spaces.Discrete)

    def _convert_state(self, state: Any) -> State:
        """Ensure Discrete obs states are plain ints (hashable for Q-tables)."""
        if self._discrete_obs:
            return int(state)
        return state

    @property
    def name(self) -> str:
        return self._name

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    def reset(self, seed: Optional[int] = None) -> State:
        state, _info = self._env.reset(seed=seed)
        return self._convert_state(state)

    def step(self, action: Action) -> Tuple[State, float, bool, bool, Dict]:
        next_state, reward, terminated, truncated, info = self._env.step(action)
        return self._convert_state(next_state), reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def render(self) -> None:
        self._env.render()
