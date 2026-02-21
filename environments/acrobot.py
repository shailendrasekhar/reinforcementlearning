"""Acrobot environment wrapper."""

from core.environment import GymEnvWrapper


def create_acrobot(**kwargs) -> GymEnvWrapper:
    return GymEnvWrapper("Acrobot-v1", **kwargs)
