"""Pendulum environment wrapper."""

from core.environment import GymEnvWrapper


def create_pendulum(**kwargs) -> GymEnvWrapper:
    return GymEnvWrapper("Pendulum-v1", **kwargs)
