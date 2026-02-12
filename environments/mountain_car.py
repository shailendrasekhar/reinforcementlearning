"""MountainCar environment wrapper."""

from core.environment import GymEnvWrapper


def create_mountain_car(**kwargs) -> GymEnvWrapper:
    return GymEnvWrapper("MountainCar-v0", **kwargs)
