"""LunarLander environment wrapper."""

from core.environment import GymEnvWrapper


def create_lunar_lander(**kwargs) -> GymEnvWrapper:
    return GymEnvWrapper("LunarLander-v3", **kwargs)
