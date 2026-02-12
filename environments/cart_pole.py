"""CartPole environment wrapper."""

from core.environment import GymEnvWrapper


def create_cart_pole(**kwargs) -> GymEnvWrapper:
    return GymEnvWrapper("CartPole-v1", **kwargs)
