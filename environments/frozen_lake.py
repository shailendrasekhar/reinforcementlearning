"""FrozenLake environment wrapper."""

from core.environment import GymEnvWrapper


def create_frozen_lake(**kwargs) -> GymEnvWrapper:
    map_name = kwargs.pop("map_name", "4x4")
    is_slippery = kwargs.pop("is_slippery", True)
    return GymEnvWrapper("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, **kwargs)
