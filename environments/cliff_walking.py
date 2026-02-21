"""CliffWalking environment wrapper."""

from core.environment import GymEnvWrapper


def create_cliff_walking(**kwargs) -> GymEnvWrapper:
    return GymEnvWrapper("CliffWalking-v1", **kwargs)
