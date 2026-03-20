"""BipedalWalker-v3 environment wrapper.

A challenging continuous control environment where an agent must control a
2D bipedal robot to walk across rough terrain without falling.

- State space: 24-dimensional continuous (hull angle, angular velocity, joints, LIDAR)
- Action space: 4-dimensional continuous (hip/knee torques for each leg)
- Reward: ~300 for reaching end, -100 for falling, small step penalty for motor use
- Solved: average reward ≥ 300 over 100 episodes

Requires box2d-py. Install: pip install box2d-py swig
"""

from core.environment import GymEnvWrapper


def create_bipedal_walker(**kwargs) -> GymEnvWrapper:
    return GymEnvWrapper("BipedalWalker-v3", **kwargs)
