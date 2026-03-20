"""Taxi-v3 environment wrapper.

The Taxi problem (Dietterich, 2000): navigate a taxi to pick up a passenger
from one of four fixed locations and drop them off at another.

- State space: 500 discrete states (25 positions × 5 passenger locations × 4 destinations)
- Action space: 6 discrete (South, North, East, West, Pickup, Dropoff)
- Reward: -1 per step, +20 for successful drop-off, -10 for illegal pickup/dropoff
"""

from core.environment import GymEnvWrapper


def create_taxi(**kwargs) -> GymEnvWrapper:
    return GymEnvWrapper("Taxi-v3", **kwargs)
