"""Blackjack-v1 environment wrapper.

The classic card game of Blackjack. The agent plays against a fixed dealer
strategy (stick on 17+) and must decide to hit or stick.

- State space: Tuple (player_sum: 4-21, dealer_showing: 1-10, usable_ace: bool)
  → encoded as a 3-dimensional observation
- Action space: 2 discrete (0=Stick, 1=Hit)
- Reward: +1 win, -1 lose, 0 draw (natural blackjack: +1.5 if not sab mode)
- Episode: single hand of Blackjack

Ideal for studying tabular RL in stochastic, episodic settings.
"""

from core.environment import GymEnvWrapper


def create_blackjack(**kwargs) -> GymEnvWrapper:
    return GymEnvWrapper("Blackjack-v1", **kwargs)
