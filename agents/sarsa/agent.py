"""SARSA agent — tabular, on-policy, value-based."""

from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np

from core.agent import BaseAgent
from core.types import Transition
from components.exploration import EpsilonGreedy, Boltzmann, ExplorationStrategy


class SarsaAgent(BaseAgent):
    """Tabular SARSA — on-policy TD(0).

    Key difference from Q-learning: uses the *actual* next action (on-policy)
    for the TD target instead of the greedy max.  The agent pre-selects the
    next action during ``update()`` and caches it for the following
    ``select_action()`` call.
    """

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        self.action_dim = env_info["action_dim"]
        self.lr = kwargs.get("learning_rate", 0.1)
        self.gamma = kwargs.get("discount_factor", 0.99)

        exp_cfg = kwargs.get("exploration", {})
        self.exploration = self._build_exploration(exp_cfg)

        self.q_table: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_dim)
        )
        self._next_action: Optional[int] = None

    def _build_exploration(self, cfg: Dict) -> ExplorationStrategy:
        strategy = cfg.get("strategy", "epsilon_greedy")
        params = cfg.get("params", {})
        if strategy == "boltzmann":
            return Boltzmann(**params)
        return EpsilonGreedy(**params)

    @property
    def name(self) -> str:
        return "sarsa"

    @property
    def epsilon(self) -> Optional[float]:
        return self.exploration.current_value

    def select_action(self, state, training: bool = True) -> int:
        if not training:
            return int(np.argmax(self.q_table[state]))
        # Use pre-selected action if available (set during previous update)
        if self._next_action is not None:
            action = self._next_action
            self._next_action = None
            return action
        return self.exploration.select(self.q_table[state])

    def update(self, transition: Transition) -> Dict[str, float]:
        s, a, r = transition.state, transition.action, transition.reward
        s_next, done = transition.next_state, transition.done

        # SARSA: pick next action on-policy and use it for the TD target
        next_action = self.exploration.select(self.q_table[s_next])
        self._next_action = next_action  # cache for next select_action

        td_target = r + self.gamma * self.q_table[s_next][next_action] * (1 - done)
        td_error = td_target - self.q_table[s][a]
        self.q_table[s][a] += self.lr * td_error

        self.exploration.decay()
        return {"td_error": abs(td_error)}

    def on_episode_start(self) -> None:
        self._next_action = None

    def get_state_dict(self) -> Dict[str, Any]:
        return {"q_table": dict(self.q_table)}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.q_table = defaultdict(
            lambda: np.zeros(self.action_dim),
            state_dict["q_table"],
        )

    def get_analysis_data(self) -> Dict[str, Any]:
        return {
            "q_table": dict(self.q_table),
            "action_dim": self.action_dim,
            "type": "tabular",
        }
