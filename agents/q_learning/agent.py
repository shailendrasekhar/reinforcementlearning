"""Q-Learning agent — tabular, off-policy, value-based."""

from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np

from core.agent import BaseAgent
from core.types import Transition
from components.exploration import EpsilonGreedy, Boltzmann, ExplorationStrategy


class QLearningAgent(BaseAgent):
    """Tabular Q-learning with configurable exploration."""

    def __init__(self, env_info: Dict[str, Any], **kwargs):
        self.action_dim = env_info["action_dim"]
        self.lr = kwargs.get("learning_rate", 0.1)
        self.gamma = kwargs.get("discount_factor", 0.99)

        # Build exploration strategy
        exp_cfg = kwargs.get("exploration", {})
        self.exploration = self._build_exploration(exp_cfg)

        # Q-table: state -> array of Q-values
        self.q_table: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_dim)
        )

    def _build_exploration(self, cfg: Dict) -> ExplorationStrategy:
        strategy = cfg.get("strategy", "epsilon_greedy")
        params = cfg.get("params", {})
        if strategy == "boltzmann":
            return Boltzmann(**params)
        return EpsilonGreedy(**params)

    @property
    def name(self) -> str:
        return "q_learning"

    @property
    def epsilon(self) -> Optional[float]:
        return self.exploration.current_value

    def select_action(self, state, training: bool = True) -> int:
        if training:
            return self.exploration.select(self.q_table[state])
        return int(np.argmax(self.q_table[state]))

    def update(self, transition: Transition) -> Dict[str, float]:
        s, a, r = transition.state, transition.action, transition.reward
        s_next, done = transition.next_state, transition.done

        td_target = r + self.gamma * np.max(self.q_table[s_next]) * (1 - done)
        td_error = td_target - self.q_table[s][a]
        self.q_table[s][a] += self.lr * td_error

        self.exploration.decay()
        return {"td_error": abs(td_error)}

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "q_table": dict(self.q_table),
            "exploration_value": self.exploration.current_value,
        }

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
