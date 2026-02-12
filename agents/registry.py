"""Agent registry — maps names to agent constructors."""

from typing import Any, Dict, Type

from core.agent import BaseAgent


class AgentRegistry:
    """Maps string names to agent classes."""

    _registry: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        cls._registry[name.lower()] = agent_class

    @classmethod
    def create(cls, name: str, env_info: Dict[str, Any], **kwargs) -> BaseAgent:
        """Create an agent by name, passing environment info + config params."""
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(f"Unknown agent '{name}'. Available: {available}")
        return cls._registry[key](env_info=env_info, **kwargs)

    @classmethod
    def list(cls):
        return sorted(cls._registry.keys())


# --- Auto-register built-in agents ---

def _register_defaults():
    from agents.q_learning.agent import QLearningAgent
    from agents.sarsa.agent import SarsaAgent
    from agents.dqn.agent import DQNAgent
    from agents.reinforce.agent import ReinforceAgent
    from agents.a2c.agent import A2CAgent
    from agents.ppo.agent import PPOAgent
    from agents.ddpg.agent import DDPGAgent
    from agents.sac.agent import SACAgent

    AgentRegistry.register("q_learning", QLearningAgent)
    AgentRegistry.register("sarsa", SarsaAgent)
    AgentRegistry.register("dqn", DQNAgent)
    AgentRegistry.register("reinforce", ReinforceAgent)
    AgentRegistry.register("a2c", A2CAgent)
    AgentRegistry.register("ppo", PPOAgent)
    AgentRegistry.register("ddpg", DDPGAgent)
    AgentRegistry.register("sac", SACAgent)


_register_defaults()
