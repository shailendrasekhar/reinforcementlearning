"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import pickle

from core.types import State, Action, Transition


class BaseAgent(ABC):
    """Abstract base class all RL agents must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this algorithm."""
        ...

    @abstractmethod
    def select_action(self, state: State, training: bool = True) -> Action:
        """Choose an action given the current state."""
        ...

    @abstractmethod
    def update(self, transition: Transition) -> Dict[str, float]:
        """Process a transition. Returns metrics dict (may be empty)."""
        ...

    # --- Lifecycle hooks (override as needed) ---

    def on_episode_start(self) -> None:
        """Called at the start of each episode."""
        pass

    def on_episode_end(self, episode_reward: float) -> Dict[str, float]:
        """Called at the end of each episode. Returns any end-of-episode metrics."""
        return {}

    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Called once before training begins."""
        pass

    def on_training_end(self) -> None:
        """Called once after training ends."""
        pass

    # --- Serialization ---

    def get_state_dict(self) -> Dict[str, Any]:
        """Return serializable state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        pass

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.get_state_dict(), f)

    def load(self, path: Path) -> None:
        with open(Path(path), "rb") as f:
            self.load_state_dict(pickle.load(f))

    # --- Introspection ---

    @property
    def epsilon(self) -> Optional[float]:
        """Current exploration rate, or None if not applicable."""
        return None

    def get_analysis_data(self) -> Dict[str, Any]:
        """Return algorithm-specific data for visualization."""
        return {}
