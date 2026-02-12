"""Core interfaces and training infrastructure."""

from core.types import Transition, EpisodeResult, TrainingResult, EvaluationResult
from core.agent import BaseAgent
from core.environment import EnvWrapper, GymEnvWrapper
from core.trainer import EpisodicTrainer
from core.rollout_trainer import RolloutTrainer
from core.evaluator import Evaluator
from core.callback import (
    Callback, CallbackList, MetricsCallback,
    CheckpointCallback, EarlyStoppingCallback, ProgressCallback,
)

__all__ = [
    "Transition", "EpisodeResult", "TrainingResult", "EvaluationResult",
    "BaseAgent", "EnvWrapper", "GymEnvWrapper",
    "EpisodicTrainer", "RolloutTrainer", "Evaluator",
    "Callback", "CallbackList", "MetricsCallback",
    "CheckpointCallback", "EarlyStoppingCallback", "ProgressCallback",
]
