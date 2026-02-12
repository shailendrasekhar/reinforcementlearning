"""Episodic training loop."""

from typing import Any, Dict, Optional

from core.types import Transition, EpisodeResult, TrainingResult
from core.agent import BaseAgent
from core.environment import EnvWrapper


class EpisodicTrainer:
    """Episode-based training loop. Algorithm-agnostic."""

    def __init__(
        self,
        env: EnvWrapper,
        agent: BaseAgent,
        config: Dict[str, Any],
        callbacks=None,
    ):
        self.env = env
        self.agent = agent
        self.config = config
        self.callbacks = callbacks

        self.num_episodes: int = config.get("num_episodes", 1000)
        self.max_steps: int = config.get("max_steps_per_episode", 1000)

    def train_episode(self) -> EpisodeResult:
        """Run one training episode."""
        state = self.env.reset()
        self.agent.on_episode_start()

        total_reward = 0.0
        step_metrics: Dict[str, float] = {}

        for step in range(self.max_steps):
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            transition = Transition(
                state=state, action=action, reward=reward,
                next_state=next_state, done=done, info=info,
            )

            metrics = self.agent.update(transition)
            if metrics:
                step_metrics.update(metrics)

            if self.callbacks:
                self.callbacks.on_step(transition, metrics)

            total_reward += reward
            state = next_state

            if done:
                break

        end_metrics = self.agent.on_episode_end(total_reward)
        step_metrics.update(end_metrics)

        return EpisodeResult(
            total_reward=total_reward,
            length=step + 1,
            metrics=step_metrics,
        )

    def train(self) -> TrainingResult:
        """Run the full training loop."""
        result = TrainingResult()
        self.agent.on_training_start(self.config)

        if self.callbacks:
            self.callbacks.on_training_start(self.config)

        for episode in range(self.num_episodes):
            ep = self.train_episode()

            result.episode_rewards.append(ep.total_reward)
            result.episode_lengths.append(ep.length)

            eps = self.agent.epsilon
            if eps is not None:
                result.epsilon_history.append(eps)

            if "loss" in ep.metrics:
                result.loss_history.append(ep.metrics["loss"])

            for key, val in ep.metrics.items():
                if key != "loss":
                    result.extra_metrics.setdefault(key, []).append(val)

            if self.callbacks:
                self.callbacks.on_episode_end(episode, ep, result)

        self.agent.on_training_end()
        if self.callbacks:
            self.callbacks.on_training_end(result)

        return result
