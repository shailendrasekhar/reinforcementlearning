"""Step-based (rollout) training loop for PPO / A2C with vectorized envs."""

from typing import Any, Dict

from core.types import Transition, EpisodeResult, TrainingResult
from core.agent import BaseAgent
from core.environment import EnvWrapper


class RolloutTrainer:
    """
    Step-based trainer that collects fixed-length rollouts across episode boundaries.
    Used by PPO, A2C, and other on-policy methods that don't align to episodes.
    """

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

        self.total_timesteps: int = config.get("total_timesteps", 100_000)
        self.max_steps: int = config.get("max_steps_per_episode", 1000)

    def train(self) -> TrainingResult:
        """Run step-based training."""
        result = TrainingResult()
        self.agent.on_training_start(self.config)
        if self.callbacks:
            self.callbacks.on_training_start(self.config)

        state = self.env.reset()
        self.agent.on_episode_start()

        episode_reward = 0.0
        episode_length = 0
        episode_num = 0

        for _step in range(self.total_timesteps):
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            transition = Transition(
                state=state, action=action, reward=reward,
                next_state=next_state, done=done, info=info,
            )

            metrics = self.agent.update(transition)

            if self.callbacks:
                self.callbacks.on_step(transition, metrics)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                end_metrics = self.agent.on_episode_end(episode_reward)

                result.episode_rewards.append(episode_reward)
                result.episode_lengths.append(episode_length)

                eps = self.agent.epsilon
                if eps is not None:
                    result.epsilon_history.append(eps)

                all_metrics = {**(metrics or {}), **(end_metrics or {})}
                if "loss" in all_metrics:
                    result.loss_history.append(all_metrics["loss"])

                if self.callbacks:
                    ep = EpisodeResult(
                        total_reward=episode_reward,
                        length=episode_length,
                        metrics=all_metrics,
                    )
                    self.callbacks.on_episode_end(episode_num, ep, result)

                episode_reward = 0.0
                episode_length = 0
                episode_num += 1

                state = self.env.reset()
                self.agent.on_episode_start()

        self.agent.on_training_end()
        if self.callbacks:
            self.callbacks.on_training_end(result)

        return result
