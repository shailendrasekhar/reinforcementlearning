"""Evaluation loop."""

from core.types import EvaluationResult
from core.agent import BaseAgent
from core.environment import EnvWrapper


class Evaluator:
    """Evaluate a trained agent in an environment."""

    def __init__(self, env: EnvWrapper, agent: BaseAgent):
        self.env = env
        self.agent = agent

    def evaluate(
        self,
        num_episodes: int = 100,
        max_steps: int = 1000,
        success_threshold: float = 0.0,
        verbose: bool = False,
    ) -> EvaluationResult:
        result = EvaluationResult()

        for ep in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            for step in range(max_steps):
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _info = self.env.step(action)
                total_reward += reward
                state = next_state
                if terminated or truncated:
                    break

            result.episode_rewards.append(total_reward)
            result.episode_lengths.append(step + 1)
            result.successes.append(total_reward > success_threshold)

            if verbose and (ep + 1) % max(1, num_episodes // 10) == 0:
                print(f"  Eval [{ep + 1}/{num_episodes}] reward={total_reward:.2f}")

        if verbose:
            print(
                f"  Evaluation: mean={result.mean_reward:.2f} +/- {result.std_reward:.2f}, "
                f"success={result.success_rate:.1f}%"
            )

        return result
