"""Callback system for training hooks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import time


class Callback:
    """Base callback — override the hooks you need."""

    def on_training_start(self, config: Dict[str, Any]) -> None:
        pass

    def on_training_end(self, result) -> None:
        pass

    def on_episode_end(self, episode: int, ep_result, result) -> None:
        pass

    def on_step(self, transition, metrics: Dict[str, float]) -> None:
        pass


class CallbackList(Callback):
    """Composite callback that fans out to a list of callbacks."""

    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def on_training_start(self, config):
        for cb in self.callbacks:
            cb.on_training_start(config)

    def on_training_end(self, result):
        for cb in self.callbacks:
            cb.on_training_end(result)

    def on_episode_end(self, episode, ep_result, result):
        for cb in self.callbacks:
            cb.on_episode_end(episode, ep_result, result)

    def on_step(self, transition, metrics):
        for cb in self.callbacks:
            cb.on_step(transition, metrics)


class ProgressCallback(Callback):
    """Prints training progress at regular intervals."""

    def __init__(self, log_frequency: int = 100, moving_avg_window: int = 100):
        self.log_freq = log_frequency
        self.window = moving_avg_window
        self.start_time: Optional[float] = None

    def on_training_start(self, config):
        self.start_time = time.time()
        total = config.get("num_episodes", config.get("total_timesteps", "?"))
        print(f"  Training started — {total} episodes/steps")

    def on_episode_end(self, episode, ep_result, result):
        if (episode + 1) % self.log_freq == 0:
            recent = result.episode_rewards[-self.window:]
            avg = sum(recent) / len(recent)
            eps_str = ""
            if result.epsilon_history:
                eps_str = f"  eps={result.epsilon_history[-1]:.4f}"
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(
                f"  Episode {episode + 1:>6d} | "
                f"avg_{self.window}={avg:>8.2f}{eps_str} | "
                f"{elapsed:.0f}s"
            )

    def on_training_end(self, result):
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"  Training complete — {result.num_episodes} episodes in {elapsed:.1f}s")


class CheckpointCallback(Callback):
    """Saves agent checkpoints during training."""

    def __init__(self, agent, artifact_store, frequency: int = 1000, save_best: bool = True):
        self.agent = agent
        self.store = artifact_store
        self.frequency = frequency
        self.save_best = save_best
        self.best_avg_reward = float("-inf")

    def on_episode_end(self, episode, ep_result, result):
        if (episode + 1) % self.frequency == 0:
            path = self.store.checkpoint_dir / f"episode_{episode + 1}.pkl"
            self.agent.save(path)

        if self.save_best and len(result.episode_rewards) >= 100:
            avg = sum(result.episode_rewards[-100:]) / 100
            if avg > self.best_avg_reward:
                self.best_avg_reward = avg
                self.agent.save(self.store.checkpoint_dir / "best.pkl")


class EarlyStoppingCallback(Callback):
    """Stops training when improvement stalls."""

    def __init__(self, patience: int = 500, min_improvement: float = 0.01, window: int = 100):
        self.patience = patience
        self.min_improvement = min_improvement
        self.window = window
        self.best_avg = float("-inf")
        self.episodes_without_improvement = 0
        self.should_stop = False

    def on_episode_end(self, episode, ep_result, result):
        if len(result.episode_rewards) >= self.window:
            avg = sum(result.episode_rewards[-self.window:]) / self.window
            if avg > self.best_avg + self.min_improvement:
                self.best_avg = avg
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1

            if self.episodes_without_improvement >= self.patience:
                self.should_stop = True


class MetricsCallback(Callback):
    """Collects per-episode metrics for CSV logging."""

    def __init__(self):
        self.step_count = 0
        self.episode_data: list = []

    def on_episode_end(self, episode, ep_result, result):
        self.episode_data.append({
            "episode": episode,
            "reward": ep_result.total_reward,
            "length": ep_result.length,
            **ep_result.metrics,
        })

    def on_step(self, transition, metrics):
        self.step_count += 1
