"""ExperimentRunner — the single orchestrator that users interact with.

Usage::

    runner = ExperimentRunner("configs/q_learning_frozenlake.yaml")
    result = runner.run()
"""

import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.types import TrainingResult, EvaluationResult
from core.trainer import EpisodicTrainer
from core.rollout_trainer import RolloutTrainer
from core.evaluator import Evaluator
from core.callback import (
    CallbackList, ProgressCallback, CheckpointCallback,
    EarlyStoppingCallback, MetricsCallback,
)
from infrastructure.config_loader import ConfigLoader
from infrastructure.artifact_store import ArtifactStore
from infrastructure.logger import Logger
from infrastructure.visualizer import Visualizer
from environments.registry import EnvRegistry
from agents.registry import AgentRegistry


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Everything produced by a single experiment run."""
    training_result: TrainingResult
    evaluation_result: EvaluationResult
    config: Dict[str, Any]
    run_dir: str
    duration_seconds: float
    best_eval_reward: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Run:          {self.run_dir}",
            f"Duration:     {self.duration_seconds:.1f}s",
            f"Episodes:     {self.training_result.num_episodes}",
            f"Eval reward:  {self.evaluation_result.mean_reward:.2f} "
            f"+/- {self.evaluation_result.std_reward:.2f}",
            f"Success rate: {self.evaluation_result.success_rate:.1f}%",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Single entry point: load config -> train -> evaluate -> save.

    Parameters
    ----------
    config_source : str or dict
        Path to a YAML config file, or an already-loaded config dict.
    """

    def __init__(self, config_source: Union[str, Dict[str, Any]]):
        if isinstance(config_source, str):
            self.config = ConfigLoader.load(config_source)
        else:
            self.config = config_source
        ConfigLoader.validate(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResult:
        """Execute a complete experiment: train -> evaluate -> visualize -> save."""
        start = time.time()

        # Seed
        seed = self.config.get("experiment", {}).get("seed", 42)
        self._set_seed(seed)

        # Environment
        env_cfg = self.config["environment"]
        env = EnvRegistry.create(env_cfg["name"], **env_cfg.get("params", {}))

        # Agent
        agent_cfg = self.config["agent"]
        env_info = {
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "is_discrete": env.is_discrete,
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }
        agent_params = dict(agent_cfg.get("params", {}))
        if "exploration" in agent_cfg:
            agent_params["exploration"] = agent_cfg["exploration"]
        if "noise" in agent_cfg:
            agent_params["noise"] = agent_cfg["noise"]

        agent = AgentRegistry.create(agent_cfg["name"], env_info=env_info, **agent_params)

        # Artifact store
        output_cfg = self.config.get("output", {})
        store = ArtifactStore(
            base_dir=output_cfg.get("base_dir", "outputs"),
            agent_name=agent.name,
            env_name=env.name,
            experiment_name=self.config.get("experiment", {}).get("name"),
        )
        store.save_config(self.config)

        # Callbacks
        callbacks = self._build_callbacks(agent, store)

        # Trainer
        train_cfg = self.config.get("training", {})
        mode = train_cfg.get("mode", "episodic")
        if mode == "rollout":
            trainer = RolloutTrainer(env, agent, train_cfg, callbacks)
        else:
            trainer = EpisodicTrainer(env, agent, train_cfg, callbacks)

        # Train
        print(f"\n[{agent.name}] Training on {env.name}...")
        train_result = trainer.train()

        # Evaluate
        eval_cfg = self.config.get("evaluation", {})
        evaluator = Evaluator(env, agent)
        eval_result = evaluator.evaluate(
            num_episodes=eval_cfg.get("num_episodes", 100),
            max_steps=train_cfg.get("max_steps_per_episode", 1000),
            success_threshold=eval_cfg.get("success_threshold", 0.0),
            verbose=True,
        )

        # Visualize
        if output_cfg.get("save_plots", True):
            self._generate_plots(train_result, eval_result, agent, store)

        # Save metrics
        if output_cfg.get("save_metrics_csv", True):
            store.save_metrics_csv("training_log", Logger.training_to_records(train_result))
            store.save_metrics_csv("evaluation_log", Logger.evaluation_to_records(eval_result))

        # Save final model
        store.save_final_model(agent)

        duration = time.time() - start
        store.save_metadata({
            "agent": agent.name,
            "environment": env.name,
            "seed": seed,
            "duration_seconds": round(duration, 2),
            "training_episodes": train_result.num_episodes,
            "eval_mean_reward": eval_result.mean_reward,
            "eval_success_rate": eval_result.success_rate,
        })

        Logger.print_summary(train_result, eval_result)
        print(f"  Artifacts saved to: {store.run_dir}\n")

        env.close()
        Visualizer.close_all()

        return ExperimentResult(
            training_result=train_result,
            evaluation_result=eval_result,
            config=self.config,
            run_dir=str(store.run_dir),
            duration_seconds=duration,
            best_eval_reward=eval_result.mean_reward,
        )

    def run_multi_seed(self, seeds: Optional[List[int]] = None) -> List[ExperimentResult]:
        """Run the same experiment with multiple seeds."""
        num_runs = self.config.get("experiment", {}).get("num_runs", 3)
        if seeds is None:
            seeds = list(range(42, 42 + num_runs))

        results = []
        for i, seed in enumerate(seeds):
            print(f"\n{'=' * 60}")
            print(f"  RUN {i + 1}/{len(seeds)} (seed={seed})")
            print(f"{'=' * 60}")
            self.config.setdefault("experiment", {})["seed"] = seed
            results.append(self.run())

        rewards = [r.evaluation_result.mean_reward for r in results]
        print(
            f"\n  Aggregate: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f} "
            f"over {len(seeds)} seeds"
        )
        return results

    @staticmethod
    def compare(config_paths: List[str]) -> List[ExperimentResult]:
        """Run multiple configs and print a comparison table."""
        results = []
        for path in config_paths:
            runner = ExperimentRunner(path)
            results.append(runner.run())

        print("\n" + "=" * 60)
        print("  COMPARISON")
        print("=" * 60)
        for r in results:
            a = r.config["agent"]["name"]
            e = r.config["environment"]["name"]
            er = r.evaluation_result
            print(
                f"  {a:>15s} on {e:>15s} : "
                f"reward={er.mean_reward:>7.2f} +/- {er.std_reward:>5.2f}, "
                f"success={er.success_rate:>5.1f}%"
            )
        print("=" * 60)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _build_callbacks(self, agent, store) -> CallbackList:
        cbs: list = []
        train_cfg = self.config.get("training", {})

        cbs.append(ProgressCallback(log_frequency=train_cfg.get("log_frequency", 100)))
        cbs.append(MetricsCallback())

        for cb_cfg in self.config.get("callbacks", []):
            cb_type = cb_cfg.get("type", "")
            params = cb_cfg.get("params", {})
            if cb_type == "checkpoint":
                cbs.append(CheckpointCallback(
                    agent, store,
                    frequency=params.get("frequency", 1000),
                    save_best=params.get("save_best", True),
                ))
            elif cb_type == "early_stopping":
                cbs.append(EarlyStoppingCallback(
                    patience=params.get("patience", 500),
                    min_improvement=params.get("min_improvement", 0.01),
                ))

        return CallbackList(cbs)

    def _generate_plots(self, train_result, eval_result, agent, store):
        fmt = self.config.get("output", {}).get("plot_format", "png")

        fig = Visualizer.plot_training_progress(train_result)
        store.save_plot(fig, "training", "reward_curve", fmt)
        plt_close(fig)

        fig = Visualizer.plot_loss(train_result)
        if fig:
            store.save_plot(fig, "training", "loss_curve", fmt)
            plt_close(fig)

        fig = Visualizer.plot_epsilon(train_result)
        if fig:
            store.save_plot(fig, "training", "epsilon_decay", fmt)
            plt_close(fig)

        fig = Visualizer.plot_evaluation(eval_result)
        store.save_plot(fig, "evaluation", "reward_distribution", fmt)
        plt_close(fig)

        analysis = agent.get_analysis_data()
        if analysis:
            fig = Visualizer.plot_tabular_policy(analysis)
            if fig:
                store.save_plot(fig, "analysis", "policy_grid", fmt)
                plt_close(fig)


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)
