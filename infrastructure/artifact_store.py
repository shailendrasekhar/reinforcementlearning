"""Artifact storage — manages the output directory tree for a single run."""

from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import json


class ArtifactStore:
    """Creates and manages output directories for an experiment run.

    Directory layout::

        outputs/<env_name>/<algo_name>/<timestamp>/
            config.yaml
            metadata.json
            checkpoints/
            metrics/
            plots/training/
            plots/evaluation/
            plots/analysis/
            gifs/
            final_model/
    """

    def __init__(
        self,
        base_dir: str,
        agent_name: str,
        env_name: str,
        experiment_name: Optional[str] = None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = timestamp
        self.run_dir = Path(base_dir) / env_name / agent_name / timestamp

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.plot_dir = self.run_dir / "plots" / "training"
        self.eval_plot_dir = self.run_dir / "plots" / "evaluation"
        self.analysis_dir = self.run_dir / "plots" / "analysis"
        self.gif_dir = self.run_dir / "gifs"
        self.model_dir = self.run_dir / "final_model"

        for d in (
            self.checkpoint_dir,
            self.metrics_dir,
            self.plot_dir,
            self.eval_plot_dir,
            self.analysis_dir,
            self.gif_dir,
            self.model_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # --- Writers ---

    def save_config(self, config: Dict[str, Any]) -> None:
        import yaml

        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save_metrics_csv(self, name: str, data: list) -> None:
        if not data:
            return
        import csv

        path = self.metrics_dir / f"{name}.csv"
        keys = data[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

    def save_plot(self, fig, category: str, name: str, fmt: str = "png") -> None:
        dir_map = {
            "training": self.plot_dir,
            "evaluation": self.eval_plot_dir,
            "analysis": self.analysis_dir,
        }
        target = dir_map.get(category, self.plot_dir)
        fig.savefig(target / f"{name}.{fmt}", dpi=150, bbox_inches="tight")

    def save_final_model(self, agent) -> None:
        agent.save(self.model_dir / "agent_state.pkl")
        info = {"agent_name": agent.name}
        with open(self.model_dir / "agent_info.json", "w") as f:
            json.dump(info, f, indent=2)
