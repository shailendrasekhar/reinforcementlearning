"""Plotting utilities for training and evaluation visualization."""

from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from core.types import TrainingResult, EvaluationResult


class Visualizer:
    """Generate and (optionally) save plots."""

    @staticmethod
    def plot_training_progress(
        result: TrainingResult,
        window: int = 100,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result.episode_rewards, alpha=0.3, color="steelblue", label="Episode reward")
        if len(result.episode_rewards) >= window:
            ma = result.moving_average(window)
            offset = len(result.episode_rewards) - len(ma)
            ax.plot(range(offset, offset + len(ma)), ma,
                    color="darkblue", linewidth=2, label=f"MA-{window}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Training Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_loss(
        result: TrainingResult,
        save_path: Optional[Path] = None,
    ) -> Optional[plt.Figure]:
        if not result.loss_history:
            return None
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result.loss_history, alpha=0.5, color="crimson")
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_epsilon(
        result: TrainingResult,
        save_path: Optional[Path] = None,
    ) -> Optional[plt.Figure]:
        if not result.epsilon_history:
            return None
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result.epsilon_history, color="green")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title("Exploration Decay")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_evaluation(
        result: EvaluationResult,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(result.episode_rewards, bins=20, color="teal", alpha=0.7, edgecolor="black")
        ax.axvline(result.mean_reward, color="red", linestyle="--",
                   label=f"Mean: {result.mean_reward:.2f}")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Frequency")
        ax.set_title("Evaluation Reward Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_tabular_policy(
        analysis_data: Dict[str, Any],
        grid_shape=(4, 4),
        save_path: Optional[Path] = None,
    ) -> Optional[plt.Figure]:
        """Plot Q-table policy grid for tabular agents."""
        if analysis_data.get("type") != "tabular":
            return None

        q_table = analysis_data["q_table"]
        rows, cols = grid_shape
        arrows = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}

        fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))
        for state in range(rows * cols):
            r, c = divmod(state, cols)
            if state in q_table:
                best = int(np.argmax(q_table[state]))
                arrow = arrows.get(best, "?")
                val = float(np.max(q_table[state]))
            else:
                arrow = "\u00b7"
                val = 0.0
            ax.text(c + 0.5, rows - r - 0.5, arrow,
                    ha="center", va="center", fontsize=18)
            ax.text(c + 0.5, rows - r - 0.2, f"{val:.2f}",
                    ha="center", va="center", fontsize=8, color="gray")

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(range(cols + 1))
        ax.set_yticks(range(rows + 1))
        ax.grid(True)
        ax.set_title("Learned Policy")
        ax.set_aspect("equal")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def close_all():
        plt.close("all")
