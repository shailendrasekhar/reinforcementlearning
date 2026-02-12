"""Training metrics logger — converts result objects to exportable records."""

from typing import Any, Dict, List

from core.types import TrainingResult, EvaluationResult


class Logger:
    """Converts result dataclasses to list-of-dicts for CSV / display."""

    @staticmethod
    def training_to_records(result: TrainingResult) -> List[Dict[str, Any]]:
        records = []
        for i in range(result.num_episodes):
            row: Dict[str, Any] = {
                "episode": i,
                "reward": result.episode_rewards[i],
                "length": result.episode_lengths[i],
            }
            if i < len(result.epsilon_history):
                row["epsilon"] = result.epsilon_history[i]
            if i < len(result.loss_history):
                row["loss"] = result.loss_history[i]
            for key, vals in result.extra_metrics.items():
                if i < len(vals):
                    row[key] = vals[i]
            records.append(row)
        return records

    @staticmethod
    def evaluation_to_records(result: EvaluationResult) -> List[Dict[str, Any]]:
        records = []
        for i in range(len(result.episode_rewards)):
            records.append({
                "episode": i,
                "reward": result.episode_rewards[i],
                "length": result.episode_lengths[i],
                "success": result.successes[i] if i < len(result.successes) else None,
            })
        return records

    @staticmethod
    def print_summary(train: TrainingResult, eval_result: EvaluationResult) -> None:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"  Training episodes:   {train.num_episodes}")
        if train.episode_rewards:
            last = train.episode_rewards[-100:]
            print(f"  Final avg reward:    {sum(last)/len(last):.2f} (last 100)")
        print(f"  Eval mean reward:    {eval_result.mean_reward:.2f} +/- {eval_result.std_reward:.2f}")
        print(f"  Eval success rate:   {eval_result.success_rate:.1f}%")
        print(f"  Eval mean length:    {eval_result.mean_length:.1f}")
        print("=" * 60)
