#!/usr/bin/env python3
"""CLI entry point for running RL experiments.

Usage:
    python run.py --config configs/q_learning_frozenlake.yaml
    python run.py --compare configs/q_learning_frozenlake.yaml configs/sarsa_frozenlake.yaml
    python run.py --config configs/dqn_cartpole.yaml --multi-seed
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infrastructure.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning Experiment Runner"
    )
    parser.add_argument("--config", type=str, help="Path to experiment YAML config")
    parser.add_argument(
        "--compare", nargs="+", type=str, help="Compare multiple configs"
    )
    parser.add_argument(
        "--multi-seed", action="store_true", help="Run with multiple seeds"
    )
    args = parser.parse_args()

    if args.compare:
        ExperimentRunner.compare(args.compare)
    elif args.config:
        runner = ExperimentRunner(args.config)
        if args.multi_seed:
            runner.run_multi_seed()
        else:
            runner.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
