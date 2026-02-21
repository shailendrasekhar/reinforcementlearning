"""
Streamlit UI for the Reinforcement Learning Framework — Modern Edition v3.

Run with:
    streamlit run app.py
"""

import sys
import time
import random
import json
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.types import Transition, TrainingResult, EvaluationResult
from core.agent import BaseAgent
from core.environment import EnvWrapper
from core.evaluator import Evaluator
from environments.registry import EnvRegistry
from agents.registry import AgentRegistry
from infrastructure.visualizer import Visualizer
from infrastructure.artifact_store import ArtifactStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_GYM_IDS = {
    "frozenlake": "FrozenLake-v1",
    "cartpole": "CartPole-v1",
    "mountaincar": "MountainCar-v0",
    "pendulum": "Pendulum-v1",
    "cliffwalking": "CliffWalking-v1",
    "acrobot": "Acrobot-v1",
    "lunarlander": "LunarLander-v3",
}

ENV_INFO = {
    "frozenlake": {
        "label": "FrozenLake",
        "icon": "🧊",
        "desc": "Navigate a frozen grid to reach the goal without falling in holes.",
        "type": "Discrete",
    },
    "cliffwalking": {
        "label": "CliffWalking",
        "icon": "🏔️",
        "desc": "Walk along a cliff edge — shortest safe path to the goal.",
        "type": "Discrete",
    },
    "acrobot": {
        "label": "Acrobot",
        "icon": "🤸",
        "desc": "Swing a double-pendulum to reach above the line.",
        "type": "Discrete",
    },
    "cartpole": {
        "label": "CartPole",
        "icon": "🏗️",
        "desc": "Balance a pole on a moving cart for as long as possible.",
        "type": "Discrete",
    },
    "mountaincar": {
        "label": "MountainCar",
        "icon": "⛰️",
        "desc": "Build momentum to drive a car up a steep mountain.",
        "type": "Discrete",
    },
    "lunarlander": {
        "label": "LunarLander",
        "icon": "🚀",
        "desc": "Land a spacecraft safely on the landing pad.",
        "type": "Discrete",
    },
    "pendulum": {
        "label": "Pendulum",
        "icon": "🔄",
        "desc": "Swing up and balance an inverted pendulum. (Continuous)",
        "type": "Continuous",
    },
}

ALGO_INFO = {
    "q_learning": {
        "label": "Q-Learning",
        "icon": "📊",
        "family": "Value-Based",
        "desc": "Tabular off-policy TD control. Best for small discrete state spaces.",
    },
    "sarsa": {
        "label": "SARSA",
        "icon": "📈",
        "family": "Value-Based",
        "desc": "Tabular on-policy TD control. More conservative than Q-Learning.",
    },
    "dqn": {
        "label": "DQN",
        "icon": "🧠",
        "family": "Value-Based",
        "desc": "Deep Q-Network with experience replay & target network.",
    },
    "reinforce": {
        "label": "Policy Gradient",
        "icon": "🎯",
        "family": "Policy Gradient",
        "desc": "Monte Carlo policy gradient (REINFORCE) with optional baseline.",
    },
    "a2c": {
        "label": "A2C",
        "icon": "🎭",
        "family": "Actor-Critic",
        "desc": "Synchronous Advantage Actor-Critic with shared backbone.",
    },
    "a3c": {
        "label": "A3C",
        "icon": "⚡",
        "family": "Actor-Critic",
        "desc": "Asynchronous Advantage Actor-Critic with n-step returns.",
    },
    "ppo": {
        "label": "PPO",
        "icon": "🏆",
        "family": "Actor-Critic",
        "desc": "Proximal Policy Optimization with clipped surrogate objective.",
    },
    "td3": {
        "label": "TD3",
        "icon": "🔧",
        "family": "Actor-Critic",
        "desc": "Twin Delayed DDPG — reduces overestimation in continuous control.",
    },
    "ddpg": {
        "label": "DDPG",
        "icon": "🎮",
        "family": "Actor-Critic",
        "desc": "Deep Deterministic Policy Gradient for continuous actions.",
    },
    "sac": {
        "label": "SAC",
        "icon": "🌡️",
        "family": "Actor-Critic",
        "desc": "Soft Actor-Critic with entropy regularization and auto-tuning.",
    },
}

# ---- Full hyperparameter definitions including network config ----

ALGO_HYPERPARAMS = {
    "q_learning": {
        "learning_rate": {"type": "slider", "min": 0.001, "max": 1.0, "default": 0.1, "step": 0.001, "help": "Step size for Q-value updates", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.8, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — how much to weight future rewards", "group": "Core"},
        "epsilon_start": {"type": "slider", "min": 0.1, "max": 1.0, "default": 1.0, "step": 0.05, "help": "Initial exploration rate", "group": "Exploration"},
        "epsilon_end": {"type": "slider", "min": 0.001, "max": 0.5, "default": 0.01, "step": 0.001, "help": "Minimum exploration rate", "group": "Exploration"},
        "decay_steps": {"type": "slider", "min": 1000, "max": 1000000, "default": 500000, "step": 1000, "help": "Steps over which epsilon decays", "group": "Exploration"},
    },
    "sarsa": {
        "learning_rate": {"type": "slider", "min": 0.001, "max": 1.0, "default": 0.1, "step": 0.001, "help": "Step size for Q-value updates", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.8, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — how much to weight future rewards", "group": "Core"},
        "epsilon_start": {"type": "slider", "min": 0.1, "max": 1.0, "default": 1.0, "step": 0.05, "help": "Initial exploration rate", "group": "Exploration"},
        "epsilon_end": {"type": "slider", "min": 0.001, "max": 0.5, "default": 0.01, "step": 0.001, "help": "Minimum exploration rate", "group": "Exploration"},
        "decay_steps": {"type": "slider", "min": 1000, "max": 1000000, "default": 500000, "step": 1000, "help": "Steps over which epsilon decays", "group": "Exploration"},
    },
    "dqn": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Adam optimizer learning rate", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor for future rewards", "group": "Core"},
        "batch_size": {"type": "select", "options": [16, 32, 64, 128, 256], "default": 64, "help": "Minibatch size for network updates", "group": "Core"},
        "buffer_size": {"type": "select", "options": [5000, 10000, 50000, 100000, 200000], "default": 100000, "help": "Replay buffer capacity", "group": "Replay Buffer"},
        "warmup_steps": {"type": "slider", "min": 100, "max": 10000, "default": 1000, "step": 100, "help": "Random actions before training starts", "group": "Replay Buffer"},
        "train_frequency": {"type": "select", "options": [1, 2, 4, 8], "default": 4, "help": "Steps between gradient updates", "group": "Core"},
        "target_update_frequency": {"type": "select", "options": [100, 500, 1000, 2000, 5000], "default": 1000, "help": "Steps between target network hard updates", "group": "Target Network"},
        "epsilon_start": {"type": "slider", "min": 0.5, "max": 1.0, "default": 1.0, "step": 0.05, "help": "Initial exploration rate", "group": "Exploration"},
        "epsilon_end": {"type": "slider", "min": 0.001, "max": 0.2, "default": 0.01, "step": 0.001, "help": "Minimum exploration rate", "group": "Exploration"},
        "epsilon_decay_steps": {"type": "slider", "min": 1000, "max": 200000, "default": 50000, "step": 1000, "help": "Steps over which epsilon decays", "group": "Exploration"},
        "gradient_clip": {"type": "slider", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1, "help": "Gradient clipping max norm (0 = disabled)", "group": "Optimization"},
        "hidden_layer_1": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Number of neurons in hidden layer 1", "group": "Network Architecture"},
        "hidden_layer_2": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Number of neurons in hidden layer 2", "group": "Network Architecture"},
        "activation_fn": {"type": "select", "options": ["relu", "tanh", "leaky_relu"], "default": "relu", "help": "Activation function for hidden layers", "group": "Network Architecture"},
        "optimizer": {"type": "select", "options": ["adam", "rmsprop", "sgd"], "default": "adam", "help": "Optimizer for gradient descent", "group": "Optimization"},
    },
    "reinforce": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Adam optimizer learning rate", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor", "group": "Core"},
        "use_baseline": {"type": "bool", "default": True, "help": "Subtract running mean return to reduce variance", "group": "Core"},
        "hidden_layer_1": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Number of neurons in hidden layer 1", "group": "Network Architecture"},
        "hidden_layer_2": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Number of neurons in hidden layer 2", "group": "Network Architecture"},
        "activation_fn": {"type": "select", "options": ["relu", "tanh", "leaky_relu"], "default": "relu", "help": "Activation function", "group": "Network Architecture"},
        "optimizer": {"type": "select", "options": ["adam", "rmsprop", "sgd"], "default": "adam", "help": "Optimizer", "group": "Optimization"},
    },
    "a2c": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Adam optimizer learning rate", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor", "group": "Core"},
        "value_coeff": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "help": "Weight of value loss in total loss", "group": "Loss Weights"},
        "entropy_coeff": {"type": "slider", "min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001, "help": "Entropy bonus coefficient for exploration", "group": "Loss Weights"},
        "gradient_clip": {"type": "slider", "min": 0.0, "max": 10.0, "default": 0.5, "step": 0.1, "help": "Max gradient norm (0 = disabled)", "group": "Optimization"},
        "hidden_layer_1": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Neurons in hidden layer 1", "group": "Network Architecture"},
        "hidden_layer_2": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Neurons in hidden layer 2", "group": "Network Architecture"},
        "activation_fn": {"type": "select", "options": ["relu", "tanh", "leaky_relu"], "default": "relu", "help": "Activation function", "group": "Network Architecture"},
        "optimizer": {"type": "select", "options": ["adam", "rmsprop", "sgd"], "default": "adam", "help": "Optimizer", "group": "Optimization"},
    },
    "a3c": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Adam optimizer learning rate", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor", "group": "Core"},
        "value_coeff": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "help": "Weight of value loss in total loss", "group": "Loss Weights"},
        "entropy_coeff": {"type": "slider", "min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001, "help": "Entropy bonus for exploration", "group": "Loss Weights"},
        "n_steps": {"type": "select", "options": [3, 5, 10, 20], "default": 5, "help": "Number of steps for n-step return bootstrapping", "group": "Core"},
        "gradient_clip": {"type": "slider", "min": 0.0, "max": 10.0, "default": 0.5, "step": 0.1, "help": "Max gradient norm (0 = disabled)", "group": "Optimization"},
        "hidden_layer_1": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Neurons in hidden layer 1", "group": "Network Architecture"},
        "hidden_layer_2": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Neurons in hidden layer 2", "group": "Network Architecture"},
        "activation_fn": {"type": "select", "options": ["relu", "tanh", "leaky_relu"], "default": "relu", "help": "Activation function", "group": "Network Architecture"},
        "optimizer": {"type": "select", "options": ["adam", "rmsprop", "sgd"], "default": "adam", "help": "Optimizer", "group": "Optimization"},
    },
    "ppo": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.0003, "step": 0.00001, "help": "Adam optimizer learning rate", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor", "group": "Core"},
        "clip_epsilon": {"type": "slider", "min": 0.05, "max": 0.5, "default": 0.2, "step": 0.01, "help": "PPO clipping range for the surrogate objective", "group": "PPO"},
        "ppo_epochs": {"type": "select", "options": [3, 5, 10, 15, 20], "default": 10, "help": "SGD epochs per rollout", "group": "PPO"},
        "batch_size": {"type": "select", "options": [32, 64, 128, 256], "default": 64, "help": "Minibatch size for PPO updates", "group": "Core"},
        "rollout_length": {"type": "select", "options": [128, 256, 512, 1024, 2048, 4096], "default": 2048, "help": "Steps collected before each PPO update", "group": "PPO"},
        "gae_lambda": {"type": "slider", "min": 0.8, "max": 1.0, "default": 0.95, "step": 0.01, "help": "GAE lambda for advantage estimation", "group": "PPO"},
        "value_coeff": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "help": "Weight of value loss", "group": "Loss Weights"},
        "entropy_coeff": {"type": "slider", "min": 0.0, "max": 0.1, "default": 0.01, "step": 0.001, "help": "Entropy bonus coefficient", "group": "Loss Weights"},
        "gradient_clip": {"type": "slider", "min": 0.0, "max": 10.0, "default": 0.5, "step": 0.1, "help": "Max gradient norm (0 = disabled)", "group": "Optimization"},
        "hidden_layer_1": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Neurons in hidden layer 1", "group": "Network Architecture"},
        "hidden_layer_2": {"type": "select", "options": [32, 64, 128, 256, 512], "default": 128, "help": "Neurons in hidden layer 2", "group": "Network Architecture"},
        "activation_fn": {"type": "select", "options": ["relu", "tanh", "leaky_relu"], "default": "relu", "help": "Activation function", "group": "Network Architecture"},
        "optimizer": {"type": "select", "options": ["adam", "rmsprop", "sgd"], "default": "adam", "help": "Optimizer", "group": "Optimization"},
    },
    "ddpg": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Learning rate for actor & critic", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor", "group": "Core"},
        "tau": {"type": "slider", "min": 0.001, "max": 0.05, "default": 0.005, "step": 0.001, "help": "Soft target update rate", "group": "Target Network"},
        "batch_size": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Minibatch size for updates", "group": "Core"},
        "buffer_size": {"type": "select", "options": [50000, 100000, 200000, 500000], "default": 100000, "help": "Replay buffer capacity", "group": "Replay Buffer"},
        "warmup_steps": {"type": "slider", "min": 100, "max": 10000, "default": 1000, "step": 100, "help": "Random actions before training starts", "group": "Replay Buffer"},
        "noise_type": {"type": "select", "options": ["ou", "gaussian"], "default": "ou", "help": "Exploration noise type (OU or Gaussian)", "group": "Exploration"},
        "noise_sigma": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.2, "step": 0.01, "help": "Noise standard deviation / sigma", "group": "Exploration"},
        "hidden_layer_1": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Neurons in hidden layer 1", "group": "Network Architecture"},
        "hidden_layer_2": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Neurons in hidden layer 2", "group": "Network Architecture"},
        "activation_fn": {"type": "select", "options": ["relu", "tanh", "leaky_relu"], "default": "relu", "help": "Activation function", "group": "Network Architecture"},
        "optimizer": {"type": "select", "options": ["adam", "rmsprop", "sgd"], "default": "adam", "help": "Optimizer", "group": "Optimization"},
    },
    "td3": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Learning rate for actor & critic", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor", "group": "Core"},
        "tau": {"type": "slider", "min": 0.001, "max": 0.05, "default": 0.005, "step": 0.001, "help": "Soft target update rate", "group": "Target Network"},
        "batch_size": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Minibatch size for updates", "group": "Core"},
        "buffer_size": {"type": "select", "options": [50000, 100000, 200000, 500000], "default": 100000, "help": "Replay buffer capacity", "group": "Replay Buffer"},
        "warmup_steps": {"type": "slider", "min": 100, "max": 10000, "default": 1000, "step": 100, "help": "Random actions before training starts", "group": "Replay Buffer"},
        "policy_delay": {"type": "select", "options": [1, 2, 3, 4], "default": 2, "help": "Actor update delay (every N critic updates)", "group": "TD3"},
        "target_noise_std": {"type": "slider", "min": 0.05, "max": 0.5, "default": 0.2, "step": 0.01, "help": "Target policy smoothing noise std", "group": "TD3"},
        "target_noise_clip": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "help": "Clipping range for target noise", "group": "TD3"},
        "noise_type": {"type": "select", "options": ["gaussian", "ou"], "default": "gaussian", "help": "Exploration noise type", "group": "Exploration"},
        "noise_sigma": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01, "help": "Exploration noise sigma", "group": "Exploration"},
        "hidden_layer_1": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Neurons in hidden layer 1", "group": "Network Architecture"},
        "hidden_layer_2": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Neurons in hidden layer 2", "group": "Network Architecture"},
        "activation_fn": {"type": "select", "options": ["relu", "tanh", "leaky_relu"], "default": "relu", "help": "Activation function", "group": "Network Architecture"},
        "optimizer": {"type": "select", "options": ["adam", "rmsprop", "sgd"], "default": "adam", "help": "Optimizer", "group": "Optimization"},
    },
    "sac": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.0003, "step": 0.00001, "help": "Adam learning rate for actor & critic", "group": "Core"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor", "group": "Core"},
        "tau": {"type": "slider", "min": 0.001, "max": 0.05, "default": 0.005, "step": 0.001, "help": "Soft target update rate", "group": "Target Network"},
        "batch_size": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Minibatch size for updates", "group": "Core"},
        "buffer_size": {"type": "select", "options": [50000, 100000, 200000, 500000], "default": 100000, "help": "Replay buffer capacity", "group": "Replay Buffer"},
        "warmup_steps": {"type": "slider", "min": 100, "max": 10000, "default": 1000, "step": 100, "help": "Random actions before training starts", "group": "Replay Buffer"},
        "init_alpha": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.2, "step": 0.01, "help": "Initial entropy temperature", "group": "SAC Entropy"},
        "auto_entropy": {"type": "bool", "default": True, "help": "Automatically tune entropy coefficient", "group": "SAC Entropy"},
        "hidden_layer_1": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Neurons in hidden layer 1", "group": "Network Architecture"},
        "hidden_layer_2": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Neurons in hidden layer 2", "group": "Network Architecture"},
        "activation_fn": {"type": "select", "options": ["relu", "tanh", "leaky_relu"], "default": "relu", "help": "Activation function", "group": "Network Architecture"},
        "optimizer": {"type": "select", "options": ["adam", "rmsprop", "sgd"], "default": "adam", "help": "Optimizer", "group": "Optimization"},
    },
}

# Which algorithms work with which environments
COMPATIBLE = {
    "frozenlake":    ["q_learning", "sarsa", "dqn"],
    "cliffwalking":  ["q_learning", "sarsa", "dqn"],
    "acrobot":       ["dqn", "reinforce", "a2c", "a3c", "ppo"],
    "cartpole":      ["dqn", "reinforce", "a2c", "a3c", "ppo"],
    "mountaincar":   ["q_learning", "sarsa", "dqn"],
    "lunarlander":   ["dqn", "reinforce", "a2c", "a3c", "ppo"],
    "pendulum":      ["ddpg", "td3", "sac"],
}

DISCRETE_ALGOS = {"q_learning", "sarsa", "dqn", "reinforce", "a2c", "a3c", "ppo"}
CONTINUOUS_ALGOS = {"ddpg", "td3", "sac"}

SUCCESS_THRESHOLDS = {
    "frozenlake": 0.0,
    "cliffwalking": -100.0,
    "acrobot": -100.0,
    "cartpole": 195.0,
    "mountaincar": -110.0,
    "lunarlander": 200.0,
    "pendulum": -300.0,
}

DEFAULT_EPISODES = {
    "frozenlake": 10000, "cliffwalking": 5000, "acrobot": 500,
    "cartpole": 500, "mountaincar": 10000, "lunarlander": 1000,
    "pendulum": 200,
}
DEFAULT_STEPS = {
    "frozenlake": 100, "cliffwalking": 200, "acrobot": 500,
    "cartpole": 500, "mountaincar": 200, "lunarlander": 1000,
    "pendulum": 200,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def build_agent_params(algo: str, hp: dict) -> dict:
    """Convert UI hyperparameters into agent constructor params."""
    params = {}

    for key in ("learning_rate", "discount_factor", "batch_size", "buffer_size",
                "warmup_steps", "train_frequency", "target_update_frequency",
                "clip_epsilon", "ppo_epochs", "rollout_length", "gae_lambda",
                "value_coeff", "entropy_coeff", "tau", "init_alpha",
                "auto_entropy", "use_baseline", "policy_delay",
                "target_noise_std", "target_noise_clip", "n_steps"):
        if key in hp:
            params[key] = hp[key]

    gc = hp.get("gradient_clip", 0.0)
    if gc > 0:
        params["gradient_clip"] = gc

    # Network architecture from hyperparams
    h1 = hp.get("hidden_layer_1", 128)
    h2 = hp.get("hidden_layer_2", 128)
    params["hidden_dims"] = [h1, h2]
    params["device"] = "auto"

    if algo in ("q_learning", "sarsa"):
        params["exploration"] = {
            "strategy": "epsilon_greedy",
            "params": {
                "epsilon_start": hp.get("epsilon_start", 1.0),
                "epsilon_end": hp.get("epsilon_end", 0.01),
                "decay_type": "linear",
                "decay_steps": int(hp.get("decay_steps", 500000)),
            },
        }
    elif algo == "dqn":
        params["exploration"] = {
            "strategy": "epsilon_greedy",
            "params": {
                "epsilon_start": hp.get("epsilon_start", 1.0),
                "epsilon_end": hp.get("epsilon_end", 0.01),
                "decay_type": "linear",
                "decay_steps": int(hp.get("epsilon_decay_steps", 50000)),
            },
        }
    elif algo in ("ddpg", "td3"):
        noise_type = hp.get("noise_type", "ou" if algo == "ddpg" else "gaussian")
        sigma = hp.get("noise_sigma", 0.2)
        params["noise"] = {"type": noise_type, "sigma": sigma}
        if noise_type == "ou":
            params["noise"]["theta"] = 0.15
        params["max_action"] = 2.0
    elif algo == "sac":
        params["max_action"] = 2.0  # Pendulum uses [-2, 2]

    return params


def create_env(env_name: str) -> EnvWrapper:
    env_params = {}
    if env_name == "frozenlake":
        env_params = {"map_name": "4x4", "is_slippery": False}
    return EnvRegistry.create(env_name, **env_params)


def get_env_kwargs(env_name: str) -> dict:
    if env_name == "frozenlake":
        return {"map_name": "4x4", "is_slippery": False}
    return {}


def train_with_live_updates(
    env: EnvWrapper,
    agent: BaseAgent,
    num_episodes: int,
    max_steps: int,
    chart_placeholder,
    metrics_placeholder,
    progress_bar,
    stop_flag_key: str = "_stop_training",
) -> TrainingResult:
    result = TrainingResult()
    agent.on_training_start({"num_episodes": num_episodes})
    update_interval = max(1, num_episodes // 200)

    for episode in range(num_episodes):
        if st.session_state.get(stop_flag_key, False):
            break

        state = env.reset()
        agent.on_episode_start()
        total_reward = 0.0
        step_metrics = {}

        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            transition = Transition(
                state=state, action=action, reward=reward,
                next_state=next_state, done=done, info=info,
            )
            metrics = agent.update(transition)
            if metrics:
                step_metrics.update(metrics)

            total_reward += reward
            state = next_state
            if done:
                break

        end_metrics = agent.on_episode_end(total_reward)
        step_metrics.update(end_metrics)

        result.episode_rewards.append(total_reward)
        result.episode_lengths.append(step + 1)

        eps = agent.epsilon
        if eps is not None:
            result.epsilon_history.append(eps)
        if "loss" in step_metrics:
            result.loss_history.append(step_metrics["loss"])

        progress_bar.progress(
            (episode + 1) / num_episodes,
            text=f"Episode {episode + 1}/{num_episodes}"
        )

        if (episode + 1) % update_interval == 0 or episode == num_episodes - 1:
            _update_live_chart(chart_placeholder, result)
            _update_live_metrics(metrics_placeholder, result, episode + 1, num_episodes)

    agent.on_training_end()
    return result


def _update_live_chart(placeholder, result: TrainingResult):
    fig, ax = plt.subplots(figsize=(10, 4))
    rewards = result.episode_rewards
    ax.plot(rewards, alpha=0.25, color="#6C63FF", linewidth=0.5)

    window = min(100, max(1, len(rewards) // 5))
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        ma = np.convolve(rewards, kernel, mode="valid")
        offset = len(rewards) - len(ma)
        ax.plot(range(offset, offset + len(ma)), ma,
                color="#3D348B", linewidth=2, label=f"MA-{window}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Reward (Live)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    placeholder.pyplot(fig)
    plt.close(fig)


def _update_live_metrics(placeholder, result: TrainingResult, episode: int, total: int):
    r = result.episode_rewards
    recent = r[-100:] if len(r) >= 100 else r
    avg = np.mean(recent)
    best = max(r) if r else 0.0

    cols = placeholder.columns(4)
    cols[0].metric("Episode", f"{episode}/{total}")
    cols[1].metric("Avg Reward (100)", f"{avg:.2f}")
    cols[2].metric("Best Episode", f"{best:.2f}")

    if result.epsilon_history:
        cols[3].metric("Epsilon", f"{result.epsilon_history[-1]:.4f}")
    elif result.loss_history:
        cols[3].metric("Loss", f"{result.loss_history[-1]:.4f}")
    else:
        cols[3].metric("Std (100)", f"{np.std(recent):.2f}")


# ---------------------------------------------------------------------------
# Output Browser — Sidebar
# ---------------------------------------------------------------------------

def render_output_browser():
    """Render the output folder browser in the sidebar.
    Structure: outputs/<env_name>/<algo_name>/<timestamp>/
    """
    outputs_root = Path("outputs")
    if not outputs_root.exists():
        st.sidebar.info("No experiments yet.")
        return

    # Collect tree
    env_dirs = sorted([d for d in outputs_root.iterdir() if d.is_dir()])
    if not env_dirs:
        st.sidebar.info("No experiments yet.")
        return

    for env_dir in env_dirs:
        env_key = env_dir.name
        env_label = ENV_INFO.get(env_key, {}).get("label", env_key)
        env_icon = ENV_INFO.get(env_key, {}).get("icon", "📁")
        algo_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir()])
        if not algo_dirs:
            continue

        with st.sidebar.expander(f"{env_icon} {env_label}", expanded=False):
            for algo_dir in algo_dirs:
                algo_key = algo_dir.name
                algo_label = ALGO_INFO.get(algo_key, {}).get("label", algo_key)
                algo_icon = ALGO_INFO.get(algo_key, {}).get("icon", "🔹")
                run_dirs = sorted([d for d in algo_dir.iterdir() if d.is_dir()], reverse=True)
                if not run_dirs:
                    continue

                st.markdown(f"**{algo_icon} {algo_label}**")
                for run_dir in run_dirs[:5]:  # Show last 5 runs
                    timestamp = run_dir.name
                    btn_key = f"load_{env_key}_{algo_key}_{timestamp}"
                    if st.button(f"🕐 {timestamp}", key=btn_key, width="stretch"):  # sidebar btn
                        st.session_state["_view_run"] = str(run_dir)
                        st.rerun()
                if len(run_dirs) > 5:
                    st.caption(f"+{len(run_dirs) - 5} more runs")
                st.markdown("---")


def render_run_viewer():
    """Display saved experiment artifacts when a run is selected."""
    run_path = Path(st.session_state.get("_view_run", ""))
    if not run_path.exists():
        return False

    parts = run_path.parts
    # outputs / env / algo / timestamp
    idx = list(parts).index("outputs")
    env_key = parts[idx + 1] if idx + 1 < len(parts) else "?"
    algo_key = parts[idx + 2] if idx + 2 < len(parts) else "?"
    timestamp = parts[idx + 3] if idx + 3 < len(parts) else "?"

    env_label = ENV_INFO.get(env_key, {}).get("label", env_key)
    algo_label = ALGO_INFO.get(algo_key, {}).get("label", algo_key)
    env_icon = ENV_INFO.get(env_key, {}).get("icon", "")
    algo_icon = ALGO_INFO.get(algo_key, {}).get("icon", "")

    st.markdown(f"""
    <div class="section-header">
        📂 Viewing: {env_icon} {env_label} / {algo_icon} {algo_label} / {timestamp}
    </div>
    """, unsafe_allow_html=True)

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Lab", width="stretch"):  # viewer back btn
            del st.session_state["_view_run"]
            st.rerun()

    # Metadata
    meta_path = run_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        mcols = st.columns(4)
        mcols[0].metric("Training Time", f"{meta.get('training_time_s', 0):.1f}s")
        mcols[1].metric("Episodes", f"{meta.get('total_episodes', '?')}")
        mcols[2].metric("Mean Reward", f"{meta.get('eval_mean_reward', 0):.2f}")
        mcols[3].metric("Success Rate", f"{meta.get('eval_success_rate', 0):.1f}%")

    # Plots
    plot_dirs = [
        ("📈 Training Plots", run_path / "plots" / "training"),
        ("🏆 Evaluation Plots", run_path / "plots" / "evaluation"),
        ("🗺️ Analysis Plots", run_path / "plots" / "analysis"),
    ]
    for title, pdir in plot_dirs:
        if pdir.exists():
            images = sorted(pdir.glob("*.png"))
            if images:
                st.markdown(f"#### {title}")
                cols = st.columns(min(len(images), 2))
                for i, img in enumerate(images):
                    cols[i % 2].image(str(img), caption=img.stem, width="stretch")

    # GIFs — full width
    gif_dir = run_path / "gifs"
    if gif_dir.exists():
        gifs = sorted(gif_dir.glob("*.gif"))
        if gifs:
            st.markdown("#### 🎬 Agent Performance")
            for gif in gifs:
                st.image(str(gif), caption=gif.stem, width="stretch")

    # Config
    cfg_path = run_path / "config.yaml"
    if cfg_path.exists():
        with st.expander("📄 Config", expanded=False):
            st.code(cfg_path.read_text(), language="yaml")

    return True


# ---------------------------------------------------------------------------
# CSS Styling
# ---------------------------------------------------------------------------

def inject_custom_css():
    """Inject CSS that styles ONLY our custom HTML elements.

    All native Streamlit components (metrics, alerts, expanders, labels,
    buttons, sidebar text, …) are left untouched so they respect whatever
    theme (light / dark) the user has active.  Streamlit exposes CSS
    custom-properties that we reference in our own elements so cards,
    headers, and badges adapt automatically.

    Key Streamlit CSS variables we rely on:
        --text-color              main body text
        --secondary-text-color    muted / secondary text
        --background-color        page background
        --secondary-background-color   card / sidebar bg
        --primary-color           accent (buttons, links)
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    /* ================================================================
       CUSTOM HTML elements — these are the ONLY things we colour.
       Everything else is Streamlit-native and inherits the active theme.
       ================================================================ */

    /* Hero header — always on gradient, always white text */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: #fff !important;
    }
    .hero-header h1 {
        margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;
        color: #fff !important;
    }
    .hero-header p {
        margin: 0.5rem 0 0 0; opacity: 0.92; font-size: 1rem;
        color: #fff !important;
    }

    /* Section separators — track the active theme text colour */
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: var(--text-color) !important;
        margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid color-mix(in srgb, var(--primary-color) 30%, transparent);
    }

    /* ---- CLICKABLE CARDS (env + algo) ---- */
    .card-grid {
        display: flex; flex-wrap: wrap; gap: 12px; margin: 0.5rem 0 1rem 0;
    }

    .env-card, .algo-card {
        background: var(--secondary-background-color);
        border: 2px solid color-mix(in srgb, var(--text-color) 15%, transparent);
        border-radius: 14px;
        cursor: pointer;
        flex: 1;
        transition: all 0.25s ease;
        color: var(--text-color) !important;
    }
    .env-card { padding: 1rem 1.2rem; text-align: center; min-width: 120px; }
    .algo-card { padding: 0.8rem 1.2rem; min-width: 150px; }

    .env-card:hover, .algo-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 4px 16px color-mix(in srgb, var(--primary-color) 20%, transparent);
    }
    .env-card.selected, .algo-card.selected {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary-color) 35%, transparent),
                    0 4px 20px color-mix(in srgb, var(--primary-color) 25%, transparent);
        background: color-mix(in srgb, var(--primary-color) 10%, var(--secondary-background-color));
    }

    .env-card .icon { font-size: 2rem; }
    .env-card .label {
        font-weight: 700; font-size: 0.9rem; margin-top: 0.3rem;
        color: var(--text-color) !important;
    }
    .env-card .type-badge {
        font-size: 0.65rem;
        background: color-mix(in srgb, var(--primary-color) 12%, transparent);
        color: var(--primary-color) !important;
        border-radius: 4px; padding: 2px 6px; margin-top: 0.3rem;
        display: inline-block;
    }
    .env-card .desc {
        font-size: 0.7rem; margin-top: 0.25rem;
        color: var(--secondary-text-color, color-mix(in srgb, var(--text-color) 60%, transparent)) !important;
    }

    .algo-card .top-row { display: flex; align-items: center; gap: 0.6rem; }
    .algo-card .icon { font-size: 1.5rem; }
    .algo-card .label {
        font-weight: 700; font-size: 0.9rem;
        color: var(--text-color) !important;
    }
    .algo-card .family {
        font-size: 0.65rem; font-weight: 600;
        color: var(--primary-color) !important;
    }
    .algo-card .desc {
        font-size: 0.7rem; margin-top: 0.25rem;
        color: var(--secondary-text-color, color-mix(in srgb, var(--text-color) 60%, transparent)) !important;
    }

    /* Config card for HP groups */
    .config-card {
        background: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--primary-color) 20%, transparent);
        border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
        color: var(--text-color) !important;
    }
    .config-card h4 {
        margin: 0 0 0.6rem 0; font-size: 0.95rem;
        color: var(--primary-color) !important;
    }

    /* ================================================================
       LIGHT ENHANCEMENTS on Streamlit-native widgets.
       Only layout / border / shadow — NO colour overrides.
       ================================================================ */

    /* Metrics — subtle card styling, colours stay theme-native */
    [data-testid="stMetric"] {
        background: var(--secondary-background-color) !important;
        border: 1px solid color-mix(in srgb, var(--primary-color) 15%, transparent);
        border-radius: 12px; padding: 1rem;
    }

    /* Primary button — branded gradient, always white text */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6C63FF, #764ba2) !important;
        border: none !important; border-radius: 10px !important;
        padding: 0.7rem 2rem !important; font-weight: 700 !important;
        font-size: 1.05rem !important; letter-spacing: 0.5px;
        transition: all 0.2s ease !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4) !important;
        transform: translateY(-1px);
    }

    /* Stop button — red accent, always white text */
    .stop-btn > button {
        background: linear-gradient(135deg, #ff4444, #cc0000) !important;
        border: none !important; border-radius: 10px !important;
        color: #fff !important; font-weight: 700 !important;
    }
    .stop-btn > button:hover {
        box-shadow: 0 6px 20px rgba(255, 68, 68, 0.4) !important;
    }

    /* Progress bar accent */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6C63FF, #764ba2) !important;
    }

    /* Matplotlib / GIF images — ensure white plot background regardless of theme */
    .stImage img, [data-testid="stImage"] img {
        background: #fff;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI Components — Clickable cards
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="hero-header">
        <h1>🎮 Reinforcement Learning Lab</h1>
        <p>Click an environment and algorithm, tune every hyperparameter, train, and visualize — all in one place.</p>
    </div>
    """, unsafe_allow_html=True)


def render_env_selector() -> str:
    """Render clickable environment cards — no dropdown."""
    st.markdown('<div class="section-header">🌍 Choose Environment</div>', unsafe_allow_html=True)

    env_order = ["frozenlake", "cliffwalking", "acrobot", "cartpole", "mountaincar", "lunarlander", "pendulum"]
    env_names = [e for e in env_order if e in ENV_INFO]

    # Initialise selection
    if "_sel_env" not in st.session_state:
        st.session_state["_sel_env"] = "cartpole"

    current = st.session_state["_sel_env"]

    # Render HTML cards
    cards_html = '<div class="card-grid">'
    for key in env_names:
        info = ENV_INFO[key]
        sel_cls = "selected" if key == current else ""
        cards_html += f"""
        <div class="env-card {sel_cls}" id="env-{key}">
            <div class="icon">{info['icon']}</div>
            <div class="label">{info['label']}</div>
            <div class="type-badge">{info['type']}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # Actual buttons (small row) for click interaction
    cols = st.columns(len(env_names))
    for i, key in enumerate(env_names):
        info = ENV_INFO[key]
        with cols[i]:
            if key == current:
                st.button(f"✓ {info['label']}", key=f"envbtn_{key}", disabled=True, width="stretch")  # env btns
            else:
                if st.button(info['label'], key=f"envbtn_{key}", width="stretch"):  # env btns
                    st.session_state["_sel_env"] = key
                    # Reset algo selection when env changes
                    st.session_state["_sel_algo"] = None
                    st.rerun()

    return st.session_state["_sel_env"]


def render_algo_selector(env_name: str) -> str:
    """Render clickable algorithm cards — no dropdown."""
    st.markdown('<div class="section-header">🧪 Choose Algorithm</div>', unsafe_allow_html=True)

    compatible = COMPATIBLE.get(env_name, list(ALGO_INFO.keys()))
    algo_keys = [a for a in compatible if a in ALGO_INFO]

    # Initialise / validate selection
    if st.session_state.get("_sel_algo") not in algo_keys:
        st.session_state["_sel_algo"] = algo_keys[0] if algo_keys else None

    current = st.session_state["_sel_algo"]

    # Render HTML cards
    cards_html = '<div class="card-grid">'
    for key in algo_keys:
        info = ALGO_INFO[key]
        sel_cls = "selected" if key == current else ""
        cards_html += f"""
        <div class="algo-card {sel_cls}">
            <div class="top-row">
                <div class="icon">{info['icon']}</div>
                <div>
                    <div class="label">{info['label']}</div>
                    <div class="family">{info['family']}</div>
                </div>
            </div>
            <div class="desc">{info['desc']}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # Button row for click interaction
    num_cols = min(len(algo_keys), 5)
    rows_needed = (len(algo_keys) + num_cols - 1) // num_cols
    for row in range(rows_needed):
        start = row * num_cols
        end = min(start + num_cols, len(algo_keys))
        cols = st.columns(num_cols)
        for col_idx, key_idx in enumerate(range(start, end)):
            key = algo_keys[key_idx]
            info = ALGO_INFO[key]
            with cols[col_idx]:
                if key == current:
                    st.button(f"✓ {info['label']}", key=f"algobtn_{key}", disabled=True, width="stretch")  # algo btns
                else:
                    if st.button(info['label'], key=f"algobtn_{key}", width="stretch"):  # algo btns
                        st.session_state["_sel_algo"] = key
                        st.rerun()

    return st.session_state["_sel_algo"]


def render_hyperparameters(algo_name: str) -> dict:
    """Render ALL hyperparameters grouped by category."""
    st.markdown('<div class="section-header">⚙️ Hyperparameters</div>', unsafe_allow_html=True)

    algo_hp_defs = ALGO_HYPERPARAMS.get(algo_name, {})
    if not algo_hp_defs:
        st.info("No tunable hyperparameters for this algorithm.")
        return {}

    hp = {}
    info = ALGO_INFO.get(algo_name, {})

    # Group params by category
    groups = {}
    for param_name, spec in algo_hp_defs.items():
        group = spec.get("group", "General")
        groups.setdefault(group, []).append((param_name, spec))

    # Render each group in its own card
    for group_name, params in groups.items():
        st.markdown(f"""<div class="config-card"><h4>{'🔧' if group_name != 'Network Architecture' else '🏗️'} {group_name}</h4></div>""", unsafe_allow_html=True)

        col_count = 2 if len(params) > 1 else 1
        for row_start in range(0, len(params), col_count):
            cols = st.columns(col_count)
            for col_idx in range(col_count):
                idx = row_start + col_idx
                if idx >= len(params):
                    break
                param_name, spec = params[idx]
                with cols[col_idx]:
                    display_name = param_name.replace("_", " ").title()

                    if spec["type"] == "slider":
                        hp[param_name] = st.slider(
                            display_name,
                            min_value=spec["min"],
                            max_value=spec["max"],
                            value=spec["default"],
                            step=spec["step"],
                            help=spec.get("help", ""),
                            key=f"hp_{algo_name}_{param_name}",
                        )
                    elif spec["type"] == "select":
                        options = spec["options"]
                        default_idx = options.index(spec["default"]) if spec["default"] in options else 0
                        hp[param_name] = st.selectbox(
                            display_name,
                            options,
                            index=default_idx,
                            help=spec.get("help", ""),
                            key=f"hp_{algo_name}_{param_name}",
                        )
                    elif spec["type"] == "bool":
                        hp[param_name] = st.checkbox(
                            display_name,
                            value=spec["default"],
                            help=spec.get("help", ""),
                            key=f"hp_{algo_name}_{param_name}",
                        )
    return hp


def render_training_config(env_name: str) -> dict:
    st.markdown('<div class="section-header">📋 Training Configuration</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        num_episodes = st.number_input(
            "Number of Episodes", min_value=10, max_value=100000,
            value=DEFAULT_EPISODES.get(env_name, 1000), step=100,
            help="Total training episodes",
        )
    with col2:
        max_steps = st.number_input(
            "Max Steps / Episode", min_value=50, max_value=10000,
            value=DEFAULT_STEPS.get(env_name, 500), step=50,
            help="Maximum steps per episode before truncation",
        )
    with col3:
        seed = st.number_input(
            "Random Seed", min_value=0, max_value=99999,
            value=42, step=1, help="Seed for reproducibility",
        )
    with col4:
        eval_episodes = st.number_input(
            "Eval Episodes", min_value=5, max_value=500,
            value=50, step=5, help="Episodes for post-training evaluation",
        )

    record_gif = st.checkbox("🎬 Record evaluation GIF", value=True, help="Create an animated GIF of the trained agent")

    return {
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "seed": seed,
        "eval_episodes": eval_episodes,
        "record_gif": record_gif,
    }


# ---------------------------------------------------------------------------
# Experiment Execution
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict):
    env_name = cfg["env_name"]
    algo_name = cfg["algo_name"]
    hp = cfg["hyperparameters"]

    set_seed(cfg["seed"])
    env = create_env(env_name)

    store = ArtifactStore(
        base_dir="outputs",
        agent_name=algo_name,
        env_name=env_name,
    )
    store.save_config(cfg)

    env_info = {
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "is_discrete": env.is_discrete,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    agent_params = build_agent_params(algo_name, hp)
    agent = AgentRegistry.create(algo_name, env_info=env_info, **agent_params)

    # ---- Training Phase ----
    st.markdown('<div class="section-header">📈 Training</div>', unsafe_allow_html=True)

    # Stop button — placed before the training loop so it's visible during training
    stop_col1, stop_col2 = st.columns([5, 1])
    with stop_col2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("⏹ Stop Training", key="_stop_btn"):
            st.session_state["_stop_training"] = True
        st.markdown('</div>', unsafe_allow_html=True)

    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    progress_bar = st.progress(0, text="Starting training…")

    start_time = time.time()

    train_result = train_with_live_updates(
        env=env,
        agent=agent,
        num_episodes=cfg["num_episodes"],
        max_steps=cfg["max_steps"],
        chart_placeholder=chart_placeholder,
        metrics_placeholder=metrics_placeholder,
        progress_bar=progress_bar,
    )

    elapsed = time.time() - start_time
    stopped_early = st.session_state.get("_stop_training", False)
    if stopped_early:
        n_done = len(train_result.episode_rewards)
        progress_bar.progress(1.0, text=f"⏹ Training stopped early at episode {n_done} — {elapsed:.1f}s")
    else:
        progress_bar.progress(1.0, text=f"✅ Training complete — {elapsed:.1f}s")

    training_records = [
        {"episode": i, "reward": r, "length": l}
        for i, (r, l) in enumerate(
            zip(train_result.episode_rewards, train_result.episode_lengths), 1
        )
    ]
    store.save_metrics_csv("training", training_records)

    # ---- Training Result Plots ----
    st.markdown('<div class="section-header">📊 Training Results</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = Visualizer.plot_training_progress(train_result)
        st.pyplot(fig)
        store.save_plot(fig, "training", "reward_curve")
        plt.close(fig)

    with col2:
        if train_result.loss_history:
            fig = Visualizer.plot_loss(train_result)
            if fig:
                st.pyplot(fig)
                store.save_plot(fig, "training", "loss")
                plt.close(fig)
        if train_result.epsilon_history:
            fig = Visualizer.plot_epsilon(train_result)
            if fig:
                st.pyplot(fig)
                store.save_plot(fig, "training", "epsilon")
                plt.close(fig)

    # Tabular policy grid
    analysis = agent.get_analysis_data()
    if analysis:
        # Determine correct grid shape for the environment
        grid_shapes = {
            "frozenlake": (4, 4),
            "cliffwalking": (4, 12),
        }
        # Arrow mappings differ by environment
        # FrozenLake: 0=left, 1=down, 2=right, 3=up
        # CliffWalking: 0=up, 1=right, 2=down, 3=left
        arrow_maps = {
            "frozenlake": {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"},
            "cliffwalking": {0: "\u2191", 1: "\u2192", 2: "\u2193", 3: "\u2190"},
        }
        grid_shape = grid_shapes.get(env_name, (4, 4))
        arrows = arrow_maps.get(env_name)
        fig = Visualizer.plot_tabular_policy(analysis, grid_shape=grid_shape, arrows=arrows)
        if fig:
            st.markdown("#### 🗺️ Learned Policy Grid")
            st.pyplot(fig)
            store.save_plot(fig, "analysis", "policy_grid")
            plt.close(fig)

    # ---- Evaluation Phase ----
    st.markdown('<div class="section-header">🏆 Evaluation</div>', unsafe_allow_html=True)

    with st.spinner(f"Evaluating over {cfg['eval_episodes']} episodes…"):
        evaluator = Evaluator(env, agent)
        eval_result = evaluator.evaluate(
            num_episodes=cfg["eval_episodes"],
            max_steps=cfg["max_steps"],
            success_threshold=SUCCESS_THRESHOLDS.get(env_name, 0.0),
        )

    ecol1, ecol2, ecol3, ecol4 = st.columns(4)
    ecol1.metric("Mean Reward", f"{eval_result.mean_reward:.2f}")
    ecol2.metric("Std Reward", f"± {eval_result.std_reward:.2f}")
    ecol3.metric("Success Rate", f"{eval_result.success_rate:.1f}%")
    ecol4.metric("Mean Length", f"{eval_result.mean_length:.1f}")

    fig = Visualizer.plot_evaluation(eval_result)
    st.pyplot(fig)
    store.save_plot(fig, "evaluation", "reward_histogram")
    plt.close(fig)

    # ---- GIF Recording — Full Width ----
    if cfg["record_gif"]:
        st.markdown('<div class="section-header">🎬 Agent Performance</div>', unsafe_allow_html=True)
        gym_id = ENV_GYM_IDS.get(env_name)
        if gym_id:
            with st.spinner("Recording evaluation GIF…"):
                try:
                    from infrastructure.gif_recorder import record_evaluation_gif
                    gif_path = store.gif_dir / f"{algo_name}_{env_name}.gif"
                    record_evaluation_gif(
                        env_id=gym_id,
                        agent=agent,
                        save_path=gif_path,
                        num_episodes=1,
                        max_steps=cfg["max_steps"],
                        fps=30,
                        env_kwargs=get_env_kwargs(env_name),
                    )
                    st.image(str(gif_path),
                             caption=f"{ALGO_INFO.get(algo_name, {}).get('label', algo_name)} on {ENV_INFO.get(env_name, {}).get('label', env_name)}",
                             width="stretch")
                    st.success(f"GIF saved to `{gif_path}`")
                except ImportError:
                    st.warning("Install `imageio` for GIF recording: `pip install imageio`")
                except Exception as e:
                    st.warning(f"GIF recording failed: {e}")
        else:
            st.info(f"GIF recording not available for `{env_name}`.")

    # ---- Save Model & Artifacts ----
    store.save_final_model(agent)
    store.save_metadata({
        "training_time_s": elapsed,
        "total_episodes": len(train_result.episode_rewards),
        "eval_mean_reward": eval_result.mean_reward,
        "eval_success_rate": eval_result.success_rate,
    })

    env.close()
    Visualizer.close_all()

    st.success(f"✅ Experiment complete! Artifacts saved to `{store.run_dir}`")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="RL Lab",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_custom_css()

    # ---- Sidebar: Output Browser (toggle-able) ----
    with st.sidebar:
        st.markdown("### 📂 Experiment Outputs")
        st.markdown("---")
        render_output_browser()

    # ---- Check if viewing a past run ----
    if "_view_run" in st.session_state and st.session_state["_view_run"]:
        render_header()
        if render_run_viewer():
            return  # We're in run-viewer mode

    render_header()

    # ---- Environment Selection (click cards) ----
    env_name = render_env_selector()

    st.markdown("")

    # ---- Algorithm Selection (click cards) ----
    algo_name = render_algo_selector(env_name)
    if algo_name is None:
        st.warning("No algorithms available for this environment.")
        return

    st.markdown("")

    # ---- Hyperparameters (all, grouped) ----
    hp = render_hyperparameters(algo_name)

    st.markdown("")

    # ---- Training Config ----
    train_cfg = render_training_config(env_name)

    # ---- Summary & Launch ----
    st.markdown("")
    st.markdown("---")

    env_label = ENV_INFO.get(env_name, {}).get("label", env_name)
    algo_label = ALGO_INFO.get(algo_name, {}).get("label", algo_name)
    env_icon = ENV_INFO.get(env_name, {}).get("icon", "")
    algo_icon = ALGO_INFO.get(algo_name, {}).get("icon", "")

    summary_cols = st.columns([3, 1])
    with summary_cols[0]:
        st.markdown(f"""
        **Ready to train:** {algo_icon} **{algo_label}** on
        {env_icon} **{env_label}** for **{train_cfg['num_episodes']:,}** episodes
        """)

    with summary_cols[1]:
        start = st.button("🚀 Start Experiment", type="primary", width="stretch")  # start btn

    if start:
        st.session_state["_stop_training"] = False
        cfg = {
            "env_name": env_name,
            "algo_name": algo_name,
            "hyperparameters": hp,
            **train_cfg,
        }
        run_experiment(cfg)


if __name__ == "__main__":
    main()
