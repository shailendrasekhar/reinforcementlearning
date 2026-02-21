"""
Streamlit UI for the Reinforcement Learning Framework — Modern Edition.

Run with:
    streamlit run app.py
"""

import sys
import time
import random
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

# Map environment registry names → gymnasium IDs (for GIF recording)
ENV_GYM_IDS = {
    "frozenlake": "FrozenLake-v1",
    "cartpole": "CartPole-v1",
    "mountaincar": "MountainCar-v0",
    "pendulum": "Pendulum-v1",
    "cliffwalking": "CliffWalking-v1",
    "acrobot": "Acrobot-v1",
    "lunarlander": "LunarLander-v3",
}

# Human-readable labels and descriptions for environments
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

# Human-readable labels and descriptions for algorithms
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
        "label": "Policy Gradient (REINFORCE)",
        "icon": "🎯",
        "family": "Policy Gradient",
        "desc": "Monte Carlo policy gradient with optional baseline.",
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
        "desc": "Twin Delayed DDPG — addresses overestimation bias in continuous control.",
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

# Complete hyperparameter definitions per algorithm
# Tuple: (min, max, default, step) → slider
# List: options → selectbox
# "bool": (default) → checkbox
ALGO_HYPERPARAMS = {
    "q_learning": {
        "learning_rate": {"type": "slider", "min": 0.001, "max": 1.0, "default": 0.1, "step": 0.001, "help": "Step size for Q-value updates"},
        "discount_factor": {"type": "slider", "min": 0.8, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — how much to weight future rewards"},
        "epsilon_start": {"type": "slider", "min": 0.1, "max": 1.0, "default": 1.0, "step": 0.05, "help": "Initial exploration rate"},
        "epsilon_end": {"type": "slider", "min": 0.001, "max": 0.5, "default": 0.01, "step": 0.001, "help": "Minimum exploration rate"},
        "decay_steps": {"type": "slider", "min": 1000, "max": 1000000, "default": 500000, "step": 1000, "help": "Steps over which epsilon decays"},
    },
    "sarsa": {
        "learning_rate": {"type": "slider", "min": 0.001, "max": 1.0, "default": 0.1, "step": 0.001, "help": "Step size for Q-value updates"},
        "discount_factor": {"type": "slider", "min": 0.8, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — how much to weight future rewards"},
        "epsilon_start": {"type": "slider", "min": 0.1, "max": 1.0, "default": 1.0, "step": 0.05, "help": "Initial exploration rate"},
        "epsilon_end": {"type": "slider", "min": 0.001, "max": 0.5, "default": 0.01, "step": 0.001, "help": "Minimum exploration rate"},
        "decay_steps": {"type": "slider", "min": 1000, "max": 1000000, "default": 500000, "step": 1000, "help": "Steps over which epsilon decays"},
    },
    "dqn": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Adam optimizer learning rate"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor for future rewards"},
        "batch_size": {"type": "select", "options": [16, 32, 64, 128, 256], "default": 64, "help": "Minibatch size for network updates"},
        "buffer_size": {"type": "select", "options": [5000, 10000, 50000, 100000, 200000], "default": 100000, "help": "Replay buffer capacity"},
        "warmup_steps": {"type": "slider", "min": 100, "max": 10000, "default": 1000, "step": 100, "help": "Random actions before training starts"},
        "train_frequency": {"type": "select", "options": [1, 2, 4, 8], "default": 4, "help": "Steps between gradient updates"},
        "target_update_frequency": {"type": "select", "options": [100, 500, 1000, 2000, 5000], "default": 1000, "help": "Steps between target network hard updates"},
        "epsilon_start": {"type": "slider", "min": 0.5, "max": 1.0, "default": 1.0, "step": 0.05, "help": "Initial exploration rate"},
        "epsilon_end": {"type": "slider", "min": 0.001, "max": 0.2, "default": 0.01, "step": 0.001, "help": "Minimum exploration rate"},
        "epsilon_decay_steps": {"type": "slider", "min": 1000, "max": 200000, "default": 50000, "step": 1000, "help": "Steps over which epsilon decays"},
        "gradient_clip": {"type": "slider", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1, "help": "Gradient clipping max norm (0 = disabled)"},
    },
    "reinforce": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Adam optimizer learning rate"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor"},
        "use_baseline": {"type": "bool", "default": True, "help": "Subtract running mean return to reduce variance"},
    },
    "a2c": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.0003, "step": 0.00001, "help": "Adam optimizer learning rate"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor"},
        "value_coeff": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "help": "Weight of value loss in total loss"},
        "entropy_coeff": {"type": "slider", "min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001, "help": "Entropy bonus coefficient for exploration"},
        "gradient_clip": {"type": "slider", "min": 0.0, "max": 10.0, "default": 0.5, "step": 0.1, "help": "Max gradient norm (0 = disabled)"},
    },
    "a3c": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Adam optimizer learning rate"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor"},
        "value_coeff": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "help": "Weight of value loss in total loss"},
        "entropy_coeff": {"type": "slider", "min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001, "help": "Entropy bonus for exploration"},
        "n_steps": {"type": "select", "options": [3, 5, 10, 20], "default": 5, "help": "Number of steps for n-step return bootstrapping"},
        "gradient_clip": {"type": "slider", "min": 0.0, "max": 100.0, "default": 40.0, "step": 1.0, "help": "Max gradient norm (0 = disabled)"},
    },
    "ppo": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.0003, "step": 0.00001, "help": "Adam optimizer learning rate"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor"},
        "clip_epsilon": {"type": "slider", "min": 0.05, "max": 0.5, "default": 0.2, "step": 0.01, "help": "PPO clipping range for the surrogate objective"},
        "ppo_epochs": {"type": "select", "options": [3, 5, 10, 15, 20], "default": 10, "help": "SGD epochs per rollout"},
        "batch_size": {"type": "select", "options": [32, 64, 128, 256], "default": 64, "help": "Minibatch size for PPO updates"},
        "rollout_length": {"type": "select", "options": [128, 256, 512, 1024, 2048, 4096], "default": 2048, "help": "Steps collected before each PPO update"},
        "gae_lambda": {"type": "slider", "min": 0.8, "max": 1.0, "default": 0.95, "step": 0.01, "help": "GAE lambda for advantage estimation"},
        "value_coeff": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "help": "Weight of value loss"},
        "entropy_coeff": {"type": "slider", "min": 0.0, "max": 0.1, "default": 0.01, "step": 0.001, "help": "Entropy bonus coefficient"},
        "gradient_clip": {"type": "slider", "min": 0.0, "max": 10.0, "default": 0.5, "step": 0.1, "help": "Max gradient norm (0 = disabled)"},
    },
    "ddpg": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Learning rate for actor & critic"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor"},
        "tau": {"type": "slider", "min": 0.001, "max": 0.05, "default": 0.005, "step": 0.001, "help": "Soft target update rate"},
        "batch_size": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Minibatch size for updates"},
        "buffer_size": {"type": "select", "options": [50000, 100000, 200000, 500000], "default": 100000, "help": "Replay buffer capacity"},
        "warmup_steps": {"type": "slider", "min": 100, "max": 10000, "default": 1000, "step": 100, "help": "Random actions before training starts"},
        "noise_type": {"type": "select", "options": ["ou", "gaussian"], "default": "ou", "help": "Exploration noise type (OU or Gaussian)"},
        "noise_sigma": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.2, "step": 0.01, "help": "Noise standard deviation / sigma"},
    },
    "td3": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.001, "step": 0.00001, "help": "Learning rate for actor & critic"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor"},
        "tau": {"type": "slider", "min": 0.001, "max": 0.05, "default": 0.005, "step": 0.001, "help": "Soft target update rate"},
        "batch_size": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Minibatch size for updates"},
        "buffer_size": {"type": "select", "options": [50000, 100000, 200000, 500000], "default": 100000, "help": "Replay buffer capacity"},
        "warmup_steps": {"type": "slider", "min": 100, "max": 10000, "default": 1000, "step": 100, "help": "Random actions before training starts"},
        "policy_delay": {"type": "select", "options": [1, 2, 3, 4], "default": 2, "help": "Actor update delay (every N critic updates)"},
        "target_noise_std": {"type": "slider", "min": 0.05, "max": 0.5, "default": 0.2, "step": 0.01, "help": "Target policy smoothing noise std"},
        "target_noise_clip": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "help": "Clipping range for target noise"},
        "noise_type": {"type": "select", "options": ["gaussian", "ou"], "default": "gaussian", "help": "Exploration noise type"},
        "noise_sigma": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01, "help": "Exploration noise sigma"},
    },
    "sac": {
        "learning_rate": {"type": "slider", "min": 0.00001, "max": 0.01, "default": 0.0003, "step": 0.00001, "help": "Adam learning rate for actor & critic"},
        "discount_factor": {"type": "slider", "min": 0.9, "max": 1.0, "default": 0.99, "step": 0.01, "help": "Gamma — discount factor"},
        "tau": {"type": "slider", "min": 0.001, "max": 0.05, "default": 0.005, "step": 0.001, "help": "Soft target update rate"},
        "batch_size": {"type": "select", "options": [64, 128, 256, 512], "default": 256, "help": "Minibatch size for updates"},
        "buffer_size": {"type": "select", "options": [50000, 100000, 200000, 500000], "default": 100000, "help": "Replay buffer capacity"},
        "warmup_steps": {"type": "slider", "min": 100, "max": 10000, "default": 1000, "step": 100, "help": "Random actions before training starts"},
        "init_alpha": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.2, "step": 0.01, "help": "Initial entropy temperature"},
        "auto_entropy": {"type": "bool", "default": True, "help": "Automatically tune entropy coefficient"},
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

# Env-specific defaults for exploration / noise
TABULAR_EXPLORATION = {
    "strategy": "epsilon_greedy",
    "params": {
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "decay_type": "linear",
        "decay_steps": 500000,
    },
}
DQN_EXPLORATION = {
    "strategy": "epsilon_greedy",
    "params": {
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "decay_type": "linear",
        "decay_steps": 50000,
    },
}
DDPG_NOISE = {"type": "ou", "theta": 0.15, "sigma": 0.2}

# Success thresholds
SUCCESS_THRESHOLDS = {
    "frozenlake": 0.0,
    "cliffwalking": -100.0,
    "acrobot": -100.0,
    "cartpole": 195.0,
    "mountaincar": -110.0,
    "lunarlander": 200.0,
    "pendulum": -300.0,
}

# Default episode / step counts per environment
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

    # Copy standard hyperparams
    for key in ("learning_rate", "discount_factor", "batch_size", "buffer_size",
                "warmup_steps", "train_frequency", "target_update_frequency",
                "clip_epsilon", "ppo_epochs", "rollout_length", "gae_lambda",
                "value_coeff", "entropy_coeff", "tau", "init_alpha",
                "auto_entropy", "use_baseline", "policy_delay",
                "target_noise_std", "target_noise_clip", "n_steps"):
        if key in hp:
            params[key] = hp[key]

    # Gradient clip
    gc = hp.get("gradient_clip", 0.0)
    if gc > 0:
        params["gradient_clip"] = gc

    params["hidden_dims"] = [128, 128]
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
        params["hidden_dims"] = [256, 256]
    elif algo == "sac":
        params["hidden_dims"] = [256, 256]

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
    """Run the training loop with real-time Streamlit chart updates."""
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
# CSS Styling
# ---------------------------------------------------------------------------

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .hero-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    /* Card styling */
    .config-card {
        background: linear-gradient(145deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 1px solid #e0e3ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .config-card h3 {
        margin-top: 0;
        color: #3D348B;
        font-size: 1.1rem;
    }

    /* Environment cards */
    .env-card {
        background: white;
        border: 2px solid #e8e8ee;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: all 0.2s ease;
        cursor: pointer;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .env-card:hover {
        border-color: #6C63FF;
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.15);
    }
    .env-card .icon { font-size: 2rem; }
    .env-card .label {
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        color: #2d2d3f;
    }
    .env-card .type-badge {
        font-size: 0.7rem;
        background: #6C63FF20;
        color: #6C63FF;
        border-radius: 4px;
        padding: 2px 6px;
        margin-top: 0.3rem;
        display: inline-block;
    }

    /* Algorithm cards */
    .algo-card {
        background: white;
        border: 2px solid #e8e8ee;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        transition: all 0.2s ease;
        min-height: 60px;
    }
    .algo-card:hover {
        border-color: #6C63FF;
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.15);
    }
    .algo-card .icon { font-size: 1.5rem; }
    .algo-card .label {
        font-weight: 600;
        font-size: 0.9rem;
        color: #2d2d3f;
    }
    .algo-card .family {
        font-size: 0.7rem;
        color: #888;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 1px solid #e0e3ff;
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #555 !important;
    }
    [data-testid="stMetricValue"] {
        color: #3D348B !important;
    }

    /* Section separators */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2d2d3f;
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e3ff;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6C63FF 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4) !important;
        transform: translateY(-1px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }

    /* Hide sidebar */
    [data-testid="stSidebar"] { display: none; }

    /* Hyperparameter section */
    .hp-section {
        background: linear-gradient(145deg, #fafbff 0%, #f3f4ff 100%);
        border: 1px solid #e0e3ff;
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6C63FF, #764ba2) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #3D348B !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="hero-header">
        <h1>🎮 Reinforcement Learning Lab</h1>
        <p>Select an environment and algorithm, tune every hyperparameter, train, and visualize — all in one place.</p>
    </div>
    """, unsafe_allow_html=True)


def render_env_selector() -> str:
    """Render environment selector as cards on main layout."""
    st.markdown('<div class="section-header">🌍 Choose Environment</div>', unsafe_allow_html=True)

    env_order = ["frozenlake", "cliffwalking", "acrobot", "cartpole", "mountaincar", "lunarlander", "pendulum"]
    env_names = [e for e in env_order if e in ENV_INFO]

    cols = st.columns(len(env_names))
    for i, env_key in enumerate(env_names):
        info = ENV_INFO[env_key]
        with cols[i]:
            st.markdown(f"""
            <div class="env-card">
                <div class="icon">{info['icon']}</div>
                <div class="label">{info['label']}</div>
                <div class="type-badge">{info['type']}</div>
            </div>
            """, unsafe_allow_html=True)

    selected = st.selectbox(
        "Select environment",
        env_names,
        format_func=lambda x: f"{ENV_INFO[x]['icon']} {ENV_INFO[x]['label']} — {ENV_INFO[x]['desc']}",
        index=env_names.index("cartpole") if "cartpole" in env_names else 0,
        label_visibility="collapsed",
    )
    return selected


def render_algo_selector(env_name: str) -> str:
    """Render algorithm selector based on compatible algorithms."""
    st.markdown('<div class="section-header">🧪 Choose Algorithm</div>', unsafe_allow_html=True)

    compatible = COMPATIBLE.get(env_name, AgentRegistry.list())
    algo_keys = [a for a in compatible if a in ALGO_INFO]

    # Display info cards in a grid
    card_cols = st.columns(min(len(algo_keys), 4))
    for i, algo_key in enumerate(algo_keys):
        info = ALGO_INFO[algo_key]
        col_idx = i % min(len(algo_keys), 4)
        with card_cols[col_idx]:
            st.markdown(f"""
            <div class="algo-card">
                <div class="icon">{info['icon']}</div>
                <div>
                    <div class="label">{info['label']}</div>
                    <div class="family">{info['family']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    selected = st.selectbox(
        "Select algorithm",
        algo_keys,
        format_func=lambda x: f"{ALGO_INFO[x]['icon']} {ALGO_INFO[x]['label']} — {ALGO_INFO[x]['desc']}",
        label_visibility="collapsed",
    )
    return selected


def render_hyperparameters(algo_name: str) -> dict:
    """Render ALL hyperparameters for the selected algorithm on the main layout."""
    st.markdown('<div class="section-header">⚙️ Hyperparameters</div>', unsafe_allow_html=True)

    algo_hp_defs = ALGO_HYPERPARAMS.get(algo_name, {})
    if not algo_hp_defs:
        st.info("No tunable hyperparameters for this algorithm.")
        return {}

    hp = {}

    # Show algorithm description
    info = ALGO_INFO.get(algo_name, {})
    st.markdown(f"""
    <div class="config-card">
        <h3>{info.get('icon', '')} {info.get('label', algo_name)} Hyperparameters</h3>
        <p style="color: #666; margin-bottom: 0;">{info.get('desc', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Arrange hyperparameters in a 2-column grid
    param_list = list(algo_hp_defs.items())
    col_count = 2
    for row_start in range(0, len(param_list), col_count):
        cols = st.columns(col_count)
        for col_idx in range(col_count):
            idx = row_start + col_idx
            if idx >= len(param_list):
                break
            param_name, spec = param_list[idx]
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
    """Render training configuration section."""
    st.markdown('<div class="section-header">📋 Training Configuration</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        num_episodes = st.number_input(
            "Number of Episodes",
            min_value=10,
            max_value=100000,
            value=DEFAULT_EPISODES.get(env_name, 1000),
            step=100,
            help="Total training episodes",
        )
    with col2:
        max_steps = st.number_input(
            "Max Steps / Episode",
            min_value=50,
            max_value=10000,
            value=DEFAULT_STEPS.get(env_name, 500),
            step=50,
            help="Maximum steps per episode before truncation",
        )
    with col3:
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
            help="Seed for reproducibility",
        )
    with col4:
        eval_episodes = st.number_input(
            "Eval Episodes",
            min_value=5,
            max_value=500,
            value=50,
            step=5,
            help="Episodes for post-training evaluation",
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
    """Execute the full experiment pipeline with live visualization."""
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
        fig = Visualizer.plot_tabular_policy(analysis)
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

    # ---- GIF Recording ----
    if cfg["record_gif"]:
        st.markdown("#### 🎬 Agent Performance")
        gym_id = ENV_GYM_IDS.get(env_name)
        if gym_id:
            with st.spinner("Recording evaluation GIF…"):
                try:
                    from infrastructure.gif_recorder import record_evaluation_gif
                    gif_path = Path("outputs") / "ui_gifs" / f"{algo_name}_{env_name}.gif"
                    record_evaluation_gif(
                        env_id=gym_id,
                        agent=agent,
                        save_path=gif_path,
                        num_episodes=1,
                        max_steps=cfg["max_steps"],
                        fps=30,
                        env_kwargs=get_env_kwargs(env_name),
                    )
                    st.image(str(gif_path), caption=f"{ALGO_INFO.get(algo_name, {}).get('label', algo_name)} on {ENV_INFO.get(env_name, {}).get('label', env_name)}")
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
    render_header()

    # ---- Environment Selection ----
    env_name = render_env_selector()

    st.markdown("")  # spacer

    # ---- Algorithm Selection ----
    algo_name = render_algo_selector(env_name)

    st.markdown("")  # spacer

    # ---- Hyperparameters ----
    hp = render_hyperparameters(algo_name)

    st.markdown("")  # spacer

    # ---- Training Config ----
    train_cfg = render_training_config(env_name)

    # ---- Summary & Launch ----
    st.markdown("")
    st.markdown("---")

    summary_cols = st.columns([3, 1])
    with summary_cols[0]:
        env_label = ENV_INFO.get(env_name, {}).get("label", env_name)
        algo_label = ALGO_INFO.get(algo_name, {}).get("label", algo_name)
        st.markdown(f"""
        **Ready to train:** {ALGO_INFO.get(algo_name, {}).get('icon', '')} **{algo_label}** on
        {ENV_INFO.get(env_name, {}).get('icon', '')} **{env_label}** for **{train_cfg['num_episodes']:,}** episodes
        """)

    with summary_cols[1]:
        start = st.button("🚀 Start Experiment", type="primary", use_container_width=True)

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
