"""
Streamlit UI for the Reinforcement Learning Framework.

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
}

# Default hyperparameters per algorithm
ALGO_DEFAULTS = {
    "q_learning": {
        "learning_rate": (0.001, 1.0, 0.1, 0.001),
        "discount_factor": (0.8, 1.0, 0.99, 0.01),
    },
    "sarsa": {
        "learning_rate": (0.001, 1.0, 0.1, 0.001),
        "discount_factor": (0.8, 1.0, 0.99, 0.01),
    },
    "dqn": {
        "learning_rate": (0.0001, 0.01, 0.001, 0.0001),
        "discount_factor": (0.9, 1.0, 0.99, 0.01),
        "batch_size": [32, 64, 128, 256],
        "buffer_size": [10000, 50000, 100000],
        "warmup_steps": (100, 5000, 500, 100),
        "train_frequency": [1, 2, 4, 8],
        "target_update_frequency": [100, 500, 1000, 2000],
    },
    "reinforce": {
        "learning_rate": (0.0001, 0.01, 0.001, 0.0001),
        "discount_factor": (0.9, 1.0, 0.99, 0.01),
        "use_baseline": [True, False],
    },
    "a2c": {
        "learning_rate": (0.0001, 0.01, 0.0003, 0.0001),
        "discount_factor": (0.9, 1.0, 0.99, 0.01),
        "value_coeff": (0.1, 1.0, 0.5, 0.1),
        "entropy_coeff": (0.001, 0.1, 0.05, 0.001),
    },
    "ppo": {
        "learning_rate": (0.0001, 0.01, 0.0003, 0.0001),
        "discount_factor": (0.9, 1.0, 0.99, 0.01),
        "clip_epsilon": (0.1, 0.4, 0.2, 0.05),
        "ppo_epochs": [3, 5, 10, 15],
        "batch_size": [32, 64, 128],
        "rollout_length": [256, 512, 1024, 2048],
        "gae_lambda": (0.9, 1.0, 0.95, 0.01),
    },
    "ddpg": {
        "learning_rate": (0.0001, 0.01, 0.001, 0.0001),
        "discount_factor": (0.9, 1.0, 0.99, 0.01),
        "tau": (0.001, 0.05, 0.005, 0.001),
        "batch_size": [64, 128, 256],
        "buffer_size": [50000, 100000, 200000],
        "warmup_steps": (100, 5000, 1000, 100),
    },
    "sac": {
        "learning_rate": (0.0001, 0.01, 0.0003, 0.0001),
        "discount_factor": (0.9, 1.0, 0.99, 0.01),
        "tau": (0.001, 0.05, 0.005, 0.001),
        "batch_size": [64, 128, 256],
        "buffer_size": [50000, 100000, 200000],
        "warmup_steps": (100, 5000, 1000, 100),
        "init_alpha": (0.05, 1.0, 0.2, 0.05),
    },
}

# Which algorithms work with which environments
COMPATIBLE = {
    "frozenlake":   ["q_learning", "sarsa", "dqn"],
    "cartpole":     ["dqn", "reinforce", "a2c", "ppo"],
    "mountaincar":  ["q_learning", "sarsa", "dqn"],
    "pendulum":     ["ddpg", "sac"],
}

DISCRETE_ALGOS = {"q_learning", "sarsa", "dqn", "reinforce", "a2c", "ppo"}
CONTINUOUS_ALGOS = {"ddpg", "sac"}

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
    "cartpole": 195.0,
    "mountaincar": -110.0,
    "pendulum": -300.0,
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
    params = dict(hp)
    params["hidden_dims"] = [128, 128]
    params["device"] = "auto"

    if algo in ("q_learning", "sarsa"):
        params["exploration"] = TABULAR_EXPLORATION
    elif algo == "dqn":
        params["exploration"] = DQN_EXPLORATION
        params["gradient_clip"] = 1.0
    elif algo == "ddpg":
        params["noise"] = DDPG_NOISE
        params["max_action"] = 2.0
        params["hidden_dims"] = [256, 256]
    elif algo == "sac":
        params["auto_entropy"] = True
        params["hidden_dims"] = [256, 256]
    elif algo == "a2c":
        params["gradient_clip"] = 0.5
    elif algo == "ppo":
        params["value_coeff"] = 0.5
        params["entropy_coeff"] = 0.01
        params["gradient_clip"] = 0.5

    return params


def create_env(env_name: str) -> EnvWrapper:
    env_params = {}
    if env_name == "frozenlake":
        env_params = {"map_name": "4x4", "is_slippery": False}
    return EnvRegistry.create(env_name, **env_params)


def get_env_kwargs(env_name: str) -> dict:
    """Return kwargs used when creating env for GIF recording."""
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

    update_interval = max(1, num_episodes // 200)  # Update chart ~200 times max

    for episode in range(num_episodes):
        # Check stop flag
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

        # Progress bar
        progress_bar.progress(
            (episode + 1) / num_episodes,
            text=f"Episode {episode + 1}/{num_episodes}"
        )

        # Live chart update
        if (episode + 1) % update_interval == 0 or episode == num_episodes - 1:
            _update_live_chart(chart_placeholder, result)
            _update_live_metrics(metrics_placeholder, result, episode + 1, num_episodes)

    agent.on_training_end()
    return result


def _update_live_chart(placeholder, result: TrainingResult):
    """Redraw the training reward chart in-place."""
    fig, ax = plt.subplots(figsize=(10, 4))
    rewards = result.episode_rewards
    ax.plot(rewards, alpha=0.25, color="steelblue", linewidth=0.5)

    # Moving average
    window = min(100, max(1, len(rewards) // 5))
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        ma = np.convolve(rewards, kernel, mode="valid")
        offset = len(rewards) - len(ma)
        ax.plot(range(offset, offset + len(ma)), ma,
                color="darkblue", linewidth=2, label=f"MA-{window}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Reward (Live)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    placeholder.pyplot(fig)
    plt.close(fig)


def _update_live_metrics(placeholder, result: TrainingResult, episode: int, total: int):
    """Update the metrics display."""
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
# Pages
# ---------------------------------------------------------------------------

def page_header():
    st.set_page_config(
        page_title="RL Framework",
        page_icon="🎮",
        layout="wide",
    )
    st.title("🎮 Reinforcement Learning Framework")
    st.markdown("Select an environment and algorithm, tune hyperparameters, train, and visualize — all from this dashboard.")


def sidebar_config() -> dict:
    """Render sidebar and return configuration dict."""
    st.sidebar.header("⚙️ Experiment Setup")

    # Environment
    env_names = EnvRegistry.list()
    env_name = st.sidebar.selectbox("Environment", env_names, index=env_names.index("cartpole"))

    # Compatible algorithms
    compatible = COMPATIBLE.get(env_name, AgentRegistry.list())
    algo_name = st.sidebar.selectbox("Algorithm", compatible)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Hyperparameters")

    defaults = ALGO_DEFAULTS.get(algo_name, {})
    hp = {}
    for param, spec in defaults.items():
        if isinstance(spec, tuple):
            # (min, max, default, step) → slider
            lo, hi, default, step = spec
            hp[param] = st.sidebar.slider(param, lo, hi, default, step)
        elif isinstance(spec, list):
            if all(isinstance(v, bool) for v in spec):
                hp[param] = st.sidebar.checkbox(param, value=spec[0])
            else:
                idx = 1 if len(spec) > 1 else 0
                hp[param] = st.sidebar.selectbox(param, spec, index=idx)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Training")

    # Reasonable defaults per env
    default_episodes = {
        "frozenlake": 10000, "cartpole": 500,
        "mountaincar": 10000, "pendulum": 200,
    }
    default_steps = {
        "frozenlake": 100, "cartpole": 500,
        "mountaincar": 200, "pendulum": 200,
    }

    num_episodes = st.sidebar.number_input(
        "Num Episodes", 10, 100000,
        value=default_episodes.get(env_name, 1000), step=100,
    )
    max_steps = st.sidebar.number_input(
        "Max Steps/Episode", 50, 10000,
        value=default_steps.get(env_name, 500), step=50,
    )
    seed = st.sidebar.number_input("Seed", 0, 99999, 42, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎬 Evaluation")
    eval_episodes = st.sidebar.number_input("Eval Episodes", 5, 500, 50, step=5)
    record_gif = st.sidebar.checkbox("Record GIF", value=True)

    return {
        "env_name": env_name,
        "algo_name": algo_name,
        "hyperparameters": hp,
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "seed": seed,
        "eval_episodes": eval_episodes,
        "record_gif": record_gif,
    }


def run_experiment(cfg: dict):
    """Execute the full experiment pipeline with live visualization."""

    env_name = cfg["env_name"]
    algo_name = cfg["algo_name"]
    hp = cfg["hyperparameters"]

    set_seed(cfg["seed"])
    env = create_env(env_name)

    # Create artifact store for this run
    store = ArtifactStore(
        base_dir="outputs",
        agent_name=algo_name,
        env_name=env_name,
    )
    store.save_config(cfg)

    # Build agent
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
    st.header("📈 Training")

    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    progress_bar = st.progress(0, text="Starting training...")

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
    progress_bar.progress(1.0, text=f"Training complete — {elapsed:.1f}s")

    # Save training metrics
    training_records = [
        {"episode": i, "reward": r, "length": l}
        for i, (r, l) in enumerate(
            zip(train_result.episode_rewards, train_result.episode_lengths), 1
        )
    ]
    store.save_metrics_csv("training", training_records)

    # ---- Training Result Plots ----
    st.header("📊 Training Results")

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
            st.subheader("🗺️ Learned Policy Grid")
            st.pyplot(fig)
            store.save_plot(fig, "analysis", "policy_grid")
            plt.close(fig)

    # ---- Evaluation Phase ----
    st.header("🏆 Evaluation")

    with st.spinner(f"Evaluating over {cfg['eval_episodes']} episodes..."):
        evaluator = Evaluator(env, agent)
        eval_result = evaluator.evaluate(
            num_episodes=cfg["eval_episodes"],
            max_steps=cfg["max_steps"],
            success_threshold=SUCCESS_THRESHOLDS.get(env_name, 0.0),
        )

    # Summary metrics
    ecol1, ecol2, ecol3, ecol4 = st.columns(4)
    ecol1.metric("Mean Reward", f"{eval_result.mean_reward:.2f}")
    ecol2.metric("Std Reward", f"± {eval_result.std_reward:.2f}")
    ecol3.metric("Success Rate", f"{eval_result.success_rate:.1f}%")
    ecol4.metric("Mean Length", f"{eval_result.mean_length:.1f}")

    # Evaluation histogram
    fig = Visualizer.plot_evaluation(eval_result)
    st.pyplot(fig)
    store.save_plot(fig, "evaluation", "reward_histogram")
    plt.close(fig)

    # ---- GIF Recording ----
    if cfg["record_gif"]:
        st.header("🎬 Agent Performance GIF")

        gym_id = ENV_GYM_IDS.get(env_name)
        if gym_id:
            with st.spinner("Recording evaluation GIF..."):
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
                    st.image(str(gif_path), caption=f"{algo_name} on {env_name}")
                    st.success(f"GIF saved to `{gif_path}`")
                except ImportError:
                    st.warning(
                        "Install `imageio` for GIF recording: `pip install imageio`"
                    )
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

    # ---- Cleanup ----
    env.close()
    Visualizer.close_all()

    st.success(f"✅ Experiment complete! Artifacts saved to `{store.run_dir}`")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    page_header()
    cfg = sidebar_config()

    # Show config summary
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f"**Ready:** `{cfg['algo_name']}` on `{cfg['env_name']}` "
            f"for {cfg['num_episodes']} episodes"
        )

    # Start button
    if st.button("🚀 Start Experiment", type="primary", use_container_width=True):
        st.session_state["_stop_training"] = False
        run_experiment(cfg)


if __name__ == "__main__":
    main()
