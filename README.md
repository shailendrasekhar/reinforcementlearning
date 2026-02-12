# Reinforcement Learning Framework

A modular, config-driven reinforcement learning framework implementing 8 algorithms across 4 environments. Built with clean abstractions, registry patterns, and a single-entry-point `ExperimentRunner` that handles the full pipeline — training, evaluation, visualization, and artifact management.

## Algorithms

| Algorithm | Type | Action Space | Environment |
|-----------|------|-------------|-------------|
| **Q-Learning** | Tabular, off-policy | Discrete | FrozenLake |
| **SARSA** | Tabular, on-policy | Discrete | FrozenLake |
| **DQN** | Deep, off-policy | Discrete | CartPole |
| **REINFORCE** | Policy gradient | Discrete | CartPole |
| **A2C** | Actor-Critic | Discrete | CartPole |
| **PPO** | Actor-Critic (clipped) | Discrete | CartPole |
| **DDPG** | Deterministic policy gradient | Continuous | Pendulum |
| **SAC** | Max-entropy, off-policy | Continuous | Pendulum |

## Quick Start

### Installation

```bash
git clone <repo-url> && cd reinforcementlearning
pip install -r requirements.txt
```

**Dependencies:** `gymnasium`, `numpy`, `matplotlib`, `pyyaml`, `torch`, `streamlit`, `imageio`
> PyTorch is only required for deep RL agents (DQN, REINFORCE, A2C, PPO, DDPG, SAC). Tabular agents (Q-Learning, SARSA) work without it.

### Interactive UI (Streamlit)

```bash
streamlit run app.py
```

Opens a browser dashboard where you can:
- **Select** any environment and compatible algorithm from dropdowns
- **Tune** hyperparameters with sliders and selectors
- **Watch** training reward curves update in real time
- **View** evaluation metrics, reward distributions, and policy grids
- **Record** an animated GIF of the trained agent's performance

### Run an Experiment

```bash
# Single experiment
python run.py --config configs/q_learning_frozenlake.yaml

# Compare multiple algorithms
python run.py --compare configs/q_learning_frozenlake.yaml configs/sarsa_frozenlake.yaml

# Multi-seed robustness check (seeds: 42, 123, 456)
python run.py --config configs/ppo_cartpole.yaml --multi-seed
```

### Available Configs

```
configs/
├── q_learning_frozenlake.yaml   # Tabular Q-Learning on FrozenLake-v1
├── sarsa_frozenlake.yaml        # Tabular SARSA on FrozenLake-v1
├── dqn_cartpole.yaml            # Deep Q-Network on CartPole-v1
├── reinforce_cartpole.yaml      # REINFORCE on CartPole-v1
├── a2c_cartpole.yaml            # A2C on CartPole-v1
├── ppo_cartpole.yaml            # PPO on CartPole-v1
├── ddpg_pendulum.yaml           # DDPG on Pendulum-v1
└── sac_pendulum.yaml            # SAC on Pendulum-v1
```

## Benchmark Results

| Algorithm | Environment | Eval Mean Reward | Success Rate |
|-----------|-------------|:----------------:|:------------:|
| Q-Learning | FrozenLake-v1 | 1.00 ± 0.00 | **100%** |
| SARSA | FrozenLake-v1 | 1.00 ± 0.00 | **100%** |
| DQN | CartPole-v1 | 272.28 ± 67.11 | **96%** |
| REINFORCE | CartPole-v1 | 250+ avg train | Learning |
| A2C | CartPole-v1 | 120+ avg train | Learning |
| PPO | CartPole-v1 | 495.16 ± 11.67 | **100%** |
| DDPG | Pendulum-v1 | -194 (improving) | Learning |
| SAC | Pendulum-v1 | -833 (improving) | Learning |

> Results from default configs with `seed: 42`. Pendulum score ranges from -1600 (worst) to 0 (best). CartPole "solved" threshold is 195.

## Project Structure

```
reinforcementlearning/
│
├── run.py                          # CLI entry point
├── app.py                          # Streamlit UI (interactive dashboard)
├── requirements.txt                # Python dependencies
├── configs/                        # YAML experiment configurations
│
├── core/                           # Framework backbone
│   ├── types.py                    # Transition, EpisodeResult, TrainingResult, EvaluationResult
│   ├── agent.py                    # BaseAgent ABC — interface all agents implement
│   ├── environment.py              # EnvWrapper ABC + GymEnvWrapper
│   ├── trainer.py                  # EpisodicTrainer — episode-based training loop
│   ├── rollout_trainer.py          # RolloutTrainer — step-based loop for PPO/A2C
│   ├── evaluator.py                # Agent evaluation (no training)
│   └── callback.py                 # ProgressCallback, CheckpointCallback, EarlyStoppingCallback, MetricsCallback
│
├── components/                     # Reusable building blocks
│   ├── replay_buffer.py            # UniformReplayBuffer (DQN, DDPG, SAC)
│   ├── rollout_buffer.py           # RolloutBuffer with GAE (PPO)
│   ├── exploration.py              # EpsilonGreedy, Boltzmann strategies
│   ├── noise.py                    # OrnsteinUhlenbeck, Gaussian action noise
│   ├── networks.py                 # MLP, QNetwork, PolicyNetwork, ActorCriticNetwork,
│   │                               # DeterministicActor, QCritic, GaussianActor
│   └── schedulers.py               # LinearScheduler, ExponentialScheduler
│
├── agents/                         # Algorithm implementations
│   ├── registry.py                 # AgentRegistry — name → class mapping
│   ├── q_learning/agent.py         # Tabular Q-Learning
│   ├── sarsa/agent.py              # Tabular SARSA
│   ├── dqn/agent.py                # Deep Q-Network
│   ├── reinforce/agent.py          # REINFORCE (Monte Carlo policy gradient)
│   ├── a2c/agent.py                # Advantage Actor-Critic
│   ├── ppo/agent.py                # Proximal Policy Optimization
│   ├── ddpg/agent.py               # Deep Deterministic Policy Gradient
│   └── sac/agent.py                # Soft Actor-Critic
│
├── environments/                   # Environment wrappers
│   ├── registry.py                 # EnvRegistry — name → factory mapping
│   ├── frozen_lake.py              # FrozenLake-v1
│   ├── cart_pole.py                # CartPole-v1
│   ├── mountain_car.py             # MountainCar-v0
│   └── pendulum.py                 # Pendulum-v1
│
├── infrastructure/                 # Orchestration & I/O
│   ├── experiment_runner.py        # ExperimentRunner — the single user-facing orchestrator
│   ├── config_loader.py            # YAML load/save/merge/validate
│   ├── artifact_store.py           # Timestamped output directory management
│   ├── logger.py                   # CSV export + console summaries
│   ├── visualizer.py               # Training curves, evaluation histograms, policy grids
│   └── gif_recorder.py             # Evaluation GIF recording (gymnasium + imageio)
│
└── outputs/                        # Auto-generated per experiment (gitignored)
    └── <experiment_name>__<timestamp>/
        ├── config.yaml             # Frozen config snapshot
        ├── metadata.json           # Run metadata (duration, host, etc.)
        ├── checkpoints/            # Periodic & best model saves
        ├── metrics/
        │   ├── training_log.csv    # Per-episode reward, length, loss, epsilon
        │   └── evaluation_log.csv  # Per-episode eval reward
        ├── plots/
        │   ├── training/           # reward_curve.png, epsilon_decay.png, loss.png
        │   ├── evaluation/         # reward_distribution.png
        │   └── analysis/           # policy_grid.png (for tabular agents)
        └── final_model/
            └── agent_state.pkl     # Serialized agent state dict
```

## Architecture

### Design Principles

1. **Registry Pattern** — Agents and environments self-register. Add a new algorithm by creating a module and registering it in `agents/registry.py`.
2. **Config-Driven** — Every hyperparameter lives in YAML. No code changes needed to tune experiments.
3. **Algorithm-Agnostic Trainer** — The `EpisodicTrainer` works with any `BaseAgent` implementation. The trainer knows nothing about Q-tables, neural networks, or replay buffers.
4. **Lifecycle Hooks** — Agents receive `on_episode_start()`, `on_episode_end()`, `on_training_start()`, `on_training_end()` callbacks. Policy gradient agents use `on_episode_end()` to flush trajectories. SARSA uses `on_episode_start()` to clear cached actions.
5. **Composable Callbacks** — `CheckpointCallback`, `EarlyStoppingCallback`, `ProgressCallback`, and `MetricsCallback` can be combined via YAML config.
6. **Artifact Isolation** — Each run gets a timestamped directory with config snapshot, metrics, plots, and model checkpoints. Nothing is overwritten.

### BaseAgent Interface

Every agent implements this contract:

```python
class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state, training: bool = True): ...

    @abstractmethod
    def update(self, transition: Transition) -> Dict[str, float]: ...

    # Lifecycle hooks (optional overrides)
    def on_episode_start(self) -> None: ...
    def on_episode_end(self, episode_reward: float) -> Dict[str, float]: ...
    def on_training_start(self, config: Dict) -> None: ...
    def on_training_end(self) -> None: ...

    # Serialization
    def get_state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...

    # Introspection
    @property
    def epsilon(self) -> Optional[float]: ...          # For exploration tracking
    def get_analysis_data(self) -> Optional[Dict]: ... # For visualization (e.g., Q-table)
```

### Training Flow

```
ExperimentRunner.run()
├── Load YAML config
├── Set seed (numpy, random, torch)
├── Create environment (via EnvRegistry)
├── Create agent (via AgentRegistry)
├── Build callbacks (from config)
├── EpisodicTrainer.train()
│   └── For each episode:
│       ├── agent.on_episode_start()
│       ├── Loop: select_action → env.step → agent.update
│       ├── agent.on_episode_end()
│       └── callbacks.on_episode_end()
├── Evaluator.evaluate()
├── Generate plots (Visualizer)
├── Export CSV logs (Logger)
├── Save model checkpoint
└── Return ExperimentResult
```

## Configuration Reference

A complete YAML config has these sections:

```yaml
experiment:
  name: my_experiment        # Used for output directory naming
  seed: 42                   # Reproducibility seed

environment:
  name: cartpole             # Registry key: frozenlake, cartpole, mountaincar, pendulum
  params:                    # Environment-specific params (optional)
    map_name: "4x4"          # FrozenLake: map size
    is_slippery: false       # FrozenLake: stochastic transitions

agent:
  name: dqn                  # Registry key: q_learning, sarsa, dqn, reinforce, a2c, ppo, ddpg, sac
  params:
    learning_rate: 0.001
    discount_factor: 0.99
    # ... algorithm-specific params (see individual configs)
  exploration:               # For discrete agents with epsilon-greedy
    strategy: epsilon_greedy  # or boltzmann
    params:
      epsilon_start: 1.0
      epsilon_end: 0.01
      decay_type: linear      # or exponential
      decay_steps: 50000      # For linear decay
  noise:                     # For continuous agents (DDPG)
    type: ou                  # or gaussian
    theta: 0.15
    sigma: 0.2

training:
  mode: episodic             # episodic (default) or rollout
  num_episodes: 1000
  max_steps_per_episode: 500
  log_frequency: 100         # Console log every N episodes

evaluation:
  num_episodes: 50
  success_threshold: 195.0   # Reward threshold for "success" counting

callbacks:
  - type: checkpoint
    params:
      frequency: 500         # Save every N episodes
      save_best: true        # Also save best-performing model
  - type: early_stopping
    params:
      patience: 200          # Stop if no improvement for N episodes
      min_delta: 1.0

output:
  base_dir: outputs
  save_plots: true
  save_metrics_csv: true
```

### Algorithm-Specific Parameters

#### Tabular (Q-Learning, SARSA)
```yaml
params:
  learning_rate: 0.1
  discount_factor: 0.99
  default_q: 0.0           # Initial Q-value
```

#### DQN
```yaml
params:
  learning_rate: 0.001
  discount_factor: 0.99
  batch_size: 64
  buffer_size: 50000        # Replay buffer capacity
  warmup_steps: 500         # Random actions before training
  train_frequency: 4        # Train every N steps
  target_update_frequency: 500  # Hard target network update
  hidden_dims: [128, 128]
  gradient_clip: 1.0
  device: auto              # auto, cpu, or cuda
```

#### REINFORCE
```yaml
params:
  learning_rate: 0.001
  discount_factor: 0.99
  use_baseline: true        # Running-mean baseline subtraction
  hidden_dims: [128, 128]
```

#### A2C
```yaml
params:
  learning_rate: 0.0003
  discount_factor: 0.99
  value_coeff: 0.5          # Value loss weight
  entropy_coeff: 0.05       # Entropy bonus weight
  gradient_clip: 0.5
  hidden_dims: [128, 128]
```

#### PPO
```yaml
params:
  learning_rate: 0.0003
  discount_factor: 0.99
  gae_lambda: 0.95          # GAE lambda
  clip_epsilon: 0.2         # Clipping range
  ppo_epochs: 10            # SGD epochs per rollout
  batch_size: 64            # Minibatch size
  rollout_length: 2048      # Steps per rollout
  value_coeff: 0.5
  entropy_coeff: 0.01
  gradient_clip: 0.5
  hidden_dims: [64, 64]
```

#### DDPG
```yaml
params:
  learning_rate: 0.001
  discount_factor: 0.99
  tau: 0.005                # Soft target update rate
  batch_size: 256
  buffer_size: 100000
  warmup_steps: 1000
  max_action: 2.0           # Action scaling (Pendulum: [-2, 2])
  hidden_dims: [256, 256]
noise:
  type: ou                   # Ornstein-Uhlenbeck
  theta: 0.15
  sigma: 0.2
```

#### SAC
```yaml
params:
  learning_rate: 0.0003
  discount_factor: 0.99
  tau: 0.005
  batch_size: 256
  buffer_size: 100000
  warmup_steps: 1000
  auto_entropy: true        # Automatic entropy tuning
  init_alpha: 0.2           # Initial entropy coefficient
  hidden_dims: [256, 256]
```

## Programmatic Usage

Beyond the CLI, you can use the framework directly in Python:

```python
from infrastructure.experiment_runner import ExperimentRunner

# From YAML file
runner = ExperimentRunner("configs/ppo_cartpole.yaml")
result = runner.run()

print(f"Mean reward: {result.evaluation_result.mean_reward:.2f}")
print(f"Success rate: {result.evaluation_result.success_rate:.1f}%")
print(f"Artifacts at: {result.run_dir}")
```

```python
# From a dict (no YAML file needed)
config = {
    "experiment": {"name": "custom_run", "seed": 42},
    "environment": {"name": "cartpole"},
    "agent": {
        "name": "dqn",
        "params": {"learning_rate": 0.001, "hidden_dims": [256, 256]},
        "exploration": {
            "strategy": "epsilon_greedy",
            "params": {"epsilon_start": 1.0, "epsilon_end": 0.01, "decay_type": "linear", "decay_steps": 50000}
        }
    },
    "training": {"num_episodes": 500, "max_steps_per_episode": 500},
    "evaluation": {"num_episodes": 50, "success_threshold": 195.0},
    "output": {"base_dir": "outputs", "save_plots": True}
}
result = ExperimentRunner(config).run()
```

```python
# Compare algorithms
ExperimentRunner.compare([
    "configs/q_learning_frozenlake.yaml",
    "configs/sarsa_frozenlake.yaml",
])

# Multi-seed runs for statistical robustness
runner = ExperimentRunner("configs/ppo_cartpole.yaml")
results = runner.run_multi_seed(seeds=[42, 123, 456, 789, 1000])
```

## Adding a New Algorithm

1. **Create the agent module:**

```
agents/
└── my_algo/
    ├── __init__.py     # from .agent import MyAlgoAgent
    └── agent.py        # class MyAlgoAgent(BaseAgent): ...
```

2. **Implement `BaseAgent`:** Override `select_action()`, `update()`, and serialization methods. Use lifecycle hooks as needed.

3. **Register it:**

```python
# agents/registry.py
from agents.my_algo import MyAlgoAgent

AgentRegistry.register("my_algo", MyAlgoAgent)
```

4. **Create a config:**

```yaml
# configs/my_algo_cartpole.yaml
experiment:
  name: my_algo_cartpole
  seed: 42
environment:
  name: cartpole
agent:
  name: my_algo
  params:
    learning_rate: 0.001
training:
  num_episodes: 1000
evaluation:
  num_episodes: 50
  success_threshold: 195.0
output:
  base_dir: outputs
  save_plots: true
```

5. **Run:** `python run.py --config configs/my_algo_cartpole.yaml`

## Adding a New Environment

1. **Create a factory function:**

```python
# environments/my_env.py
from core.environment import GymEnvWrapper

def create_my_env(**kwargs):
    return GymEnvWrapper("MyEnv-v0", **kwargs)
```

2. **Register it:**

```python
# environments/registry.py
from environments.my_env import create_my_env

EnvRegistry.register("myenv", create_my_env)
```

3. **Use it in configs:** `environment: { name: myenv }`

## Output Artifacts

Every run generates a timestamped directory:

```
outputs/ppo_cartpole__2026-02-12_08-26-59/
├── config.yaml             # Exact config used (for reproducibility)
├── metadata.json           # Timestamp, duration, system info
├── checkpoints/
│   ├── checkpoint_ep100.pkl
│   ├── checkpoint_ep200.pkl
│   └── best_model.pkl      # Highest evaluation reward
├── metrics/
│   ├── training_log.csv    # episode, reward, length, loss, epsilon
│   └── evaluation_log.csv  # episode, reward, length
├── plots/
│   ├── training/
│   │   ├── reward_curve.png   # Reward per episode + moving average
│   │   ├── epsilon_decay.png  # Exploration schedule
│   │   └── loss.png           # Training loss curve
│   ├── evaluation/
│   │   └── reward_distribution.png  # Histogram of eval rewards
│   └── analysis/
│       └── policy_grid.png   # Q-value arrow grid (tabular agents)
└── final_model/
    └── agent_state.pkl     # Full agent state for resuming/deploying
```

## Design Decisions

- **Why YAML configs?** — Hyperparameter changes shouldn't require code changes. YAML is human-readable and diffs cleanly in version control.
- **Why a registry pattern?** — Open/closed principle. New algorithms and environments are added without modifying existing code.
- **Why lifecycle hooks?** — Different algorithm families need different boundary behaviors (SARSA caches next-action, REINFORCE flushes trajectories, DDPG resets noise). Hooks solve this without polluting the trainer.
- **Why `Transition.extra`?** — Algorithm-specific data (log probabilities, value estimates) can ride alongside standard transition data without breaking the common interface.
- **Why per-episode A2C instead of rollout-based?** — Simplicity. PPO uses the `RolloutBuffer` for cross-episode step collection. A2C uses single-episode updates which trades sample efficiency for simpler code. Both approaches are valid.

## License

MIT
