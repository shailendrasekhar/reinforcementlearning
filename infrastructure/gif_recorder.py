"""GIF recorder — captures evaluation episodes as animated GIFs."""

from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym

from core.agent import BaseAgent


def _coerce_action(action, action_space: gym.Space):
    """Make sure the action has the right type/shape for the env."""
    if isinstance(action_space, gym.spaces.Discrete):
        return int(action) if not isinstance(action, (int, np.integer)) else action
    # Continuous — ensure numpy array with correct shape
    action = np.asarray(action, dtype=np.float32).flatten()
    action = np.clip(action, action_space.low, action_space.high)
    if action.shape != action_space.shape:
        action = action.reshape(action_space.shape)
    return action


def record_evaluation_gif(
    env_id: str,
    agent: BaseAgent,
    save_path: str | Path,
    num_episodes: int = 1,
    max_steps: int = 500,
    fps: int = 30,
    env_kwargs: Optional[dict] = None,
    max_frames: int = 600,
) -> Path:
    """Run evaluation episodes and save frames as an animated GIF.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (e.g. "CartPole-v1").
    agent : BaseAgent
        Trained agent (used in eval mode).
    save_path : str or Path
        Output GIF file path.
    num_episodes : int
        Number of episodes to record.
    max_steps : int
        Maximum steps per episode.
    fps : int
        Frames per second in the output GIF.
    env_kwargs : dict, optional
        Extra kwargs for gym.make.
    max_frames : int
        Hard cap on total frames to avoid huge GIFs.

    Returns
    -------
    Path
        Saved GIF file path.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for GIF recording. "
            "Install with: pip install imageio"
        )

    env_kwargs = env_kwargs or {}
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    discrete_obs = isinstance(env.observation_space, gym.spaces.Discrete)
    discrete_act = isinstance(env.action_space, gym.spaces.Discrete)

    def _convert(s):
        return int(s) if discrete_obs else np.asarray(s, dtype=np.float32)

    frames = []
    hit_cap = False
    for _ep in range(num_episodes):
        if hit_cap:
            break
        state, _info = env.reset()
        state = _convert(state)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))

        for _step in range(max_steps):
            if len(frames) >= max_frames:
                hit_cap = True
                break
            try:
                action = agent.select_action(state, training=False)
                action = _coerce_action(action, env.action_space)
            except Exception:
                # Fallback to random action if agent fails
                action = env.action_space.sample()

            state, _reward, terminated, truncated, _info = env.step(action)
            state = _convert(state)

            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))

            if terminated or truncated:
                break

    env.close()

    if not frames:
        raise RuntimeError(
            f"No frames captured for {env_id} — "
            "environment may not support rgb_array rendering."
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Use imageio v2 mimsave API for animated GIF writing
    duration_s = 1.0 / fps
    imageio.mimsave(str(save_path), frames, duration=duration_s, loop=0)

    return save_path
