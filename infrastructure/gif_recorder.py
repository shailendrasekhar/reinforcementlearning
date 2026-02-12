"""GIF recorder — captures evaluation episodes as animated GIFs."""

from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym

from core.agent import BaseAgent


def record_evaluation_gif(
    env_id: str,
    agent: BaseAgent,
    save_path: str | Path,
    num_episodes: int = 1,
    max_steps: int = 500,
    fps: int = 30,
    env_kwargs: Optional[dict] = None,
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

    def _convert(s):
        return int(s) if discrete_obs else s

    frames = []
    for _ep in range(num_episodes):
        state, _info = env.reset()
        state = _convert(state)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))

        for _step in range(max_steps):
            action = agent.select_action(state, training=False)
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
            "No frames captured — environment may not support rgb_array rendering."
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Use imageio v2 mimsave API for animated GIF writing
    duration_s = 1.0 / fps
    imageio.mimsave(str(save_path), frames, duration=duration_s, loop=0)

    return save_path
