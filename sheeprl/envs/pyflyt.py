import PyFlyt.gym_envs # noqa
from PyFlyt.gym_envs import FlattenWaypointEnv
import gymnasium as gym
from gymnasium import spaces

import numpy as np


class PyFlytGymWrapper(gym.Wrapper):
    def __init__(self, id, render_mode):
        env1 = gym.make("PyFlyt/QuadX-Waypoints-v2", render_mode=render_mode, sparse_reward=True, render_resolution=(64, 64))
        env = FlattenWaypointEnv(env1, context_length=2)
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.observation_space.shape[0],), dtype=np.float32),
                "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            }
        )
        self.action_space = self.env.action_space
        
    def _convert_obs(self, obs: np.ndarray):
        return {"vector": obs, "image": self.render()}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_obs(obs), reward, terminated, truncated, info
        
    def reset(
        self, *, seed, options):
        obs, info = self.env.reset()
        return self._convert_obs(obs), info
    
    def render(self):
        rgba_img = super().render()[:, :, :3]
        # remove alpha channel
        return rgba_img
