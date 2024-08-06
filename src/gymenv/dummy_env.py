# src/environments/dummy_env.py
from typing import Optional

import gym
from gym import spaces
import numpy as np


class DummyEnv(gym.Env):
    def __init__(self, env_cfg):
        super().__init__()
        self.item_nbr = env_cfg['item_nbr']
        self.max_steps = env_cfg.get('max_steps', 100)
        self.state_size = env_cfg.get('state_size', 4)
        self.action_size = env_cfg.get('action_size', 2)

        self.current_step = 0
        self.state = None

        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)

    def reset(self, seed: Optional[int]=None, options: Optional[dict] = None):
        super().reset(seed =seed)
        self.np_random = np.random.default_rng(seed)
        self.current_step = 0
        self.state = self.np_random.uniform(low=-1, high=1, size=(self.state_size,))
        return self.state

    def step(self, action):
        self.current_step += 1

        # Simple dynamics: add a small random change to the state
        self.state += self.np_random.uniform(low=-0.1, high=0.1, size=(self.state_size,))

        # Simple reward: based on the action and current state
        reward = np.sum(self.state) * action

        done = self.current_step >= self.max_steps
        info = {"item_nbr": self.item_nbr}

        return self.state, reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, State: {self.state}")