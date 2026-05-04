from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        size = env.unwrapped.size
        self.observation_space = spaces.Box(
            low=-(size - 1),
            high=(size - 1),
            shape=(2,),
            dtype=int,
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return observation["target"] - observation["agent"]
