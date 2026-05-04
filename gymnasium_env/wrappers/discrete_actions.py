from __future__ import annotations

import gymnasium as gym
import numpy as np


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env, actions: list[np.ndarray]) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("DiscreteActions wrapper expects Box action space in wrapped env.")

        self._actions = [np.asarray(action, dtype=float) for action in actions]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action: int) -> np.ndarray:
        return self._actions[action]
