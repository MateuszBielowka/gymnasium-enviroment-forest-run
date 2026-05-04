from __future__ import annotations

import gymnasium as gym


class DistanceShapingReward(gym.Wrapper):
    def __init__(self, env: gym.Env, distance_weight: float = 0.1) -> None:
        super().__init__(env)
        self.distance_weight = distance_weight

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        distance = float(info.get("distance", 0.0))
        shaped_reward = float(reward - self.distance_weight * distance)
        return observation, shaped_reward, terminated, truncated, info
