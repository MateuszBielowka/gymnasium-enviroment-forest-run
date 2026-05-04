from __future__ import annotations

import gymnasium as gym


class ClipReward(gym.RewardWrapper):
    def __init__(self, env: gym.Env, min_reward: float = -1.0, max_reward: float = 1.0) -> None:
        super().__init__(env)
        self._min_reward = min_reward
        self._max_reward = max_reward

    def reward(self, reward: float) -> float:
        return float(max(self._min_reward, min(self._max_reward, reward)))
