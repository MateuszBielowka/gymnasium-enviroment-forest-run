import gymnasium as gym
import gymnasium_env
import numpy as np


def test_env_registers_and_resets():
    env = gym.make("gymnasium_env/GridWorld-v0", size=5)
    obs, info = env.reset(seed=123)

    assert "agent" in obs
    assert "target" in obs
    assert "distance" in info
    assert obs["agent"].shape == (2,)
    assert obs["target"].shape == (2,)

    env.close()


def test_step_output_contract():
    env = gym.make("gymnasium_env/GridWorld-v0", size=5)
    obs, _ = env.reset(seed=123)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "distance" in info
        assert np.all(obs["agent"] >= 0)
        assert np.all(obs["agent"] <= 4)

    env.close()
