import gymnasium as gym
import gymnasium_env
import numpy as np


def progress_bonus(base_env, before: np.ndarray, after: np.ndarray) -> float:
    before_distance = float(np.linalg.norm(before - base_env._target_location, ord=1))
    after_distance = float(np.linalg.norm(after - base_env._target_location, ord=1))
    return base_env.approach_reward_weight * (before_distance - after_distance)


def test_env_registers_and_resets():
    env = gym.make("gymnasium_env/GridWorld-v0", size=5, tree_count=3, bush_count=2)
    obs, info = env.reset(seed=123)
    base_env = env.unwrapped

    assert "agent" in obs
    assert "target" in obs
    assert "trees" in obs
    assert "bushes" in obs
    assert "distance" in info
    assert obs["agent"].shape == (2,)
    assert obs["target"].shape == (2,)
    assert obs["trees"].shape == (5, 5)
    assert obs["bushes"].shape == (5, 5)
    np.testing.assert_array_equal(base_env._agent_location, np.array([4, 0]))
    np.testing.assert_array_equal(base_env._target_location, np.array([0, 4]))
    assert obs["trees"][4, 0] == 0
    assert obs["trees"][0, 4] == 0
    assert obs["bushes"][4, 0] == 0
    assert obs["bushes"][0, 4] == 0
    assert int(np.count_nonzero(obs["trees"])) == 3
    assert int(np.count_nonzero(obs["bushes"])) == 2
    assert np.all((obs["trees"] + obs["bushes"]) <= 1)

    env.close()


def test_step_output_contract():
    env = gym.make("gymnasium_env/GridWorld-v0", size=5, tree_count=3, bush_count=2)
    obs, _ = env.reset(seed=123)
    base_env = env.unwrapped

    tree_cells = np.argwhere(obs["trees"] == 1)
    bush_cells = np.argwhere(obs["bushes"] == 1)

    tree_tested = False
    for tree_row, tree_column in tree_cells:
        candidates = [
            (np.array([tree_row, tree_column - 1]), 0),
            (np.array([tree_row, tree_column + 1]), 2),
            (np.array([tree_row - 1, tree_column]), 3),
            (np.array([tree_row + 1, tree_column]), 1),
        ]
        for candidate_location, action in candidates:
            if np.any(candidate_location < 0) or np.any(candidate_location >= 5):
                continue
            if obs["trees"][tuple(candidate_location)] == 1 or obs["bushes"][tuple(candidate_location)] == 1:
                continue
            if np.array_equal(candidate_location, base_env._target_location):
                continue

            base_env._agent_location = candidate_location.copy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            assert np.array_equal(next_obs["agent"], candidate_location)
            assert np.isclose(reward, base_env.step_penalty + base_env.tree_penalty)
            assert not terminated
            assert not truncated
            assert "distance" in info
            tree_tested = True
            break
        if tree_tested:
            break

    assert tree_tested

    bush_tested = False
    for bush_row, bush_column in bush_cells:
        candidates = [
            (np.array([bush_row, bush_column - 1]), 0),
            (np.array([bush_row, bush_column + 1]), 2),
            (np.array([bush_row - 1, bush_column]), 3),
            (np.array([bush_row + 1, bush_column]), 1),
        ]
        for candidate_location, action in candidates:
            if np.any(candidate_location < 0) or np.any(candidate_location >= 5):
                continue
            if obs["trees"][tuple(candidate_location)] == 1 or obs["bushes"][tuple(candidate_location)] == 1:
                continue
            if np.array_equal(candidate_location, base_env._target_location):
                continue

            base_env._agent_location = candidate_location.copy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            assert np.array_equal(next_obs["agent"], np.array([bush_row, bush_column]))
            expected_reward = (
                base_env.step_penalty
                + base_env.bush_penalty
                + progress_bonus(base_env, candidate_location, np.array([bush_row, bush_column]))
            )
            assert np.isclose(reward, expected_reward)
            assert not terminated
            assert not truncated
            assert "distance" in info
            bush_tested = True
            break
        if bush_tested:
            break

    assert bush_tested

    env.close()


def test_goal_is_in_corners_and_terminates():
    env = gym.make("gymnasium_env/GridWorld-v0", size=5, tree_count=3, bush_count=2)
    obs, _ = env.reset(seed=123)
    base_env = env.unwrapped

    base_env._agent_location = np.array([0, 3])
    obs, reward, terminated, truncated, _ = env.step(0)
    assert np.array_equal(obs["agent"], np.array([0, 4]))
    assert terminated is True
    assert truncated is False
    assert np.isclose(reward, base_env.goal_reward)

    env.close()


def test_chase_zone_penalizes_and_grows():
    env = gym.make(
        "gymnasium_env/GridWorld-v0",
        size=5,
        tree_count=0,
        bush_count=0,
        chase_activation_step=0,
        chase_growth_interval=1,
        chase_growth_rate=1.0,
        chase_speed=0,
        chase_penalty=-0.07,
    )
    obs, _ = env.reset(seed=123)
    base_env = env.unwrapped

    obs, reward, terminated, truncated, info = env.step(0)

    assert np.isclose(info["chase_radius"], 1.0)
    expected_reward = (
        base_env.step_penalty
        + base_env.chase_penalty
        + progress_bonus(base_env, np.array([4, 0]), np.array([4, 1]))
    )
    assert np.isclose(reward, expected_reward)
    assert terminated is False
    assert truncated is False
    assert np.array_equal(obs["agent"], np.array([4, 1]))

    env.close()


def test_revisit_penalty_discourages_backtracking():
    env = gym.make("gymnasium_env/GridWorld-v0", size=5, tree_count=0, bush_count=0)
    obs, _ = env.reset(seed=123)
    base_env = env.unwrapped

    obs, reward_forward, terminated, truncated, _ = env.step(0)
    assert np.array_equal(obs["agent"], np.array([4, 1]))
    assert terminated is False
    assert truncated is False

    obs, reward_back, terminated, truncated, _ = env.step(2)
    expected_back_reward = (
        base_env.step_penalty
        + base_env.revisit_penalty
        + base_env.approach_reward_weight * (27.0 - 28.0)
    )
    assert np.array_equal(obs["agent"], np.array([4, 0]))
    assert reward_forward > reward_back
    assert np.isclose(reward_back, expected_back_reward)
    assert terminated is False
    assert truncated is False

    env.close()
