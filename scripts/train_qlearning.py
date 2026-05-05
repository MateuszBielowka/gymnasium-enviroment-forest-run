from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium_env
import gymnasium as gym
import numpy as np


def _cell_type(base_env, row: int, column: int) -> int:
    if row < 0 or row >= base_env.size or column < 0 or column >= base_env.size:
        return 3
    if base_env._tree_map[row, column] == 1:
        return 1
    if base_env._bush_map[row, column] == 1:
        return 2
    return 0


def to_state(obs: dict[str, np.ndarray], base_env) -> tuple[int, ...]:
    agent_row, agent_column = (int(value) for value in obs["agent"])
    target_row, target_column = (int(value) for value in obs["target"])

    relative_row = target_row - agent_row
    relative_column = target_column - agent_column

    local_features = (
        _cell_type(base_env, agent_row - 1, agent_column),
        _cell_type(base_env, agent_row, agent_column + 1),
        _cell_type(base_env, agent_row + 1, agent_column),
        _cell_type(base_env, agent_row, agent_column - 1),
    )

    chase_active = int(base_env._step_count >= base_env.chase_activation_step)
    chase_radius = int(round(base_env._chase_radius))
    chase_distance = int(
        np.linalg.norm(np.array([agent_row, agent_column]) - base_env._chase_center, ord=2)
    )

    current_cell = _cell_type(base_env, agent_row, agent_column)

    return (
        relative_row,
        relative_column,
        *local_features,
        current_cell,
        chase_active,
        chase_radius,
        chase_distance,
    )


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))
    return int(np.argmax(q_values))


def make_train_env(size: int = 5) -> gym.Env:
    return gym.make("gymnasium_env/GridWorld-v0", size=size)


def train(
    size: int = 5,
    episodes: int = 5000,
    alpha: float = 0.15,
    gamma: float = 0.98,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
):
    env = make_train_env(size=size)
    rng = np.random.default_rng(123)

    q_table = defaultdict(lambda: np.full(env.action_space.n, 0.1, dtype=float))

    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        base_env = env.unwrapped
        state = to_state(obs, base_env)

        done = False
        while not done:
            action = epsilon_greedy(q_table[state], epsilon=epsilon, rng=rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = to_state(next_obs, base_env)

            td_target = reward + gamma * np.max(q_table[next_state]) * (0.0 if terminated else 1.0)
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            done = terminated or truncated

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} epsilon={epsilon:.3f}")

    env.close()
    return q_table


def evaluate(q_table, size: int = 5, episodes: int = 20):
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None, size=size)
    wins = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=10_000 + episode)
        base_env = env.unwrapped
        state = to_state(obs, base_env)

        for _ in range(100):
            action = int(np.argmax(q_table[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            state = to_state(obs, base_env)
            if terminated:
                wins += 1
                break
            if truncated:
                break

    env.close()
    print(f"Skutecznosc: {wins}/{episodes} = {wins / episodes:.2%}")


if __name__ == "__main__":
    q = train()
    evaluate(q)
