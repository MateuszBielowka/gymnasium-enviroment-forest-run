from collections import defaultdict

import gymnasium as gym
import numpy as np
import gymnasium_env


def to_state(obs: dict[str, np.ndarray]) -> tuple[int, int, int, int]:
    a0, a1 = obs["agent"]
    t0, t1 = obs["target"]
    return int(a0), int(a1), int(t0), int(t1)


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))
    return int(np.argmax(q_values))


def train(episodes: int = 1000, alpha: float = 0.2, gamma: float = 0.95, epsilon: float = 0.1):
    env = gym.make("gymnasium_env/GridWorld-v0", size=5)
    rng = np.random.default_rng(123)

    q_table = defaultdict(lambda: np.zeros(env.action_space.n, dtype=float))

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        state = to_state(obs)

        done = False
        while not done:
            action = epsilon_greedy(q_table[state], epsilon=epsilon, rng=rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = to_state(next_obs)

            td_target = reward + gamma * np.max(q_table[next_state])
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            done = terminated or truncated

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}")

    env.close()
    return q_table


def evaluate(q_table, episodes: int = 20):
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None, size=5)
    wins = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=10_000 + episode)
        state = to_state(obs)

        for _ in range(100):
            action = int(np.argmax(q_table[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            state = to_state(obs)
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
