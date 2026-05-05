import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium_env.envs.grid_world import GridWorldEnv
from scripts.train_qlearning import to_state, train
import numpy as np


def run_random_episode(
    render_mode: str = "human",
    size: int = 15,
    max_steps: int = 80,
    tree_count: int = 24,
    bush_count: int = 16,
) -> None:
    env = GridWorldEnv(
        render_mode=render_mode,
        size=size,
        tree_count=tree_count,
        bush_count=bush_count,
    )
    obs, info = env.reset(seed=42)
    terrain_trees = int(obs["trees"].sum())
    terrain_bushes = int(obs["bushes"].sum())
    print("START")
    print("obs:", obs)
    print("info:", info)
    print(f"start={obs['agent'].tolist()} goal={obs['target'].tolist()} trees={terrain_trees} bushes={terrain_bushes}")

    total_reward = 0.0
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"step={step:02d} action={action} reward={reward:.2f} terminated={terminated} truncated={truncated} distance={info['distance']:.1f}"
        )

        if terminated or truncated:
            break

        if render_mode == "human":
            time.sleep(0.15)

    print("SUMA NAGROD:", total_reward)
    env.close()


def run_trained_episode(
    render_mode: str = "human",
    size: int = 30,
    max_steps: int = 200,
    tree_count: int | None = None,
    bush_count: int | None = None,
    train_episodes: int = 1500,
) -> None:
    q_table = train(size=size, episodes=train_episodes)
    env = GridWorldEnv(
        render_mode=render_mode,
        size=size,
        tree_count=tree_count,
        bush_count=bush_count,
    )
    obs, info = env.reset(seed=42)

    terrain_trees = int(obs["trees"].sum())
    terrain_bushes = int(obs["bushes"].sum())
    print("START TRAINED RUN")
    print("obs:", obs)
    print("info:", info)
    print(
        f"start={obs['agent'].tolist()} goal={obs['target'].tolist()} trees={terrain_trees} bushes={terrain_bushes}"
    )

    total_reward = 0.0
    for step in range(max_steps):
        state = to_state(obs, env)
        action = int(np.argmax(q_table[state]))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"step={step:02d} action={action} reward={reward:.2f} terminated={terminated} truncated={truncated} distance={info['distance']:.1f}"
        )

        if terminated or truncated:
            break

        if render_mode == "human":
            time.sleep(0.15)

    print("SUMA NAGROD:", total_reward)
    env.close()


if __name__ == "__main__":
    run_trained_episode()
