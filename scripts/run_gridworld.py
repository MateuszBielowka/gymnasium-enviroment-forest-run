import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium_env.envs.grid_world import GridWorldEnv, n_agents
from scripts.train_qlearning import to_state
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
    obs_list, infos = env.reset(seed=42)
    terrain_trees = int(obs_list[0]["trees"].sum())
    terrain_bushes = int(obs_list[0]["bushes"].sum())
    print("START")
    print("obs:", obs_list[0])
    print("info:", infos[0])
    print(f"start={obs_list[0]['agent'].tolist()} goal={obs_list[0]['target'].tolist()} trees={terrain_trees} bushes={terrain_bushes}")

    total_rewards = [0.0 for _ in range(n_agents)]
    for step in range(max_steps):
        # action = env.action_space.sample()
        # obs, reward, terminated, truncated, info = env.step(action)
        actions = [env.action_space.sample() for _ in range(n_agents)]
        obs_list, rewards, terminateds, truncated, infos = env.step(actions)
        for i, (obs, reward, action, terminated, info) in enumerate(zip(obs_list, rewards, actions, terminateds, infos)):
            total_rewards[i] += reward

            print(
                f"step={step:02d} action={action} reward={reward:.2f} terminated={terminated} truncated={truncated} distance={info['distance']:.1f}"
            )

        if all(terminateds) or truncated:
            break

        if render_mode == "human":
            time.sleep(0.15)

    print("SUMA NAGROD:", total_rewards)
    env.close()


# def run_trained_episode(
#     render_mode: str = "human",
#     size: int = 30,
#     max_steps: int = 200,
#     tree_count: int | None = None,
#     bush_count: int | None = None,
#     train_episodes: int = 1500,
# ) -> None:
#     q_table = train(size=size, episodes=train_episodes)
#     env = GridWorldEnv(
#         render_mode=render_mode,
#         size=size,
#         tree_count=tree_count,
#         bush_count=bush_count,
#     )
#     obs_list, infos = env.reset(seed=42)

#     terrain_trees = int(obs_list[0]["trees"].sum())
#     terrain_bushes = int(obs_list[0]["bushes"].sum())
#     print("START TRAINED RUN")
#     print("obs:", obs_list[0])
#     print("info:", infos[0])
#     print(
#         f"start={obs_list[0]['agent'].tolist()} goal={obs_list[0]['target'].tolist()} trees={terrain_trees} bushes={terrain_bushes}"
#     )

#     total_reward = 0.0
#     for step in range(max_steps):
#         state = to_state(obs_list[0], env)
#         action = int(np.argmax(q_table[state]))
#         obs, reward, terminated, truncated, info = env.step(action)
#         total_reward += reward

#         print(
#             f"step={step:02d} action={action} reward={reward:.2f} terminated={terminated} truncated={truncated} distance={info['distance']:.1f}"
#         )

#         if terminated or truncated:
#             break

#         if render_mode == "human":
#             time.sleep(0.15)

#     print("SUMA NAGROD:", total_reward)
#     env.close()

def run_trained_episode(
    render_mode: str = "human",
    size: int = 30,
    max_steps: int = 200,
    tree_count: int | None = None,
    bush_count: int | None = None,
    train_episodes: int = 1500,
) -> None:
    from scripts.train_qlearning import train_multi
    q_tables = train_multi(size=size, episodes=train_episodes)

    env = GridWorldEnv(
        render_mode=render_mode,
        size=size,
        tree_count=tree_count,
        bush_count=bush_count,
    )
    obs_list, infos = env.reset(seed=42)
    base_env = env

    total_rewards = [0.0] * n_agents
    for step in range(max_steps):
        states = [to_state(obs_list[i], base_env) for i in range(n_agents)]
        actions = [int(np.argmax(q_tables[i][states[i]])) for i in range(n_agents)]
        obs_list, rewards, terminateds, truncated, infos = env.step(actions)

        for i in range(n_agents):
            total_rewards[i] += rewards[i]

        print(f"step={step:02d} " + " | ".join(
            f"a{i} r={rewards[i]:.2f} done={terminateds[i]}" for i in range(n_agents)
        ))

        if all(terminateds) or truncated:
            break

        if render_mode == "human":
            time.sleep(0.15)

    print("SUMA NAGROD:", total_rewards)
    env.close()


if __name__ == "__main__":
    run_trained_episode()
