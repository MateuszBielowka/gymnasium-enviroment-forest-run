import time
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium_env.envs.grid_world import GridWorldEnv, n_agents
from scripts.train_qlearning import DEFAULT_MODEL_PATH, load_q_tables, to_state
import numpy as np


def run_random_episode(
    render_mode: str = "human",
    size: int = 30,
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


def run_trained_episode(
    q_tables,
    render_mode: str = "human",
    size: int = 30,
    max_steps: int = 200,
    tree_count: int | None = None,
    bush_count: int | None = None,
) -> None:
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
    parser = argparse.ArgumentParser(description="Run GridWorld using a saved Q-learning model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved Q-learning model.",
    )
    parser.add_argument("--size", type=int, default=None, help="Override grid size used for the run.")
    parser.add_argument("--render-mode", type=str, default="human", help="Gymnasium render mode.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum number of steps to run.")
    args = parser.parse_args()

    q_tables, payload = load_q_tables(args.model)
    run_size = args.size if args.size is not None else int(payload["size"])
    run_trained_episode(
        q_tables,
        render_mode=args.render_mode,
        size=run_size,
        max_steps=args.max_steps,
    )
