import time

import gymnasium as gym
import gymnasium_env


def run_random_episode(render_mode: str = "human", size: int = 30, max_steps: int = 50) -> None:
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode=render_mode, size=size)
    obs, info = env.reset(seed=42)
    print("START")
    print("obs:", obs)
    print("info:", info)

    total_reward = 0.0
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"step={step:02d} action={action} reward={reward:.1f} terminated={terminated} truncated={truncated} distance={info['distance']:.1f}"
        )

        if terminated or truncated:
            break

        if render_mode == "human":
            time.sleep(0.15)

    print("SUMA NAGROD:", total_reward)
    env.close()


if __name__ == "__main__":
    run_random_episode()
