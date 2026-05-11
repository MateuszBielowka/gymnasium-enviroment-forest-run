from collections import defaultdict
import argparse
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium_env
import gymnasium as gym
import numpy as np
from gymnasium_env.envs.grid_world import GridWorldEnv

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gridworld_q_tables.pkl"


def default_q_row() -> np.ndarray:
    return np.full(4, 0.1, dtype=float)


def make_q_table() -> defaultdict[tuple[int, ...], np.ndarray]:
    return defaultdict(default_q_row)


def _cell_type(base_env, row: int, column: int) -> int:
    if row < 0 or row >= base_env.size or column < 0 or column >= base_env.size:
        return 3
    if base_env._tree_map[row, column] == 1:
        return 1
    if base_env._bush_map[row, column] == 1:
        return 2
    return 0


# def to_state(obs: dict[str, np.ndarray], base_env) -> tuple[int, ...]:
#     agent_row, agent_column = (int(value) for value in obs["agent"])
#     target_row, target_column = (int(value) for value in obs["target"])

#     relative_row = target_row - agent_row
#     relative_column = target_column - agent_column

#     local_features = (
#         _cell_type(base_env, agent_row - 1, agent_column),
#         _cell_type(base_env, agent_row, agent_column + 1),
#         _cell_type(base_env, agent_row + 1, agent_column),
#         _cell_type(base_env, agent_row, agent_column - 1),
#     )

#     chase_active = int(base_env._step_count >= base_env.chase_activation_step)
#     chase_radius = int(round(base_env._chase_radius))
#     chase_distance = int(
#         np.linalg.norm(np.array([agent_row, agent_column]) - base_env._chase_center, ord=2)
#     )

#     current_cell = _cell_type(base_env, agent_row, agent_column)

#     return (
#         relative_row,
#         relative_column,
#         *local_features,
#         current_cell,
#         chase_active,
#         chase_radius,
#         chase_distance,
#     )

# def to_state(obs: dict[str, np.ndarray], base_env) -> tuple[int, ...]:
#     agent_row, agent_column = (int(value) for value in obs["agent"])
#     target_row, target_column = (int(value) for value in obs["target"])

#     relative_row = target_row - agent_row
#     relative_column = target_column - agent_column

#     r, c = agent_row, agent_column

#     ring_features = (
#         _cell_type(base_env, r - 1, c),      # N
#         _cell_type(base_env, r - 1, c + 1),  # NE
#         _cell_type(base_env, r,     c + 1),  # E
#         _cell_type(base_env, r + 1, c + 1),  # SE
#         _cell_type(base_env, r + 1, c),      # S
#         _cell_type(base_env, r + 1, c - 1),  # SW
#         _cell_type(base_env, r,     c - 1),  # W
#         _cell_type(base_env, r - 1, c - 1),  # NW
#     )

#     far_features = (
#         _cell_type(base_env, r - 2, c),  # N2
#         _cell_type(base_env, r,     c + 2),  # E2
#         _cell_type(base_env, r + 2, c),  # S2
#         _cell_type(base_env, r,     c - 2),  # W2
#     )

#     local_features = ring_features + far_features

#     chase_active = int(base_env._step_count >= base_env.chase_activation_step)
#     chase_radius = int(round(base_env._chase_radius))
#     # chase_distance = int(
#     #     np.linalg.norm(np.array([r, c]) - base_env._chase_center, ord=2)
#     # )
#     dist_to_center = float(np.linalg.norm(np.array([r, c]) - base_env._chase_center, ord=2))
#     chase_distance_to_edge = int(np.clip(
#         round(dist_to_center - base_env._chase_radius), -5, 10
#     ))

#     current_cell = _cell_type(base_env, r, c)

#     return (
#         relative_row,
#         relative_column,
#         *local_features,
#         current_cell,
#         chase_active,
#         chase_radius,
#         chase_distance_to_edge
#     )


def to_state(obs: dict[str, np.ndarray], base_env) -> tuple[int, ...]:
    agent_row, agent_column = (int(value) for value in obs["agent"])
    target_row, target_column = (int(value) for value in obs["target"])

    dir_row = int(np.sign(target_row - agent_row))
    dir_col = int(np.sign(target_column - agent_column))

    r, c = agent_row, agent_column

    local_features = (
        _cell_type(base_env, r - 1, c),  # N
        _cell_type(base_env, r,     c + 1),  # E
        _cell_type(base_env, r + 1, c),  # S
        _cell_type(base_env, r,     c - 1),  # W
    )

    chase_active = int(base_env._step_count >= base_env.chase_activation_step)
    dist_to_center = float(np.linalg.norm(np.array([r, c]) - base_env._chase_center, ord=2))
    in_danger = int(chase_active and dist_to_center <= base_env._chase_radius + 2)

    return (
        dir_row,
        dir_col,
        *local_features,
        in_danger
    )


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))
    return int(np.argmax(q_values))


def make_train_env(size: int = 30) -> gym.Env:
    return gym.make("gymnasium_env/GridWorld-v0", size=size)


# def train(
#     size: int = 5,
#     episodes: int = 5000,
#     alpha: float = 0.15,
#     gamma: float = 0.98,
#     epsilon_start: float = 1.0,
#     epsilon_end: float = 0.05,
#     epsilon_decay: float = 0.995,
# ):
#     env = make_train_env(size=size)
#     rng = np.random.default_rng(123)

#     q_table = defaultdict(lambda: np.full(env.action_space.n, 0.1, dtype=float))

#     epsilon = epsilon_start

#     for episode in range(episodes):
#         obs, _ = env.reset(seed=episode)
#         base_env = env.unwrapped
#         state = to_state(obs, base_env)

#         done = False
#         while not done:
#             action = epsilon_greedy(q_table[state], epsilon=epsilon, rng=rng)
#             next_obs, reward, terminated, truncated, _ = env.step(action)
#             next_state = to_state(next_obs, base_env)

#             td_target = reward + gamma * np.max(q_table[next_state]) * (0.0 if terminated else 1.0)
#             td_error = td_target - q_table[state][action]
#             q_table[state][action] += alpha * td_error

#             state = next_state
#             done = terminated or truncated

#         epsilon = max(epsilon_end, epsilon * epsilon_decay)

#         if (episode + 1) % 100 == 0:
#             print(f"Episode {episode + 1}/{episodes} epsilon={epsilon:.3f}")

#     env.close()
#     return q_table


def train_multi(
    size: int = 30,
    episodes: int = 8000,
    agent_configs: list[dict] | None = None,
):
    if agent_configs is None:
        agent_configs = [
            {"alpha": 0.15, "gamma": 0.98, "epsilon_decay": 0.995},
            {"alpha": 0.10, "gamma": 0.95, "epsilon_decay": 0.990},
            {"alpha": 0.20, "gamma": 0.99, "epsilon_decay": 0.998},
        ]

    n = len(agent_configs)
    env = GridWorldEnv(size=size)
    rng = np.random.default_rng(123)

    q_tables = [make_q_table() for _ in range(n)]
    epsilons = [1.0] * n

    for episode in range(episodes):
        obs_list, _ = env.reset(seed=episode)
        base_env = env.unwrapped
        states = [to_state(obs_list[i], base_env) for i in range(n)]

        done = False
        while not done:
            actions = [
                epsilon_greedy(q_tables[i][states[i]], epsilons[i], rng)
                for i in range(n)
            ]
            next_obs_list, rewards, terminateds, truncated, _ = env.step(actions)
            next_states = [to_state(next_obs_list[i], base_env) for i in range(n)]

            for i in range(n):
                cfg = agent_configs[i]
                td_target = rewards[i] + cfg["gamma"] * np.max(q_tables[i][next_states[i]]) * (0.0 if terminateds[i] else 1.0)
                td_error = td_target - q_tables[i][states[i]][actions[i]]
                q_tables[i][states[i]][actions[i]] += cfg["alpha"] * td_error

            states = next_states
            done = all(terminateds) or truncated

        for i in range(n):
            epsilons[i] = max(0.05, epsilons[i] * agent_configs[i]["epsilon_decay"])

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} epsilons={[f'{e:.3f}' for e in epsilons]}")

    env.close()
    return q_tables


def save_q_tables(
    path: Path,
    q_tables,
    *,
    size: int,
    agent_configs: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "size": size,
        "agent_configs": agent_configs,
        "q_tables": [dict(q_table) for q_table in q_tables],
    }
    with path.open("wb") as file:
        pickle.dump(payload, file)


def load_q_tables(path: Path):
    with path.open("rb") as file:
        payload = pickle.load(file)

    q_tables = [make_q_table() for _ in payload["q_tables"]]
    for index, table_data in enumerate(payload["q_tables"]):
        q_tables[index].update(table_data)

    return q_tables, payload


# def evaluate(q_table, size: int = 5, episodes: int = 20):
#     env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None, size=size)
#     wins = 0

#     for episode in range(episodes):
#         obs, _ = env.reset(seed=10_000 + episode)
#         base_env = env.unwrapped
#         state = to_state(obs, base_env)

#         for _ in range(100):
#             action = int(np.argmax(q_table[state]))
#             obs, reward, terminated, truncated, _ = env.step(action)
#             state = to_state(obs, base_env)
#             if terminated:
#                 wins += 1
#                 break
#             if truncated:
#                 break

#     env.close()
#     print(f"Skutecznosc: {wins}/{episodes} = {wins / episodes:.2%}")


def evaluate_multi(q_tables, size: int = 30, episodes: int = 20):
    from gymnasium_env.envs.grid_world import GridWorldEnv, n_agents
    env = GridWorldEnv(size=size)
    wins_per_agent = [0] * n_agents

    for episode in range(episodes):
        obs_list, _ = env.reset(seed=10_000 + episode)

        for _ in range(200):
            prev_done = env._done_agents[:]
            states = [to_state(obs_list[i], env) for i in range(n_agents)]
            actions = [int(np.argmax(q_tables[i][states[i]])) for i in range(n_agents)]
            obs_list, rewards, terminateds, truncated, _ = env.step(actions)

            for i in range(n_agents):
                if terminateds[i] and not prev_done[i]:
                    wins_per_agent[i] += 1

            if all(terminateds) or truncated:
                break

    env.close()
    for i in range(n_agents):
        print(f"Agent {i}: {wins_per_agent[i]}/{episodes} = {wins_per_agent[i]/episodes:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-learning agents for GridWorld.")
    parser.add_argument("--size", type=int, default=30, help="Grid size used during training.")
    parser.add_argument("--episodes", type=int, default=8000, help="Number of training episodes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after training.")
    args = parser.parse_args()

    agent_configs = [
        {"alpha": 0.15, "gamma": 0.98, "epsilon_decay": 0.995},
        {"alpha": 0.10, "gamma": 0.95, "epsilon_decay": 0.990},
        {"alpha": 0.20, "gamma": 0.99, "epsilon_decay": 0.998},
    ]

    q_tables = train_multi(size=args.size, episodes=args.episodes, agent_configs=agent_configs)
    save_q_tables(args.output, q_tables, size=args.size, agent_configs=agent_configs)
    print(f"Saved trained model to {args.output}")

    if args.evaluate:
        evaluate_multi(q_tables, size=args.size)
