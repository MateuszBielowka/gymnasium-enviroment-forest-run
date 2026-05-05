from __future__ import annotations
from pathlib import Path

from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


TEXTURES_DIR = Path(__file__).parents[2] / "textures"

n_agents = 3

class Actions(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        size: int = 5,
        tree_count: int | None = None,
        bush_count: int | None = None,
        tree_density: float = 0.12,
        bush_density: float = 0.08,
        step_penalty: float = -0.1,
        tree_penalty: float = -1.0,
        bush_penalty: float = -0.2,
        chase_activation_step: int = 8,
        chase_growth_interval: int = 4,
        chase_growth_rate: float = 1.0,
        chase_speed: int = 1,
        chase_penalty: float = -1.0,
        approach_reward_weight: float = 1.0,
        revisit_penalty: float = -0.1,
        max_steps: int = 200,
    ) -> None:
        self.size = size
        self.window_size = 768
        self.tree_count = tree_count
        self.bush_count = bush_count
        self.tree_density = tree_density
        self.bush_density = bush_density
        self.step_penalty = step_penalty
        self.tree_penalty = tree_penalty
        self.bush_penalty = bush_penalty
        self.chase_activation_step = chase_activation_step
        self.chase_growth_interval = chase_growth_interval
        self.chase_growth_rate = chase_growth_rate
        self.chase_speed = chase_speed
        self.chase_penalty = chase_penalty
        self.approach_reward_weight = approach_reward_weight
        self.revisit_penalty = revisit_penalty
        self.goal_reward = 100.0
        self.max_steps = max_steps

        self._start_location = np.array([self.size - 1, 0], dtype=int)
        self._target_anchor = np.array([0, self.size - 1], dtype=int)
        self._chase_origin = self._start_location.copy()

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "trees": spaces.Box(0, 1, shape=(size, size), dtype=np.int8),
                "bushes": spaces.Box(0, 1, shape=(size, size), dtype=np.int8),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([0, 1]),
            Actions.UP.value: np.array([-1, 0]),
            Actions.LEFT.value: np.array([0, -1]),
            Actions.DOWN.value: np.array([1, 0]),
        }

        # self._agent_location = np.array([-1, -1], dtype=int)

        self._agent_locations = [np.array([-1, -1], dtype=int) for _ in range(n_agents)]
        self._done_agents = [False] * n_agents
        self._finish_order = []

        self._target_location = np.array([-1, -1], dtype=int)
        self._tree_map = np.zeros((self.size, self.size), dtype=np.int8)
        self._bush_map = np.zeros((self.size, self.size), dtype=np.int8)
        self._chase_center = self._chase_origin.copy()
        self._chase_radius = 0.0
        self._step_count = 0
        self._visited_locations: set[tuple[int, int]] = set()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self, agent_idx: int) -> dict[str, np.ndarray]:
        return {
            "agent": self._agent_locations[agent_idx].copy(),
            "target": self._target_location.copy(),
            "trees": self._tree_map.copy(),
            "bushes": self._bush_map.copy(),
        }

    def _get_info(self, agent_idx: int) -> dict[str, float]:
        return {
            "distance": float(
                np.linalg.norm(self._agent_locations[agent_idx] - self._target_location, ord=1)
            ),
            "chase_radius": float(self._chase_radius),
            "chase_active": float(self._step_count >= self.chase_activation_step),
        }

    def _is_inside_chase_zone(self, location: np.ndarray) -> bool:
        return float(np.linalg.norm(location - self._chase_center, ord=2)) <= self._chase_radius

    def _update_chase_zone(self) -> None:
        if self._step_count < self.chase_activation_step:
            return

        steps_since_activation = self._step_count - self.chase_activation_step + 1
        if steps_since_activation > 0 and steps_since_activation % self.chase_growth_interval == 0:
            self._chase_radius += self.chase_growth_rate

    def _build_terrain(self) -> None:
        self._tree_map.fill(0)
        self._bush_map.fill(0)

        available_cells = [
            (row, column)
            for row in range(self.size)
            for column in range(self.size)
            if not np.array_equal((row, column), self._start_location)
            and not np.array_equal((row, column), self._target_location)
        ]
        if not available_cells:
            return

        available_cells_array = np.array(available_cells, dtype=int)
        remaining_cells = len(available_cells)

        tree_count = self.tree_count
        if tree_count is None:
            tree_count = int(round(self.size * self.size * self.tree_density))
        tree_count = int(np.clip(tree_count, 0, remaining_cells))

        tree_indices = (
            self.np_random.choice(remaining_cells, size=tree_count, replace=False)
            if tree_count > 0
            else np.array([], dtype=int)
        )
        tree_cells = available_cells_array[tree_indices]

        for row, column in tree_cells:
            self._tree_map[row, column] = 1

        remaining_cells = [
            (row, column)
            for row, column in available_cells
            if self._tree_map[row, column] == 0
        ]
        if not remaining_cells:
            return

        remaining_cells_array = np.array(remaining_cells, dtype=int)
        bush_count = self.bush_count
        if bush_count is None:
            bush_count = int(round(self.size * self.size * self.bush_density))
        bush_count = int(np.clip(bush_count, 0, len(remaining_cells)))

        bush_indices = (
            self.np_random.choice(len(remaining_cells), size=bush_count, replace=False)
            if bush_count > 0
            else np.array([], dtype=int)
        )
        bush_cells = remaining_cells_array[bush_indices]

        for row, column in bush_cells:
            self._bush_map[row, column] = 1

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        super().reset(seed=seed)

        # self._agent_location = self._start_location.copy()
        self._agent_locations = [self._start_location.copy() for _ in range(n_agents)]
        self._done_agents = [False] * n_agents
        self._finish_order = []
        self._target_location = self._target_anchor.copy()
        self._build_terrain()
        self._chase_center = self._chase_origin.copy()
        self._chase_radius = 0.0
        self._step_count = 0
        self._visited_locations = {tuple(self._agent_locations[i]) for i in range(n_agents)}

        observations = [self._get_obs(i) for i in range(n_agents)]
        infos = [self._get_info(i) for i in range(n_agents)]

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    def step(self, actions: list[int]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, float]]:
        self._step_count += 1
        self._update_chase_zone()

        observations = []
        rewards = []
        terminateds = []
        infos = []
        for i, action in enumerate(actions):
            if self._done_agents[i]:
                observations.append(self._get_obs(i))
                rewards.append(0.0)
                terminateds.append(True)
                infos.append(self._get_info(i))
                continue

            previous_distance = float(
                np.linalg.norm(self._agent_locations[i] - self._target_location, ord=1)
            )

            direction = self._action_to_direction[action]
            proposed_location = np.clip(self._agent_locations[i] + direction, 0, self.size - 1)

            reward = self.step_penalty
            if self._tree_map[tuple(proposed_location)] == 1:
                proposed_location = self._agent_locations[i].copy()
                reward += self.tree_penalty
            else:
                self._agent_locations[i] = proposed_location
                if self._bush_map[tuple(self._agent_locations[i])] == 1:
                    reward += self.bush_penalty

            agent_location_key = tuple(self._agent_locations[i])
            if agent_location_key in self._visited_locations:
                reward += self.revisit_penalty
            else:
                self._visited_locations.add(agent_location_key)

            current_distance = float(
                np.linalg.norm(self._agent_locations[i] - self._target_location, ord=1)
            )
            distance_progress = previous_distance - current_distance
            reward += self.approach_reward_weight * distance_progress

            # if self._step_count >= self.chase_activation_step and self._is_inside_chase_zone(self._agent_locations[i]):
            #     reward += self.chase_penalty

            if self._step_count >= self.chase_activation_step:
                dist_to_center = float(np.linalg.norm(
                    self._agent_locations[i] - self._chase_center, ord=2
                ))
                if dist_to_center <= self._chase_radius:
                    # w strefie — pełna kara
                    reward += self.chase_penalty
                elif dist_to_center <= self._chase_radius + 3:
                    # strefa zagrożenia 3 komórki przed granicą — kara proporcjonalna
                    proximity = (self._chase_radius + 3 - dist_to_center) / 3
                    reward += self.chase_penalty * proximity * 0.4

            terminated = bool(np.array_equal(self._agent_locations[i], self._target_location))
            if terminated:
                reward = self.goal_reward
                self._done_agents[i] = True
                self._finish_order.append(i)

            observation = self._get_obs(i)    
            info = self._get_info(i)

            observations.append(observation)
            rewards.append(reward)
            terminateds.append(terminated)
            infos.append(info)

        all_done = all(self._done_agents)
        any_truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminateds, any_truncated, infos

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray | None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

            cell_size = int(self.window_size / self.size)
            self._tree_icon = pygame.transform.scale(
                pygame.image.load(TEXTURES_DIR / "tree.png").convert_alpha(),
                (cell_size, cell_size),
            )
            self._bush_icon = pygame.transform.scale(
                pygame.image.load(TEXTURES_DIR / "bush.png").convert_alpha(),
                (cell_size, cell_size),
            )
            self._goal_icon = pygame.transform.scale(
                pygame.image.load(TEXTURES_DIR / "house.png").convert_alpha(),
                (cell_size, cell_size),
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((30, 150, 65))
        pix_square_size = self.window_size / self.size

        if self._step_count >= self.chase_activation_step:
            chase_overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            chase_center = ((self._chase_center[::-1] + 0.5) * pix_square_size).astype(int)
            chase_radius = max(1, int(self._chase_radius * pix_square_size))
            pygame.draw.circle(
                chase_overlay,
                # (255, 120, 0, 70),
                # (255, 255, 255, 70),
                (200, 210, 220, 55),
                chase_center,
                chase_radius,
            )

            pygame.draw.circle(
                chase_overlay,
                (180, 190, 200, 90),
                chase_center,
                int(chase_radius * 0.6),
            )

            pygame.draw.circle(
                chase_overlay,
                # (255, 90, 0, 180),
                # (255, 255, 255, 180),
                (220, 225, 230, 130),
                chase_center,
                chase_radius,
                width=4,
            )
            canvas.blit(chase_overlay, (0, 0))

        # for row, column in np.argwhere(self._bush_map == 1):
        #     pygame.draw.rect(
        #         canvas,
        #         (112, 173, 71),
        #         pygame.Rect(
        #             pix_square_size * np.array([column, row]),
        #             (pix_square_size, pix_square_size),
        #         ),
        #     )

        # for row, column in np.argwhere(self._tree_map == 1):
        #     pygame.draw.rect(
        #         canvas,
        #         (64, 93, 37),
        #         pygame.Rect(
        #             pix_square_size * np.array([column, row]),
        #             (pix_square_size, pix_square_size),
        #         ),
        #     )

        for row, column in np.argwhere(self._bush_map == 1):
            canvas.blit(
                self._bush_icon,
                (pix_square_size * column, pix_square_size * row),
            )

        for row, column in np.argwhere(self._tree_map == 1):
            canvas.blit(
                self._tree_icon,
                (pix_square_size * column, pix_square_size * row),
            )

        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location[::-1],
        #         (pix_square_size, pix_square_size),
        #     ),
        # )

        canvas.blit(
            self._goal_icon,
            pix_square_size * self._target_location[::-1],
        )

        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location[::-1] + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )

        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
        for i, loc in enumerate(self._agent_locations):
            if not self._done_agents[i]:
                pygame.draw.circle(
                    canvas,
                    colors[i % len(colors)],
                    (loc[::-1] + 0.5) * pix_square_size,
                    pix_square_size / 3,
                )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
