"""A small, deterministic-friendly Gymnasium implementation of 2048.

The observation is a flattened log-scaled board plus normalized move count.
Actions use the common ordering ``UP, RIGHT, DOWN, LEFT``.  ``action_masks``
returns moves that actually change the board when masking is enabled.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..envs.renderer import TwentyFortyEightRenderer

BOARD_SIZE = 4
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
TARGET_TILE = 2048
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT = range(4)


class TwentyFortyEightEnv(gym.Env[np.ndarray, int]):
    """Classic 4x4 2048 with optional legal-action masking."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        *,
        max_steps: int = 2_000,
        mask_actions: bool = False,
        invalid_action_penalty: float = -1.0,
        reward_scale: float = 0.01,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if invalid_action_penalty > 0:
            raise ValueError("invalid_action_penalty must be zero or negative")
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"unsupported render_mode={render_mode!r}")

        self.max_steps = int(max_steps)
        self.mask_actions = bool(mask_actions)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.reward_scale = float(reward_scale)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        # 16 log2 tile values (2048 maps to 1) + normalized move count.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(BOARD_CELLS + 1,),
            dtype=np.float32,
        )

        self._board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
        self._score = 0
        self._step_count = 0
        self._invalid_action_count = 0
        self._last_action: int | None = None
        self._last_invalid_action = False
        self._last_move_score = 0
        self._last_reward = 0.0
        self._won = False
        self._episode_done = False
        self._cached_legal_mask: np.ndarray | None = None
        self._renderer: TwentyFortyEightRenderer | None = None

    @property
    def board(self) -> np.ndarray:
        """Return a copy of the current board for debugging and evaluation."""

        return self._board.copy()

    @property
    def score(self) -> int:
        return int(self._score)

    @property
    def max_tile(self) -> int:
        return int(self._board.max(initial=0))

    @property
    def empty_cells(self) -> int:
        return int(np.count_nonzero(self._board == 0))

    @property
    def step_count(self) -> int:
        return int(self._step_count)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self._board.fill(0)
        self._score = 0
        self._step_count = 0
        self._invalid_action_count = 0
        self._last_action = None
        self._last_invalid_action = False
        self._last_move_score = 0
        self._last_reward = 0.0
        self._won = False
        self._episode_done = False
        self._cached_legal_mask = None
        self._spawn_tile()
        self._spawn_tile()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}; expected integer in [0, 3]")

        legal_mask = (
            self._cached_legal_mask.copy()
            if self._cached_legal_mask is not None
            else self.legal_action_mask()
        )
        self._cached_legal_mask = None
        self._last_action = action
        self._last_move_score = 0
        self._step_count += 1
        invalid_action = not bool(legal_mask[action])
        self._last_invalid_action = invalid_action

        if invalid_action:
            self._invalid_action_count += 1
            reward = self.invalid_action_penalty
            terminated = not bool(legal_mask.any())
        else:
            moved_board, gained, changed = self._move_board(self._board, action)
            if not changed:
                # Keep this invariant explicit: legal_action_mask and step must agree.
                raise RuntimeError("action was marked legal but did not change the board")
            self._board = moved_board
            self._score += gained
            self._last_move_score = gained
            self._spawn_tile()
            self._won = self._won or self.max_tile >= TARGET_TILE
            terminated = not bool(self.legal_action_mask().any())
            reward = self.reward_scale * gained

        truncated = not terminated and self._step_count >= self.max_steps
        self._episode_done = terminated or truncated
        self._last_reward = float(reward)
        if self.render_mode is not None:
            self.render()
        return self._observation(), float(reward), terminated, truncated, self._info()

    def legal_action_mask(self) -> np.ndarray:
        """Return the true board-changing mask, regardless of training settings."""

        return np.asarray(
            [self._move_board(self._board, action)[2] for action in range(len(ACTION_NAMES))],
            dtype=bool,
        )

    def action_masks(self) -> np.ndarray:
        """Return the mask consumed by MaskablePPO.

        With masking disabled, all actions remain available so experiments can
        compare the exact same environment with and without masking.
        """

        legal_mask = self.legal_action_mask()
        self._cached_legal_mask = legal_mask
        return legal_mask.copy() if self.mask_actions else np.ones(len(ACTION_NAMES), dtype=bool)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = TwentyFortyEightRenderer(self.render_mode)
        return self._renderer.render(
            board=self._board,
            score=self._score,
            step_count=self._step_count,
            max_steps=self.max_steps,
            last_action=self._last_action,
            last_reward=self._last_reward,
            invalid_action_count=self._invalid_action_count,
            mask_actions=self.mask_actions,
        )

    def wait_for_human_action(self) -> bool:
        """Pause human playback until its controller requests one action."""

        if self.render_mode != "human":
            return True
        if self._renderer is None:
            self.render()
        assert self._renderer is not None
        return self._renderer.wait_for_action()

    def wait_for_human_close(self) -> None:
        if self.render_mode == "human" and self._renderer is not None:
            self._renderer.wait_for_close()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @staticmethod
    def _move_line(line: np.ndarray) -> tuple[np.ndarray, int]:
        """Slide a line left and merge equal neighbors once, as in 2048."""

        compact = [int(value) for value in line if value != 0]
        result: list[int] = []
        gained = 0
        index = 0
        while index < len(compact):
            if index + 1 < len(compact) and compact[index] == compact[index + 1]:
                merged = compact[index] * 2
                result.append(merged)
                gained += merged
                index += 2
            else:
                result.append(compact[index])
                index += 1
        result.extend([0] * (BOARD_SIZE - len(result)))
        return np.asarray(result, dtype=np.int64), gained

    @classmethod
    def _move_board(cls, board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
        """Apply one action without mutating ``board``."""

        if action not in range(len(ACTION_NAMES)):
            raise ValueError(f"invalid action {action}")
        moved = np.zeros_like(board)
        gained = 0
        if action in (ACTION_LEFT, ACTION_RIGHT):
            for row_index in range(BOARD_SIZE):
                line = board[row_index, :]
                if action == ACTION_RIGHT:
                    line = line[::-1]
                new_line, line_score = cls._move_line(line)
                if action == ACTION_RIGHT:
                    new_line = new_line[::-1]
                moved[row_index, :] = new_line
                gained += line_score
        else:
            for column_index in range(BOARD_SIZE):
                line = board[:, column_index]
                if action == ACTION_DOWN:
                    line = line[::-1]
                new_line, line_score = cls._move_line(line)
                if action == ACTION_DOWN:
                    new_line = new_line[::-1]
                moved[:, column_index] = new_line
                gained += line_score
        return moved, gained, not np.array_equal(moved, board)

    def _spawn_tile(self) -> None:
        empty = np.argwhere(self._board == 0)
        if len(empty) == 0:
            return
        row, column = empty[int(self.np_random.integers(len(empty)))]
        self._board[row, column] = 4 if self.np_random.random() < 0.1 else 2

    def _observation(self) -> np.ndarray:
        observation = np.zeros(BOARD_CELLS + 1, dtype=np.float32)
        flattened = self._board.reshape(-1)
        occupied = flattened > 0
        observation[:BOARD_CELLS][occupied] = np.clip(
            np.log2(flattened[occupied]).astype(np.float32) / np.log2(TARGET_TILE),
            0.0,
            1.0,
        )
        observation[BOARD_CELLS] = self._step_count / self.max_steps
        return observation

    def _info(self) -> dict[str, Any]:
        legal_mask = self.legal_action_mask()
        return {
            "action": None if self._last_action is None else ACTION_NAMES[self._last_action],
            "action_index": self._last_action,
            "legal_actions": int(legal_mask.sum()),
            "legal_action_mask": legal_mask.copy(),
            "invalid_action": self._last_invalid_action,
            "invalid_actions": self._invalid_action_count,
            "last_move_score": self._last_move_score,
            "last_reward": self._last_reward,
            "score": self._score,
            "max_tile": self.max_tile,
            "empty_cells": self.empty_cells,
            "step": self._step_count,
            "mask_actions": self.mask_actions,
            "won": self._won,
            "game_over": self._episode_done,
        }
