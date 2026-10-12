"""Pygame renderer for the 2048 environment."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pygame

ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT = range(4)


class TwentyFortyEightRenderer:
    """Shared human and RGB-array renderer with a compact action indicator."""

    BOARD_SIZE = 4
    TILE_SIZE = 100
    GAP = 12
    MARGIN = 16
    HUD_HEIGHT = 72
    BOARD_PIXELS = BOARD_SIZE * TILE_SIZE + (BOARD_SIZE - 1) * GAP
    CONTROL_TOP = HUD_HEIGHT + BOARD_PIXELS + MARGIN
    CONTROL_HEIGHT = 64
    WIDTH = MARGIN * 2 + BOARD_SIZE * TILE_SIZE + (BOARD_SIZE - 1) * GAP
    HEIGHT = CONTROL_TOP + CONTROL_HEIGHT

    _BACKGROUND = (250, 248, 239)
    _BOARD_BACKGROUND = (187, 173, 160)
    _TEXT_DARK = (119, 110, 101)
    _TEXT_LIGHT = (249, 246, 242)
    _TILE_COLORS: ClassVar[dict[int, tuple[int, int, int]]] = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
    }

    def __init__(self, render_mode: str) -> None:
        if render_mode not in {"human", "rgb_array"}:
            raise ValueError(f"unsupported render mode {render_mode!r}")
        self.render_mode = render_mode
        pygame.init()
        self._screen: pygame.Surface | None = None
        if render_mode == "human":
            self._screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("2048 — reinforcement learning demo")
        self._font = pygame.font.SysFont("sans", 20, bold=True)
        self._tile_font = pygame.font.SysFont("sans", 42, bold=True)
        self._small_font = pygame.font.SysFont("sans", 15)
        self.auto_play = False
        self._step_requested = False
        self._quit_requested = False
        self._clock = pygame.time.Clock()
        self._auto_button = pygame.Rect(16, self.CONTROL_TOP + 14, 116, 36)
        self._next_button = pygame.Rect(144, self.CONTROL_TOP + 14, 142, 36)

    def render(
        self,
        *,
        board: np.ndarray,
        score: int,
        step_count: int,
        max_steps: int,
        last_action: int | None,
        last_reward: float,
        invalid_action_count: int,
        mask_actions: bool,
    ) -> np.ndarray | None:
        surface = self._screen or pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self._BACKGROUND)
        pygame.draw.rect(
            surface,
            self._BOARD_BACKGROUND,
            pygame.Rect(
                self.MARGIN,
                self.HUD_HEIGHT,
                self.WIDTH - self.MARGIN * 2,
                self.BOARD_PIXELS,
            ),
            border_radius=6,
        )

        action_name = "—" if last_action is None else ("UP", "RIGHT", "DOWN", "LEFT")[last_action]
        self._draw_text(surface, f"Score {score:>7}   Move {step_count}/{max_steps}", 16, 14)
        self._draw_text(
            surface,
            f"Action {action_name:<5}  reward {last_reward:+.2f}  {'MASKED' if mask_actions else 'UNMASKED'}",
            16,
            40,
            small=True,
        )
        if last_action is not None:
            self._draw_action_triangle(surface, (self.WIDTH - 24, 34), last_action)

        for row in range(self.BOARD_SIZE):
            for column in range(self.BOARD_SIZE):
                value = int(board[row, column])
                x = self.MARGIN + column * (self.TILE_SIZE + self.GAP)
                y = self.HUD_HEIGHT + row * (self.TILE_SIZE + self.GAP)
                pygame.draw.rect(
                    surface,
                    self._TILE_COLORS.get(value, self._color_for_large_tile(value)),
                    pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE),
                    border_radius=6,
                )
                if value:
                    font = self._tile_font
                    if value >= 1_000:
                        font = pygame.font.SysFont("sans", 31, bold=True)
                    rendered = font.render(str(value), True, self._TEXT_LIGHT if value > 4 else self._TEXT_DARK)
                    text_rect = rendered.get_rect(
                        center=(x + self.TILE_SIZE // 2, y + self.TILE_SIZE // 2)
                    )
                    surface.blit(rendered, text_rect)

        pygame.draw.rect(
            surface,
            (232, 225, 215),
            pygame.Rect(0, self.CONTROL_TOP, self.WIDTH, self.CONTROL_HEIGHT),
        )
        self._draw_button(
            surface,
            self._auto_button,
            f"Auto: {'ON' if self.auto_play else 'OFF'}",
            active=self.auto_play,
        )
        self._draw_button(surface, self._next_button, "Next action", active=False)
        self._draw_text(surface, f"invalid: {invalid_action_count}", 304, self.CONTROL_TOP + 25, small=True)
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()

    def wait_for_action(self) -> bool:
        """Wait until auto-play or the user requests one model action."""

        if self.render_mode != "human":
            return True
        while True:
            self._poll_events()
            if self._quit_requested:
                return False
            if self.auto_play:
                self._step_requested = False
                self._clock.tick(10)
                return True
            if self._step_requested:
                self._step_requested = False
                return True
            self._clock.tick(60)

    def wait_for_close(self) -> None:
        """Keep the final human frame visible until the window is closed."""

        if self.render_mode != "human":
            return
        while not self._quit_requested:
            self._poll_events()
            self._clock.tick(60)

    def _poll_events(self) -> None:
        controls_changed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._auto_button.collidepoint(event.pos):
                    self.auto_play = not self.auto_play
                    controls_changed = True
                elif self._next_button.collidepoint(event.pos):
                    self._step_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self._quit_requested = True
                elif event.key == pygame.K_SPACE:
                    self.auto_play = not self.auto_play
                    controls_changed = True
                elif event.key == pygame.K_n:
                    self._step_requested = True
        if controls_changed:
            self._refresh_auto_button()

    def _refresh_auto_button(self) -> None:
        if self._screen is None:
            return
        background = self._auto_button.inflate(4, 4)
        pygame.draw.rect(self._screen, (232, 225, 215), background, border_radius=8)
        self._draw_button(
            self._screen,
            self._auto_button,
            f"Auto: {'ON' if self.auto_play else 'OFF'}",
            active=self.auto_play,
        )
        pygame.display.flip()

    def _draw_button(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        *,
        active: bool,
    ) -> None:
        color = (120, 170, 116) if active else (187, 173, 160)
        pygame.draw.rect(surface, color, rect, border_radius=6)
        text = self._small_font.render(label, True, (255, 255, 255))
        surface.blit(text, text.get_rect(center=rect.center))

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        x: int,
        y: int,
        *,
        small: bool = False,
    ) -> None:
        font = self._small_font if small else self._font
        surface.blit(font.render(text, True, self._TEXT_DARK), (x, y))

    @staticmethod
    def _color_for_large_tile(value: int) -> tuple[int, int, int]:
        if value <= 0:
            return (205, 193, 180)
        hue = (math.log2(value) * 23) % 360
        # A readable, muted HSV-to-RGB approximation without another dependency.
        import colorsys

        red, green, blue = colorsys.hsv_to_rgb(hue / 360, 0.65, 0.92)
        return int(red * 255), int(green * 255), int(blue * 255)

    @staticmethod
    def _draw_action_triangle(surface: pygame.Surface, center: tuple[int, int], action: int) -> None:
        x, y = center
        if action == ACTION_UP:
            points = [(x, y - 10), (x - 8, y + 7), (x + 8, y + 7)]
        elif action == ACTION_RIGHT:
            points = [(x + 10, y), (x - 7, y - 8), (x - 7, y + 8)]
        elif action == ACTION_DOWN:
            points = [(x, y + 10), (x - 8, y - 7), (x + 8, y - 7)]
        else:
            points = [(x - 10, y), (x + 7, y - 8), (x + 7, y + 8)]
        pygame.draw.polygon(surface, (119, 110, 101), points)

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
        pygame.quit()
