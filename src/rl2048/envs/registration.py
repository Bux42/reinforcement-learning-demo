"""Gymnasium registration for the 2048 environment."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

ENV_ID = "TwentyFortyEight-v0"


def register_envs() -> None:
    """Register the environment once per Python process."""

    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="rl2048.envs.twenty_forty_eight:TwentyFortyEightEnv",
        )
