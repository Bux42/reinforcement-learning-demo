"""2048 reinforcement-learning demo package."""

from .determinism import configure_determinism
from .envs.registration import ENV_ID, register_envs

configure_determinism()
register_envs()

__all__ = ["ENV_ID", "configure_determinism", "register_envs"]
