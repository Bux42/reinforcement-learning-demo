"""Evaluation and human/RGB playback helpers."""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from . import ENV_ID


def evaluate(
    model_path: str | Path,
    *,
    episodes: int = 3,
    render_mode: str | None = "human",
    video_dir: str | Path | None = None,
    seed: int = 100,
    mask_actions: bool = False,
    max_steps: int = 2_000,
    invalid_action_penalty: float = -1.0,
    reward_scale: float = 0.01,
    wandb_project: str | None = None,
    use_wandb: bool = False,
) -> list[dict[str, int | float | bool]]:
    """Run deterministic episodes and print score, tile, and invalid-action data."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    wandb_project = wandb_project or os.environ.get("WANDB_PROJECT", "rl-2048")
    if render_mode == "none":
        render_mode = None
    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        mask_actions=mask_actions,
        max_steps=max_steps,
        invalid_action_penalty=invalid_action_penalty,
        reward_scale=reward_scale,
    )
    if video_dir is not None:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: True,
        )
    model = MaskablePPO.load(model_path, env=env)
    wandb_run = None
    results: list[dict[str, int | float | bool]] = []

    try:
        if use_wandb:
            import wandb

            wandb_run = wandb.init(project=wandb_project, job_type="evaluation")
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            if render_mode == "human":
                env.render()
            terminated = truncated = False
            quit_requested = False
            while not (terminated or truncated):
                if render_mode == "human" and not env.unwrapped.wait_for_human_action():
                    quit_requested = True
                    break
                masks = get_action_masks(env) if mask_actions else None
                action, _ = model.predict(
                    observation,
                    action_masks=masks,
                    deterministic=True,
                )
                observation, _, terminated, truncated, info = env.step(action)
            if quit_requested:
                break
            result = {
                "episode": episode,
                "score": int(info["score"]),
                "max_tile": int(info["max_tile"]),
                "moves": int(info["step"]),
                "invalid_actions": int(info["invalid_actions"]),
                "won": bool(info["won"]),
            }
            results.append(result)
            print(
                f"episode={episode} score={result['score']} max_tile={result['max_tile']} "
                f"moves={result['moves']} invalid_actions={result['invalid_actions']} "
                f"won={result['won']}"
            )
            if wandb_run is not None:
                wandb_run.log(result)
        if render_mode == "human":
            env.unwrapped.wait_for_human_close()
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()
    return results
