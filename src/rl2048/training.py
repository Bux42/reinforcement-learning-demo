"""MaskablePPO training entry point for 2048."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from . import ENV_ID
from .determinism import configure_determinism


def make_env(
    *,
    render_mode: str | None = None,
    seed: int | None = None,
    mask_actions: bool = False,
    max_steps: int = 2_000,
    invalid_action_penalty: float = -1.0,
    reward_scale: float = 0.01,
) -> Monitor:
    """Create a monitored 2048 environment."""

    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        mask_actions=mask_actions,
        max_steps=max_steps,
        invalid_action_penalty=invalid_action_penalty,
        reward_scale=reward_scale,
    )
    monitored = Monitor(env)
    if seed is not None:
        monitored.reset(seed=seed)
        monitored.action_space.seed(seed)
    return monitored


class _ProgressCallback(BaseCallback):
    """Print compact progress without taking over SB3's logger."""

    def __init__(self, report_every: int = 10_000) -> None:
        super().__init__()
        self.report_every = report_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.report_every == 0:
            print(f"timesteps={self.num_timesteps}")
        return True


class _ActionMetricsCallback(BaseCallback):
    """Expose 2048-specific metrics in TensorBoard/W&B."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for index, info in enumerate(infos):
            if index >= len(dones) or not dones[index]:
                continue
            steps = max(int(info.get("step", 0)), 1)
            invalid_actions = int(info.get("invalid_actions", 0))
            self.logger.record("rollout/score", int(info.get("score", 0)))
            self.logger.record("rollout/max_tile", int(info.get("max_tile", 0)))
            self.logger.record("rollout/empty_cells", int(info.get("empty_cells", 0)))
            self.logger.record("rollout/invalid_actions", invalid_actions)
            self.logger.record("rollout/invalid_action_rate", invalid_actions / steps)
            self.logger.record("rollout/mask_actions", float(bool(info.get("mask_actions", False))))
            self.logger.record("rollout/won", float(bool(info.get("won", False))))
            self.logger.record("rollout/moves", steps)
        return True


def _make_model(
    env: Monitor,
    *,
    seed: int,
    tensorboard_log: Path,
    device: str,
) -> MaskablePPO:
    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(tensorboard_log),
        learning_rate=3e-4,
        n_steps=2_048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
    )


def train(
    *,
    total_timesteps: int = 200_000,
    seed: int = 0,
    output_dir: str | Path = "artifacts",
    run_name: str | None = None,
    mask_actions: bool = False,
    max_steps: int = 2_000,
    invalid_action_penalty: float = -1.0,
    reward_scale: float = 0.01,
    eval_freq: int = 10_000,
    device: str = "auto",
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_mode: str = "online",
) -> Path:
    """Train MaskablePPO with or without legal-action masking."""

    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    if eval_freq < 1:
        raise ValueError("eval_freq must be positive")

    configure_determinism(seed)
    wandb_project = wandb_project or os.environ.get("WANDB_PROJECT", "rl-2048")
    mask_label = "masked" if mask_actions else "unmasked"
    run_name = run_name or f"ppo-{mask_label}-seed-{seed}"
    root = Path(output_dir) / run_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "best").mkdir(exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)

    config = {
        "algorithm": "MaskablePPO",
        "environment": ENV_ID,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "mask_actions": mask_actions,
        "max_steps": max_steps,
        "invalid_action_penalty": invalid_action_penalty,
        "reward_scale": reward_scale,
        "eval_freq": eval_freq,
        "run_name": run_name,
    }

    train_env = make_env(
        seed=seed,
        mask_actions=mask_actions,
        max_steps=max_steps,
        invalid_action_penalty=invalid_action_penalty,
        reward_scale=reward_scale,
    )
    eval_env = make_env(
        seed=seed + 1,
        mask_actions=mask_actions,
        max_steps=max_steps,
        invalid_action_penalty=invalid_action_penalty,
        reward_scale=reward_scale,
    )
    wandb_run: Any | None = None

    try:
        if use_wandb:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config=config,
                sync_tensorboard=True,
                monitor_gym=False,
                save_code=True,
                mode=wandb_mode,
            )

        model = _make_model(
            train_env,
            seed=seed,
            tensorboard_log=root / "tensorboard",
            device=device,
        )
        callbacks: list[BaseCallback] = [
            MaskableEvalCallback(
                eval_env,
                best_model_save_path=str(root / "best"),
                log_path=str(root / "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=10,
                deterministic=True,
                render=False,
            ),
            _ActionMetricsCallback(),
            _ProgressCallback(),
        ]
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            use_masking=mask_actions,
            progress_bar=False,
        )
        final_path = root / "final_model"
        model.save(final_path)
        return final_path.with_suffix(".zip")
    finally:
        train_env.close()
        eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()
