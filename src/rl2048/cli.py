"""Command-line interfaces for the 2048 reinforcement-learning demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

from . import ENV_ID
from .envs.twenty_forty_eight import ACTION_NAMES, TwentyFortyEightEnv
from .evaluation import evaluate
from .training import train


def _add_common_game_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-steps", type=int, default=2_000, help="episode move budget")
    parser.add_argument(
        "--invalid-action-penalty",
        type=float,
        default=-1.0,
        help="reward for a swipe that does not change the board",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.01,
        help="multiplier applied to standard 2048 merge score",
    )


def train_command() -> None:
    parser = argparse.ArgumentParser(description="Train a MaskablePPO agent on 2048.")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--device", default="auto", help="SB3 device: auto, cpu, or cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--mask-actions",
        action="store_true",
        help="hide swipes that would leave the board unchanged",
    )
    _add_common_game_args(parser)
    args = parser.parse_args()
    train(
        total_timesteps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
        mask_actions=args.mask_actions,
        max_steps=args.max_steps,
        invalid_action_penalty=args.invalid_action_penalty,
        reward_scale=args.reward_scale,
        eval_freq=args.eval_freq,
        device=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
    )


def _evaluation_command(*, force_video: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Play or evaluate a trained 2048 agent.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", choices=("human", "rgb_array", "none"), default="human")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--mask-actions", action="store_true")
    _add_common_game_args(parser)
    args = parser.parse_args()

    video_dir = args.video_dir
    render_mode = None if args.render == "none" else args.render
    if force_video and video_dir is None:
        video_dir = Path("artifacts/videos")
    if force_video and video_dir is not None and render_mode == "human":
        render_mode = "rgb_array"
    if video_dir is not None and render_mode == "human":
        parser.error("--video-dir requires --render rgb_array")

    evaluate(
        args.model,
        episodes=args.episodes,
        render_mode=render_mode,
        video_dir=video_dir,
        seed=args.seed,
        mask_actions=args.mask_actions,
        max_steps=args.max_steps,
        invalid_action_penalty=args.invalid_action_penalty,
        reward_scale=args.reward_scale,
        wandb_project=args.wandb_project,
        use_wandb=args.wandb,
    )


def play_command() -> None:
    _evaluation_command()


def eval_command() -> None:
    _evaluation_command(force_video=True)


def manual_command() -> None:
    """Play 2048 with arrows/WASD in a pygame window."""

    parser = argparse.ArgumentParser(description="Play 2048 manually.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-actions", action="store_true", help="block swipes that cannot change the board")
    _add_common_game_args(parser)
    args = parser.parse_args()

    import pygame

    env = TwentyFortyEightEnv(
        render_mode="human",
        max_steps=args.max_steps,
        mask_actions=args.mask_actions,
        invalid_action_penalty=args.invalid_action_penalty,
        reward_scale=args.reward_scale,
    )
    key_to_action = {
        pygame.K_UP: 0,
        pygame.K_RIGHT: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_w: 0,
        pygame.K_d: 1,
        pygame.K_s: 2,
        pygame.K_a: 3,
    }
    current_seed = args.seed
    terminated = truncated = False
    try:
        env.reset(seed=current_seed)
        env.render()
        print("Arrows/WASD: move | R: reset | Q/Esc: quit")
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        current_seed += 1
                        env.reset(seed=current_seed)
                        terminated = truncated = False
                        env.render()
                        print(f"reset seed={current_seed}")
                    elif event.key in key_to_action and not (terminated or truncated):
                        action = key_to_action[event.key]
                        if args.mask_actions and not env.legal_action_mask()[action]:
                            print(f"blocked: {ACTION_NAMES[action]} cannot change the board")
                            continue
                        _, _, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            reason = "game over" if terminated else "move limit"
                            print(
                                f"{reason}: score={info['score']} max_tile={info['max_tile']} "
                                "press R to restart or Q/Esc to quit"
                            )
            clock.tick(60)
    finally:
        env.close()


def check_env_command() -> None:
    parser = argparse.ArgumentParser(description="Validate the 2048 Gymnasium API.")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--mask-actions", action="store_true")
    parser.add_argument("--invalid-action-penalty", type=float, default=-1.0)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    args = parser.parse_args()
    env = TwentyFortyEightEnv(
        max_steps=args.max_steps,
        mask_actions=args.mask_actions,
        invalid_action_penalty=args.invalid_action_penalty,
        reward_scale=args.reward_scale,
    )
    try:
        check_env(env, warn=True, skip_render_check=True)
        wrapped = gym.make(
            ENV_ID,
            render_mode=None,
            max_steps=args.max_steps,
            mask_actions=args.mask_actions,
            invalid_action_penalty=args.invalid_action_penalty,
            reward_scale=args.reward_scale,
        )
        observation, info = wrapped.reset(seed=0)
        print(f"environment={ENV_ID} observation_shape={observation.shape} info={info}")
        wrapped.close()
        print("environment check passed")
    finally:
        env.close()
