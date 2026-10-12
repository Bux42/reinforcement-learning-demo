# 2048 reinforcement-learning demo

Small, teachable 2048 environment built with Gymnasium, Stable-Baselines3,
SB3-Contrib `MaskablePPO`, and Weights & Biases. Training is headless and fast;
playback supports a human window, RGB frames, and video recording.

## Quick start

```bash
uv sync
uv run rl-check-env --mask-actions
uv run rl-train --steps 200000 --mask-actions --run-name ppo-masked
uv run rl-play \
  --model artifacts/ppo-masked/final_model.zip \
  --mask-actions --render human
```

Play the game yourself, without a model:

```bash
uv run rl-manual
```

Use arrow keys or `WASD`; `R` starts a new seeded game and `Q`/`Esc` quits.
Add `--mask-actions` to block swipes that cannot change the board. Without it,
those swipes are accepted as invalid actions and receive the configured penalty.

Compare masking directly:

```bash
uv run rl-train --steps 200000 --run-name ppo-unmasked
uv run rl-train --steps 200000 --mask-actions --run-name ppo-masked
```

The human renderer shows the last selected action in the HUD and with a small
direction triangle. RGB playback can be saved as video:

```bash
uv run rl-play \
  --model artifacts/ppo-masked/final_model.zip \
  --mask-actions --render rgb_array --video-dir artifacts/videos
```

`rl-eval` is the video-producing shortcut. Use `--render none` for completely
headless evaluation. Human model playback starts paused: click `Auto` to let it
run continuously, or click `Next action` to advance exactly one model decision.
`Space` toggles auto-play and `N` is a keyboard shortcut for one action.

## Environment API

Environment ID: `TwentyFortyEight-v0`

Actions are a `Discrete(4)` space:

```text
0 UP    1 RIGHT    2 DOWN    3 LEFT
```

The observation is a `Box(17,)` vector:

- 16 flattened board cells, encoded as `clip(log2(tile) / log2(2048), 0, 1)`;
  empty is `0`, tile 2 is `1/11`, and tile 2048 is `1`.
- 1 normalized move-count feature, so the time limit is observable.

The reward is the standard merge score multiplied by `--reward-scale` (default
`0.01`). A valid move spawns one random 2 or 4. An action that leaves the board
unchanged is not a valid 2048 move; in unmasked runs it receives
`--invalid-action-penalty` (default `-1`) and does not spawn a tile.

With `--mask-actions`, `MaskablePPO` only sees actions that change the current
board. Without it, all four actions remain selectable, making invalid-action
rate and learning speed easy to compare in W&B. The environment uses
`MaskablePPO`; DQN is intentionally not included because the project’s main
comparison is legal-action masking.

Episodes terminate when no move is possible and truncate at `--max-steps`
(default 2,000). `info` exposes score, maximum tile, empty cells, legal mask,
invalid-action count, move score, and whether 2048 was reached.

## W&B and determinism

After `wandb login`:

```bash
uv run rl-train \
  --steps 1000000 --mask-actions --wandb \
  --wandb-project showcase --run-name ppo-masked
```

Use `--wandb-mode offline` when you want local logs only. Seeds are applied to
Python/NumPy, the environment, action spaces, and PyTorch. Same seed,
configuration, and device are intended to reproduce the same run; different
hardware or BLAS/CUDA versions can still introduce numerical differences.

## Profiling

`py-spy` is a dev dependency and training remains a normal Python process:

```bash
uv run py-spy record --output artifacts/train.svg -- \
  uv run rl-train --steps 100000 --mask-actions --device cpu
```

## Layout

```text
src/rl2048/
├── envs/
│   ├── twenty_forty_eight.py  Gymnasium game, mechanics, masks
│   ├── renderer.py            human/RGB pygame renderer
│   └── registration.py        Gymnasium registration
├── training.py                MaskablePPO, callbacks, W&B/TensorBoard
├── evaluation.py              deterministic playback and video
└── cli.py                     rl-train, rl-play, rl-eval, rl-manual, rl-check-env
tests/                         mechanics, API, renderer, determinism
```
