import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

import rl2048
from rl2048.envs.twenty_forty_eight import ACTION_NAMES, TwentyFortyEightEnv


def test_environment_is_registered() -> None:
    env = gym.make(rl2048.ENV_ID, render_mode=None)
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert info["score"] == 0
    assert info["max_tile"] in (2, 4)
    env.close()


def test_environment_passes_sb3_checker() -> None:
    env = TwentyFortyEightEnv(render_mode=None)
    check_env(env, warn=False, skip_render_check=True)
    env.close()


def test_seeded_rollouts_are_deterministic() -> None:
    first = TwentyFortyEightEnv(render_mode=None, mask_actions=False)
    second = TwentyFortyEightEnv(render_mode=None, mask_actions=False)
    first_observation, _ = first.reset(seed=123)
    second_observation, _ = second.reset(seed=123)
    np.testing.assert_array_equal(first_observation, second_observation)

    for action in (0, 1, 2, 3, 0, 3, 1, 2):
        first_result = first.step(action)
        second_result = second.step(action)
        np.testing.assert_array_equal(first_result[0], second_result[0])
        assert first_result[1] == second_result[1]
        assert first_result[2:4] == second_result[2:4]
        assert first_result[4].keys() == second_result[4].keys()
        for key in first_result[4]:
            if isinstance(first_result[4][key], np.ndarray):
                np.testing.assert_array_equal(first_result[4][key], second_result[4][key])
            else:
                assert first_result[4][key] == second_result[4][key]
        if first_result[2] or first_result[3]:
            break
    first.close()
    second.close()


def test_merge_rules_match_2048() -> None:
    moved, gained = TwentyFortyEightEnv._move_line(np.array([2, 0, 2, 2]))
    np.testing.assert_array_equal(moved, [4, 2, 0, 0])
    assert gained == 4

    moved, gained = TwentyFortyEightEnv._move_line(np.array([2, 2, 2, 2]))
    np.testing.assert_array_equal(moved, [4, 4, 0, 0])
    assert gained == 8


def test_action_mask_only_contains_board_changing_moves() -> None:
    env = TwentyFortyEightEnv(mask_actions=True, render_mode=None)
    env.reset(seed=0)
    env._board = np.array(
        [
            [2, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [4096, 8192, 16384, 32768],
        ],
        dtype=np.int64,
    )
    env._cached_legal_mask = None
    assert env.action_masks().tolist() == [False, True, False, True]
    assert list(ACTION_NAMES) == ["UP", "RIGHT", "DOWN", "LEFT"]
    env.close()


def test_unmasked_illegal_action_is_penalized_without_spawning() -> None:
    env = TwentyFortyEightEnv(
        mask_actions=False,
        invalid_action_penalty=-1.5,
        render_mode=None,
    )
    env.reset(seed=0)
    env._board = np.array(
        [
            [2, 0, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    before = env.board
    _, reward, terminated, truncated, info = env.step(0)  # UP cannot move a top-row tile.
    np.testing.assert_array_equal(env.board, before)
    assert reward == -1.5
    assert not terminated
    assert not truncated
    assert info["invalid_action"]
    assert info["invalid_actions"] == 1
    env.close()


def test_valid_move_merges_and_spawns_exactly_one_tile() -> None:
    env = TwentyFortyEightEnv(render_mode=None, reward_scale=0.01)
    env.reset(seed=1)
    env._board = np.array(
        [
            [2, 0, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    env._score = 0
    env._cached_legal_mask = None
    _, reward, _, _, info = env.step(3)  # LEFT
    np.testing.assert_array_equal(env.board[0, :2], [4, 2])
    assert np.count_nonzero(env.board) == 3
    assert env.score == 4
    assert reward == 0.04
    assert info["last_move_score"] == 4
    env.close()


def test_game_over_has_no_legal_actions() -> None:
    env = TwentyFortyEightEnv(render_mode=None)
    env.reset(seed=0)
    env._board = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ],
        dtype=np.int64,
    )
    assert not env.legal_action_mask().any()
    env.close()
