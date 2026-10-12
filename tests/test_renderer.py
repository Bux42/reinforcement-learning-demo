import numpy as np

from rl2048.envs.twenty_forty_eight import TwentyFortyEightEnv


def test_rgb_array_render_has_expected_shape() -> None:
    env = TwentyFortyEightEnv(render_mode="rgb_array")
    env.reset(seed=4)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (588, 468, 3)
    assert frame.dtype == np.uint8
    env.close()
