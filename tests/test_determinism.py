import numpy as np
from sb3_contrib import MaskablePPO

from rl2048.training import train


def test_same_seed_reproduces_policy(tmp_path) -> None:
    first_path = train(
        total_timesteps=2_048,
        eval_freq=2_048,
        seed=17,
        output_dir=tmp_path / "first",
        run_name="run",
        mask_actions=True,
        max_steps=40,
        device="cpu",
    )
    second_path = train(
        total_timesteps=2_048,
        eval_freq=2_048,
        seed=17,
        output_dir=tmp_path / "second",
        run_name="run",
        mask_actions=True,
        max_steps=40,
        device="cpu",
    )

    first_model = MaskablePPO.load(first_path, device="cpu")
    second_model = MaskablePPO.load(second_path, device="cpu")
    for name, first_parameter in first_model.policy.state_dict().items():
        second_parameter = second_model.policy.state_dict()[name]
        np.testing.assert_array_equal(
            first_parameter.detach().numpy(),
            second_parameter.detach().numpy(),
        )
