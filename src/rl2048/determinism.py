"""Process-wide deterministic settings used by the demo."""

from __future__ import annotations

import os
import random

import numpy as np


def configure_determinism(seed: int | None = None) -> None:
    """Configure reproducible Python, NumPy, PyTorch, and BLAS behavior."""

    if seed is None:
        os.environ.setdefault("PYTHONHASHSEED", "0")
    else:
        os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only permits this before its thread pool is initialized.
            pass
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
