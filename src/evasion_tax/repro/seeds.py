"""Deterministic seeding across Python ``random``, NumPy, and (optionally) torch.

Seeding is the first reproducibility invariant (CLAUDE.md): every run pins and
records its seed. ``torch`` is soft-imported so this module works on the local M1
machine where torch is not installed.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> dict:
    """Seed Python ``random``, NumPy, and torch (if importable).

    Args:
        seed: The integer seed to apply to every available RNG.

    Returns:
        A fresh dict recording the applied ``seed`` and the sorted list of
        libraries that were actually ``seeded`` (torch appears only if installed).
    """
    seeded: list[str] = []

    random.seed(seed)
    seeded.append("random")

    np.random.seed(seed)
    seeded.append("numpy")

    try:
        # Optional/soft import by design: torch is absent on the local M1 machine.
        import torch  # type: ignore[import-not-found]
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded.append("torch")

    return {"seed": seed, "seeded": sorted(seeded)}
