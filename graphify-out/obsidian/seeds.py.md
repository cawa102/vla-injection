---
source_file: "src/evasion_tax/repro/seeds.py"
type: "code"
community: "Deterministic Seeding"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Deterministic_Seeding
---

# seeds.py

## Connections
- [[A process-stable 64-bit seed derived from arbitrary parts.      Hashes ``.joi]] - `defined_in` [EXTRACTED]
- [[Deterministic seeding across Python ``random``, NumPy, and (optionally) torch.]] - `rationale_for` [EXTRACTED]
- [[Seed Python ``random``, NumPy, and torch (if importable).      Args         see]] - `defined_in` [EXTRACTED]
- [[seed_everything]] - `defined_in` [EXTRACTED]
- [[seed_everything()]] - `contains` [EXTRACTED]
- [[stable_seed_1]] - `defined_in` [EXTRACTED]
- [[stable_seed()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Deterministic_Seeding

## 📄 Source

`src/evasion_tax/repro/seeds.py`

```python
"""Deterministic seeding across Python ``random``, NumPy, and (optionally) torch.

Seeding is the first reproducibility invariant (CLAUDE.md): every run pins and
records its seed. ``torch`` is soft-imported so this module works on a local dev
host where torch is not installed. Also exposes :func:`stable_seed`, a
process-stable seed derived from arbitrary parts (never the salted built-in
``hash``).
"""

from __future__ import annotations

import hashlib
import random

import numpy as np


def stable_seed(*parts: object) -> int:
    """A process-stable 64-bit seed derived from arbitrary parts.

    Hashes ``"|".join(map(str, parts))`` with SHA-256 and takes the first 8
    bytes, so identical parts always yield the same seed across processes and
    runs — never the salted built-in ``hash`` (reproducibility invariant).
    """
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


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
        # Optional/soft import by design: torch is absent on a local dev host without CUDA.
        import torch  # type: ignore[import-not-found]
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded.append("torch")

    return {"seed": seed, "seeded": sorted(seeded)}
```

