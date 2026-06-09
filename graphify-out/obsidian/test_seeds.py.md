---
source_file: "tests/evasion_tax/repro/test_seeds.py"
type: "code"
community: "Deterministic Seeding"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Deterministic_Seeding
---

# test_seeds.py

## Connections
- [[Tests for the deterministic seeding helper.]] - `rationale_for` [EXTRACTED]
- [[seed_everything()]] - `references` [EXTRACTED]
- [[test_different_seeds_give_different_numpy_draws()]] - `contains` [EXTRACTED]
- [[test_does_not_mutate_inputs_and_returns_fresh_dict()]] - `contains` [EXTRACTED]
- [[test_returns_dict_recording_applied_seed()]] - `contains` [EXTRACTED]
- [[test_same_seed_gives_identical_numpy_draws()]] - `contains` [EXTRACTED]
- [[test_same_seed_gives_identical_python_random_draws()]] - `contains` [EXTRACTED]
- [[test_torch_absent_is_recorded_not_seeded()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Deterministic_Seeding

## 📄 Source

`tests/evasion_tax/repro/test_seeds.py`

```python
"""Tests for the deterministic seeding helper."""

import numpy as np

from evasion_tax.repro import seed_everything


def test_returns_dict_recording_applied_seed():
    result = seed_everything(123)
    assert isinstance(result, dict)
    assert result["seed"] == 123
    # Records which libraries were seeded.
    assert "seeded" in result
    assert "random" in result["seeded"]
    assert "numpy" in result["seeded"]


def test_same_seed_gives_identical_numpy_draws():
    seed_everything(42)
    first = np.random.rand(5)

    seed_everything(42)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)


def test_same_seed_gives_identical_python_random_draws():
    import random

    seed_everything(7)
    first = [random.random() for _ in range(5)]

    seed_everything(7)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_different_seeds_give_different_numpy_draws():
    seed_everything(1)
    first = np.random.rand(5)

    seed_everything(2)
    second = np.random.rand(5)

    assert not np.array_equal(first, second)


def test_torch_absent_is_recorded_not_seeded():
    # On this machine torch is not installed; the helper must still succeed
    # and must not claim to have seeded torch.
    result = seed_everything(0)
    assert "torch" not in result["seeded"]


def test_does_not_mutate_inputs_and_returns_fresh_dict():
    a = seed_everything(5)
    b = seed_everything(5)
    assert a is not b
    assert a == b
```

