"""Tests for the deterministic seeding helper."""

import sys

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


def test_torch_absent_is_recorded_not_seeded(monkeypatch):
    # When torch is not importable, the helper must still succeed and must not
    # claim to have seeded torch. Simulate absence so this holds on any host
    # (the model-free suite must not depend on whether torch is installed).
    monkeypatch.setitem(sys.modules, "torch", None)
    result = seed_everything(0)
    assert "torch" not in result["seeded"]


def test_does_not_mutate_inputs_and_returns_fresh_dict():
    a = seed_everything(5)
    b = seed_everything(5)
    assert a is not b
    assert a == b
