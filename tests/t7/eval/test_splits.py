"""Tests for calibration/test split disjointness (Task 7, invariant #3).

The harness must *assert* that the calibration and test manifests share no ids
on any axis (tasks / scenes / seeds), guarding against calib<->test leakage.
"""

import pytest

from t7.eval.splits import assert_disjoint


def _manifest(tasks, scenes, seeds):
    return {"tasks": tasks, "scenes": scenes, "seeds": seeds}


def test_disjoint_manifests_pass():
    calib = _manifest(["t1", "t2"], ["s1"], [0, 1])
    test = _manifest(["t3", "t4"], ["s2"], [2, 3])
    assert assert_disjoint(calib, test) is None  # no raise


def test_shared_seed_raises_naming_axis():
    calib = _manifest(["t1"], ["s1"], [0, 1])
    test = _manifest(["t2"], ["s2"], [1, 2])  # seed 1 overlaps
    with pytest.raises(ValueError, match="seeds"):
        assert_disjoint(calib, test)


def test_shared_task_raises_naming_axis():
    calib = _manifest(["t1", "t2"], ["s1"], [0])
    test = _manifest(["t2", "t3"], ["s2"], [1])  # task t2 overlaps
    with pytest.raises(ValueError, match="tasks"):
        assert_disjoint(calib, test)


def test_shared_scene_raises_naming_axis():
    calib = _manifest(["t1"], ["s1", "s2"], [0])
    test = _manifest(["t2"], ["s2", "s3"], [1])  # scene s2 overlaps
    with pytest.raises(ValueError, match="scenes"):
        assert_disjoint(calib, test)


def test_overlap_value_appears_in_message():
    calib = _manifest(["t1"], ["s1"], [0, 7])
    test = _manifest(["t2"], ["s2"], [7, 8])
    with pytest.raises(ValueError, match="7"):
        assert_disjoint(calib, test)


def test_missing_axis_raises():
    calib = {"tasks": ["t1"], "scenes": ["s1"]}  # no seeds
    test = _manifest(["t2"], ["s2"], [1])
    with pytest.raises(ValueError):
        assert_disjoint(calib, test)


def test_empty_axes_are_disjoint():
    calib = _manifest([], [], [])
    test = _manifest([], [], [])
    assert assert_disjoint(calib, test) is None
