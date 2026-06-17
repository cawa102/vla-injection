"""Tests for the predicted-action validator (CSB bring-up step 3).

``validate_action_vector`` is the model-free "valid action vector" gate the
OpenVLA load smoke (``scripts/smoke_openvla_load.py``) and the later benign
rollout body assert against. It must accept a finite 7-DoF vector and reject
everything else loudly — so a silent NaN / wrong-shape model output never passes
for a real action.
"""

import numpy as np
import pytest

from evasion_tax.policy import validate_action_vector


def test_accepts_finite_seven_dof_vector_and_returns_float_array():
    # Arrange: a plausible OpenVLA 7-DoF output (x, y, z, roll, pitch, yaw, gripper).
    action = [0.01, -0.02, 0.03, 0.0, -0.1, 0.2, 1.0]

    # Act
    result = validate_action_vector(action)

    # Assert
    assert isinstance(result, np.ndarray)
    assert result.shape == (7,)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, action)


@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_accepts_list_tuple_and_ndarray(container):
    action = container([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    result = validate_action_vector(action)
    assert result.shape == (7,)


def test_rejects_wrong_dimensionality_naming_expected_dim():
    with pytest.raises(ValueError, match="7"):
        validate_action_vector([0.1, 0.2, 0.3])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_values(bad):
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, bad]
    with pytest.raises(ValueError, match="finite"):
        validate_action_vector(action)


def test_rejects_non_numeric_payload():
    with pytest.raises(ValueError):
        validate_action_vector(["a", "b", "c", "d", "e", "f", "g"])  # type: ignore[arg-type]


def test_rejects_non_one_dimensional_array():
    # predict_action returns a flat (7,) vector; a 2-D blob is a wiring bug.
    action = np.zeros((2, 7))
    with pytest.raises(ValueError, match="1-D"):
        validate_action_vector(action)


def test_expected_dim_is_configurable_for_other_action_spaces():
    action = [0.0, 1.0, 2.0]
    result = validate_action_vector(action, expected_dim=3)
    assert result.shape == (3,)
