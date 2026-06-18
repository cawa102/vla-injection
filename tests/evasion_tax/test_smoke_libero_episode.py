"""Tests for ``scripts/smoke_libero_episode.py`` (CSB bring-up step 4).

Two contracts are exercised **without** CUDA or LIBERO, the only parts a local
(CUDA-free) host can run:

* the model-free seam ``build_rollout_step`` — one ``(camera obs, action)`` pair
  -> a canonical :class:`~evasion_tax.records.RolloutStep` via the real
  :class:`~evasion_tax.metric.state_libero.LiberoStateAdapter`, with the camera
  image key and the ``_to_`` relative deltas excluded from ``object_poses``;
* the GPU guard — with no CUDA the script prints the GPU-required message and
  exits ``2`` (never a silent no-op), the shared contract of the Task-9 scripts.

Importing the script must not import torch/LIBERO (heavy imports live inside
``main`` after the guard), so this whole module runs in the core ``.venv``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from evasion_tax.metric.state_libero import LiberoStateAdapter
from evasion_tax.records import RolloutStep

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def smoke_module():
    """Import ``scripts/smoke_libero_episode.py`` (scripts/ on sys.path for _bootstrap)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module("smoke_libero_episode")


def _camera_obs() -> dict:
    """A synthetic ``libero_spatial`` camera obs dict.

    Grounded in the real key set (docs/setup/libero-local-notes.md): the
    robosuite/LIBERO state keys plus the camera image keys the EGL env adds. The
    ``_to_robot0_eef_pos`` relative delta and ``robot0_*`` proprio keys must be
    filtered out of ``object_poses``; the image keys (not ``*_pos``) too.
    """
    return {
        "agentview_image": [[0, 0, 0]],  # excluded: not a *_pos key
        "robot0_eye_in_hand_image": [[0, 0, 0]],  # excluded
        "robot0_eef_pos": (-0.105, -0.001, 1.013),
        "robot0_eef_quat": (0.0, 0.0, 0.0, 1.0),
        "robot0_gripper_qpos": (0.021, -0.021),  # sum|qpos| = 0.042 > 0.04 -> open
        "akita_black_bowl_1_pos": (-0.011, 0.026, 0.831),
        "akita_black_bowl_1_to_robot0_eef_pos": (0.09, 0.02, -0.18),  # excluded: relative delta
        "plate_1_pos": (0.10, 0.20, 0.83),
        "robot0_joint_pos": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),  # excluded: robot0_ proprio
    }


def _adapter() -> LiberoStateAdapter:
    # target_region = last obj_of_interest = "plate_1"
    return LiberoStateAdapter(["akita_black_bowl_1", "plate_1"])


def test_build_rollout_step_returns_valid_record(smoke_module):
    step = smoke_module.build_rollout_step(
        _camera_obs(),
        (0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 1.0),
        adapter=_adapter(),
        run_id="r0",
        seed=42,
        git_commit="abc123",
        suite="libero_spatial",
        task_id="task_0",
        step=3,
        instruction="pick up the bowl and put it on the plate",
        trusted_goal="pick up the bowl and put it on the plate",
    )
    assert isinstance(step, RolloutStep)
    assert step.step == 3
    assert step.attacked is False
    assert step.suffix_ref is None
    assert step.observation_ref == "libero_spatial/task_0/3"
    assert step.action == (0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 1.0)


def test_build_rollout_step_filters_camera_and_relative_keys(smoke_module):
    step = smoke_module.build_rollout_step(
        _camera_obs(),
        (0.0,) * 7,
        adapter=_adapter(),
        run_id="r0",
        seed=42,
        git_commit=None,
        suite="libero_spatial",
        task_id="task_0",
        step=0,
        instruction="i",
        trusted_goal="g",
    )
    poses = step.privileged_state["object_poses"]
    assert set(poses) == {"akita_black_bowl_1", "plate_1"}
    assert "agentview_image" not in poses
    assert "akita_black_bowl_1_to_robot0_eef" not in poses
    assert step.privileged_state["ee_pos"] == (-0.105, -0.001, 1.013)
    assert step.privileged_state["gripper_open"] is True
    assert step.privileged_state["target_region"] == "plate_1"


def test_build_rollout_step_rejects_wrong_length_action(smoke_module):
    with pytest.raises(ValueError):
        smoke_module.build_rollout_step(
            _camera_obs(),
            (0.0, 0.0, 0.0),  # only 3 dims -> RolloutStep boundary check fires
            adapter=_adapter(),
            run_id="r0",
            seed=42,
            git_commit=None,
            suite="libero_spatial",
            task_id="task_0",
            step=0,
            instruction="i",
            trusted_goal="g",
        )


def test_main_guards_without_cuda(smoke_module, monkeypatch, capsys):
    monkeypatch.setattr(smoke_module, "cuda_available", lambda: False)
    rc = smoke_module.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert smoke_module.STAGE in err
