"""Tests for ``evasion_tax.eval.rollout_io`` (CSB bring-up step 5).

The JSON→``Rollout`` deserializer (the one missing seam) plus the source-run
provenance binding (D-5). All model-free: these run in the core ``.venv`` against
the committed step-4 smoke run, no CUDA / OpenVLA / LIBERO.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evasion_tax.eval.rollout_io import (
    SourceProvenance,
    load_rollout_log,
    rollout_from_log,
    validate_run_dir,
)
from evasion_tax.metric.state import _REQUIRED_KEYS
from evasion_tax.records import Rollout

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_TRIMMED = _FIXTURES / "steps_trimmed.json"
_REAL_RUN_DIR = _REPO_ROOT / "results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke"


def _step_dict(*, step: int, action: list[float], gripper_open: bool = True) -> dict:
    """A minimal RolloutStep-shaped row (the exact keys the step-4 log writes)."""
    return {
        "run_id": "r0",
        "seed": 42,
        "git_commit": "abc123",
        "suite": "libero_spatial",
        "task_id": "task_0",
        "step": step,
        "observation_ref": f"libero_spatial/task_0/{step}",
        "action": action,
        "privileged_state": {
            "ee_pos": [0.0, 0.0, 1.0 - 0.1 * step],
            "gripper_open": gripper_open,
            "object_poses": {"plate_1": [0.1, 0.2, 0.83], "bowl_1": [-0.1, 0.0, 0.83]},
            "target_region": "plate_1",
        },
        "instruction": "put the bowl on the plate",
        "trusted_goal": "put the bowl on the plate",
        "attacked": False,
        "suffix_ref": None,
    }


def _log_obj(n: int = 3) -> dict:
    """A well-formed rollout log dict (``{"n_steps", "steps"}``)."""
    steps = [
        _step_dict(step=i, action=[0.01 * i, -0.02, 0.03, 0.0, 0.0, 0.0, 1.0])
        for i in range(n)
    ]
    return {"n_steps": n, "steps": steps}


def test_rollout_from_log_round_trips_steps():
    obj = _log_obj(3)

    rollout = rollout_from_log(obj)

    assert isinstance(rollout, Rollout)
    assert len(rollout) == 3
    # action tuples survive the JSON list -> tuple coercion.
    assert rollout.steps[1].action == (0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 1.0)
    # privileged_state dict is carried through unchanged.
    assert rollout.steps[0].privileged_state["target_region"] == "plate_1"
    assert set(rollout.steps[0].privileged_state["object_poses"]) == {"plate_1", "bowl_1"}


def test_rollout_from_log_rejects_missing_steps_key():
    with pytest.raises((KeyError, ValueError)):
        rollout_from_log({"n_steps": 0})


def test_rollout_from_log_rejects_empty_or_non_list_steps():
    with pytest.raises(ValueError):
        rollout_from_log({"steps": []})
    with pytest.raises((TypeError, ValueError)):
        rollout_from_log({"steps": 7})


def test_rollout_from_log_propagates_bad_row_not_drops_it():
    obj = _log_obj(2)
    # A 6-element action must surface as a ValueError from RolloutStep, never be
    # silently skipped (boundary validation: never trust external data).
    obj["steps"][1]["action"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        rollout_from_log(obj)


def test_rollout_from_log_on_real_trimmed_log_matches_state_contract():
    obj = json.loads(_TRIMMED.read_text())

    rollout = rollout_from_log(obj)

    assert len(rollout) == obj["n_steps"]
    for step in rollout.steps:
        ps = step.privileged_state
        # The 4 keys the SyntheticStateAdapter (the metric's default) requires.
        assert set(ps) == set(_REQUIRED_KEYS)
        # target_region resolvable: the goal anchor is in object_poses, so the
        # PrivilegedGoalResolver will NOT abstain on this real benign log.
        assert ps["target_region"] == "plate_1"
        assert ps["target_region"] in ps["object_poses"]
        assert len(step.action) == 7


def test_validate_run_dir_passes_on_real_step4_run():
    prov = validate_run_dir(_REAL_RUN_DIR)

    assert isinstance(prov, SourceProvenance)
    assert prov.run_id == "2026-06-18T14-21-51Z-libero-episode-smoke"
    assert prov.success is True
    assert prov.n_steps == 90
    assert prov.model == "openvla/openvla-7b-finetuned-libero-spatial"
    assert prov.git_commit == "25266b289a65c2ad21b4043bc51e1edab8b43629"
    # steps_sha256 binds the report to the exact bytes ingested (D-5).
    expected = hashlib.sha256((_REAL_RUN_DIR / "steps.json").read_bytes()).hexdigest()
    assert prov.steps_sha256 == expected


def _copy_run(tmp_path: Path) -> Path:
    """Copy the real step-4 run dir into ``tmp_path`` so siblings can be mutated."""
    dst = tmp_path / "run"
    shutil.copytree(_REAL_RUN_DIR, dst)
    return dst


def _mutate_json(path: Path, **overrides) -> None:
    obj = json.loads(path.read_text())
    obj.update(overrides)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def test_validate_run_dir_rejects_unsuccessful_episode(tmp_path):
    run = _copy_run(tmp_path)
    _mutate_json(run / "episode_meta.json", success=False)
    with pytest.raises(ValueError):
        validate_run_dir(run)


def test_validate_run_dir_rejects_n_steps_mismatch(tmp_path):
    run = _copy_run(tmp_path)
    _mutate_json(run / "episode_meta.json", n_steps=89)  # log has 90 rows
    with pytest.raises(ValueError):
        validate_run_dir(run)


def test_validate_run_dir_rejects_model_mismatch(tmp_path):
    run = _copy_run(tmp_path)
    _mutate_json(run / "episode_meta.json", model="openvla/some-other-checkpoint")
    with pytest.raises(ValueError):
        validate_run_dir(run)


def test_validate_run_dir_rejects_wrong_stage(tmp_path):
    run = _copy_run(tmp_path)
    _mutate_json(run / "episode_meta.json", stage="some_other_stage")
    with pytest.raises(ValueError):
        validate_run_dir(run)


def _mutate_first_step_row(path: Path, **overrides) -> None:
    obj = json.loads(path.read_text())
    obj["steps"][0].update(overrides)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def test_validate_run_dir_rejects_inconsistent_run_id_across_rows(tmp_path):
    run = _copy_run(tmp_path)
    _mutate_first_step_row(run / "steps.json", run_id="some-other-run")
    with pytest.raises(ValueError):
        validate_run_dir(run)


def test_validate_run_dir_rejects_seed_mismatch_across_rows(tmp_path):
    run = _copy_run(tmp_path)
    _mutate_first_step_row(run / "steps.json", seed=999)
    with pytest.raises(ValueError):
        validate_run_dir(run)


def test_validate_run_dir_rejects_suite_mismatch_across_rows(tmp_path):
    run = _copy_run(tmp_path)
    _mutate_first_step_row(run / "steps.json", suite="libero_object")
    with pytest.raises(ValueError):
        validate_run_dir(run)


def test_validate_run_dir_rejects_task_mismatch_across_rows(tmp_path):
    run = _copy_run(tmp_path)
    _mutate_first_step_row(run / "steps.json", task_id="some_other_task")
    with pytest.raises(ValueError):
        validate_run_dir(run)


def test_validate_run_dir_rejects_git_commit_mismatch(tmp_path):
    # run.json git_commit disagreeing with the (consistent) step rows means the
    # protocol block was written against different code than the rollout (D-5).
    run = _copy_run(tmp_path)
    _mutate_json(run / "run.json", git_commit="0" * 40)
    with pytest.raises(ValueError):
        validate_run_dir(run)


def test_load_rollout_log_on_run_dir_returns_rollout_and_provenance():
    rollout, prov = load_rollout_log(_REAL_RUN_DIR)

    assert len(rollout) == 90
    assert isinstance(prov, SourceProvenance)
    assert prov.run_id == "2026-06-18T14-21-51Z-libero-episode-smoke"


def test_load_rollout_log_rejects_bare_steps_json_without_unverified():
    with pytest.raises(ValueError):
        load_rollout_log(_REAL_RUN_DIR / "steps.json")


def test_load_rollout_log_accepts_bare_steps_json_when_unverified(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        rollout, prov = load_rollout_log(_REAL_RUN_DIR / "steps.json", unverified=True)

    assert len(rollout) == 90
    assert prov is None  # no provenance binding when unverified
    assert any("unverified" in r.message.lower() for r in caplog.records)
