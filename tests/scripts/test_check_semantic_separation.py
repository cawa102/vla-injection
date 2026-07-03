"""Tests for ``scripts/check_semantic_separation.py`` (Task 6) — the pre-registration
separation/validity guard for Tier B targets. Model-free geometry only (the
GPU-gated benign/adversary reach-floor lives behind the Task-8 GPU gate).

DM-3: these checks validate *target choice*, never a detector threshold.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from evasion_tax.metric.consistency_a import SchemaA

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


def _load():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module("check_semantic_separation")


def test_separated_pair_is_ok():
    m = _load()
    # SchemaA defaults: engagement 0.05 + grasp 0.10 -> separation threshold 0.15 m.
    r = m.check_separation(
        "task_0",
        "blue_cup",
        distractor_pose=(0.5, 0.5, 0.05),  # 0.5 m from the benign goal (> 0.15)
        benign_goal_pose=(0.5, 0.0, 0.05),
        ee_pos=(0.0, 0.0, 0.30),  # far from the distractor
        schema=SchemaA(),
    )
    assert r.separable is True
    assert r.ee_clear is True
    assert r.ok is True


def test_overlapping_pair_fails_and_names_the_offender():
    m = _load()
    r = m.check_separation(
        "task_1",
        "wine_bottle",
        distractor_pose=(0.5, 0.05, 0.05),  # only 0.05 m from the goal (< 0.15)
        benign_goal_pose=(0.5, 0.0, 0.05),
        ee_pos=(0.0, 0.0, 0.30),
        schema=SchemaA(),
    )
    assert r.separable is False
    assert r.ok is False
    with pytest.raises(ValueError, match="wine_bottle"):
        m.assert_all_separable([r])


def test_ee_already_at_distractor_fails():
    m = _load()
    r = m.check_separation(
        "task_2",
        "mug",
        distractor_pose=(0.5, 0.5, 0.05),
        benign_goal_pose=(0.0, 0.0, 0.05),  # separable
        ee_pos=(0.5, 0.5, 0.06),  # 0.01 m from the distractor (< 0.05)
        schema=SchemaA(),
    )
    assert r.ee_clear is False
    assert r.ok is False


def test_artifact_is_write_once_and_captures_distances_and_hash(tmp_path):
    m = _load()
    r = m.check_separation(
        "task_0",
        "blue_cup",
        distractor_pose=(0.5, 0.5, 0.05),
        benign_goal_pose=(0.5, 0.0, 0.05),
        ee_pos=(0.0, 0.0, 0.30),
        schema=SchemaA(),
    )
    out = tmp_path / "run" / "semantic_separation.json"
    m.write_separation_artifact(out, [r], registry_hash="deadbeef")

    data = json.loads(out.read_text())
    assert data["registry_hash"] == "deadbeef"
    task = data["tasks"][0]
    assert task["task_key"] == "task_0"
    assert task["distractor_object"] == "blue_cup"
    assert task["distractor_to_goal"] == pytest.approx(0.5)
    assert task["ee_to_distractor"] > 0.0
    assert task["ok"] is True

    # write-once: a second write to the same path is refused.
    with pytest.raises(FileExistsError):
        m.write_separation_artifact(out, [r], registry_hash="deadbeef")


def test_registry_hash_is_stable_sha256(tmp_path):
    m = _load()
    path = tmp_path / "libero_spatial.json"
    path.write_text('{"suite": "libero_spatial", "tasks": {}}')
    h1 = m.registry_hash(path)
    h2 = m.registry_hash(path)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest
