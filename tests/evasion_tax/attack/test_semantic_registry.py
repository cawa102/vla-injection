"""Tests for the pre-registered adversary-instruction registry (Task 3, Tier B).

The registry maps a benign LIBERO scene to the adversary's goal (the instruction
the attacker wants executed + the distractor object it drives toward). These tests
pin the pure resolver logic against **fixture** configs written to ``tmp_path`` —
the real pre-registered ``configs/semantic_targets/<suite>.json`` (a research
pre-registration artifact) is validated separately.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from evasion_tax.attack.semantic_registry import AdversarySpec, adversary_spec_for


def _write_config(config_dir: Path, suite: str = "libero_spatial") -> Path:
    """Write a minimal two-task fixture registry and return its path."""
    config = {
        "suite": suite,
        "tasks": {
            "pick_up_the_black_bowl": {
                "task_index": 0,
                "libero_task_name": "KITCHEN_SCENE1_pick_up_the_black_bowl",
                "adv_instruction": "pick up the blue cup",
                "distractor_object": "blue_cup_1",
            },
            "put_the_bowl_on_the_plate": {
                "task_index": 1,
                "libero_task_name": "KITCHEN_SCENE2_put_the_bowl_on_the_plate",
                "adv_instruction": "pick up the wine bottle",
                "distractor_object": "wine_bottle_1",
            },
        },
    }
    path = config_dir / f"{suite}.json"
    path.write_text(json.dumps(config))
    return path


def test_resolves_by_symbolic_task_id(tmp_path):
    _write_config(tmp_path)
    spec = adversary_spec_for("libero_spatial", "task_0", config_dir=tmp_path)
    assert isinstance(spec, AdversarySpec)
    assert spec.task_index == 0
    assert spec.adv_instruction == "pick up the blue cup"
    assert spec.distractor_object == "blue_cup_1"
    assert spec.libero_task_name == "KITCHEN_SCENE1_pick_up_the_black_bowl"


def test_symbolic_id_and_libero_task_name_resolve_to_same_entry(tmp_path):
    # The two accepted key forms (run_attack's task_<i> and the LIBERO task.name)
    # must resolve to the identical AdversarySpec (Codex R1 canonical key).
    _write_config(tmp_path)
    by_id = adversary_spec_for("libero_spatial", "task_1", config_dir=tmp_path)
    by_name = adversary_spec_for(
        "libero_spatial", "KITCHEN_SCENE2_put_the_bowl_on_the_plate", config_dir=tmp_path
    )
    assert by_id == by_name
    assert by_id.distractor_object == "wine_bottle_1"


def test_unknown_task_key_raises_clear_error(tmp_path):
    _write_config(tmp_path)
    with pytest.raises(KeyError):
        adversary_spec_for("libero_spatial", "task_9", config_dir=tmp_path)
    with pytest.raises(KeyError):
        adversary_spec_for("libero_spatial", "NOT_A_REAL_TASK_NAME", config_dir=tmp_path)


def test_unknown_suite_raises(tmp_path):
    _write_config(tmp_path)  # only libero_spatial exists
    with pytest.raises(FileNotFoundError):
        adversary_spec_for("libero_goal", "task_0", config_dir=tmp_path)


def test_adversary_spec_is_frozen(tmp_path):
    _write_config(tmp_path)
    spec = adversary_spec_for("libero_spatial", "task_0", config_dir=tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.distractor_object = "tampered"  # type: ignore[misc]


def test_real_libero_spatial_registry_is_well_formed():
    # The committed pre-registration config (DRAFT) must be structurally valid: every
    # eval task_0/1/2 resolves; the distractor is an object-name key form (not a
    # robot0_* proprio key, not a *_to_* relative delta) and differs from the benign
    # target_region (plate_1 for this suite); the two key forms agree. Robust to the
    # author changing which distractor is chosen. (Presence-after-reset is the driver's
    # runtime check — extract_object_poses returns the stripped object name.)
    config_dir = Path(__file__).resolve().parents[3] / "configs" / "semantic_targets"
    for i in range(3):
        spec = adversary_spec_for("libero_spatial", f"task_{i}", config_dir=config_dir)
        assert spec.adv_instruction.strip(), "adv_instruction must be non-empty"
        assert spec.distractor_object != "plate_1", "distractor must differ from benign target"
        assert not spec.distractor_object.startswith("robot0_"), "not a proprio key"
        assert "_to_" not in spec.distractor_object, "not a relative-delta key"
        by_name = adversary_spec_for(
            "libero_spatial", spec.libero_task_name, config_dir=config_dir
        )
        assert by_name == spec, "task_<i> and the LIBERO task.name must resolve identically"
