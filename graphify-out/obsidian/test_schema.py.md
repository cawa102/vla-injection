---
source_file: "tests/evasion_tax/config/test_schema.py"
type: "code"
community: "Config Schema & Immutability"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Config_Schema__Immutability
---

# test_schema.py

## Connections
- [[Config]] - `references` [EXTRACTED]
- [[Path_2]] - `defined_in` [EXTRACTED]
- [[Tests for the pinned-config schema (Task 9).  ``Config`` is a frozen pydantic mo]] - `rationale_for` [EXTRACTED]
- [[_valid_dict()]] - `contains` [EXTRACTED]
- [[_write()]] - `contains` [EXTRACTED]
- [[load_config()]] - `references` [EXTRACTED]
- [[one_variable_diff()]] - `references` [EXTRACTED]
- [[test_committed_example_config_is_valid()]] - `contains` [EXTRACTED]
- [[test_config_is_immutable()]] - `contains` [EXTRACTED]
- [[test_empty_fpr_targets_raises()]] - `contains` [EXTRACTED]
- [[test_fpr_target_out_of_unit_interval_raises()]] - `contains` [EXTRACTED]
- [[test_missing_required_field_raises()]] - `contains` [EXTRACTED]
- [[test_one_variable_diff_detects_single_leaf()]] - `contains` [EXTRACTED]
- [[test_one_variable_diff_empty_for_identical()]] - `contains` [EXTRACTED]
- [[test_one_variable_diff_reports_multiple_changes()]] - `contains` [EXTRACTED]
- [[test_out_of_range_metric_k_raises()]] - `contains` [EXTRACTED]
- [[test_unknown_field_is_forbidden()]] - `contains` [EXTRACTED]
- [[test_valid_config_loads()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Config_Schema__Immutability

## 📄 Source

`tests/evasion_tax/config/test_schema.py`

```python
"""Tests for the pinned-config schema (Task 9).

``Config`` is a frozen pydantic model that validates a run's pinned parameters
at the system boundary (reject missing / out-of-range / unknown fields), and
``one_variable_diff`` enforces the "change exactly one variable per run"
discipline (Playbook §8) by reporting the dotted paths of leaves that differ.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from evasion_tax.config.schema import Config, load_config, one_variable_diff

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE = _REPO_ROOT / "configs" / "example_m2.yaml"


def _valid_dict() -> dict:
    return {
        "seed": 42,
        "model": {
            "name": "openvla-7b",
            "checkpoint": "openvla/openvla-7b-finetuned-libero-spatial",
            "unnorm_key": "libero_spatial_no_noops",
            "quantization": "bf16",
        },
        "env": {"suite": "libero_spatial", "tasks": ["task_0", "task_1"], "max_steps": 200},
        "attack": {"name": "robogcg", "targets_per_task": 20, "persistence_steps": 5},
        "metric": {"k": 5},
        "detector": {"fpr_targets": [0.01, 0.05]},
        "eval": {
            "matrix": ["clean_ceiling", "coarse_goal"],
            "splits": {
                "calib": {"tasks": ["task_0"], "scenes": ["scene_0"], "seeds": [42]},
                "test": {"tasks": ["task_1"], "scenes": ["scene_1"], "seeds": [43]},
            },
        },
    }


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_valid_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, _valid_dict()))
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert cfg.metric.k == 5
    assert cfg.detector.fpr_targets == [0.01, 0.05]
    assert cfg.eval.splits.calib.seeds == [42]


def test_committed_example_config_is_valid():
    # The shipped example must always validate (it is an M2 deliverable).
    cfg = load_config(_EXAMPLE)
    assert isinstance(cfg, Config)


def test_config_is_immutable(tmp_path):
    cfg = load_config(_write(tmp_path, _valid_dict()))
    with pytest.raises(ValidationError):
        cfg.seed = 7  # frozen model — mutation is rejected


def test_missing_required_field_raises(tmp_path):
    data = _valid_dict()
    del data["metric"]
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_out_of_range_metric_k_raises(tmp_path):
    data = _valid_dict()
    data["metric"]["k"] = 0  # window must be >= 1
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_fpr_target_out_of_unit_interval_raises(tmp_path):
    data = _valid_dict()
    data["detector"]["fpr_targets"] = [0.01, 1.5]  # FPR must be in (0, 1)
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_empty_fpr_targets_raises(tmp_path):
    data = _valid_dict()
    data["detector"]["fpr_targets"] = []
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_unknown_field_is_forbidden(tmp_path):
    data = _valid_dict()
    data["metric"]["window"] = 9  # typo for `k` — must be rejected, not ignored
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_one_variable_diff_detects_single_leaf(tmp_path):
    base = load_config(_write(tmp_path, _valid_dict()))
    changed = _valid_dict()
    changed["metric"]["k"] = 8
    other = Config.model_validate(changed)
    assert one_variable_diff(base, other) == ["metric.k"]


def test_one_variable_diff_empty_for_identical(tmp_path):
    a = load_config(_write(tmp_path, _valid_dict()))
    b = Config.model_validate(_valid_dict())
    assert one_variable_diff(a, b) == []


def test_one_variable_diff_reports_multiple_changes():
    a = Config.model_validate(_valid_dict())
    changed = _valid_dict()
    changed["seed"] = 99
    # Keep 0.05 in the list so the default primary_fpr stays valid + unchanged
    # (the diff under test is seed + fpr_targets, not primary_fpr).
    changed["detector"]["fpr_targets"] = [0.02, 0.05]
    b = Config.model_validate(changed)
    assert one_variable_diff(a, b) == ["detector.fpr_targets", "seed"]
```

