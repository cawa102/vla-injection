---
source_file: "tests/evasion_tax/repro/test_run_logger.py"
type: "code"
community: "Run Logging & Rollout Demo"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Run_Logging__Rollout_Demo
---

# test_run_logger.py

## Connections
- [[RunLogger]] - `references` [EXTRACTED]
- [[Tests for the write-once RunLogger (invariant 5).]] - `rationale_for` [EXTRACTED]
- [[datetime]] - `imports_from` [EXTRACTED]
- [[fixed_now()]] - `contains` [EXTRACTED]
- [[test_default_now_is_utc_and_succeeds()]] - `contains` [EXTRACTED]
- [[test_run_json_contains_required_protocol_fields()]] - `contains` [EXTRACTED]
- [[test_second_start_with_colliding_dir_raises()]] - `contains` [EXTRACTED]
- [[test_start_creates_timestamped_dir()]] - `contains` [EXTRACTED]
- [[test_start_does_not_mutate_passed_config()]] - `contains` [EXTRACTED]
- [[test_write_array_refuses_to_overwrite()]] - `contains` [EXTRACTED]
- [[test_write_array_saves_npy_and_round_trips()]] - `contains` [EXTRACTED]
- [[test_write_json_creates_file_and_returns_path()]] - `contains` [EXTRACTED]
- [[test_write_refuses_to_overwrite_existing_file()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Run_Logging__Rollout_Demo

## 📄 Source

`tests/evasion_tax/repro/test_run_logger.py`

```python
"""Tests for the write-once RunLogger (invariant #5)."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from evasion_tax.repro import RunLogger

FIXED = datetime(2026, 5, 31, 14, 30, 5, tzinfo=timezone.utc)


def fixed_now():
    return FIXED


def test_start_creates_timestamped_dir(tmp_path):
    logger = RunLogger(tmp_path, now=fixed_now)
    handle = logger.start("benign-smoke", config={"k": 4}, seed=42)

    expected_dir = tmp_path / "2026-05-31T14-30-05Z-benign-smoke"
    assert expected_dir.exists()
    assert handle.dir == expected_dir


def test_run_json_contains_required_protocol_fields(tmp_path):
    logger = RunLogger(tmp_path, now=fixed_now)
    config = {"k": 4, "fpr_targets": [0.01, 0.05]}
    handle = logger.start("run-a", config=config, seed=7)

    run_json = json.loads((handle.dir / "run.json").read_text())

    # Filled-now fields.
    assert run_json["run_id"] == "2026-05-31T14-30-05Z-run-a"
    assert run_json["seed"] == 7
    assert run_json["config"] == config
    assert run_json["created_utc"] == "2026-05-31T14:30:05+00:00"
    assert "hardware" in run_json
    assert "git_commit" in run_json

    # Empty placeholders to fill after the run.
    for placeholder in ("hypothesis", "expected", "observed", "decision", "one_variable"):
        assert placeholder in run_json
        assert run_json[placeholder] is None


def test_start_does_not_mutate_passed_config(tmp_path):
    logger = RunLogger(tmp_path, now=fixed_now)
    config = {"k": 4, "nested": {"a": 1}}
    snapshot = {"k": 4, "nested": {"a": 1}}
    handle = logger.start("run", config=config, seed=1)

    # Caller's config object is unchanged.
    assert config == snapshot
    # Mutating the logged snapshot must not reach back into the caller's dict.
    run_json = json.loads((handle.dir / "run.json").read_text())
    run_json["config"]["nested"]["a"] = 999
    assert config["nested"]["a"] == 1


def test_second_start_with_colliding_dir_raises(tmp_path):
    logger = RunLogger(tmp_path, now=fixed_now)
    logger.start("dup", config={}, seed=0)

    # Same fixed timestamp + same slug → same target dir → must refuse.
    with pytest.raises(FileExistsError):
        logger.start("dup", config={}, seed=0)


def test_write_json_creates_file_and_returns_path(tmp_path):
    logger = RunLogger(tmp_path, now=fixed_now)
    handle = logger.start("run", config={}, seed=0)

    path = handle.write("scores", {"auc": 0.97})
    assert path.exists()
    assert path == handle.dir / "scores.json"
    assert json.loads(path.read_text()) == {"auc": 0.97}


def test_write_refuses_to_overwrite_existing_file(tmp_path):
    logger = RunLogger(tmp_path, now=fixed_now)
    handle = logger.start("run", config={}, seed=0)
    handle.write("scores", {"auc": 0.97})

    with pytest.raises(FileExistsError):
        handle.write("scores", {"auc": 0.5})


def test_write_array_saves_npy_and_round_trips(tmp_path):
    logger = RunLogger(tmp_path, now=fixed_now)
    handle = logger.start("run", config={}, seed=0)

    arr = np.array([0.1, 0.2, 0.3])
    path = handle.write_array("benign_scores", arr)
    assert path == handle.dir / "benign_scores.npy"
    np.testing.assert_array_equal(np.load(path), arr)


def test_write_array_refuses_to_overwrite(tmp_path):
    logger = RunLogger(tmp_path, now=fixed_now)
    handle = logger.start("run", config={}, seed=0)
    handle.write_array("arr", np.zeros(3))

    with pytest.raises(FileExistsError):
        handle.write_array("arr", np.ones(3))


def test_default_now_is_utc_and_succeeds(tmp_path):
    # No injected clock: must still produce a UTC-stamped dir without raising.
    logger = RunLogger(tmp_path)
    handle = logger.start("default-clock", config={}, seed=0)
    assert handle.dir.exists()
    assert handle.dir.name.endswith("-default-clock")
    assert "Z-" in handle.dir.name
```

