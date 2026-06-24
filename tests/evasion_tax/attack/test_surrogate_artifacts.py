from __future__ import annotations

import json
from typing import Any

import pytest

from evasion_tax.attack.surrogate_artifacts import (
    SCHEMA_VERSION,
    SurrogateSuffixArtifact,
    TransferEvalRecord,
    assert_transfer_compatible,
    read_suffix_artifact,
    token_l2_distance,
    write_json_record,
)


def _artifact(**overrides) -> SurrogateSuffixArtifact:
    base: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION,
        artifact_id="a0",
        surrogate_precision="int8",
        model_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
        suite="libero_spatial",
        task_id="task_0",
        target_action_tokens=(1, 2, 3, 4, 5, 6, 7),
        seed=42,
        gcg_config={
            "suffix_len": 20,
            "n_steps": 500,
            "top_k": 256,
            "search_width": 512,
            "early_stop": True,
        },
        suffix_token_ids=(9, 9, 9),
        suffix_path="artifacts/untrusted/run/a0.txt",
        suffix_sha256="a" * 64,
        source_run_dir="results/run",
        git_commit="abc123",
        gpu_id="RTX A5000",
        environment={"torch": "2.2.0", "cuda": "12.1", "dependencies": {}},
        load_record={"precision": "int8"},
        surrogate_target_hit=True,
        surrogate_steps_to_success=12,
        surrogate_censored=False,
        surrogate_best_loss=0.123,
        surrogate_wall_seconds=60.0,
        surrogate_peak_vram_gib=18.0,
        failure_reason=None,
        created_utc="2026-06-24T00:00:00+00:00",
    )
    base.update(overrides)
    return SurrogateSuffixArtifact.from_dict(base)


def _transfer(**overrides) -> TransferEvalRecord:
    base: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION,
        transfer_id="t0",
        artifact_id="a0",
        artifact_path="artifacts/untrusted/run/a0.json",
        suffix_sha256="a" * 64,
        surrogate_precision="int8",
        surrogate_target_hit=True,
        victim_precision="bf16",
        model_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
        suite="libero_spatial",
        task_id="task_0",
        seed=42,
        target_action_tokens=(1, 2, 3, 4, 5, 6, 7),
        victim_target_hit=True,
        predicted_target_tokens=(1, 2, 3, 4, 5, 6, 7),
        action_distance_to_target=0.0,
        persistence_window=1,
        rollout_evaluated=False,
        rollout_success=None,
        wall_seconds=5.0,
        failure_reason=None,
        censored=False,
        source_run_dir="results/run",
        git_commit="abc123",
        environment={"torch": "2.2.0"},
        created_utc="2026-06-24T00:00:00+00:00",
    )
    base.update(overrides)
    return TransferEvalRecord.from_dict(base)


def test_suffix_artifact_round_trips_json(tmp_path):
    artifact = _artifact()
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact.to_dict()))

    loaded = SurrogateSuffixArtifact.from_dict(json.loads(path.read_text()))

    assert loaded == artifact
    assert loaded.surrogate_gpu_hours == pytest.approx(60.0 / 3600.0)


def test_suffix_artifact_missing_provenance_rejected():
    with pytest.raises(ValueError, match="git_commit"):
        _artifact(git_commit="")


def test_suffix_artifact_path_must_stay_under_quarantine():
    with pytest.raises(ValueError, match="artifacts/untrusted"):
        _artifact(suffix_path="results/run/suffix.txt")


def test_write_and_read_suffix_artifact_requires_quarantined_json_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = _artifact()
    path = "artifacts/untrusted/test-artifact/a0.json"
    written = write_json_record(artifact, path)

    loaded = read_suffix_artifact(written)

    assert loaded == artifact


def test_transfer_record_round_trips_json():
    record = _transfer(victim_target_hit=False, predicted_target_tokens=(1, 2, 3, 4, 5, 6, 8))
    loaded = TransferEvalRecord.from_dict(json.loads(json.dumps(record.to_dict())))

    assert loaded == record
    assert loaded.censored is False  # caller records censoring policy explicitly


def test_transfer_record_requires_bf16_victim():
    with pytest.raises(ValueError, match="bf16"):
        _transfer(victim_precision="int8")


def test_transfer_compatibility_rejects_mismatched_checkpoint_suite_task():
    artifact = _artifact()
    with pytest.raises(ValueError, match="model_checkpoint"):
        assert_transfer_compatible(
            artifact,
            model_checkpoint="other",
            suite="libero_spatial",
            task_id="task_0",
        )


def test_transfer_compatibility_override_allows_mismatch():
    assert_transfer_compatible(
        _artifact(),
        model_checkpoint="other",
        suite="libero_goal",
        task_id="task_9",
        override=True,
    )


def test_token_l2_distance_uses_token_space():
    assert token_l2_distance((1, 2, 3), (1, 2, 6)) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        token_l2_distance((1, 2), (1, 2, 3))
