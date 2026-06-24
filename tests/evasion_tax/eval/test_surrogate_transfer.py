from __future__ import annotations

import pytest

from evasion_tax.attack.surrogate_artifacts import SCHEMA_VERSION, TransferEvalRecord
from evasion_tax.eval.surrogate_transfer import summarize_transfer, write_summary_outputs


def _record(
    precision: str,
    transfer_id: str,
    *,
    hit: bool,
    surrogate_hit=True,
    censored=False,
    failed=False,
):
    return TransferEvalRecord(
        schema_version=SCHEMA_VERSION,
        transfer_id=transfer_id,
        artifact_id=f"a-{transfer_id}",
        artifact_path=f"artifacts/untrusted/run/a-{transfer_id}.json",
        suffix_sha256="a" * 64,
        surrogate_precision=precision,  # type: ignore[arg-type]
        surrogate_target_hit=surrogate_hit,
        victim_precision="bf16",
        model_checkpoint="m",
        suite="libero_spatial",
        task_id="task_0",
        seed=42,
        target_action_tokens=(1, 2, 3),
        victim_target_hit=hit,
        predicted_target_tokens=(1, 2, 3) if hit else (1, 2, 4),
        action_distance_to_target=0.0 if hit else 1.0,
        persistence_window=1,
        rollout_evaluated=False,
        rollout_success=None,
        wall_seconds=1800.0,
        failure_reason="RuntimeError: failed" if failed else None,
        censored=censored,
        source_run_dir="results/run",
        git_commit="abc123",
        environment={},
        created_utc="2026-06-24T00:00:00+00:00",
    )


def test_summarize_transfer_groups_asr_gap_cost_and_censoring():
    records = [
        _record("int8", "i0", hit=True, surrogate_hit=True),
        _record("int8", "i1", hit=False, surrogate_hit=True, censored=True),
        _record("nf4_4bit", "n0", hit=False, surrogate_hit=False, censored=True, failed=True),
    ]

    summary = summarize_transfer(records)
    by_precision = {row["surrogate_precision"]: row for row in summary["by_precision"]}

    int8 = by_precision["int8"]
    assert int8["n"] == 2
    assert int8["victim_asr"] == pytest.approx(0.5)
    assert int8["surrogate_asr"] == pytest.approx(1.0)
    assert int8["transfer_gap"] == pytest.approx(0.5)
    assert int8["censored_fraction"] == pytest.approx(0.5)
    assert int8["gpu_hours"] == pytest.approx(1.0)
    assert int8["cost_normalized_asr"] == pytest.approx(0.5)
    assert len(int8["victim_asr_ci"]) == 2

    nf4 = by_precision["nf4_4bit"]
    assert nf4["failed"] == 1
    assert nf4["censored"] == 1


def test_summarize_transfer_rejects_empty():
    with pytest.raises(ValueError):
        summarize_transfer([])


def test_write_summary_outputs_refuses_existing_dir(tmp_path):
    summary = summarize_transfer([_record("bf16", "b0", hit=True)])
    out = tmp_path / "summary"

    written = write_summary_outputs(summary, out)

    assert (out / "summary.json") in written
    assert (out / "asr_by_precision.csv").exists()
    assert (out / "steps_to_success_censored.png").exists()
    with pytest.raises(FileExistsError):
        write_summary_outputs(summary, out)
