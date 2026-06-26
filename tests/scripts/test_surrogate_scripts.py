from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from evasion_tax.attack.surrogate_artifacts import (
    SCHEMA_VERSION,
    SurrogateSuffixArtifact,
    is_under_quarantine,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
_CONFIG_INT8 = _REPO_ROOT / "configs" / "surrogate" / "libero_spatial_int8.yaml"
_CONFIG_BF16 = _REPO_ROOT / "configs" / "surrogate" / "libero_spatial_bf16.yaml"


def _load(name: str):
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module(name)


def _artifact_base(**overrides) -> dict:
    base = dict(
        schema_version=SCHEMA_VERSION,
        artifact_id="a0",
        surrogate_precision="int8",
        model_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
        suite="libero_spatial",
        task_id="task_0",
        target_action_tokens=[1, 2, 3, 4, 5, 6, 7],
        seed=42,
        gcg_config={
            "suffix_len": 20,
            "n_steps": 500,
            "top_k": 256,
            "search_width": 512,
            "early_stop": True,
        },
        suffix_token_ids=[9, 9, 9],
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
        surrogate_gradient_health=None,
        failure_reason=None,
        created_utc="2026-06-24T00:00:00+00:00",
    )
    base.update(overrides)
    return base


def _artifact_json(tmp_path: Path, **overrides) -> Path:
    base = _artifact_base(**overrides)
    path = tmp_path / "artifacts" / "untrusted" / "run" / "a0.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(base))
    return path


def test_run_surrogate_gcg_defines_gate_samples_constant():
    mod = _load("run_surrogate_gcg")

    assert mod._GATE_SAMPLES == 32


def test_results_pointer_carries_metrics_without_leaking_suffix():
    mod = _load("run_surrogate_gcg")
    health = {
        "grad_absmax": 1.18,
        "grad_nonzero": True,
        "grad_finite": True,
        "recommended_mean_delta": -0.44,
        "random_mean_delta": -0.22,
        "faithfulness_passed": True,
        "gate_samples": 32,
    }
    artifact = SurrogateSuffixArtifact.from_dict(
        _artifact_base(
            surrogate_gradient_health=health,
            surrogate_wall_seconds=1113.19,
            surrogate_peak_vram_gib=17.2,
            surrogate_steps_to_success=25,
            surrogate_target_hit=False,
            surrogate_censored=True,
        )
    )

    pointer = mod._results_pointer(artifact, "artifacts/untrusted/run/a0.json")

    # Metrics land in the committable results/ record so a run is analysable from version
    # control, not stranded in the gitignored quarantine.
    assert pointer["surrogate_wall_seconds"] == 1113.19
    assert pointer["surrogate_peak_vram_gib"] == 17.2
    assert pointer["surrogate_gradient_health"] == health
    assert pointer["surrogate_steps_to_success"] == 25
    assert pointer["surrogate_censored"] is True
    assert pointer["suffix_sha256"] == "a" * 64  # provenance pointer kept
    # The suffix payload itself stays quarantined — it must NOT leak into results/.
    assert "suffix_token_ids" not in pointer
    assert "suffix_path" not in pointer


class _FakeReport:
    recommended_mean_delta = -1.45
    random_mean_delta = -0.44
    passed = True


class _FakeTarget:
    """Minimal stand-in for OpenVlaGcgTarget exercising the off-GPU diagnostic seam."""

    def __init__(self, grad, *, raise_exc=None):
        self._grad = grad
        self._raise_exc = raise_exc

    def init_suffix_ids(self):
        return np.zeros(3, dtype=np.int64)

    def token_gradient(self, ids):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._grad

    def gradient_agrees_with_swaps(self, n_samples, rng):
        assert n_samples == 32  # the diagnostic must use _GATE_SAMPLES
        return _FakeReport()


def test_gradient_health_records_the_shared_contract_dict():
    mod = _load("run_surrogate_gcg")
    grad = np.array([[0.0, 1.135], [0.5, -0.2]])

    health = mod._gradient_health(_FakeTarget(grad), seed=42)

    assert health == {
        "grad_absmax": pytest.approx(1.135),
        "grad_nonzero": True,
        "grad_finite": True,
        "recommended_mean_delta": -1.45,
        "random_mean_delta": -0.44,
        "faithfulness_passed": True,
        "gate_samples": 32,
    }


def test_gradient_health_captures_probe_failure_without_raising():
    # "record, never gate": a dead-gradient probe must be logged, not abort the run.
    mod = _load("run_surrogate_gcg")
    target = _FakeTarget(None, raise_exc=RuntimeError("dead gradient"))

    health = mod._gradient_health(target, seed=42)

    assert health == {"error": "RuntimeError: dead gradient"}


def test_progress_line_reports_step_budget_and_loss():
    mod = _load("run_surrogate_gcg")

    line = mod._progress_line(25, 500, 6.357123, 1113.4)

    assert line.startswith("[gcg]")
    assert "step 25/500" in line
    assert "best_loss=6.357" in line
    assert "1113" in line  # elapsed seconds, no suffix payload


def test_checkpoint_dict_carries_the_shared_contract_and_round_trips():
    mod = _load("run_surrogate_gcg")
    suffix = np.array([7, 8, 9], dtype=np.int64)

    record = mod._checkpoint_dict(
        step=50,
        n_steps=500,
        best_suffix=suffix,
        best_loss=6.123,
        precision="int8",
        updated_utc="2026-06-26T12:00:00+00:00",
    )

    assert record == {
        "step": 50,
        "n_steps": 500,
        "best_loss": 6.123,
        "best_suffix_ids": [7, 8, 9],  # the attack payload — why the file is quarantined
        "precision": "int8",
        "updated_utc": "2026-06-26T12:00:00+00:00",
    }
    # JSON-safe: numpy ids became plain ints, so the dict round-trips losslessly.
    assert json.loads(json.dumps(record)) == record


def test_atomic_write_json_overwrites_in_place_and_round_trips(tmp_path):
    mod = _load("run_surrogate_gcg")
    path = tmp_path / "checkpoint.json"

    mod._atomic_write_json(path, {"step": 50, "best_loss": 6.1})
    assert json.loads(path.read_text()) == {"step": 50, "best_loss": 6.1}

    # The checkpoint is MUTABLE: a second write overwrites without FileExistsError.
    mod._atomic_write_json(path, {"step": 100, "best_loss": 5.0})
    assert json.loads(path.read_text()) == {"step": 100, "best_loss": 5.0}
    # No stray temp file left behind.
    assert list(tmp_path.iterdir()) == [path]


def test_make_on_step_throttles_progress_and_checkpoint(tmp_path):
    mod = _load("run_surrogate_gcg")
    out = io.StringIO()
    ckpt = tmp_path / "checkpoint.json"
    on_step = mod._make_on_step(
        n_steps=500, precision="int8", checkpoint_path=ckpt, t0=0.0, out=out
    )

    checkpoint_steps: list[int] = []
    for step in range(1, 101):
        suffix = np.array([step, step + 1, step + 2], dtype=np.int64)
        on_step(step, suffix, float(500 - step))
        if ckpt.exists():
            checkpoint_steps.append(json.loads(ckpt.read_text())["step"])

    # A progress line every _PROGRESS_EVERY (25) steps.
    progress_steps = [
        int(line.split("step ")[1].split("/")[0]) for line in out.getvalue().splitlines()
    ]
    assert progress_steps == [25, 50, 75, 100]

    # The checkpoint is refreshed every _CHECKPOINT_EVERY (50) steps: it first appears at
    # step 50 and is overwritten at step 100 — its "step" field only ever holds 50 or 100.
    assert checkpoint_steps[0] == 50  # nothing written before the first interval
    assert set(checkpoint_steps) == {50, 100}

    final = json.loads(ckpt.read_text())
    assert final["step"] == 100
    assert final["best_suffix_ids"] == [100, 101, 102]  # the suffix passed at step 100
    assert final["best_loss"] == 400.0


def test_checkpoint_path_is_quarantined_under_the_run_dir():
    mod = _load("run_surrogate_gcg")

    path = mod._checkpoint_path(mod.QUARANTINE_ROOT / "run123")

    assert path.name == "checkpoint.json"
    assert path.parent.name == "run123"
    assert is_under_quarantine(path)  # the suffix-bearing checkpoint never leaves quarantine


def test_checkpoint_path_rejects_a_non_quarantined_run_dir():
    mod = _load("run_surrogate_gcg")

    with pytest.raises(ValueError):
        mod._checkpoint_path(Path("results/run123"))


def test_run_surrogate_gcg_dry_run_uses_config_precision(capsys):
    mod = _load("run_surrogate_gcg")

    rc = mod.main(["--config", str(_CONFIG_INT8), "--dry-run"])

    assert rc == 0
    record = json.loads(capsys.readouterr().out)
    assert record["precision"] == "int8"
    assert record["gcg_config"]["suffix_len"] == 20
    assert record["load_record"]["quantization_config"] == {"load_in_8bit": True}
    # The GPU-only gradient-health diagnostic must never leak into the CUDA-free dry-run.
    assert "surrogate_gradient_health" not in record


def test_run_surrogate_gcg_no_cuda_exits_nonzero(monkeypatch, tmp_path, capsys):
    mod = _load("run_surrogate_gcg")
    monkeypatch.setattr(mod, "cuda_available", lambda: False)

    rc = mod.main(["--config", str(_CONFIG_BF16), "--results-root", str(tmp_path)])

    assert rc == 2
    assert mod.STAGE in capsys.readouterr().err


def test_evaluate_transfer_rejects_mismatched_artifact_before_cuda(tmp_path, monkeypatch):
    mod = _load("evaluate_surrogate_transfer")
    monkeypatch.chdir(tmp_path)
    artifact = _artifact_json(tmp_path)

    with pytest.raises(ValueError, match="suite"):
        mod.main(["--artifact", str(artifact), "--suite", "libero_goal"])


def test_evaluate_transfer_override_reaches_cuda_guard(tmp_path, monkeypatch, capsys):
    mod = _load("evaluate_surrogate_transfer")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "cuda_available", lambda: False)
    artifact = _artifact_json(tmp_path)

    rc = mod.main(["--artifact", str(artifact), "--suite", "libero_goal", "--override-mismatch"])

    assert rc == 2
    assert mod.STAGE in capsys.readouterr().err


def test_summarizer_expands_paths_and_writes_once(tmp_path, monkeypatch):
    mod = _load("summarize_surrogate_transfer")
    monkeypatch.chdir(tmp_path)
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    record = {
        "schema_version": SCHEMA_VERSION,
        "transfer_id": "t0",
        "artifact_id": "a0",
        "artifact_path": "artifacts/untrusted/run/a0.json",
        "suffix_sha256": "a" * 64,
        "surrogate_precision": "int8",
        "surrogate_target_hit": True,
        "victim_precision": "bf16",
        "model_checkpoint": "m",
        "suite": "libero_spatial",
        "task_id": "task_0",
        "seed": 42,
        "target_action_tokens": [1, 2, 3],
        "victim_target_hit": True,
        "predicted_target_tokens": [1, 2, 3],
        "action_distance_to_target": 0.0,
        "persistence_window": 1,
        "rollout_evaluated": False,
        "rollout_success": None,
        "wall_seconds": 10.0,
        "failure_reason": None,
        "censored": False,
        "source_run_dir": "results/run",
        "git_commit": "abc123",
        "environment": {},
        "created_utc": "2026-06-24T00:00:00+00:00",
    }
    (records_dir / "transfer_eval.json").write_text(json.dumps(record))

    rc = mod.main(
        ["--records", str(records_dir / "*.json"), "--out-dir", str(tmp_path / "summary")]
    )

    assert rc == 0
    assert (tmp_path / "summary" / "summary.json").exists()
    with pytest.raises(FileExistsError):
        mod.main(["--records", str(records_dir / "*.json"), "--out-dir", str(tmp_path / "summary")])
