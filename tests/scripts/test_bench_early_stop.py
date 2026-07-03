"""Tests for ``scripts/bench_early_stop.py`` — the model-free driver glue (bench Task 6).

The GPU loop (loading OpenVLA, running ``run_gcg`` per target) runs on the CSB A5000;
here we pin the pieces that are testable off-GPU with the GPU call mocked:

* the CUDA guard exits 2 (no silent no-op) on a host without CUDA;
* :func:`process_targets` — the resume-aware loop: skip a target whose checkpoint exists,
  run a missing one, write its outcome write-once, quarantine exactly one suffix per fresh
  target (DE-4 / DE-5) — driven by an injected fake ``run_one`` (no GPU);
* :func:`aggregate_run` — reads ALL per-target JSONs (fresh + pre-existing from a prior
  restart) and reproduces :func:`steps_to_success_summary` + the realistic cost (DE-4);
* the arg parser defaults (``--resume`` on by default, the DE-2 budget knobs).

The script must import without torch (clean-import test below).
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from evasion_tax.attack.early_stop_bench import (
    TargetOutcome,
    outcome_to_record,
    realistic_s_per_target,
    steps_to_success_summary,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
_CONFIG = _REPO_ROOT / "configs" / "example_m2.yaml"


def _load_bench():
    """Import ``scripts/bench_early_stop.py`` (scripts/ on sys.path for _bootstrap)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module("bench_early_stop")


def _outcome(target_id, *, reached, steps, censored, best_loss=0.5):
    return TargetOutcome(
        target_id=target_id,
        reached=reached,
        steps_to_success=steps,
        censored=censored,
        best_loss=best_loss,
        wall_seconds=1.0,
        peak_vram_gib=14.5,
        suffix_sha256="0" * 64,
    )


# --------------------------------------------------------------------------- #
# CUDA guard                                                                   #
# --------------------------------------------------------------------------- #


def test_main_exits_with_gpu_required_when_no_cuda(tmp_path, monkeypatch):
    # The guard must exit 2 (no silent no-op) when CUDA is absent. Force cuda_available
    # False so this holds on ANY host: on a real GPU box the *unmocked* guard falls
    # through and main() runs the actual multi-hour bench — a unit test must never do GPU
    # work. (Same cuda_available monkeypatch pattern as tests/scripts/test_run_attack.py.)
    bench = _load_bench()
    monkeypatch.setattr(bench, "cuda_available", lambda: False)
    rc = bench.main(["--config", str(_CONFIG), "--results-root", str(tmp_path)])
    assert rc == 2


# --------------------------------------------------------------------------- #
# process_targets: resume skip + write-once + quarantine (GPU mocked)          #
# --------------------------------------------------------------------------- #


def test_process_targets_resumes_done_and_runs_missing(tmp_path):
    bench = _load_bench()
    run_dir = tmp_path / "run"
    targets_dir = run_dir / "targets"
    targets_dir.mkdir(parents=True)
    # A pre-existing checkpoint for target index 0 (seed 100 -> "t100").
    (targets_dir / "t100.json").write_text(
        json.dumps(outcome_to_record(_outcome("t100", reached=True, steps=42, censored=False)))
    )
    ran: list[str] = []

    def run_one(i, tid):
        ran.append(tid)
        return _outcome(tid, reached=True, steps=10, censored=False), f"suffix-{tid}"

    bench.process_targets(
        n_targets=2,
        seed=100,
        run_dir=run_dir,
        quarantine_dir=tmp_path / "q",
        resume=True,
        run_one=run_one,
        log=lambda *_: None,
    )

    assert ran == ["t101"]  # t100 skipped (checkpoint exists), t101 run
    assert (targets_dir / "t101.json").exists()  # fresh outcome written write-once


def test_process_targets_quarantines_one_suffix_per_fresh_target(tmp_path):
    bench = _load_bench()
    run_dir = tmp_path / "run"
    q_dir = tmp_path / "q"

    def run_one(i, tid):
        return _outcome(tid, reached=True, steps=10, censored=False), f"suffix-{tid}"

    bench.process_targets(
        n_targets=3,
        seed=0,
        run_dir=run_dir,
        quarantine_dir=q_dir,
        resume=True,
        run_one=run_one,
        log=lambda *_: None,
    )

    files = sorted(p.name for p in q_dir.glob("*.txt"))
    assert files == ["t0.txt", "t1.txt", "t2.txt"]  # exactly one quarantined suffix per target
    assert (q_dir / "t0.txt").read_text() == "suffix-t0"  # the optimised suffix text (DE-5)


def test_process_targets_writes_outcome_write_once(tmp_path):
    # Write-once: a fresh target whose JSON already exists must not be silently overwritten
    # when resume is OFF (the write-once invariant guards the primary record).
    bench = _load_bench()
    run_dir = tmp_path / "run"
    targets_dir = run_dir / "targets"
    targets_dir.mkdir(parents=True)
    (targets_dir / "t0.json").write_text("{}")

    def run_one(i, tid):
        return _outcome(tid, reached=True, steps=10, censored=False), "s"

    with pytest.raises(FileExistsError):
        bench.process_targets(
            n_targets=1,
            seed=0,
            run_dir=run_dir,
            quarantine_dir=tmp_path / "q",
            resume=False,  # no skip -> tries to write t0.json which already exists
            run_one=run_one,
            log=lambda *_: None,
        )


# --------------------------------------------------------------------------- #
# aggregate_run: multi-restart equivalence (fresh + pre-existing JSONs)         #
# --------------------------------------------------------------------------- #


def test_aggregate_run_equals_summary_over_all_per_target_jsons(tmp_path):
    bench = _load_bench()
    run_dir = tmp_path / "run"
    targets_dir = run_dir / "targets"
    targets_dir.mkdir(parents=True)
    outcomes = [
        _outcome("t1", reached=True, steps=40, censored=False),
        _outcome("t2", reached=True, steps=80, censored=False),
        _outcome("t3", reached=False, steps=500, censored=True),  # censored
    ]
    for o in outcomes:
        (targets_dir / f"{o.target_id}.json").write_text(json.dumps(outcome_to_record(o)))

    agg = bench.aggregate_run(run_dir, n_steps_cap=500, s_per_step=33.19)

    expected = steps_to_success_summary(outcomes, n_steps_cap=500)
    assert agg["steps_to_success"] == expected
    assert agg["realistic_s_per_target"] == pytest.approx(
        realistic_s_per_target(expected["median_steps"], 33.19)
    )


def test_aggregate_run_all_censored_has_no_realistic_cost(tmp_path):
    # All censored -> median None -> realistic cost None (the worst-case bound stays the
    # honest number, DE-2); the hint must say so rather than emit a misleading number.
    bench = _load_bench()
    run_dir = tmp_path / "run"
    targets_dir = run_dir / "targets"
    targets_dir.mkdir(parents=True)
    for tid in ("t1", "t2"):
        rec = outcome_to_record(_outcome(tid, reached=False, steps=500, censored=True))
        (targets_dir / f"{tid}.json").write_text(json.dumps(rec))

    agg = bench.aggregate_run(run_dir, n_steps_cap=500, s_per_step=33.19)

    assert agg["realistic_s_per_target"] is None
    assert "censored" in agg["branch_hint"].lower()


def test_aggregate_run_rejects_empty_targets_dir(tmp_path):
    bench = _load_bench()
    run_dir = tmp_path / "run"
    (run_dir / "targets").mkdir(parents=True)
    with pytest.raises(ValueError):
        bench.aggregate_run(run_dir, n_steps_cap=500, s_per_step=33.19)


# --------------------------------------------------------------------------- #
# arg parser                                                                   #
# --------------------------------------------------------------------------- #


def test_arg_parser_defaults_resume_on_and_de2_budget():
    bench = _load_bench()
    args = bench.build_arg_parser().parse_args(["--config", "c.yaml"])
    assert args.resume is True  # resume on by default (DE-4)
    assert args.search_width == 512  # DE-2
    assert args.n_steps == 500  # DE-2 cap
    assert args.eval_batch == 32  # the measured max-B chunk so sw=512 fits (Task 5 / DC-2)


def test_arg_parser_no_resume_flag_disables_resume():
    bench = _load_bench()
    args = bench.build_arg_parser().parse_args(["--config", "c.yaml", "--no-resume"])
    assert args.resume is False


# --------------------------------------------------------------------------- #
# build_run_header: the §8 repro contract                                       #
# --------------------------------------------------------------------------- #


def test_build_run_header_carries_repro_fields():
    bench = _load_bench()
    args = bench.build_arg_parser().parse_args(
        ["--config", "c.yaml", "--seed", "7", "--exclusive-gpu"]
    )
    env = {"git_commit": "abc123", "torch": "2.2.0", "cuda": "12.1", "driver": "12.4"}

    header = bench.build_run_header(args, env)

    assert header["dtype"] == "bfloat16"
    assert header["seed"] == 7
    assert header["early_stop"] is True
    assert header["exclusive_gpu"] is True
    assert header["eval_batch"] == 32
    assert header["gcg_config"]["search_width"] == 512
    assert header["env"] == env  # full §8 capture (git/torch/cuda/driver) rides along


def test_quarantine_suffix_is_write_once_idempotent(tmp_path):
    # A resumed restart must not clobber an already-quarantined suffix (DE-4/DE-5).
    bench = _load_bench()
    bench._quarantine_suffix(tmp_path, "t0", "first")
    bench._quarantine_suffix(tmp_path, "t0", "second")
    assert (tmp_path / "t0.txt").read_text() == "first"


# --------------------------------------------------------------------------- #
# Clean import: the model-free driver must not pull torch at import time         #
# --------------------------------------------------------------------------- #


def test_bench_module_imports_without_torch():
    # sys.modules is process-global, so an in-process check flakes on suite ordering (an
    # earlier test that imports torch fails it regardless of this module). Verify the real
    # invariant — "importing bench_early_stop pulls in no torch" — in a fresh interpreter
    # whose sys.modules starts clean. _SCRIPTS is passed as argv[1] (its _bootstrap then
    # puts src/ on the path).
    check = textwrap.dedent(
        """
        import importlib
        import sys

        sys.path.insert(0, sys.argv[1])
        importlib.import_module("bench_early_stop")

        heavy = sorted(n for n in sys.modules if n == "torch" or n.startswith("torch."))
        assert not heavy, "import pulled in torch: " + repr(heavy)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", check, str(_SCRIPTS)],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
