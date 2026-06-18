"""Tests for ``scripts/attach_l2_to_rollout.py`` (CSB bring-up step 5).

The L2-attach driver, in two honest halves (D-2):

* **state half** — ``score_rollout_l2`` runs the real privileged-*state* trajectory
  through metric (A);
* **action half** — ``action_stream_check`` independently proves the real 7-DoF
  *action* stream is well-formed and runs the D2 scorer, so the gate cannot pass on
  a zeroed/corrupted action log.

All model-free: runs in the core ``.venv`` (no torch / LIBERO / CUDA). Importing the
script must not import torch/LIBERO (clean-import test below).
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import sys
from pathlib import Path

import pytest

from evasion_tax.detector.decide import rollout_fires
from evasion_tax.eval.rollout_io import rollout_from_log
from evasion_tax.records import Decision, Rollout, RolloutStep, Score

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
_TRIMMED = _REPO_ROOT / "tests/evasion_tax/eval/fixtures/steps_trimmed.json"
_REAL_RUN_DIR = _REPO_ROOT / "results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke"


@pytest.fixture(scope="module")
def attach_module():
    """Import ``scripts/attach_l2_to_rollout.py`` (scripts/ on sys.path for _bootstrap)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module("attach_l2_to_rollout")


def _real_rollout() -> Rollout:
    """The real (benign) step-4 rollout, trimmed to 3 steps (reuses the seam)."""
    return rollout_from_log(json.loads(_TRIMMED.read_text()))


def test_score_rollout_l2_scores_every_step(attach_module):
    rollout = _real_rollout()

    scores = attach_module.score_rollout_l2(rollout, k=len(rollout))

    assert len(scores) == len(rollout)
    for t, score in enumerate(scores):
        assert isinstance(score, Score)
        assert math.isfinite(score.value)
        assert 0.0 <= score.value <= 1.0
        assert score.window_end == t


def test_action_stream_check_passes_on_real_actions(attach_module):
    rollout = _real_rollout()

    info = attach_module.action_stream_check(rollout)

    assert info["n_steps"] == len(rollout)
    assert info["dims"] == 7
    assert info["all_finite"] is True
    assert info["degenerate"] is False
    assert len(info["per_dim_min"]) == 7
    assert len(info["per_dim_max"]) == 7
    # the D2 path was exercised on the real vectors (illustrative target).
    assert "reached_window" in info
    assert "completion_step" in info
    assert "illustrative_target" in info


def test_action_half_fails_on_zeroed_stream_but_state_half_unchanged(attach_module):
    rollout = _real_rollout()
    zeroed = Rollout(
        steps=tuple(dataclasses.replace(s, action=(0.0,) * 7) for s in rollout.steps)
    )

    # The action half now genuinely fails — the gate depends on the real actions,
    # not just the state geometry (anti-false-confidence, Codex [high]).
    with pytest.raises(ValueError):
        attach_module.action_stream_check(zeroed)

    # The state half is byte-for-byte unchanged: metric (A) ignores RolloutStep.action.
    before = [s.value for s in attach_module.score_rollout_l2(rollout, k=len(rollout))]
    after = [s.value for s in attach_module.score_rollout_l2(zeroed, k=len(zeroed))]
    assert before == after


def _gripper_step(step: int, ee: tuple[float, float, float], gripper_open: bool) -> RolloutStep:
    return RolloutStep(
        run_id="g",
        seed=0,
        git_commit=None,
        suite="s",
        task_id="t",
        step=step,
        observation_ref=f"s/t/{step}",
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 if gripper_open else 1.0),
        privileged_state={
            "ee_pos": ee,
            "gripper_open": gripper_open,
            "object_poses": {"target_obj": (0.0, 0.0, 0.0), "distractor": (1.0, 0.0, 0.0)},
            "target_region": "target_obj",
        },
        instruction="i",
        trusted_goal="g",
        attacked=False,
        suffix_ref=None,
    )


def test_geometry_stats_counts_grasp_transitions_and_finite_distances(attach_module):
    # gripper: open, then open->close (exactly one transition), then stays closed.
    rollout = Rollout(
        steps=(
            _gripper_step(0, (0.0, 0.0, 0.5), gripper_open=True),
            _gripper_step(1, (0.0, 0.0, 0.1), gripper_open=False),
            _gripper_step(2, (0.0, 0.0, 0.05), gripper_open=False),
        )
    )

    stats = attach_module.geometry_stats(rollout)

    assert stats["grasp_transitions"] == 1
    assert stats["target_resolvable"] is True
    assert len(stats["grasp_ee_target_distances"]) == 1
    assert math.isfinite(stats["min_ee_distractor_distance"])
    assert len(stats["ee_target_distance_per_step"]) == 3
    assert all(math.isfinite(d) for d in stats["ee_target_distance_per_step"])


def test_decision_path_on_scored_real_rollout_is_well_typed(attach_module):
    rollout = _real_rollout()
    scores = attach_module.score_rollout_l2(rollout, k=len(rollout))

    decision = rollout_fires(scores, 0.5)  # tau ILLUSTRATIVE, not calibrated

    assert isinstance(decision, Decision)
    assert isinstance(decision.hold, bool)
    assert isinstance(decision.step, int)


def test_main_runs_gate_on_real_run_dir_and_writes_report(attach_module, tmp_path, capsys):
    rc = attach_module.main(
        ["--rollout", str(_REAL_RUN_DIR), "--results-root", str(tmp_path)]
    )

    assert rc == 0
    reports = list(tmp_path.rglob("l2_attach_report.json"))
    assert reports, "no l2_attach_report.json written"
    report = json.loads(reports[0].read_text())

    # provenance bound to the verified step-4 run (D-5).
    assert report["provenance_verified"] is True
    assert report["steps_sha256"]
    assert report["source_run_id"] == "2026-06-18T14-21-51Z-libero-episode-smoke"
    assert report["n_steps"] == 90

    # state half.
    assert len(report["per_step_scores"]) == 90
    assert set(report["score_summary"]) == {"min", "mean", "max"}
    assert "decision" in report

    # action half.
    assert report["action_stream"]["n_steps"] == 90
    assert report["action_stream"]["degenerate"] is False
    assert report["action_stream"]["all_finite"] is True

    # report-only geometry + the honesty claim line.
    assert "geometry_stats" in report
    assert "wiring de-risk only" in report["claim"]

    assert "PASS" in capsys.readouterr().out


def test_action_stream_check_fails_on_non_finite_actions(attach_module):
    rollout = _real_rollout()
    bad = Rollout(
        steps=(
            dataclasses.replace(rollout.steps[0], action=(float("inf"), 0, 0, 0, 0, 0, 0)),
            *rollout.steps[1:],
        )
    )
    with pytest.raises(ValueError):
        attach_module.action_stream_check(bad)


def test_main_accepts_unverified_bare_steps_json(attach_module, tmp_path, capsys):
    rc = attach_module.main(
        [
            "--rollout", str(_REAL_RUN_DIR / "steps.json"),
            "--unverified",
            "--results-root", str(tmp_path),
        ]
    )

    assert rc == 0
    report = json.loads(next(tmp_path.rglob("l2_attach_report.json")).read_text())
    assert report["provenance_verified"] is False
    assert report["steps_sha256"] is None
    assert "unverified" in capsys.readouterr().out.lower()


def test_module_imports_clean_without_torch_or_libero(attach_module):
    # Loaded in the core .venv — the model-free gate must not pull heavy deps.
    assert "torch" not in sys.modules
    assert "libero" not in sys.modules
    for name in ("score_rollout_l2", "action_stream_check", "geometry_stats", "main"):
        assert callable(getattr(attach_module, name))
