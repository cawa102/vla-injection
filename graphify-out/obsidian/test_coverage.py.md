---
source_file: "tests/evasion_tax/metric/test_coverage.py"
type: "code"
community: "Coverage Manifest"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Coverage_Manifest
---

# test_coverage.py

## Connections
- [[CoverageManifest]] - `references` [EXTRACTED]
- [[Tests for the metric-A coverage manifest (Codex review 2 6).  Metric A v1 cove]] - `rationale_for` [EXTRACTED]
- [[_mixed_cells()]] - `contains` [EXTRACTED]
- [[build_manifest()]] - `references` [EXTRACTED]
- [[cell()]] - `contains` [EXTRACTED]
- [[classify_cell()]] - `references` [EXTRACTED]
- [[test_assert_covers_flags_a_missing_cell()]] - `contains` [EXTRACTED]
- [[test_assert_covers_passes_when_keys_match()]] - `contains` [EXTRACTED]
- [[test_build_manifest_classifies_every_cell()]] - `contains` [EXTRACTED]
- [[test_build_manifest_rejects_duplicate_cells()]] - `contains` [EXTRACTED]
- [[test_coverage_cell_is_immutable()]] - `contains` [EXTRACTED]
- [[test_is_supported_and_status_of()]] - `contains` [EXTRACTED]
- [[test_limitation_report_lists_uncovered_with_reasons()]] - `contains` [EXTRACTED]
- [[test_only_single_anchor_object_is_supported_in_v1()]] - `contains` [EXTRACTED]
- [[test_out_of_scope_kinds_are_unsupported()]] - `contains` [EXTRACTED]
- [[test_predicate_honours_custom_task_accessor()]] - `contains` [EXTRACTED]
- [[test_predicate_keeps_supported_excludes_others()]] - `contains` [EXTRACTED]
- [[test_predicate_target_with_unresolvable_anchor_is_excluded()]] - `contains` [EXTRACTED]
- [[test_status_of_unknown_cell_raises()]] - `contains` [EXTRACTED]
- [[test_summary_counts_every_status()]] - `contains` [EXTRACTED]
- [[test_supported_kind_with_resolvable_anchor_is_supported()]] - `contains` [EXTRACTED]
- [[test_supported_kind_with_unresolvable_anchor_abstains()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Coverage_Manifest

## 📄 Source

`tests/evasion_tax/metric/test_coverage.py`

```python
"""Tests for the metric-A coverage manifest (Codex review #2 #6).

Metric A v1 covers **single-anchor object reach/pick** goals only (frozen schema
§6). Placement-*region* anchors, pure-orientation deviations, and multi-phase
goals are **out of v1 scope** -> pre-registered exclusions (a headline
limitation). A supported-kind goal whose anchor the resolver cannot resolve is
**abstained** (the metric scores 0.0 -- "no goal to be inconsistent with") and
must be surfaced, never silently scored. The manifest classifies every declared
D4-matrix cell into supported / unsupported / abstained and yields the predicate
that constrains the idealized attacker (§4b-II) to supported targets.
"""

import dataclasses
import types

import pytest

from evasion_tax.metric.coverage import (
    SUPPORTED_GOAL_KINDS,
    CoverageCell,
    CoverageManifest,
    CoverageStatus,
    GoalKind,
    build_manifest,
    classify_cell,
)

# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #


def cell(task="t0", target="g0", kind=GoalKind.SINGLE_ANCHOR_OBJECT, resolvable=True):
    return CoverageCell(
        task_id=task, target_id=target, goal_kind=kind, anchor_resolvable=resolvable
    )


# --------------------------------------------------------------------------- #
# Schema scope: which kinds v1 supports                                        #
# --------------------------------------------------------------------------- #


def test_only_single_anchor_object_is_supported_in_v1():
    assert SUPPORTED_GOAL_KINDS == frozenset({GoalKind.SINGLE_ANCHOR_OBJECT})


# --------------------------------------------------------------------------- #
# classify_cell                                                               #
# --------------------------------------------------------------------------- #


def test_supported_kind_with_resolvable_anchor_is_supported():
    assert classify_cell(cell(resolvable=True)) is CoverageStatus.SUPPORTED


def test_supported_kind_with_unresolvable_anchor_abstains():
    # The key #6 distinction: a reach/pick goal whose anchor cannot be resolved
    # is ABSTAINED (surfaced), NOT silently scored as consistent.
    assert classify_cell(cell(resolvable=False)) is CoverageStatus.ABSTAINED


@pytest.mark.parametrize(
    "kind",
    [GoalKind.PLACEMENT_REGION, GoalKind.ORIENTATION_ONLY, GoalKind.MULTI_PHASE],
)
def test_out_of_scope_kinds_are_unsupported(kind):
    # Unsupported is decided by kind regardless of anchor resolvability.
    assert classify_cell(cell(kind=kind, resolvable=True)) is CoverageStatus.UNSUPPORTED
    assert classify_cell(cell(kind=kind, resolvable=False)) is CoverageStatus.UNSUPPORTED


def test_coverage_cell_is_immutable():
    c = cell()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.anchor_resolvable = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# build_manifest                                                              #
# --------------------------------------------------------------------------- #


def _mixed_cells():
    return [
        cell(task="t0", target="reach_cube", kind=GoalKind.SINGLE_ANCHOR_OBJECT, resolvable=True),
        cell(task="t1", target="reach_ghost", kind=GoalKind.SINGLE_ANCHOR_OBJECT, resolvable=False),
        cell(task="t2", target="place_region", kind=GoalKind.PLACEMENT_REGION, resolvable=True),
        cell(task="t3", target="rotate_only", kind=GoalKind.ORIENTATION_ONLY, resolvable=True),
    ]


def test_build_manifest_classifies_every_cell():
    m = build_manifest(_mixed_cells())
    assert isinstance(m, CoverageManifest)
    assert {c.target_id for c in m.supported()} == {"reach_cube"}
    assert {c.target_id for c in m.abstained()} == {"reach_ghost"}
    assert {c.target_id for c in m.unsupported()} == {"place_region", "rotate_only"}


def test_summary_counts_every_status():
    m = build_manifest(_mixed_cells())
    summary = m.summary()
    assert summary[CoverageStatus.SUPPORTED] == 1
    assert summary[CoverageStatus.ABSTAINED] == 1
    assert summary[CoverageStatus.UNSUPPORTED] == 2
    assert sum(summary.values()) == 4  # no cell dropped


def test_build_manifest_rejects_duplicate_cells():
    with pytest.raises(ValueError):
        build_manifest([cell(task="t0", target="g0"), cell(task="t0", target="g0")])


def test_is_supported_and_status_of():
    m = build_manifest(_mixed_cells())
    assert m.is_supported("t0", "reach_cube") is True
    assert m.is_supported("t1", "reach_ghost") is False  # abstained
    assert m.is_supported("t2", "place_region") is False  # unsupported
    assert m.status_of("t1", "reach_ghost") is CoverageStatus.ABSTAINED


def test_status_of_unknown_cell_raises():
    m = build_manifest(_mixed_cells())
    with pytest.raises(KeyError):
        m.status_of("nope", "nope")


# --------------------------------------------------------------------------- #
# Limitation report (pre-registered headline limitation)                       #
# --------------------------------------------------------------------------- #


def test_limitation_report_lists_uncovered_with_reasons():
    m = build_manifest(_mixed_cells())
    report = m.limitation_report()
    # Every non-supported cell appears, each with a status + a reason string.
    keys = {(r["task_id"], r["target_id"]) for r in report}
    assert keys == {("t1", "reach_ghost"), ("t2", "place_region"), ("t3", "rotate_only")}
    assert all(r["reason"] for r in report)
    uncovered = {CoverageStatus.UNSUPPORTED, CoverageStatus.ABSTAINED}
    assert all(r["status"] in uncovered for r in report)


# --------------------------------------------------------------------------- #
# assert_covers — completeness guard over the declared D4 matrix               #
# --------------------------------------------------------------------------- #


def test_assert_covers_passes_when_keys_match():
    m = build_manifest(_mixed_cells())
    expected = {
        ("t0", "reach_cube"),
        ("t1", "reach_ghost"),
        ("t2", "place_region"),
        ("t3", "rotate_only"),
    }
    m.assert_covers(expected)  # no raise


def test_assert_covers_flags_a_missing_cell():
    m = build_manifest(_mixed_cells())
    expected = {("t0", "reach_cube"), ("t9", "missing")}
    with pytest.raises(ValueError):
        m.assert_covers(expected)


# --------------------------------------------------------------------------- #
# predicate_for_target — the seam constraining the idealized attacker          #
# --------------------------------------------------------------------------- #


def test_predicate_keeps_supported_excludes_others():
    # trace_frontier(supported=...) calls a Callable[[scenario], bool]; the
    # predicate maps scenario.task_id (+ a fixed target) to supported-ness.
    # (trace_frontier honouring any such predicate is already covered in
    # test_idealized_frontier; here we pin the predicate's own correctness.)
    m = build_manifest(_mixed_cells())
    keep = m.predicate_for_target("reach_cube")
    assert keep(types.SimpleNamespace(task_id="t0")) is True
    # A task whose (task, target) cell is unknown is conservatively excluded.
    assert keep(types.SimpleNamespace(task_id="t2")) is False


def test_predicate_target_with_unresolvable_anchor_is_excluded():
    m = build_manifest(_mixed_cells())
    keep = m.predicate_for_target("reach_ghost")
    assert keep(types.SimpleNamespace(task_id="t1")) is False  # abstained -> excluded


def test_predicate_honours_custom_task_accessor():
    m = build_manifest(_mixed_cells())
    keep = m.predicate_for_target("reach_cube", task_of=lambda s: s["task"])
    assert keep({"task": "t0"}) is True
```

