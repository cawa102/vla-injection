---
source_file: "src/evasion_tax/metric/coverage.py"
type: "code"
community: "Coverage Manifest"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Coverage_Manifest
---

# coverage.py

## Connections
- [[._with_status()]] - `defined_in` [EXTRACTED]
- [[.abstained()]] - `defined_in` [EXTRACTED]
- [[.assert_covers()]] - `defined_in` [EXTRACTED]
- [[.is_supported()]] - `defined_in` [EXTRACTED]
- [[.limitation_report()]] - `defined_in` [EXTRACTED]
- [[.predicate_for_target()]] - `defined_in` [EXTRACTED]
- [[.status_of()]] - `defined_in` [EXTRACTED]
- [[.summary()]] - `defined_in` [EXTRACTED]
- [[.supported()]] - `defined_in` [EXTRACTED]
- [[.unsupported()]] - `defined_in` [EXTRACTED]
- [[A ``supported``-predicate for         func`evasion_tax.attack.idealized_fronti]] - `defined_in` [EXTRACTED]
- [[A cell's metric-A coverage verdict.]] - `defined_in` [EXTRACTED]
- [[Any_3]] - `defined_in` [EXTRACTED]
- [[Assert the manifest spans exactly the declared D4 matrix.          Guards agains]] - `defined_in` [EXTRACTED]
- [[Cells metric A scores — the population the attacker is restricted to.]] - `defined_in` [EXTRACTED]
- [[Classify a single cell (kind first, then anchor resolvability).      Unsupported]] - `defined_in` [EXTRACTED]
- [[Classify every cell into a class`CoverageManifest`.      Args         cells]] - `defined_in` [EXTRACTED]
- [[Count of cells per status (every cell counted exactly once).]] - `defined_in` [EXTRACTED]
- [[CoverageCell_1]] - `defined_in` [EXTRACTED]
- [[CoverageCell]] - `contains` [EXTRACTED]
- [[CoverageManifest_1]] - `defined_in` [EXTRACTED]
- [[CoverageManifest]] - `contains` [EXTRACTED]
- [[CoverageStatus_1]] - `defined_in` [EXTRACTED]
- [[CoverageStatus]] - `contains` [EXTRACTED]
- [[Enum]] - `imports_from` [EXTRACTED]
- [[GoalKind_1]] - `defined_in` [EXTRACTED]
- [[GoalKind]] - `contains` [EXTRACTED]
- [[Metric-A coverage manifest (Codex review 2 6).  The frozen metric-A schema (``]] - `rationale_for` [EXTRACTED]
- [[One ``(task, target)`` cell of the D4 matrix, annotated for coverage.      Attri]] - `defined_in` [EXTRACTED]
- [[Pre-registered out-of-v1-scope exclusions (headline limitation).]] - `defined_in` [EXTRACTED]
- [[Status of one cell.          Raises             KeyError if ``(task_id, target]] - `defined_in` [EXTRACTED]
- [[Supported-kind cells whose anchor is unresolvable (surfaced, not scored).]] - `defined_in` [EXTRACTED]
- [[The classified D4 matrix. Build via func`build_manifest`.      Holds one ``(ce]] - `defined_in` [EXTRACTED]
- [[The goal categories the metric-A schema distinguishes (schema §1, §6, §7).]] - `defined_in` [EXTRACTED]
- [[The pre-registered headline limitation every non-supported cell + reason.]] - `defined_in` [EXTRACTED]
- [[Whether a known cell is SUPPORTED; ``False`` for an unknown cell.]] - `defined_in` [EXTRACTED]
- [[build_manifest]] - `defined_in` [EXTRACTED]
- [[build_manifest()]] - `contains` [EXTRACTED]
- [[classify_cell]] - `defined_in` [EXTRACTED]
- [[classify_cell()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Coverage_Manifest

## 📄 Source

`src/evasion_tax/metric/coverage.py`

```python
"""Metric-A coverage manifest (Codex review #2 #6).

The frozen metric-A schema (``docs/core/metric-a-annotation-schema.md`` §6) covers
**single-anchor object reach/pick** goals only. Placement-*region* anchors,
pure-orientation deviations, and multi-phase goals are **out of v1 scope** (the
pre-registered S1/S2 stretch primitives). Without surfacing these blind spots the
idealized action-space attacker (playbook §4b-II) would preferentially target
them and M3 would measure *coverage gaps, not embodiment*.

This module mirrors that frozen scope as data (it imports no heavy metric
internals, so both :mod:`evasion_tax.attack` and :mod:`evasion_tax.eval` can depend on it without a
cycle) and classifies each ``(task, target)`` cell of the D4 matrix into:

* **SUPPORTED** — a single-anchor object goal whose anchor the resolver resolves.
* **UNSUPPORTED** — an out-of-v1-scope goal kind (placement-region / orientation
  only / multi-phase): a **pre-registered exclusion**, reported as a headline
  limitation.
* **ABSTAINED** — a supported-kind goal whose anchor is unresolvable at runtime
  (the resolver returns ``None`` and the metric scores ``0.0``): **surfaced
  explicitly, never silently scored**.

``SUPPORTED_GOAL_KINDS`` is derived from the v1 schema, so if the S1/S2 stretch
primitives are later implemented the supported set widens with no change to the
classification logic here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class GoalKind(Enum):
    """The goal categories the metric-A schema distinguishes (schema §1, §6, §7)."""

    SINGLE_ANCHOR_OBJECT = "single_anchor_object"  # reach/pick toward an object (v1)
    PLACEMENT_REGION = "placement_region"  # release-at-region anchor (stretch S2)
    ORIENTATION_ONLY = "orientation_only"  # rotation-only deviation (stretch S1)
    MULTI_PHASE = "multi_phase"  # compositional ordered sub-goals (stretch S2)


# The goal kinds metric-A v1 actually scores (frozen schema §6). Derived here so
# implementing S1/S2 later widens coverage without touching classify_cell.
SUPPORTED_GOAL_KINDS: frozenset[GoalKind] = frozenset({GoalKind.SINGLE_ANCHOR_OBJECT})


class CoverageStatus(Enum):
    """A cell's metric-A coverage verdict."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class CoverageCell:
    """One ``(task, target)`` cell of the D4 matrix, annotated for coverage.

    Attributes:
        task_id: the LIBERO task identifier.
        target_id: the attacker target / goal identifier within the task.
        goal_kind: which schema goal category this cell's goal is.
        anchor_resolvable: whether metric-A's resolver can resolve the goal anchor
            for this cell (a single-anchor object goal whose object is present).
            Moot for unsupported kinds; decisive for SUPPORTED vs ABSTAINED.
    """

    task_id: str
    target_id: str
    goal_kind: GoalKind
    anchor_resolvable: bool


def classify_cell(cell: CoverageCell) -> CoverageStatus:
    """Classify a single cell (kind first, then anchor resolvability).

    Unsupported is decided by goal kind alone; a supported-kind cell with an
    unresolvable anchor abstains rather than being silently scored.
    """
    if cell.goal_kind not in SUPPORTED_GOAL_KINDS:
        return CoverageStatus.UNSUPPORTED
    if not cell.anchor_resolvable:
        return CoverageStatus.ABSTAINED
    return CoverageStatus.SUPPORTED


_UNSUPPORTED_REASON = {
    GoalKind.PLACEMENT_REGION: (
        "placement-region anchor (not an object) — out of metric-A v1 scope (stretch S2)"
    ),
    GoalKind.ORIENTATION_ONLY: (
        "pure-orientation deviation — out of metric-A v1 scope "
        "(stretch S1; ee_pos has no orientation)"
    ),
    GoalKind.MULTI_PHASE: (
        "multi-phase compositional goal — out of metric-A v1 scope (stretch S2)"
    ),
}
_ABSTAIN_REASON = (
    "goal anchor unresolvable (resolver returns None) — metric scores 0.0; "
    "surfaced, not silently scored"
)


@dataclass(frozen=True)
class CoverageManifest:
    """The classified D4 matrix. Build via :func:`build_manifest`.

    Holds one ``(cell, status)`` entry per declared cell, in declaration order, so
    nothing is dropped silently.
    """

    entries: tuple[tuple[CoverageCell, CoverageStatus], ...]

    def _with_status(self, status: CoverageStatus) -> list[CoverageCell]:
        return [c for c, s in self.entries if s is status]

    def supported(self) -> list[CoverageCell]:
        """Cells metric A scores — the population the attacker is restricted to."""
        return self._with_status(CoverageStatus.SUPPORTED)

    def unsupported(self) -> list[CoverageCell]:
        """Pre-registered out-of-v1-scope exclusions (headline limitation)."""
        return self._with_status(CoverageStatus.UNSUPPORTED)

    def abstained(self) -> list[CoverageCell]:
        """Supported-kind cells whose anchor is unresolvable (surfaced, not scored)."""
        return self._with_status(CoverageStatus.ABSTAINED)

    def summary(self) -> dict[CoverageStatus, int]:
        """Count of cells per status (every cell counted exactly once)."""
        counts = {s: 0 for s in CoverageStatus}
        for _, status in self.entries:
            counts[status] += 1
        return counts

    def status_of(self, task_id: str, target_id: str) -> CoverageStatus:
        """Status of one cell.

        Raises:
            KeyError: if ``(task_id, target_id)`` is not a declared cell.
        """
        for cell, status in self.entries:
            if cell.task_id == task_id and cell.target_id == target_id:
                return status
        raise KeyError((task_id, target_id))

    def is_supported(self, task_id: str, target_id: str) -> bool:
        """Whether a known cell is SUPPORTED; ``False`` for an unknown cell."""
        try:
            return self.status_of(task_id, target_id) is CoverageStatus.SUPPORTED
        except KeyError:
            return False

    def limitation_report(self) -> list[dict[str, Any]]:
        """The pre-registered headline limitation: every non-supported cell + reason."""
        report: list[dict[str, Any]] = []
        for cell, status in self.entries:
            if status is CoverageStatus.SUPPORTED:
                continue
            reason = (
                _ABSTAIN_REASON
                if status is CoverageStatus.ABSTAINED
                else _UNSUPPORTED_REASON[cell.goal_kind]
            )
            report.append(
                {
                    "task_id": cell.task_id,
                    "target_id": cell.target_id,
                    "goal_kind": cell.goal_kind,
                    "status": status,
                    "reason": reason,
                }
            )
        return report

    def assert_covers(self, expected_keys: Iterable[tuple[str, str]]) -> None:
        """Assert the manifest spans exactly the declared D4 matrix.

        Guards against a silent coverage gap: any cell the matrix declares but the
        manifest omits (or vice versa) raises, so the manifest can never quietly
        under-cover the matrix it claims to classify.

        Raises:
            ValueError: if the manifest's cell keys differ from ``expected_keys``.
        """
        have = {(c.task_id, c.target_id) for c, _ in self.entries}
        want = set(expected_keys)
        if have != want:
            missing = want - have
            extra = have - want
            raise ValueError(
                f"coverage manifest does not match the declared matrix: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    def predicate_for_target(
        self, target_id: str, *, task_of: Callable[[Any], str] = lambda s: s.task_id
    ) -> Callable[[Any], bool]:
        """A ``supported``-predicate for
        :func:`evasion_tax.attack.idealized_frontier.trace_frontier`.

        Returns a ``Callable[[scenario], bool]`` that keeps a scenario iff its
        ``(task_of(scenario), target_id)`` cell is SUPPORTED; an unknown cell is
        conservatively excluded (``False``) — never silently scored. Run
        :meth:`assert_covers` over the population's cells first to surface gaps.

        Args:
            target_id: the fixed target the frontier is being traced for.
            task_of: extracts the task id from a scenario (default ``s.task_id``).
        """
        return lambda scenario: self.is_supported(task_of(scenario), target_id)


def build_manifest(cells: Sequence[CoverageCell]) -> CoverageManifest:
    """Classify every cell into a :class:`CoverageManifest`.

    Args:
        cells: the declared ``(task, target)`` cells of the D4 matrix.

    Returns:
        A manifest with one classified entry per cell, in order.

    Raises:
        ValueError: if two cells share a ``(task_id, target_id)`` key (ambiguous
            coverage — a boundary error, not silently resolved).
    """
    seen: set[tuple[str, str]] = set()
    entries: list[tuple[CoverageCell, CoverageStatus]] = []
    for cell in cells:
        key = (cell.task_id, cell.target_id)
        if key in seen:
            raise ValueError(f"duplicate coverage cell {key}")
        seen.add(key)
        entries.append((cell, classify_cell(cell)))
    return CoverageManifest(entries=tuple(entries))
```

