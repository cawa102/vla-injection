"""JSON→``Rollout`` deserializer + source-run provenance binding (step 5).

The one missing seam in the L2 stack: nothing in the repo loaded a logged
``steps.json`` back into :class:`~evasion_tax.records.Rollout` records (the eval
CLIs consume *already-scored* rollouts; the demos score in-memory rollouts they
generate). This module adds that seam — pure dict→records plus the boundary I/O
that binds an offline L2-attach to the *verified* step-4 source run (decision
D-5): a run directory is validated against its sibling ``run.json`` /
``episode_meta.json`` and the ingested ``steps.json`` is SHA-256-bound.

Model-free: no torch / LIBERO / CUDA. Runs in the core ``.venv``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from evasion_tax.records import Rollout, RolloutStep

_log = logging.getLogger(__name__)

StrPath = str | PathLike[str]

# The step-4 smoke writes its protocol block with this stage tag; the offline
# gate binds only to a run produced by that stage (D-5).
_EXPECTED_STAGE = "smoke_libero_episode"


def rollout_from_log(obj: Mapping) -> Rollout:
    """Build a :class:`Rollout` from a parsed rollout-log mapping.

    Args:
        obj: A mapping with a non-empty ``"steps"`` list of
            :class:`~evasion_tax.records.RolloutStep`-shaped dicts (the exact
            shape ``smoke_libero_episode.py`` writes). The JSON list→tuple
            coercion for ``action`` is handled downstream by
            ``RolloutStep.__post_init__``.

    Returns:
        The reconstructed rollout.

    Raises:
        ValueError / KeyError / TypeError: On a malformed log (a missing
            ``"steps"`` key, a non-list / empty ``"steps"``, or a row that is
            not a valid :class:`RolloutStep`). Never silently drops a row.
    """
    if "steps" not in obj:
        raise KeyError("rollout log is missing the required 'steps' key")
    steps = obj["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("rollout log 'steps' must be a non-empty list")
    return Rollout(steps=tuple(RolloutStep(**row) for row in steps))


def _single(steps: list[dict], key: str):
    """Return the one value of ``key`` shared by every step row, or raise.

    A row that disagrees means the log is a splice of two runs — reject it
    (D-5: the gate binds to one consistent source run).
    """
    values = {row[key] for row in steps}
    if len(values) != 1:
        raise ValueError(f"step rows disagree on {key!r}: {sorted(map(str, values))}")
    return next(iter(values))


@dataclass(frozen=True)
class SourceProvenance:
    """Identity + integrity of the source run an offline L2-attach binds to (D-5).

    ``steps_sha256`` is the SHA-256 of the ingested ``steps.json`` bytes; the
    other fields are the cross-checked facts of the step-4 run.
    """

    run_id: str
    git_commit: str | None
    steps_sha256: str
    model: str
    suite: str
    task_id: str
    seed: int
    n_steps: int
    success: bool


def validate_run_dir(run_dir: StrPath) -> SourceProvenance:
    """Validate a step-4 run directory and return its :class:`SourceProvenance`.

    Reads the sibling ``run.json`` / ``episode_meta.json`` / ``steps.json`` and
    cross-checks them (stage, model, suite, task, seed, success, ``n_steps``,
    ``run_id``) so a stale / trimmed / hand-edited / unsuccessful smoke log
    cannot be cited as "the real attach" (D-5). Computes the ``steps.json``
    SHA-256 and binds it into the returned provenance.

    Raises:
        ValueError: On any cross-file / step-row inconsistency (boundary check).
    """
    run_dir = Path(run_dir)
    run_json = json.loads((run_dir / "run.json").read_text())
    meta = json.loads((run_dir / "episode_meta.json").read_text())
    steps_path = run_dir / "steps.json"
    steps = json.loads(steps_path.read_text())["steps"]
    cfg = run_json.get("config", {})

    if meta.get("stage") != _EXPECTED_STAGE or cfg.get("stage") != _EXPECTED_STAGE:
        raise ValueError(
            f"stage mismatch: expected {_EXPECTED_STAGE!r}, got "
            f"run.json={cfg.get('stage')!r} episode_meta={meta.get('stage')!r}"
        )
    if meta.get("success") is not True:
        raise ValueError(
            f"source run unsuccessful: episode_meta.success={meta.get('success')!r}"
        )
    if meta.get("n_steps") != len(steps):
        raise ValueError(
            f"n_steps mismatch: episode_meta.n_steps={meta.get('n_steps')} "
            f"!= len(steps)={len(steps)}"
        )
    if cfg.get("model") != meta.get("model"):
        raise ValueError(
            f"model mismatch: run.json config.model={cfg.get('model')!r} "
            f"!= episode_meta.model={meta.get('model')!r}"
        )

    # Cross-check suite / task / seed across run.json, episode_meta AND the step
    # rows; run_id and git_commit between run.json and the rows (D-5).
    suite = _single(steps, "suite")
    task_id = _single(steps, "task_id")
    seed = _single(steps, "seed")
    run_id = _single(steps, "run_id")
    git_commit = _single(steps, "git_commit")
    if not (cfg.get("suite") == meta.get("suite") == suite):
        raise ValueError("suite mismatch across run.json / episode_meta / step rows")
    if not (cfg.get("task_id") == meta.get("task_id") == task_id):
        raise ValueError("task_id mismatch across run.json / episode_meta / step rows")
    if not (run_json.get("seed") == meta.get("seed") == seed):
        raise ValueError("seed mismatch across run.json / episode_meta / step rows")
    if run_json.get("run_id") != run_id:
        raise ValueError(
            f"run_id mismatch: run.json={run_json.get('run_id')!r} != step rows {run_id!r}"
        )
    if run_json.get("git_commit") != git_commit:
        raise ValueError("git_commit mismatch between run.json and step rows")

    steps_sha256 = hashlib.sha256(steps_path.read_bytes()).hexdigest()
    return SourceProvenance(
        run_id=run_id,
        git_commit=git_commit,
        steps_sha256=steps_sha256,
        model=meta["model"],
        suite=suite,
        task_id=task_id,
        seed=seed,
        n_steps=len(steps),
        success=meta["success"],
    )


def load_rollout_log(
    path: StrPath, *, unverified: bool = False
) -> tuple[Rollout, SourceProvenance | None]:
    """Load a logged rollout, bound to its source run unless explicitly waived.

    Args:
        path: A **run directory** (the verified default — its siblings are
            validated and the ``steps.json`` SHA-256 is bound), or a bare
            ``steps.json`` file (accepted only with ``unverified=True``).
        unverified: When ``True``, accept a bare ``steps.json`` with no
            provenance binding (a warning is logged). Ignored for a run dir.

    Returns:
        ``(rollout, provenance)`` for a run dir; ``(rollout, None)`` for an
        accepted bare log.

    Raises:
        ValueError: If ``path`` is a bare ``steps.json`` and ``unverified`` is
            ``False`` (the gate must not bind to an unvalidated log).
    """
    p = Path(path)
    if p.is_dir():
        provenance = validate_run_dir(p)
        obj = json.loads((p / "steps.json").read_text())
        return rollout_from_log(obj), provenance

    if not unverified:
        raise ValueError(
            f"{p} is a bare steps.json with no sibling run.json/episode_meta.json "
            "to validate; pass unverified=True (CLI --unverified) to accept it "
            "without provenance binding."
        )
    _log.warning(
        "loading UNVERIFIED rollout log %s — no provenance binding "
        "(report provenance_verified=false)",
        p,
    )
    obj = json.loads(p.read_text())
    return rollout_from_log(obj), None
