"""Pre-registered adversary-instruction registry (Task 3, Tier B).

Maps a benign LIBERO scene to the adversary's goal: the ``adv_instruction`` the
attacker wants executed and the ``distractor_object`` it drives the end-effector
toward. The mapping is **pinned + deterministic** — the pre-registration artifact
for the semantic wrong-object redirect.

Circularity guard (DM-3): this registry feeds the **attacker only**. No detector
threshold is derived from it; the detector's ``SchemaA`` radii stay benign-pinned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_SYMBOLIC_TASK_RE = re.compile(r"^task_(\d+)$")
# src/evasion_tax/attack/semantic_registry.py -> repo root is parents[3].
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "semantic_targets"


@dataclass(frozen=True)
class AdversarySpec:
    """The adversary's pre-registered goal for one benign scene (Tier B target).

    Attributes:
        adv_instruction: The instruction the attacker wants the policy to execute.
        distractor_object: The ``object_poses`` key of the real distractor object
            the adversary drives the end-effector toward.
        task_index: The benign task's index in the eval matrix (``task_<i>``).
        libero_task_name: The exact LIBERO ``task.name`` for this scene.
    """

    adv_instruction: str
    distractor_object: str
    task_index: int
    libero_task_name: str


def adversary_spec_for(
    suite: str, task_key: str, *, config_dir: Path | None = None
) -> AdversarySpec:
    """Resolve the pre-registered :class:`AdversarySpec` for ``(suite, task_key)``.

    ``task_key`` may be the symbolic ``task_<i>`` id (as ``run_attack`` uses) — matched
    on ``task_index`` — or the LIBERO ``task.name`` (matched on ``libero_task_name``).
    Both forms resolve to the same entry. Raises on an unknown suite/key.
    """
    config_dir = Path(config_dir) if config_dir is not None else _DEFAULT_CONFIG_DIR
    entry = _find_entry(_load_tasks(suite, config_dir), task_key)
    return AdversarySpec(
        adv_instruction=str(entry["adv_instruction"]),
        distractor_object=str(entry["distractor_object"]),
        task_index=int(entry["task_index"]),
        libero_task_name=str(entry["libero_task_name"]),
    )


def _load_tasks(suite: str, config_dir: Path) -> dict:
    path = config_dir / f"{suite}.json"
    if not path.exists():
        raise FileNotFoundError(f"no semantic-target registry for suite {suite!r} at {path}")
    return dict(json.loads(path.read_text())["tasks"])


def _find_entry(tasks: dict, task_key: str) -> dict:
    symbolic = _SYMBOLIC_TASK_RE.match(task_key)
    if symbolic is not None:
        wanted = int(symbolic.group(1))
        for entry in tasks.values():
            if int(entry["task_index"]) == wanted:
                return entry
    else:
        for entry in tasks.values():
            if str(entry["libero_task_name"]) == task_key:
                return entry
    raise KeyError(f"no adversary spec for task_key {task_key!r} (known: {sorted(tasks)})")
