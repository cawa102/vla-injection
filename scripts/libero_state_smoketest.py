#!/usr/bin/env python3
"""Task 10 — time-boxed state-only LIBERO smoke test (plan: t7-local-prep-plan.md).

Attempt, in an isolated env, to spin up a **state-only** (no rendering, no policy,
8 GB-safe) MuJoCo-backed robot env, dump its *real* ground-truth schema, and check
that the Task-4 ``PrivilegedState`` contract is constructible from it.

Tiered + graceful (rabbit-hole guard, ~90 min time-box):

* **Tier L — LIBERO**: a real LIBERO task env. Preferred, because it also yields the
  BDDL object names and goal regions a concrete adapter ultimately needs.
* **Tier R — robosuite**: fallback. LIBERO sits *on* robosuite, so the MuJoCo
  ground-truth API (``obs['robot0_eef_pos']``, gripper qpos, object ``*_pos``) is the
  same; validating it here de-risks the adapter mapping even without full LIBERO.
* **Tier 0 — neither importable**: clean SKIP. Keep synthetic fixtures; defer the
  concrete adapter to the GPU node.

This is a *validation bonus*, not a blocker for Tasks 0–9: it never raises on a
missing dependency and writes **no** experiment record under ``results/``. The
printed report + JSON summary are transcribed by hand into
``docs/setup/libero-local-notes.md``.

Run (inside the isolated smoke venv that has mujoco+robosuite[+libero]):

    python scripts/libero_state_smoketest.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

# State-only: make absolutely sure no GL/offscreen context is requested anywhere.
os.environ.setdefault("MUJOCO_GL", "disable")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STATE_PATH = _REPO_ROOT / "src" / "t7" / "metric" / "state.py"
_REPORT_PATH = Path.home() / ".cache" / "t7-libero-smoke" / "smoke_report.json"

# The four fields the Task-4 PrivilegedState contract requires (see src/t7/metric/state.py).
_CONTRACT_FIELDS = ("ee_pos", "gripper_open", "object_poses", "target_region")


def _load_privileged_state_cls() -> Any:
    """Load ``PrivilegedState`` directly from its file (decoupled from the t7 package).

    Avoids importing the whole ``t7`` package (and its numpy/scipy deps) into the
    smoke venv: ``state.py`` itself imports only the stdlib.
    """
    spec = importlib.util.spec_from_file_location("t7_state_smoke", _STATE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load PrivilegedState from {_STATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass(frozen=True) resolves types via
    # sys.modules[cls.__module__] (Py 3.11), which is None unless we register it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PrivilegedState


def _as_tuple3(value: Any) -> tuple[float, float, float]:
    """Coerce a length-3 array/sequence to a float triple (positions only)."""
    items = [float(x) for x in list(value)]
    if len(items) != 3:
        raise ValueError(f"expected length-3 position, got length {len(items)}")
    return (items[0], items[1], items[2])


def _gripper_open_heuristic(gripper_qpos: Any) -> tuple[bool, str]:
    """Best-effort open/closed call from raw gripper joint positions.

    Panda's two-finger gripper reads ~[+0.04, -0.04] (|sum of |q|| large) when open
    and ~0 when closed. We report the raw values too so the LIBERO-side mapping can
    be set precisely on the GPU node; this is a *heuristic*, clearly labelled.
    """
    qpos = [float(x) for x in list(gripper_qpos)]
    span = sum(abs(x) for x in qpos)
    return (span > 0.04, f"sum|qpos|={span:.4f} over {len(qpos)} joint(s) (heuristic)")


def _extract_ground_truth(obs: dict[str, Any]) -> dict[str, Any]:
    """Pull the PrivilegedState-relevant fields out of a robosuite/LIBERO obs dict.

    The ground-truth extraction mechanism is identical for robosuite and LIBERO
    (LIBERO returns a robosuite observation dict).
    """
    keys = sorted(obs.keys())

    ee_pos = _as_tuple3(obs["robot0_eef_pos"]) if "robot0_eef_pos" in obs else None

    gripper_open: bool | None = None
    gripper_note = "no robot0_gripper_qpos key"
    if "robot0_gripper_qpos" in obs:
        gripper_open, gripper_note = _gripper_open_heuristic(obs["robot0_gripper_qpos"])

    # Object positions = every "<name>_pos" obs that is 3-D and is not the EE itself.
    object_poses: dict[str, tuple[float, float, float]] = {}
    for key in keys:
        if not key.endswith("_pos") or key == "robot0_eef_pos":
            continue
        try:
            object_poses[key[: -len("_pos")]] = _as_tuple3(obs[key])
        except (TypeError, ValueError):
            continue  # not a 3-vector (e.g. joint pos) — skip

    return {
        "obs_keys": keys,
        "ee_pos": ee_pos,
        "gripper_open": gripper_open,
        "gripper_note": gripper_note,
        "object_poses": object_poses,
        # robosuite has no explicit goal region; LIBERO carries it in the BDDL.
        "target_region": None,
        "target_region_note": (
            "robosuite has no explicit goal region; in LIBERO it comes from the BDDL "
            "task definition (deferred to the GPU node where the suites live)."
        ),
    }


def _validate_against_contract(gt: dict[str, Any]) -> dict[str, Any]:
    """Try to build a real ``PrivilegedState`` from the extracted ground truth."""
    privileged_state_cls = _load_privileged_state_cls()
    missing = [f for f in _CONTRACT_FIELDS if gt.get(f) is None and f != "target_region"]
    report: dict[str, Any] = {"contract_fields": list(_CONTRACT_FIELDS), "missing": missing}
    if missing:
        report["constructed"] = False
        report["reason"] = f"env did not expose: {', '.join(missing)}"
        return report
    try:
        state = privileged_state_cls(
            ee_pos=gt["ee_pos"],
            gripper_open=bool(gt["gripper_open"]),
            object_poses=gt["object_poses"],
            target_region=gt["target_region"],
        )
        report["constructed"] = True
        report["privileged_state_repr"] = repr(state)
    except Exception as exc:  # noqa: BLE001 - report any contract violation, don't crash
        report["constructed"] = False
        report["reason"] = f"{type(exc).__name__}: {exc}"
    return report


def try_libero() -> dict[str, Any] | None:
    """Tier L: a real LIBERO task env, state-only. Returns None if LIBERO is absent.

    Best-effort and fully guarded: any failure (import, asset, GL) returns a blocker
    dict rather than raising, so the caller can fall back to robosuite.
    """
    try:
        from libero.libero import benchmark, get_libero_path  # type: ignore[import-not-found]
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]
    except ImportError:
        return None  # LIBERO not installed -> caller falls back to Tier R.

    try:

        bench = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = bench.get_task(0)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        # State-only: no camera obs. OffScreenRenderEnv still needs a GL context; if that
        # fails on this headless/Mac box it is a documented blocker (fall back to Tier R).
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=0, camera_widths=0)
        obs = env.reset()
        env.close()
        gt = _extract_ground_truth(obs)
        return {
            "tier": "L",
            "stack": "libero",
            "task": getattr(task, "name", None) or getattr(task, "bddl_file", None),
            "ground_truth": gt,
            "contract": _validate_against_contract(gt),
        }
    except Exception as exc:  # noqa: BLE001
        return {"tier": "L", "blocker": f"{type(exc).__name__}: {exc}"}


def try_robosuite() -> dict[str, Any] | None:
    """Tier R: a state-only robosuite env (Lift/Panda). Returns None if robosuite absent."""
    try:
        import robosuite as suite  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        # No renderer, no offscreen renderer, no camera obs => pure state, 8 GB-safe.
        # controller_configs omitted => robosuite applies the robot's default controller.
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            horizon=10,
        )
        obs = env.reset()
        env.close()
        gt = _extract_ground_truth(obs)
        return {
            "tier": "R",
            "stack": f"robosuite {getattr(suite, '__version__', '?')}",
            "task": "Lift/Panda",
            "ground_truth": gt,
            "contract": _validate_against_contract(gt),
        }
    except Exception as exc:  # noqa: BLE001
        return {"tier": "R", "blocker": f"{type(exc).__name__}: {exc}"}


def _print_report(result: dict[str, Any]) -> None:
    line = "=" * 72
    print(line)
    print("T7 Task 10 — state-only env smoke test")
    print(line)
    tier = result.get("tier", "0")
    if tier == "0":
        print("RESULT: SKIP — no state-only env stack importable.")
        print("  -> Keep synthetic fixtures; defer the concrete LIBERO adapter to the GPU node.")
        print(line)
        return

    if "blocker" in result:
        print(f"RESULT: BLOCKED at Tier {tier} ({result.get('stack', '?')}).")
        print(f"  blocker: {result['blocker']}")
        print("  -> Documented blocker; keep synthetic fixtures (validation bonus, not a gate).")
        print(line)
        return

    gt = result["ground_truth"]
    contract = result["contract"]
    print(f"RESULT: env reset OK at Tier {tier} ({result['stack']}, task={result['task']}).")
    print(f"\nObservation keys ({len(gt['obs_keys'])}):")
    for key in gt["obs_keys"]:
        print(f"    {key}")
    print("\nExtracted ground truth (PrivilegedState candidates):")
    print(f"    ee_pos        : {gt['ee_pos']}")
    print(f"    gripper_open  : {gt['gripper_open']}   [{gt['gripper_note']}]")
    print(f"    object_poses  : {gt['object_poses']}")
    print(f"    target_region : {gt['target_region']}   [{gt['target_region_note']}]")
    print("\nContract check (Task-4 PrivilegedState):")
    if contract.get("constructed"):
        print("    CONSTRUCTED OK ->", contract.get("privileged_state_repr"))
    else:
        print(f"    NOT constructed: {contract.get('reason')}")
    print(line)


def main() -> int:
    result = try_libero()
    if result is None:
        result = try_robosuite()
    if result is None:
        result = {"tier": "0"}

    _print_report(result)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
    print(f"(machine-readable report written to {_REPORT_PATH})")
    # Validation bonus: a SKIP/BLOCKED outcome is an acceptable, documented result.
    return 0


if __name__ == "__main__":
    sys.exit(main())
