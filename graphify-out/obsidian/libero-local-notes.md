---
source_file: "docs/setup/libero-local-notes.md"
type: "document"
community: "GPU Runbook & Kelvin2"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/GPU_Runbook__Kelvin2
---

# libero-local-notes.md

## Connections
- [[LIBERO state-only smoke test — local notes]] - `defined_in` [EXTRACTED]
- [[PrivilegedState contract (Task-4 schema)]] - `defined_in` [EXTRACTED]
- [[libero_state_smoketest (Tier R  Tier L)]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/GPU_Runbook__Kelvin2

## 📄 Source

`docs/setup/libero-local-notes.md`

# LIBERO state-only smoke test — local notes (Task 10)

> **What this is.** The honest record of the time-boxed Task-10 validation bonus
> (`local-prep-plan.md` Task 10): can we stand up a **state-only** (no render, no
> policy, 8 GB-safe) MuJoCo-backed robot env on the M1 box, dump its real
> ground-truth schema, and confirm the Task-4 `PrivilegedState` contract is
> constructible from it? Script: [`../../scripts/libero_state_smoketest.py`](../../scripts/libero_state_smoketest.py).
>
> **Date:** 2026-06-03. **Box:** Apple Silicon (arm64), 8 GB, Python 3.11.14, `uv`.
> **Isolation:** throwaway venv at `~/.cache/evasion_tax-libero-smoke/venv` — the core `.venv`
> was **not** touched (plan: "never let this destabilise the core env").

---

## 2026-06-09 UPDATE — Tier L now RUNS locally (the deferral is overturned)

The "Tier L BLOCKED / deferred to the GPU node" conclusion below was **env-assembly,
not fundamental**. A state-only LIBERO env now runs on this mac, **GL-free and
torch-free**, by avoiding the two blocking entry points:

- bypass `libero.libero.benchmark` (hard-imports torch) → locate the BDDL from the
  package dir directly;
- bypass `OffScreenRenderEnv` (GL) → construct the lower-level
  `libero.libero.envs.env_wrapper.ControlEnv(use_camera_obs=False, has_renderer=False,
  has_offscreen_renderer=False)`.

The remaining real requirement was **robosuite 1.4.0** (LIBERO imports
`robosuite...single_arm_env.SingleArmEnv`, removed in 1.5) → a dedicated Py3.10 venv;
full recipe in [`libero-local-env.md`](./libero-local-env.md).

**What this unblocked (built locally against real LIBERO ground truth):**
- `src/evasion_tax/metric/state_libero.py` (`LiberoStateAdapter`) — was deferred to the
  GPU node; now built + unit-tested against frozen real-obs fixtures
  (`tests/evasion_tax/metric/fixtures/`).
- The real BDDL **`target_region`** the demo faked: `obj_of_interest[-1]`
  (`plate_1` for the spatial-0 pick-place; `(On bowl plate_1)`).
- A **bug** the synthetic/robosuite path hid: naive `*_pos` extraction ingests
  `<obj>_to_robot0_eef_pos` **relative deltas** as phantom objects (corrupts the metric's
  distractor primitive). The adapter + this script now filter `_to_` keys.
- Tier L of this smoke test now **PASSES** (was: fell back to Tier R).

**Still on the GPU node:** the production pinned stack (Py3.10 / robosuite-pin / GL) +
re-validation of the gripper threshold, object naming, and the relative-key filter across
**all** suites; and anything needing a policy (OpenVLA rollouts). The local adapter is a
**draft re-validated on the GPU node** (repro rule), not the final.

The original 2026-06-03 record is kept verbatim below (honest history).

---

## Outcome (headline)

**Tier R (robosuite) — PASS.** A state-only robosuite env (`Lift`/`Panda`,
`has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False`) reset
cleanly on the M1, and the Task-4 `PrivilegedState` contract **constructed from the
real MuJoCo ground truth**. The extraction mechanism LIBERO sits on is validated.

**Tier L (real LIBERO) — BLOCKED, deferred to the GPU node** (documented below).
This is acceptable per the plan: a validation bonus, not a gate. **No
`state_libero.py` was created** (the conditional adapter ships only on a real
LIBERO success); **synthetic fixtures remain the test contract.**

Installed in the isolated venv: `mujoco 3.9.0`, `robosuite 1.5.2`.

---

## What Tier R proved

State-only reset returned 15 observation keys; the four `PrivilegedState` fields
all map directly from the robosuite/MuJoCo ground truth (this API is identical
under LIBERO, which returns a robosuite observation dict):

| `PrivilegedState` field | Source obs key | Example value (one reset) |
|---|---|---|
| `ee_pos` (x,y,z) | `robot0_eef_pos` | `(-0.105, -0.001, 1.013)` |
| `gripper_open` (bool) | `robot0_gripper_qpos` (heuristic: `sum|qpos|>0.04`) | `True` (`sum|qpos|=0.0417`, 2 joints) |
| `object_poses` (name→xyz) | every 3-vector `*_pos` key | `{cube: (-0.011, 0.026, 0.831), gripper_to_cube: (...)}` |
| `target_region` (str\|None) | — (LIBERO-only; see below) | `None` |

Full obs key set: `cube_pos, cube_quat, gripper_to_cube_pos, object-state,
robot0_eef_pos, robot0_eef_quat, robot0_eef_quat_site, robot0_gripper_qpos,
robot0_gripper_qvel, robot0_joint_acc, robot0_joint_pos, robot0_joint_pos_cos,
robot0_joint_pos_sin, robot0_joint_vel, robot0_proprio-state`.

**Contract check:** `PrivilegedState(ee_pos=…, gripper_open=True, object_poses=…,
target_region=None)` constructed and validated without modification. The Task-4
schema needs **no changes** to accept real env ground truth.

### Caveats the GPU node must close
- **`target_region`** has no robosuite equivalent — in LIBERO it comes from the
  **BDDL** task definition (goal predicate / target object). The concrete adapter
  must read it from the LIBERO task spec, not the obs dict. Until then it is `None`.
- **`gripper_open`** uses a `sum|qpos|>0.04` heuristic on Panda's two finger joints;
  confirm the exact open/closed threshold against the LIBERO env's gripper on the
  GPU node (raw qpos is logged so the threshold can be pinned precisely).
- **Object names** here are robosuite's (`cube`); LIBERO uses BDDL object names
  (e.g. per-suite scene objects). The adapter maps LIBERO names → `object_poses`.

---

## Why Tier L (real LIBERO) was blocked locally

LIBERO installed (`libero==0.1.0`, editable from source) plus `bddl==1.0.1`,
`easydict`. Building a LIBERO env then hit a stack of blockers, **all of which the
GPU-node pinned stack resolves** — so none is a project risk:

1. **`benchmark` hard-imports `torch`.** `libero/libero/benchmark/__init__.py` does
   `import torch` at module load, so even *enumerating tasks / locating a BDDL file*
   needs the full training stack. We deliberately did not install torch locally
   (heavy, and unneeded for a state-only reset).
2. **`OffScreenRenderEnv` needs an offscreen GL context.** LIBERO's env wrapper
   always offscreen-renders; headless on this M1 with `MUJOCO_GL=disable` there is no
   GL context, and a render path is contrary to the intended *state-only* test.
   → On the GPU node, the concrete adapter should build a **state-only** env
   (`has_offscreen_renderer=False`) via LIBERO's lower-level problem/`TASK_MAPPING`
   path rather than `OffScreenRenderEnv`, to honour "no rendering".
3. **Version drift.** LIBERO pins `robosuite==1.4.0`, `numpy==1.22.4` (no Py-3.11
   wheels); the local box has `robosuite 1.5.2` / Py 3.11. The GPU node installs
   LIBERO's pinned stack under **Python 3.10** (see `gpu-runbook.md` Step 1), so the
   local combo is **not representative** of production anyway.
4. **uv editable `.pth` unreliable on this host** (already documented in the plan):
   `import libero` failed after `pip install -e` and needed a `PYTHONPATH` shim — the
   same M1 quirk noted for the `evasion_tax` editable install.

The smoke-test script handles this gracefully: a `torch`/LIBERO `ImportError`
(Tier L) silently falls back to Tier R, and a box with neither lib falls back to a
clean SKIP. On the GPU node (torch + GL + LIBERO present), Tier L runs automatically.

---

## Decision (recorded)

- **Keep synthetic fixtures** (`tests/evasion_tax/metric/fixtures_state.py`) as the metric-(A)
  test contract — unchanged.
- **Do not create `src/evasion_tax/metric/state_libero.py` yet.** Wire the concrete LIBERO
  `StateAdapter` on the GPU node (runbook Step 6 / metric-(A) signal check), where
  the real LIBERO env + checkpoints + BDDL goal regions exist. The smoke test proved
  the *target* `PrivilegedState` schema is sound, so that adapter is a mapping job,
  not a redesign.
- **Schema verdict:** Task-4 `PrivilegedState` is confirmed buildable from real
  MuJoCo ground truth; **no schema change required** (so the Task-5 metric-(A)
  freeze stands).

### GPU-node reproduction (for the runbook)
```bash
# inside the pinned evasion_tax-openvla env (Py3.10, LIBERO + robosuite 1.4, torch, GL):
python scripts/libero_state_smoketest.py     # Tier L should now run (torch + GL present)
```
Then add `target_region` (from the BDDL goal) + the LIBERO object-name mapping and,
on a clean reset, materialise `src/evasion_tax/metric/state_libero.py`.

### Cleanup
The isolated env is disposable: `rm -rf ~/.cache/evasion_tax-libero-smoke` removes the venv +
LIBERO clone (nothing in the repo depends on it).

