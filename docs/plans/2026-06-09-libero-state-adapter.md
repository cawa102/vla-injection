# LIBERO State Adapter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans / `/tdd` to implement this plan task-by-task.

**Goal:** Build the concrete LIBERO `StateAdapter` (`src/evasion_tax/metric/state_libero.py`) that maps a real LIBERO observation dict + task metadata to the FROZEN `PrivilegedState` contract, with the BDDL-derived `target_region` the demo previously faked — pulling forward work that was deferred to the GPU node, now that state-only LIBERO is proven to run locally.

**Architecture:** A small, pure mapping module behind the existing `StateAdapter` protocol (`src/evasion_tax/metric/state.py`). All LIBERO-specific knowledge lives here; the metric still depends only on `PrivilegedState` (Dependency Inversion, unchanged). Unit tests run in the **core `.venv`** against a **frozen real-LIBERO obs fixture** (captured once from the live env) — no LIBERO import in the test suite. A thin, guarded live-smoke path proves end-to-end construction against the real env.

**Tech Stack:** Python 3.10, NumPy. Local LIBERO env (disposable): `~/.cache/evasion_tax-libero14/venv` (Py3.10 + `robosuite==1.4.0` + `mujoco` + `numpy<1.24` + `pyyaml easydict bddl termcolor scipy pillow opencv-python pynput h5py matplotlib cloudpickle gym==0.25.2`); LIBERO source at `~/.cache/t7-libero-smoke/LIBERO` on `PYTHONPATH`; constructed **state-only** via `libero.libero.envs.env_wrapper.ControlEnv(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)` — **GL-free, torch-free** (bypasses `benchmark`'s torch import and `OffScreenRenderEnv`'s GL context).

---

## Grounded findings (from live probing 2026-06-09 — the contract this plan encodes)

- **`PrivilegedState` is FROZEN** (`state.py`); this plan implements an adapter *to* it, it does **not** change the schema (Task-5 freeze stands).
- **`target_region` semantics** (`metric/consistency_a.py:PrivilegedGoalResolver`): `anchor = state.object_poses[state.target_region]`. So `target_region` **must be a key in `object_poses`**, and its source is the BDDL goal. LIBERO exposes `env.obj_of_interest`:
  - pick-place `(On bowl plate_1)` → `obj_of_interest=[bowl, plate_1]`; goal/placement = **`plate_1`** = last element.
  - unary `(Open wooden_cabinet_1_middle_region)` → `obj_of_interest=[region]`; goal = **last element**.
  - **Rule (default):** `target_region = obj_of_interest[-1]`. If that name has no `*_pos` obs (abstract regions), leave it — the resolver already **abstains gracefully** (scores 0.0, "runtime abstain"). Do **not** synthesise a pose. (Pre-registered alternative if `[-1]` proves wrong for some predicate type: parse the BDDL `(:goal …)` reference object directly — heavier, only if needed.)
- **`object_poses` extraction must EXCLUDE relative keys.** Real LIBERO obs contains `<obj>_to_robot0_eef_pos` deltas alongside absolute `<obj>_pos`. The naive "every `*_pos` 3-vector" rule (demo/smoke) wrongly ingests these as phantom objects and corrupts P2 (distractor engagement). **Filter:** keep `*_pos` 3-vectors where the stem is **not** `robot0_*` and does **not** contain `_to_`.
- **`gripper_open`:** `sum(|robot0_gripper_qpos|) > 0.04`; confirmed against real open value `0.0417` (Panda 2-finger). Keep raw value available for GPU-node threshold pinning.
- **`ee_pos`** ← `obs["robot0_eef_pos"]`.

---

- [ ] Task 1: Freeze real-LIBERO obs fixtures + provenance

**Files:**
- Create: `tests/evasion_tax/metric/fixtures/libero_obs_spatial0.json` (pick-place / binary predicate)
- Create: `tests/evasion_tax/metric/fixtures/libero_obs_goal_opendrawer.json` (unary predicate / region target with no pose — the abstain case)
- Create: `tests/evasion_tax/metric/fixtures/PROVENANCE.md`

**What:** Capture one real `reset()` obs dict + `{language_instruction, obj_of_interest, problem_info}` from the live local LIBERO env for two task types, and commit them as frozen JSON test fixtures (data, not code). These are the contract the unit tests assert against, so the suite needs no LIBERO/robosuite/GL.

**Interface (fixture JSON shape):**
- `{ "problem_info": {...}, "language_instruction": str, "obj_of_interest": [str, ...], "obs": { "<key>": number | [numbers] } }`

**Test scenarios:** (none — this is data capture; a reusable dump helper already exists at `/tmp/libero_dump_obs.py`, re-point its bddl to each suite/task.)

**Dependencies:** local `evasion_tax-libero14` venv + LIBERO clone on `PYTHONPATH`.

**Notes:** `PROVENANCE.md` records: source = `Lifelong-Robot-Learning/LIBERO` clone (path + the bddl file), captured 2026-06-09, env recipe (above), seed/default reset. Keep values exactly as emitted (float precision) so the test is a true regression anchor. The spatial0 fixture's `obj_of_interest=[akita_black_bowl_1, plate_1]`; the open-drawer fixture's target (`wooden_cabinet_1_middle_region`) is expected to be **absent** from `object_poses` (the abstain case).

**Commit:** `test(fixtures): freeze real-LIBERO obs for state-adapter tests`

---

- [ ] Task 2: `state_libero.py` — extraction helpers + `LiberoStateAdapter`

**Files:**
- Create: `src/evasion_tax/metric/state_libero.py`
- Test: `tests/evasion_tax/metric/test_state_libero.py`

**What:** Pure functions that map a LIBERO obs dict to the `PrivilegedState` fields, and a `LiberoStateAdapter` implementing the existing `StateAdapter` protocol. No LIBERO import — operates on the plain obs dict, so it is fully unit-testable in the core `.venv` against Task 1 fixtures.

**Interface:**
- `extract_ee_pos(obs: Mapping) -> tuple[float, float, float]` — from `robot0_eef_pos`.
- `gripper_open_from_qpos(qpos: Sequence[float], threshold: float = GRIPPER_OPEN_SUM_THRESHOLD) -> bool` — `sum(|qpos|) > threshold`.
- `extract_object_poses(obs: Mapping) -> dict[str, tuple[float,float,float]]` — `*_pos` 3-vectors; drop `robot0_*` and any stem containing `_to_`.
- `target_region_from_obj_of_interest(obj_of_interest: Sequence[str]) -> str | None` — last element, or `None` if empty.
- `class LiberoStateAdapter` — constructed with the per-rollout task metadata `LiberoStateAdapter(obj_of_interest: Sequence[str])` (target_region is fixed at scene setup); `to_privileged_state(raw: object) -> PrivilegedState` maps a per-step obs `Mapping` (validates `raw` is a mapping; reuses the helpers; delegates field validation to `PrivilegedState`).

**Constants:**
- `GRIPPER_OPEN_SUM_THRESHOLD = 0.04`

**Test scenarios:**
- spatial0 fixture → `PrivilegedState` with `ee_pos≈(-0.208, ~0, 1.173)`, `gripper_open is True`, `target_region == "plate_1"`, and `target_region in object_poses`.
- `object_poses` from spatial0 **excludes** every `*_to_robot0_eef` key and every `robot0_*` joint/proprio key; includes `akita_black_bowl_1`, `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`, `plate_1` (5 absolute objects).
- open-drawer fixture → `target_region == "wooden_cabinet_1_middle_region"` and that name is **absent** from `object_poses` (documents the abstain case; not an error).
- `gripper_open_from_qpos([0.020833, -0.020833])` is `True`; `gripper_open_from_qpos([0.0, 0.0])` is `False`.
- empty `obj_of_interest` → `target_region is None`.
- `to_privileged_state` on a non-mapping raises `TypeError`; missing `robot0_eef_pos` raises a clear error.
- **Metric integration:** build `PrivilegedState` from spatial0 fixture, pass `[state]` to `PrivilegedGoalResolver().resolve(...)` (or a 1-step `ConsistencyMetricA.score_rollout`) → anchor resolves to `plate_1`'s position (proves the adapter feeds the frozen metric correctly).

**Dependencies:** `evasion_tax.metric.state` (`PrivilegedState`, `StateAdapter`), `numpy` (optional), Task 1 fixtures.

**Notes:** Mirror the established extraction in `scripts/libero_state_smoketest.py:_extract_ground_truth` / `demo_rollout.py:_extract_privileged_state` but **add the `_to_` relative-key filter** (the bug those two have on multi-object scenes). Keep the module < 150 lines (KISS); no LIBERO/robosuite import (Dependency Inversion preserved).

**Commit:** `feat(metric): concrete LiberoStateAdapter → PrivilegedState (real BDDL target_region)`

---

- [ ] Task 3: Live Tier-L smoke — GL-free `ControlEnv` path

**Files:**
- Modify: `scripts/libero_state_smoketest.py` (`try_libero()` ~line 158)

**What:** Replace the blocked Tier-L path (which imported `benchmark` → torch and `OffScreenRenderEnv` → GL) with the lower-level **state-only** `ControlEnv` construction proven to work locally, then build a `PrivilegedState` via `LiberoStateAdapter` and print it. Tier L now PASSES locally instead of falling back to robosuite.

**Interface:** `try_libero()` returns the existing report dict, now with `tier="L"`, the real `language_instruction`, `obj_of_interest`, and `contract.constructed = True` via the new adapter.

**Test scenarios:** (manual smoke, not pytest) run `PYTHONPATH=~/.cache/t7-libero-smoke/LIBERO ~/.cache/evasion_tax-libero14/venv/bin/python scripts/libero_state_smoketest.py` → prints Tier L PASS with `target_region="plate_1"`.

**Dependencies:** local libero14 venv + LIBERO clone (runtime only; the script stays graceful — `ImportError`/asset/GL failure still falls back to Tier R, so the core suite is unaffected).

**Notes:** Construct the bddl path directly from the clone's `bddl_files/<suite>/` (avoid `benchmark.get_benchmark_dict` so torch is never imported). Keep the tiered+graceful guard contract (never raise on missing deps).

**Commit:** `feat(smoke): Tier-L LIBERO now runs locally via state-only ControlEnv`

---

- [ ] Task 4: Docs — record the overturned deferral + env recipe

**Files:**
- Modify: `docs/setup/libero-local-notes.md`
- Modify: `docs/core/execution-playbook.md` (§1 You-Are-Here + §10 decision ledger)
- Create: `docs/setup/libero-local-env.md` (the reproducible local LIBERO env recipe)

**What:** Record that state-only LIBERO **runs locally** (the 2026-06-03 "Tier L BLOCKED / deferred to GPU" conclusion was env-assembly, not fundamental); document the exact env recipe; note the `*_to_robot0_eef` extraction bug caught against real data and the `target_region = obj_of_interest[-1]` convention; state what the GPU node still must re-validate (gripper threshold + object naming + relative-key filter across **all** suites; production pinned stack Py3.10/robosuite-pin).

**Test scenarios:** (none — docs.)

**Notes:** Keep the honest-record tone of `libero-local-notes.md`. Add a dated "2026-06-09 UPDATE" block rather than rewriting history. In the playbook §10 ledger, link the commit and mark `state_libero.py` **built locally** (was: deferred to GPU node, runbook Step 6). Per reproducibility rules, the local adapter is a **draft re-validated on the GPU node**, not the final.

**Commit:** `docs: state-only LIBERO runs locally; state_libero.py pulled forward`

---

## Out of scope (do NOT do here)

- OpenVLA / GCG / LIBERO **policy rollouts** — still GPU-node only (this is ground-truth state extraction, no model).
- Changing `PrivilegedState` or metric (A) — both FROZEN.
- Committing the local venv, the LIBERO clone, or any checkpoint/dataset (gitignored; provenance only).
- Wiring `LiberoStateAdapter` into a full rollout-recording run — that needs the policy (GPU).
