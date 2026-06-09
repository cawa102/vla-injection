---
source_file: "tests/evasion_tax/metric/fixtures/PROVENANCE.md"
type: "document"
community: "LIBERO State Adapter Plan"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/LIBERO_State_Adapter_Plan
---

# PROVENANCE.md

## Connections
- [[LIBERO obs fixtures provenance]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/LIBERO_State_Adapter_Plan

## 📄 Source

`tests/evasion_tax/metric/fixtures/PROVENANCE.md`

# LIBERO obs fixtures — provenance

Frozen real-LIBERO `reset()` observations used as the regression contract for
`src/evasion_tax/metric/state_libero.py` (`LiberoStateAdapter`). The test suite asserts
the adapter against these **without importing LIBERO/robosuite/GL**, so it runs in the
core `.venv`.

| Field | Value |
|---|---|
| Source | `Lifelong-Robot-Learning/LIBERO` clone at `~/.cache/t7-libero-smoke/LIBERO` |
| Captured | 2026-06-09 |
| Construction | state-only `libero.libero.envs.env_wrapper.ControlEnv(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)` — GL-free, torch-free (bypasses `benchmark`/`OffScreenRenderEnv`) |
| Env recipe | `~/.cache/evasion_tax-libero14/venv` — Py3.10 + `robosuite==1.4.0` + `mujoco` + `numpy<1.24` + small deps (see `docs/setup/libero-local-env.md`) |
| Reset | default seed / default initialization |

Values are kept at full float precision as emitted, so the fixtures are a true
regression anchor. **Caution:** these are captured on a local mac stack, **not** the
production GPU pinned stack (Py3.10 / robosuite-pin) — the GPU node re-validates the
gripper threshold, object naming, and the relative-key filter across all suites.

## Files

- `libero_obs_spatial0.json` — `libero_spatial` task 0,
  *"pick up the black bowl between the plate and the ramekin and place it on the plate"*.
  Binary goal `(On akita_black_bowl_1 plate_1)` → `obj_of_interest=[akita_black_bowl_1, plate_1]`.
  **Resolvable case:** `target_region="plate_1"` IS present in `object_poses`
  (5 absolute objects after the `_to_robot0_eef` relative-key filter).

- `libero_obs_goal_opendrawer.json` — `libero_goal`,
  *"open the middle drawer of the cabinet"*.
  Unary goal `(Open wooden_cabinet_1_middle_region)` → `obj_of_interest=[wooden_cabinet_1_middle_region]`.
  **Abstain case:** `target_region="wooden_cabinet_1_middle_region"` is NOT in `object_poses`
  (abstract region, no pose obs) → the metric's `PrivilegedGoalResolver` abstains (scores 0.0).
  This is expected/handled behaviour, not an adapter error.

## Reproduce

```bash
PYTHONPATH=~/.cache/t7-libero-smoke/LIBERO \
  ~/.cache/evasion_tax-libero14/venv/bin/python /tmp/libero_capture_fixtures.py
```

