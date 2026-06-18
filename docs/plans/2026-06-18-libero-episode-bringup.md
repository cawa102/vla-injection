# CSB Bring-up Step 4 — One LIBERO Episode (EGL) with the bf16 Policy — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use `test-driven-development` (`/tdd`) to implement Task 1 task-by-task.

**Goal:** Run one real `libero_spatial` episode on the CSB A5000 with the bf16 LIBERO-finetuned OpenVLA
policy driving a camera-rendering (EGL) env, completing the loop end-to-end and logging each step in the
frozen `RolloutStep` / `PrivilegedState` schema so the metric/state side ingests real OpenVLA-driven data.

**Architecture:** A focused bring-up script `scripts/smoke_libero_episode.py` mirroring
`scripts/smoke_openvla_load.py` (CUDA guard + `RunLogger` repro header). The CUDA-touching body (load policy,
build EGL env, run the episode) lives behind the guard; a **model-free seam** — turning one `(obs, action,
metadata)` triple into a `RolloutStep` via the real `LiberoStateAdapter` — is extracted as a pure function so
it is unit-testable locally without CUDA/LIBERO. Image preprocessing + action transforms are **ported from
OpenVLA's pinned LIBERO eval** (`run_libero_eval.py` @ `c8f03f48`), not hand-rolled.

**Tech Stack:** Python 3.10, PyTorch 2.2.0+cu121 (bf16), `transformers` 4.40.1 (OpenVLA via `trust_remote_code`),
LIBERO (`OffScreenRenderEnv`, MuJoCo + robosuite, `MUJOCO_GL=egl`), the project's `evasion_tax` package
(`records`, `metric.state_libero`, `repro`, `config`).

**Scope boundary (do not exceed):** This is the **wiring de-risk** — verify gate = *one episode completes +
log schema matches the state-adapter / metric side* (`docs/gpu/CSB/plan.md` step 4). It is **NOT** the benign
success-rate measurement (≥300 benign rollouts vs published numbers) — that is the later **M1** task /
`docs/setup/gpu-runbook.md` Step 3. One task, one episode.

**References:** @docs/gpu/CSB/plan.md (step 4 + how-to) · @docs/setup/gpu-runbook.md (Steps 1–3, canonical
`run_libero_eval.py` recipe) · @scripts/smoke_openvla_load.py (guard + repro pattern to mirror) ·
@scripts/demo_rollout.py (the record schema this must match) · @src/evasion_tax/metric/state_libero.py (the
real adapter).

---

## Task 1: `scripts/smoke_libero_episode.py` + model-free seam (TDD — mac, now)

**Files:**
- Create: `scripts/smoke_libero_episode.py`
- Test: `tests/evasion_tax/test_smoke_libero_episode.py`

**What:** The bring-up script. Locally (no CUDA) it **guards** — prints `gpu_required_message(STAGE)` and exits
`2`, never a silent no-op (same contract as `smoke_openvla_load.py` / `run_benign.py`). On the box it runs the
episode (Task 2). The extracted pure helper is the only part exercised by local tests.

**Interface (public, model-free — the locally-testable seam):**
- `build_rollout_step(obs: Mapping, action: Sequence[float], *, adapter: LiberoStateAdapter, run_id: str,
  seed: int, git_commit: str | None, suite: str, task_id: str, step: int, instruction: str,
  trusted_goal: str) -> RolloutStep` — build one canonical record: `privileged_state =
  dataclasses.asdict(adapter.to_privileged_state(obs))`, `action` coerced to 7-tuple by `RolloutStep`,
  `attacked=False`, `suffix_ref=None`, `observation_ref=f"{suite}/{task_id}/{step}"`. Pure: no CUDA, no LIBERO.

**Interface (CUDA-guarded body — box only, Task 2):**
- `main(argv: list[str] | None = None) -> int` — argparse (`--model` default
  `openvla/openvla-7b-finetuned-libero-spatial`, `--unnorm-key` default `libero_spatial_no_noops`,
  `--task-suite` default `libero_spatial`, `--task-id` default `0`, `--max-steps` default `220`,
  `--num-settle-steps` default per OpenVLA recipe `[VERIFY]`, `--attn-impl` default `sdpa`, `--device`
  `cuda:0`, `--seed` `42`, `--results-root` `results/_smoke`). Guard → heavy imports → run episode → log.

**Test scenarios (Task 1 only — the model-free seam):**
- `build_rollout_step` on a synthetic **camera** obs dict (contains `agentview_image` + state keys
  `robot0_eef_pos`, `robot0_gripper_qpos`, `<obj>_pos`, `<obj>_to_robot0_eef_pos`) returns a valid
  `RolloutStep`; its `privileged_state["object_poses"]` **excludes** `agentview_image` and the `_to_` delta
  and `robot0_*` keys (camera-env obs parity with the state-only adapter contract).
- `privileged_state["ee_pos"]` equals the obs `robot0_eef_pos`; `target_region` equals the adapter's
  `obj_of_interest[-1]`.
- `action` of wrong length (≠7) propagates `ValueError` from `RolloutStep` (boundary check, not swallowed).
- Importing the script on a **CUDA-free** host does not import torch/LIBERO at module load (heavy imports are
  inside `main` after the guard) — assert the module imports cleanly in the core `.venv`.
- Running `main([])` with `cuda_available()` monkeypatched `False` returns `2` and prints the GPU-required
  message (mirror `test_runtime.py` / the `run_benign` guard contract).

**Dependencies:** `evasion_tax.records` (`RolloutStep`), `evasion_tax.metric.state_libero`
(`LiberoStateAdapter`), `evasion_tax.repro` (`RunLogger`, `seed_everything`, `capture_env`),
`evasion_tax.config` (`cuda_available`, `gpu_required_message`), `scripts/_bootstrap`.

**Notes:**
- Mirror `smoke_openvla_load.py` exactly for: the `_bootstrap` import, the post-guard heavy imports, the
  `RunLogger().start(...).write(...)` log path, the peak-VRAM capture (`max_memory_reserved`), and the
  `fits_one_card` gate (reuse — the EGL render context shares the card; ~9 GB headroom from step 3).
- Synthetic obs fixture: hand-author a dict matching the real `libero_spatial` obs key set documented in
  `docs/setup/libero-local-notes.md` (so the fixture is grounded, not invented).
- `git_commit` comes from `capture_env()` (already used by `RunLogger`); pass through, do not re-shell git.

**Commit:** `feat: CSB bring-up step 4 — smoke_libero_episode.py guard + model-free RolloutStep seam (TDD)`

---

## Task 2: On-box episode body + execution runbook (box — next CSB session)

> Procedural, on the A5000; each sub-step has a verify gate (mirrors the `docs/gpu/CSB/plan.md` ladder). The
> CUDA-guarded body of `smoke_libero_episode.py` is filled in here against the real env; preprocessing/transforms
> are **imported or ported verbatim** from OpenVLA's pinned eval — research first (Context7 / the repo), do not
> reconstruct constants from memory.

**2a. Env assembly — add LIBERO (camera/EGL) to the working uv `.venv` (lock-external, like steps 1–3).**
- Clone `LIBERO` and `openvla` @ `c8f03f48`; `uv pip install -e LIBERO` + `uv pip install -r
  openvla/experiments/robot/libero/libero_requirements.txt` into the existing `.venv`.
- Default = stay in uv (single env, consistent with steps 1–3). **Fallback if deps conflict** with the
  installed inference stack (robosuite/mujoco vs transformers/timm pins) = a dedicated `micromamba` env per
  `docs/setup/gpu-runbook.md` Step 1 — record whichever is used (invariant #8).
- *Verify:* `import libero`; `OffScreenRenderEnv(...)` builds; **`MUJOCO_GL=egl` initialises on the A5000** —
  a 1-frame `env.render`/obs grab returns an `agentview_image` array (this retires the EGL headless risk,
  `docs/gpu/CSB/plan.md` §Environment).
- **Done 2026-06-18 (box) — import gate GREEN, two gotchas resolved (canonical commands now in
  `docs/gpu/CSB/plan.md` Step 4 how-to; pins in `configs/env/requirements-gpu.txt`):**
  1. `uv pip install -e ~/LIBERO` does **not** make `import libero` work — LIBERO's top `libero/` is a PEP-420
     namespace package (no `__init__.py`); uv's PEP-660 editable finder won't expose it (pip's legacy `-e`
     would, since it puts the repo root on `sys.path`). **→ `PYTHONPATH=~/LIBERO`** at import **and** run time
     (the script only adds `--openvla-root`, so LIBERO still needs PYTHONPATH).
  2. The eval-helper import chain (`robot_utils`→`prismatic`→`dlimp`→`tensorflow_datasets`→`tensorflow_metadata`)
     needs the TF stack; tfds 4.9.3 caps nothing on tensorflow-metadata, so pip pulled tfmd 1.21.0 (wants
     protobuf≥5.26) → `runtime_version` ImportError on protobuf 4.25.9. **→ `tensorflow-metadata<1.16` +
     `protobuf<5`** (tensorflow 2.15.0 / tfds 4.9.3 already correct). `import libero, experiments.robot.robot_utils`
     → `helpers OK`.
  - **Still pending (2c–2d):** EGL `agentview_image` render on the A5000 + the actual episode run.

**2b. Checkpoint switch + provenance.**
- `huggingface-cli download openvla/openvla-7b-finetuned-libero-spatial` to `HF_HOME` (~14 GB, roomy disk).
- Fill its row in `docs/references/README.md` (source URL, **SHA-256** of resolved weights, date, **licence —
  VERIFY on the model card**, Llama-2 backbone note).
- *Verify:* loads bf16 + fits one card (reuse the step-3 VRAM gate).

**2c. Episode loop (`main`'s guarded body — written + verified vs `c8f03f48` on the mac 2026-06-18; runs on the box).**
- Build env for `libero_spatial` task 0 via OpenVLA's `get_libero_env` (`resolution=256` per recipe,
  `MUJOCO_GL=egl`); `--center_crop True` semantics applied to the image (ESSENTIAL — fine-tunes used
  random-crop aug, `gpu-runbook` Step 3).
- Settle: run the recipe's no-op/dummy actions for `--num-steps-wait` — **VERIFIED = 10** (`GenerateConfig.num_steps_wait`
  @ c8f03f48). The settle steps are **extra** (recipe loops `t < max_steps + num_steps_wait`), so the policy keeps
  the full `--max-steps` (220) budget — corrected in the script (was `range(max_steps)`, losing 10 policy steps).
- Per step: preprocess `obs["agentview_image"]` (flip `[::-1, ::-1]` + resize 224 + center-crop) → `prompt =
  "In: What action should the robot take to {instruction}?\nOut:"` → `vla.predict_action(..., unnorm_key=
  libero_spatial_no_noops, do_sample=False)` → apply OpenVLA's gripper transforms
  (`normalize_gripper_action(…, binarize=True)` then `invert_gripper_action`) — **VERIFIED order** @ c8f03f48 →
  `env.step` → `build_rollout_step(obs, action, ...)`. Stop at LIBERO `done` or `--max-steps`.
- **Model-load divergence (VERIFIED + deliberate):** OpenVLA's `get_vla` **hardcodes**
  `attn_implementation="flash_attention_2"`, which the box has not built (step 3 ran `sdpa`, caveat L5). The script
  therefore loads directly (mirroring `smoke_openvla_load.py`) to honour `--attn-impl sdpa`, **not** via `get_model`.
  For a hub id this is equivalent w.r.t. `norm_stats` (embedded; `get_vla`'s local `dataset_statistics.json` branch
  is skipped for hub ids anyway). The preprocess/settle/gripper/action path stays verbatim (`get_action`,
  `get_libero_image`, `quat2axisangle`). **If you instead want flash-attn on the box**, build/verify `flash-attn`
  first (L5), then pass `--attn-impl flash_attention_2`.
- *Verify (on the box):* loop completes N steps with **no crash**; `done`/success flag captured.

**2d. Write-once log + verify gate.**
- `RunLogger("results/_smoke").start("libero-episode-smoke", config=..., seed=...)`; write: `steps` (the
  `RolloutStep` list), `episode_meta` (suite, task_id, instruction, `obj_of_interest`, n_steps, success,
  settle_steps, center_crop, camera size, `MUJOCO_GL`), `peak_vram_*`, `fits_one_card`, full repro header.
- *Verify gate (step 4 PASS):* one `PrivilegedState` constructs from the **camera-env** obs (metric-side
  ingestion confirmed) **and** the record round-trips (re-read JSON → reconstruct `RolloutStep`); episode
  completed within `--max-steps`. Logged under `results/_smoke/` (tracked, non-registered).

**Commit (box):** `result: CSB bring-up step 4 GREEN — one libero_spatial episode (EGL) with bf16 finetuned policy`

---

## Done-when (Step 4 exit)
- [x] Task 1 green locally (core `.venv`): guard returns `2` CUDA-free, model-free seam tests pass (4), ruff
      clean; full suite 410 passed / 0 failed (2026-06-18).
- [x] Box: `import libero` + EGL `agentview_image` render OK on the A5000 (2a verify). *(2026-06-18: import chain GREEN via `PYTHONPATH=~/LIBERO` + `tensorflow-metadata<1.16`/`protobuf<5`; EGL `MUJOCO_GL=egl` initialised + rendered during the episode → headless-render risk retired.)*
- [x] Box: one `libero_spatial` task-0 episode completes; `results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke`
      written with the `RolloutStep` schema + repro header. *(90 policy steps, success=True, sdpa, peak VRAM 14.50 GiB / 23.5 GiB → fits one card. Third gotcha: `--unnorm-key` = `libero_spatial` not `*_no_noops` — script default fixed. Commit the run log from the box: `results/_smoke` is tracked.)*
- [x] Tick `docs/gpu/CSB/plan.md` step 4 `[x]` (run_id + peak VRAM done); playbook §1 updated (next = step 5: attach the L2 detector to the real rollout). *(Checkpoint provenance row — 2b SHA-256/licence — still TODO before any **registered** run; the smoke run is non-registered.)*
