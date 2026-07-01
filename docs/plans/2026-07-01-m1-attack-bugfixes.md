# M1 Attack-Path Bug Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task,
> and `test-driven-development` (`/tdd`) for every pure/model-free piece (GPU bodies stay behind the existing
> CUDA guard, like `scripts/run_attack.py`).

**Goal:** Make `scripts/run_attack.py` produce a **complete, valid, resumable** M1 registered run (10 units × 500
GCG steps, bf16 OpenVLA-7B + LIBERO, A5000) so the GO/NO-GO gate result is trustworthy — fixing 5 bugs found by a
2026-07-01 adversarial audit (Codex + verified against code), after a 2-day/50h run was wasted by a deterministic
crash masked as transient OOM.

**Architecture:** Small surgical fixes to existing seams. Most are **model-free (mac, TDD)**: incremental aggregate
write, benign re-scoring against the re-pinned schema, resume-header validation. Two touch the GPU body
(`run_attack.py`): the post-settle target frame and per-unit `empty_cache`. No new subsystems.

**Tech Stack:** Python 3.10; NumPy + pure eval helpers (mac, pytest); torch 2.2.0+cu121 / transformers 4.40.1 /
bf16 OpenVLA-7B + LIBERO (EGL) behind the CUDA guard.

---

## Context — already fixed (do NOT redo)

- **BUG0 (crash, DONE `d876ccd`):** `run_attack.py` passed a numpy image (`get_libero_image`) to a processor that
  needs PIL → `AttributeError: 'numpy.ndarray' has no attribute 'convert'`. Fixed via `_to_pil()` in
  `src/evasion_tax/attack/gcg_openvla.py` (numpy→`Image.fromarray`, PIL passthrough) + a unit test. Verified correct.

## Findings (verified against code, 2026-07-01)

| # | Bug | Evidence (file:line) | Class |
|---|-----|----------------------|-------|
| 1 | `attack_records.json` written only **after** the whole loop → partial/crashed run leaves no aggregate; `m1_gate_report` `FileNotFoundError`; no early-look | `run_attack.py:314-315` (after `run_attack_loop`) | completion / usability |
| 2 | benign scored on the **placeholder** SchemaA (runs before re-pin, box step-1 has no `--schema-from`), attack scored on the **re-pinned** schema; `m1_verdict` compares **stored** scores (no re-score) → separation AUC is cross-scale = **invalid** | `run_benign.py:229,270-291` + `m1_gate.py:67-69,178-182` + `configs/m1_viability.yaml` step 1 | **validity (critical)** |
| 3 | `--resume` reuses units on `path.exists()` only — no schema/commit/n_steps/model check → dry-run or stale units silently mixed | `run_attack.py:116-118` | robustness (low if dry-runs use a separate dir) |
| 4 | GCG target image captured **before** settle (t=0); `run_episode` acts **after** `num_steps_wait=10` dummy steps → suffix optimized on the wrong frame | `run_attack.py:267` vs `rollout_runner.py:217-226` | validity/efficacy (not a crash) |
| 5 | No per-unit `torch.cuda.empty_cache()` (the bench has it) → fragmentation OOM over 10 units at `sw=512` | `run_attack.py` loop vs `bench_early_stop.py` | stall/OOM (partly mitigated: `expandable_segments:True` set via env) |

**Priority:** BUG1 + BUG2 are mandatory before the next launch (BUG2 = invalid result even if the run completes).
BUG5 strongly recommended. BUG3/4 small but real.

---

## Tasks

- [ ] **Task 1: BUG1 — incremental `attack_records.json`**

  **Files:**
  - Modify: `scripts/run_attack.py` (`run_attack_loop` ~113-131; move/duplicate the aggregate write from ~314-315)
  - Modify (belt-and-suspenders): `scripts/m1_gate_report.py` / `_load_list` (~33-37) — fall back to reading
    `units/*.json` when `attack_records.json` is absent
  - Test: `tests/scripts/test_run_attack.py`

  **What:** Write the derived `attack_records.json` **after every completed unit** (and after each reloaded unit on
  resume), so an interrupted run leaves a valid aggregate of finished units and early-look works. Per-unit
  `units/<uid>.json` stays the write-once source of truth; the aggregate is an overwrite-safe view.

  **Interface:** keep `run_attack_loop(...)` signature; write `units_dir.parent / "attack_records.json"` inside the
  loop after each `records.append(...)`. (Optionally accept an explicit `aggregate_path`.)

  **Test scenarios:** after k of N units the aggregate holds exactly k records; an `attack_fn` that raises on unit k
  leaves the aggregate with k-1 records (no exception-swallow — the raise still propagates to the retry loop);
  resume includes reloaded units; `m1_gate_report` reads `units/` when the aggregate file is missing.

  **Dependencies:** none new.

  **Commit:** `fix(attack): write attack_records.json incrementally per unit (partial-run + early-look safe)`

- [ ] **Task 2: BUG2 — re-score benign against the re-pinned SchemaA (Approach A, DM-3 same-scale)**

  **Files:**
  - Create: `src/evasion_tax/eval/rescore_benign.py` + `tests/evasion_tax/eval/test_rescore_benign.py`
  - Create: `scripts/rescore_benign.py` (thin CLI; model-free, no GPU guard)
  - Modify: `configs/m1_viability.yaml` header (insert step 2b) — see Task 6

  **What:** After re-pin, re-score the benign baseline's **logged rollouts** (`results/<benign>/episodes/`) against
  the **re-pinned** `SchemaA`, writing a `benign_records_repinned.json` so benign and attack are on the **same
  scale**. The gate report then consumes the re-pinned benign records.

  **Interface:**
  - `rescore_benign_records(rollouts: Sequence[Rollout], *, success: Sequence[bool], is_calibration: Sequence[bool],
    schema: SchemaA, k: int) -> list[dict]` — pure; returns the `BenignRecord` dicts (`success`,
    `metric_a_per_step`, `is_calibration`) that `benign_records_from_dicts` reads.
  - CLI: `scripts/rescore_benign.py --run results/<benign> --schema-from <schema_repinned.json> --out
    results/<benign>/benign_records_repinned.json`.

  **Test scenarios:** re-scoring with the run's **original** schema reproduces `run_benign`'s stored
  `metric_a_per_step`; a **different** schema changes the scores; `success` + `is_calibration` are preserved
  unchanged; empty input raises (no silent default).

  **Dependencies:** `eval/rollout_io.py` (JSON→`Rollout` loader), `metric/consistency_a.ConsistencyMetricA/SchemaA`,
  `eval/schema_repin` output.

  **Notes / OPEN before coding:** confirm what `episodes/ep*.json` contains — the raw `RolloutStep` stream (needed
  to re-score) **and** where per-episode `success` + `is_calibration` live (episodes vs the aggregate
  `benign_records.json`). Reuse the existing metric; do **not** re-roll benign (rollouts are already on disk).

  **Commit:** `feat(eval): re-score benign against the re-pinned SchemaA (DM-3 same-scale separation)`

- [ ] **Task 3: BUG3 — validate run header before `--resume` reuse**

  **Files:**
  - Modify: `scripts/run_attack.py` (`_run` / `run_attack_loop`, ~113-131 + the run.json write ~235)
  - Test: `tests/scripts/test_run_attack.py`

  **What:** On `--resume`, before reusing any `units/*.json`, assert the existing `run.json` header matches the
  current run (schema hash from `--schema-from`, `git_commit`, `n_steps`, `search_width`, `eval_batch`, model,
  `seed`). Abort with a clear message on mismatch instead of silently mixing incompatible units.

  **Interface:** `assert_resume_compatible(run_dir: Path, current_header: Mapping) -> None` (raises `SystemExit`/
  `ValueError` with the offending field on mismatch; no-op when `run.json` is absent = fresh run).

  **Test scenarios:** identical header → proceeds; changed `n_steps`/`search_width`/schema/commit → raises naming
  the field; absent `run.json` → no-op.

  **Notes:** also document: **dry-runs MUST use a separate `--results-root` / `--run-name`** (never the registered
  `m1-robogcg-redirect` dir).

  **Commit:** `fix(attack): validate run header before --resume reuse (no cross-config unit mixing)`

- [ ] **Task 4: BUG4 — build the GCG target on the post-settle rollout-start frame**

  **Files:**
  - Modify: `src/evasion_tax/eval/rollout_runner.py` (extract the reset+settle seam from `run_episode` ~217-226)
  - Modify: `scripts/run_attack.py` (~265-267: capture `start_image` via the shared seam)
  - Test: `tests/evasion_tax/eval/test_rollout_runner.py`

  **What:** Share one helper that resets, applies `num_steps_wait` dummy steps, and returns the **first policy obs**,
  used by BOTH the GCG target capture and `run_episode`, so the suffix is optimized on the exact frame the rollout
  acts from.

  **Interface:** `reset_and_settle(env, *, init_state, model_family, num_steps_wait=10) -> obs` (GPU-guarded body;
  the settle-count/dummy-action logic is unit-tested with a mock env).

  **Test scenarios (mock env):** exactly `num_steps_wait` dummy steps applied; returns the obs after settle;
  `run_episode` behaviour byte-unchanged when refactored to call it (existing rollout test still green).

  **Dependencies:** OpenVLA eval helpers (`get_libero_dummy_action`); the step-4 loop.

  **Commit:** `fix(attack): build GCG target on the post-settle rollout-start frame (match run_episode)`

- [ ] **Task 5: BUG5 — `torch.cuda.empty_cache()` between units**

  **Files:**
  - Modify: `scripts/run_attack.py` (attack loop / GPU body; mirror `bench_early_stop.py`)
  - Test: `tests/scripts/test_run_attack.py` (GPU call mocked — assert the cache-clear hook fires per unit boundary)

  **What:** After each unit (post suffix-quarantine, before the next target build) call `torch.cuda.empty_cache()`
  to curb fragmentation OOM over 10 sequential `sw=512` searches. Keep documenting `expandable_segments:True` (set
  via env) in the config/runbook.

  **Test scenarios:** GPU-mocked — one cache-clear per unit boundary; off-GPU guard unchanged.

  **Commit:** `fix(attack): torch.cuda.empty_cache() between units (A5000 fragmentation at sw=512)`

- [ ] **Task 6: box sequence + docs**

  **Files:**
  - Modify: `configs/m1_viability.yaml` (header launch sequence: insert **step 2b** `rescore_benign.py`; gate report
    consumes `benign_records_repinned.json`; note the dry-run separate dir)
  - Modify: `docs/plans/2026-06-24-m1-viability-gate.md` (DM-3 flow: benign re-score step) + `docs/core/execution-playbook.md` §1 (bugfix note)

  **What:** Update the box runbook so the corrected order is: benign → re-pin → **re-score benign** → attack dry-run
  (separate dir) → attack registered → gate report (on the re-pinned benign records).

  **Commit:** `docs: M1 box sequence — benign re-score step + attack bugfix notes`

---

## Build order & where each runs

1. **Mac (TDD, `/tdd`):** Task 1 → Task 2 → Task 3 (model-free) → Task 4 pure seam → Task 5 (GPU call mocked). Suite
   green, ruff + pyright clean.
2. **Box:** attack **dry-run** (`--n-attacked 1 --n-steps 20 --results-root results/_smoke --run-name attack-dry`)
   → confirm it completes, writes `attack_records.json`, quarantines the suffix, and `m1_gate_report` runs on the
   re-scored benign. THEN the registered attack (detached, resume) → re-score/report.
3. **Mac:** gate report interpretation.

**Verify gates:** (a) aggregate exists after each unit + survives a mid-loop crash; (b) benign re-scored on the
re-pinned schema — benign & attack on the SAME scale; (c) resume aborts on header mismatch; (d) GCG target image ==
post-settle rollout-start frame; (e) `empty_cache` fires per unit; (f) dry-run (1 unit, 20 steps) completes and the
whole benign→re-pin→re-score→attack→report chain runs end-to-end before spending the registered hours.

---

## References

@scripts/run_attack.py · @scripts/run_benign.py · @scripts/m1_gate_report.py · @src/evasion_tax/eval/m1_gate.py ·
@src/evasion_tax/eval/rollout_runner.py · @src/evasion_tax/eval/rollout_io.py ·
@src/evasion_tax/attack/gcg_openvla.py (BUG0 fix) · @src/evasion_tax/attack/early_stop_bench.py ·
@src/evasion_tax/metric/consistency_a.py · @configs/m1_viability.yaml ·
@docs/plans/2026-06-24-m1-viability-gate.md (DM-1..DM-8) · @docs/core/execution-playbook.md (§1, §6 D8).
