# Early-Stop Steps-to-Success Bench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task,
> and `test-driven-development` (`/tdd`) for every pure/model-free piece (the GPU bodies stay behind the existing
> CUDA guard, exactly like `scripts/microbench_gcg.py` / the `OpenVlaGcgTarget` seam).

**Goal:** Measure the **A5000 GCG steps-to-success distribution with early_stop ON** over a small target set, so the
realistic per-target attack cost (`median_steps × s/step`) can replace the registered **early_stop-OFF worst case**
(16,595 s ≈ 4.61 h) that currently drives `branch_select` to the conservative provisional Branch N−.

**Architecture:** `run_gcg(fn, cfg, *, reached_fn=…)` in `src/evasion_tax/attack/gcg.py` **already** stops early when
`reached_fn` reports the target reached — nothing currently passes one. We add (1) a **pure** RoboGCG-style success
predicate (argmax forced-decode match over the target span), (2) a GPU-guarded `OpenVlaGcgTarget.reached()` that
forwards once and calls the pure predicate, (3) a **pure** bench-bookkeeping module (per-target outcome / summary /
resume helpers), and (4) a **checkpointed, resumable** driver script that runs each target through `run_gcg` with
`reached_fn=target.reached`, writes each outcome write-once on completion (skip-if-exists on restart), quarantines
each suffix, and aggregates the distribution. A prerequisite (5) **chunks `loss_of` candidate evaluation at
`eval_batch`** so a *real* `search_width=512` attack fits 24 GB — without it `run_gcg(sw=512)` OOMs (the single
512-wide forward materialises ~8 GiB of logits on top of the 14 GiB model; observed 2026-06-23). Built to run
**unattended** (`nohup` + `systemd-inhibit`; per-target checkpoint + a shell `until` auto-restart loop), per
`docs/gpu/CSB/plan.md` *Unattended runs*.

**Tech Stack:** Python 3.10; NumPy (pure predicate + bookkeeping + tests); torch 2.2.0+cu121 / transformers 4.40.1 /
bf16 OpenVLA-7B behind the CUDA guard (identical to `smoke_openvla_gradient.py` and the existing GCG seam).

---

## Decisions (pre-registered)

- **DE-1 — Success criterion = RoboGCG-style exact forced-decode match.** "Target reached" ⇔ the **greedy/argmax**
  next-token decode over **every non-ignored target-span position** equals the target action-token ids (the same
  causal shift `per_sequence_ce` uses). Chosen over a loss-threshold (no calibration-free threshold exists) and
  over top-k membership (weaker). Author-agreed default 2026-06-23.
- **DE-2 — `n_steps` cap = 500, `search_width` = 512 (RoboGCG-faithful budget).** A target that does not reach
  within 500 steps is **censored**, not a failure — recorded as `censored=True`, `steps_to_success=500`, and
  **excluded from the median** (reported separately as `censored_fraction`). This keeps the realistic cost honest
  (a high censored fraction → the worst-case bound is the right planning number after all).
- **DE-3 — `reached_fn` is checked every step (RoboGCG parity).** One extra B=1 forward/step (~`t_fwd`≈2 s on top of
  ~33 s/step ⇒ ~6 % overhead) — acceptable, and it is what makes `steps_to_success` meaningful. Do **not**
  micro-optimise to target-position-only logits (out of scope, mirrors DB-5).
- **DE-4 — Per-target write-once checkpoint + resume.** Each finished target writes one JSON immediately under the
  run dir; a restart **skips** any target whose JSON already exists (idempotent). This is what makes the unattended
  `until`-loop / OOM auto-restart safe (`docs/gpu/CSB/plan.md`). The aggregate reads **all** target JSONs (fresh +
  resumed) so a multi-restart run still produces one clean distribution.
- **DE-5 — Ethics: every optimised suffix is an adversarial artifact → quarantine.** Suffixes are written under
  `artifacts/untrusted/` (gitignored), never committed, never auto-run (D6-6). Only **outcome metadata** (steps,
  reached, loss, vram, `suffix_sha256`) goes to `results/`.
- **DE-6 — Registered run, but `--exclusive-gpu` only nice-to-have.** This bench is **not** a timing bench (we want
  the steps distribution, not sub-second wall-time precision), but the box is author-exclusive so run exclusive and
  log the full §8 header; the realistic-cost number it yields **updates `branch_select`** so it is a real result.
- **DE-7 — `loss_of` candidate-eval chunking is a PREREQUISITE for the faithful sw=512 attack (un-defers DB-5's
  "eval mini-batching").** The current `loss_of` runs **one** forward over **all** `B` candidates, so `run_gcg` at
  `search_width=512` materialises `logits [512, seq, V]` (~8 GiB) on top of the ~14 GiB model → **OOM on 24 GB**
  (observed 2026-06-23: 23.47 GiB in use, 604 MiB alloc fails). The registered D8 only ran the direct loop at
  **sw=32** (1 chunk, fits) and computed the sw=512 cost **analytically** (`⌈512/eval_batch⌉·t_fwd`). To run a
  *real* early_stop sw=512 attack we must forward candidates **`eval_batch` at a time and concatenate the `[B]`
  losses** — peak VRAM then becomes one chunk's, so sw=512 fits. This was explicitly deferred in the step-6 plan
  (DB-5 / "eval mini-batching deferred to M1/Task D"); the bench un-defers it. **The D8 registered numbers are
  unaffected** (they never ran `loss_of(512)`; `loss_of(32)` = 1 chunk = identical).

---

## Shared contract (the pure success predicate both the GPU path and tests depend on)

```python
# src/evasion_tax/attack/early_stop.py  (pure, numpy — unit-tested off-GPU)
def target_span_argmax_matches(
    logits: np.ndarray,        # [T, V] next-token logits for ONE sequence
    labels: np.ndarray,        # [T] target ids, ignore_index where masked
    *, ignore_index: int = -100,
) -> bool:
    """Greedy forced-decode check (DE-1). Causal shift: position t is predicted from
    logits[t-1]; for EVERY non-ignored target position argmax(logits[t-1]) == labels[t].
    Returns True iff all non-ignored positions match (== 'target reached'). Mirrors the
    shift in per_sequence_ce so the GPU `reached()` agrees with the loss seam."""
```

---

## Tasks

- [x] **Task 1: pure success predicate (`target_span_argmax_matches`) — mac**

  **Files:**
  - Create: `src/evasion_tax/attack/early_stop.py`
  - Test: `tests/evasion_tax/attack/test_early_stop.py`

  **What:** Implement the pure predicate above. No torch at module top (keep the existing "attack module imports no
  torch" invariant test green).

  **Interface:**
  - `target_span_argmax_matches(logits, labels, *, ignore_index=-100) -> bool` (contract above).

  **Test scenarios:**
  - All non-ignored positions argmax-match → `True`; one mismatch → `False`.
  - Fully-ignored `labels` → documented behaviour (vacuously `True`, no crash) — note it explicitly.
  - Causal shift is correct (position t predicted from t-1), matching `per_sequence_ce`'s shift on a shared fixture.
  - Single non-ignored position works (the minimal case).

  **Dependencies:** numpy.

  **Notes:** Pair the fixture with `per_sequence_ce` so a CE≈0 sequence also argmax-matches (sanity cross-link).

  **Commit:** `feat: pure RoboGCG-style forced-decode success predicate for GCG early-stop`

- [x] **Task 2: GPU `OpenVlaGcgTarget.reached()` (the `reached_fn`) — mac (guarded) + box wiring**

  **Files:**
  - Modify: `src/evasion_tax/attack/gcg_openvla.py` (add `reached`)
  - Test: `tests/evasion_tax/attack/test_gcg_openvla.py` (guard/shape contract; pure logic already in Task 1)

  **What:** Add `reached(self, suffix_ids) -> bool`: build `_full_ids(suffix_ids)` + `_labels(...)`, one
  `torch.no_grad()` forward → `logits [1, seq, V]`, take row 0, call `target_span_argmax_matches`. This is exactly
  the `reached_fn` `run_gcg` expects (`Callable[[np.ndarray], bool]`), so callers pass `reached_fn=target.reached`.
  Reuse the existing `_full_ids`/`_labels`/forward machinery (same as `_loss_single`); no new forward path.

  **Interface:**
  - `OpenVlaGcgTarget.reached(suffix_ids: np.ndarray) -> bool`.

  **Test scenarios:**
  - Off-GPU ⇒ the method's torch use is reached only under the guard (module still imports torch-free).
  - On-box wiring (verify gate = a tiny run): for the **init** suffix `reached` is `False`; after `run_gcg` reports
    `reached=True`, `target.reached(best_suffix)` independently returns `True` (consistency with `run_gcg`'s flag).
  - `reached` agrees with "loss≈0": when `_loss_single(best) ≈ 0` the predicate is `True` (cross-check, DE-1).

  **Dependencies:** Task 1; the existing seam (`_full_ids`, `_labels`, the model forward).

  **Notes:** Keep `token_gradient`/`loss_of` unchanged. One extra B=1 forward per step is intended (DE-3).

  **Commit:** `feat: OpenVlaGcgTarget.reached() — GPU forced-decode predicate as run_gcg reached_fn`

- [x] **Task 3: pure bench bookkeeping (outcome / summary / resume / realistic cost) — mac**

  **Files:**
  - Create: `src/evasion_tax/attack/early_stop_bench.py`
  - Test: `tests/evasion_tax/attack/test_early_stop_bench.py`

  **What:** Pure dataclass + aggregation + resume helpers (no torch). The driver (Task 6) does only I/O + the GPU
  loop around these.

  **Interface:**
  ```python
  @dataclass(frozen=True)
  class TargetOutcome:
      target_id: str
      reached: bool
      steps_to_success: int     # run_gcg.n_steps_run (== n_steps when censored)
      censored: bool            # not reached within n_steps (DE-2)
      best_loss: float
      wall_seconds: float
      peak_vram_gib: float
      suffix_sha256: str

  def steps_to_success_summary(outcomes: Sequence[TargetOutcome], *, n_steps_cap: int) -> dict:
      """n, reached_fraction, censored_fraction, and median/IQR of steps_to_success over
      REACHED targets only (censored excluded from the median, reported separately, DE-2)."""

  def realistic_s_per_target(median_steps: float, s_per_step: float) -> float:
      """median_steps * s_per_step — the early-stop realistic per-target cost that replaces
      the early_stop-OFF worst case fed to branch_select (s_per_step from D8 = 33.19 s)."""

  def target_id_for(seed: int, index: int) -> str:              # stable id, e.g. f"t{seed+index}"
  def is_target_done(targets_dir: str | Path, target_id: str) -> bool   # skip-if-exists (DE-4)
  def outcome_to_record(o: TargetOutcome) -> dict                # JSON-safe per-target record
  ```

  **Test scenarios:**
  - `steps_to_success_summary` excludes censored targets from the median, reports `censored_fraction` /
    `reached_fraction`; all-censored → median `None` (or NaN-free sentinel) + `censored_fraction == 1.0`.
  - `realistic_s_per_target(60, 33.19) ≈ 1991 s`; rejects non-positive inputs (no silent default).
  - `is_target_done` True iff the per-target JSON exists; `target_id_for` is deterministic.

  **Dependencies:** numpy; stdlib (`statistics`/`pathlib`/`hashlib` in the driver, not here).

  **Notes:** Keep this module torch-free and I/O-light (path existence only); heavy I/O lives in the driver.

  **Commit:** `feat: pure early-stop bench bookkeeping (outcome, distribution summary, resume helpers)`

- [x] **Task 4: extract the shared frozen-OpenVLA loader/target-builder — mac**

  **Files:**
  - Create: `src/evasion_tax/attack/openvla_loader.py` (move `_load_frozen_openvla` + `_build_target` +
    `_DEFAULT_INSTRUCTION` out of `scripts/microbench_gcg.py`, GPU-guarded as today)
  - Modify: `scripts/microbench_gcg.py` (import the two helpers from the new module instead of defining them)
  - Test: `tests/evasion_tax/attack/test_openvla_loader.py` (guard/contract only; GPU body unchanged)

  **What:** DRY: the bench driver (Task 6) needs the exact same frozen-bf16 load + target build as the microbench;
  avoid a second copy that can drift. Pure move + re-import; **no behaviour change** to the registered microbench.

  **Interface:** `load_frozen_openvla(...)`, `build_target(...)` (public names; keep signatures identical to the
  current private ones).

  **Test scenarios:**
  - Existing `microbench_gcg` unit tests stay green (no behaviour change).
  - Off-GPU ⇒ guard exits non-zero (module importable torch-free).

  **Dependencies:** none new.

  **Notes:** Surgical refactor — if it grows beyond a move + re-import, stop and reassess. Do **not** touch the
  registered `results/2026-06-23T13-34-55Z-gcg-microbench/` record.

  **Commit:** `refactor: extract shared frozen-OpenVLA loader/target-builder for reuse by the bench`

- [x] **Task 5: `loss_of` candidate-eval chunking (eval mini-batching, DE-7) — mac (TDD) + box fit-check**

  **Files:**
  - Modify: `src/evasion_tax/attack/gcg_openvla.py` (`OpenVlaGcgTarget.__init__` + `loss_of`)
  - Test: `tests/evasion_tax/attack/test_gcg_openvla.py` (chunked-vs-single equivalence; pure where possible)

  **What:** Make `loss_of` forward candidates in **chunks of `eval_batch`** and concatenate the per-chunk `[B]`
  losses, so `run_gcg` at `search_width=512` fits 24 GB (DE-7). Add an `eval_batch: int | None = None` attribute to
  `OpenVlaGcgTarget.__init__`: `None` ⇒ **one forward over all B** (current behaviour — preserves the D8 sw=32
  path and the existing `batched_matches_single`/`per_sequence_ce` tests bit-for-bit); an int ⇒ loop over
  `ceil(B/eval_batch)` chunks, `torch.cuda.empty_cache()` between chunks, peak VRAM = the **max single-chunk** peak
  (that is the real 24 GB ceiling). Output `[B]` is identical to the single-forward path within `atol` regardless
  of chunk size (chunking is a pure reassociation of independent per-row CE).

  **Interface:**
  - `OpenVlaGcgTarget(__init__ …, eval_batch: int | None = None)`.
  - `loss_of(candidate_suffixes: np.ndarray) -> np.ndarray` — `[B, L] → [B]`, chunked at `eval_batch` when set.

  **Test scenarios:**
  - **Chunked == single:** `loss_of` with `eval_batch ∈ {1, 8, 32}` equals the `eval_batch=None` result (and the
    per-candidate `_loss_single`) within `atol=1e-3`, across multiple B (incl. B not divisible by `eval_batch`,
    e.g. B=50 / chunk 32 → chunks 32+18) and ≥2 suffix lengths.
  - **Rank order preserved:** `argmin` of the chunked `[B]` == `argmin` of the single path (DB-4 spirit).
  - **Determinism:** same seed/dtype ⇒ identical `[B]` across repeats.
  - **Back-compat:** `eval_batch=None` is byte-for-byte the current behaviour (existing tests untouched & green).
  - **(box fit-check, verify gate = the run):** a `run_gcg` at `search_width=512`, `eval_batch=32` completes with
    **peak VRAM < 24 GiB** (the OOM at sw=512 is gone); a few steps suffice.

  **Dependencies:** the existing `loss_of`/`_target_span_ce_torch`/`per_sequence_ce` seam; numpy.

  **Notes:** Un-defers DB-5's "eval mini-batching" **only** for candidate evaluation — the `token_gradient`
  (B=1 backward) path and target-position-only logits stay out of scope. The chunk size is the **HW-adapted
  `batch_size`** (the measured max-B ≈ 32–43, DC-2). **Do not re-measure or alter the registered D8 record.**

  **Commit:** `feat: chunk loss_of candidate eval at eval_batch (sw=512 fits 24 GB; chunked==single gated)`

- [x] **Task 6: the early-stop bench driver `scripts/bench_early_stop.py` — mac core + box run**

  **Files:**
  - Create: `scripts/bench_early_stop.py`
  - Test: `tests/scripts/test_bench_early_stop.py` (pure arg-parse + the guard + the resume/aggregate glue with the
    GPU call mocked)

  **What:** GPU-guarded driver. Parse args; if no CUDA → `gpu_required_message` + exit 2 (no silent no-op). Load the
  model once (Task 4). For `i in range(n_targets)`: `tid = target_id_for(seed, i)`; if `--resume` and
  `is_target_done` → log skip + continue (DE-4); else build target (seed+i) **with `eval_batch` chunking so the
  sw=512 attack fits (Task 5)**, reset CUDA peak, time `run_gcg(target,
  cfg, reached_fn=target.reached)` (early_stop ON, DE-3), build a `TargetOutcome`, **write its JSON write-once** to
  `results/<run>/targets/<tid>.json`, and **quarantine** the suffix to `artifacts/untrusted/<run>/<tid>.txt`
  (DE-5). After the loop, read **all** per-target JSONs, `steps_to_success_summary(...)`, print
  `realistic_s_per_target(median, 33.19)` + a one-line branch hint, and write `bench_result.json` + `run.json`
  (§8 repro header: env, pinned seed, **which A5000**, bf16, CUDA/driver/torch, git commit).

  **Interface (CLI):**
  - `--config configs/example_m2.yaml --n-targets 5 --search-width 512 --n-steps 500 --seed 42`
  - `--device cuda:0 --attn-impl flash_attention_2 --eval-batch 32 --results-root results --exclusive-gpu`
    (`--eval-batch` = the candidate-eval chunk size from Task 5 → `--search-width 512` fits 24 GB, DE-7).
  - `--resume/--no-resume` (default **resume on**, DE-4).

  **Test scenarios (GPU mocked):**
  - Off-GPU ⇒ guard exits 2.
  - `--resume` skips a target whose JSON already exists; a missing one is run.
  - The final aggregate equals `steps_to_success_summary` over fresh + pre-existing per-target JSONs (multi-restart
    equivalence).
  - Each run produces exactly one quarantined suffix file per *fresh* target under `artifacts/untrusted/`.

  **Dependencies:** Tasks 1–5; `run_gcg`/`GcgConfig` (`src/evasion_tax/attack/gcg.py`); `cuda_available` /
  `gpu_required_message` (`src/evasion_tax/config.py`); the repro `run.json` writer used by the other scripts.

  **Notes:** **Unattended launch** (box): `nohup systemd-inhibit … uv run python scripts/bench_early_stop.py …
  > results/<run>/log.txt 2>&1 & disown`, wrapped in `until uv run … ; do sleep 10; done` for OOM auto-restart —
  the per-target checkpoint makes restarts cheap (`docs/gpu/CSB/plan.md` *Unattended runs*). Validate with a
  **non-registered dry run** first (`--n-targets 1 --n-steps 20 --results-root results/_smoke`) before the
  registered run.

  **Commit:** `feat: early-stop steps-to-success bench driver (checkpoint+resume, quarantine, registered aggregate)`

- [ ] **Task 7: record the realistic cost → re-run `branch_select` → update docs — mac (after the box run)**

  **Files:**
  - Modify: `docs/core/execution-playbook.md` (§6 D8 + §10: add the **measured early-stop** median, the realistic
    `s/target`, the re-derived provisional branch; **hard-F default still stands** until the *adaptive* cost lands)
  - Modify: `docs/gpu/CSB/plan.md` (note the early-stop result under the step-6 / M1 context)

  **What:** Feed `realistic_s_per_target(measured_median, 33.19)` into `branch_select.affordable_matrix` (same
  calendar as the 2026-06-23 Task-5 record) and record the updated **provisional** branch + the censored_fraction
  caveat. If `censored_fraction` is high, state that the worst-case bound remains the honest planning number.

  **Test scenarios:** docs not unit-tested; `branch_select` already covered. Recorded branch stays
  `provisional`, `locked=False`, `default_if_unconfirmed="F"` (the adaptive cost still gates the lock).

  **Dependencies:** Task 6's registered `bench_result.json`.

  **Notes:** This is the early-stop analogue of step-6 Task 5; it **refines** the provisional branch, it does not
  lock it. **M3/H6-A delivered in every branch** regardless.

  **Commit:** `docs: record measured A5000 early-stop cost → refine provisional branch (still hard-F default)`

---

## Build order & where each runs

1. **On the mac (TDD, `/tdd`):** Task 1 (pure predicate) → Task 3 (pure bookkeeping) → Task 4 (loader extraction) →
   Task 5 (`loss_of` chunking — pure chunked==single equivalence) → Task 2 *guarded body* + Task 6 *core/glue* with
   the GPU call mocked. Full `src/evasion_tax` stays type-clean + ruff-clean; suite green.
2. **On the CSB A5000 box (one session, unattended):** Task 2 on-box wiring (`reached` agrees with `run_gcg`) +
   Task 5 **box fit-check** (`run_gcg` at sw=512/eval_batch=32 completes < 24 GiB — the OOM is gone) → Task 6
   **non-registered dry run** (1 target, 20 steps) → Task 6 **registered run** (`nohup` + `systemd-inhibit`,
   per-target checkpoint, auto-restart `until`-loop) → push results.
3. **On the mac:** Task 7 — realistic-cost record + `branch_select` re-run + doc updates.

**Verify gates:** (a) pure predicate matches hand-computed argmax + agrees with `per_sequence_ce≈0`; (b) on-box
`target.reached` agrees with `run_gcg.reached`; (c) **chunked `loss_of` == single within `atol` (all B / chunk
sizes / suffix lengths) AND `run_gcg` at sw=512 fits 24 GiB**; (d) registered `bench_result.json` carries the
steps-to-success distribution (median/IQR/censored_fraction) under the §8 header, with per-target resume proven;
(e) the realistic `s/target` re-feeds `branch_select` and the **provisional** branch + **hard-F default** are written.

---

## References

@src/evasion_tax/attack/gcg.py (`run_gcg(..., reached_fn=…)`, `GcgResult.n_steps_run/reached` — early-stop already
supported) · @src/evasion_tax/attack/gcg_openvla.py (`OpenVlaGcgTarget`: `_full_ids`, `_labels`, `loss_of`,
`token_gradient`, `per_sequence_ce`) · @scripts/microbench_gcg.py (`_load_frozen_openvla`, `_build_target`,
`steps_to_success` recording — the D8 harness this bench complements) · @src/evasion_tax/eval/branch_select.py
(`affordable_matrix` / `provisional_branch` — Task 7 re-feeds it) · @docs/gpu/CSB/plan.md (*Unattended runs* +
step 6) · @docs/core/execution-playbook.md (§6 D7/D8, §7 M1 "A5000 early-stop steps-to-success bench", §10).
Registered cost input: `results/2026-06-23T13-34-55Z-gcg-microbench/` (s/step = 33.19 s; s/target worst = 16,595 s).
