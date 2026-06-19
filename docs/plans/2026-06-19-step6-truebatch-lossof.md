# CSB Step 6 (follow-up) — True-batch `loss_of` + D8 Re-registration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use `test-driven-development` (`/tdd`) for the **model-free**
> per-sequence-CE reference + the strengthened wiring assertions; the GPU `loss_of` body runs on the **CSB
> A5000 box**, guarded exactly like the existing seam. REQUIRED on execution: `executing-plans`.

**Goal:** Replace the GCG seam's batch-1 **loop** `loss_of` with a single **true-batch** forward, then make the
**true-batch** harness the official **D8** registered measurement (loop kept as an ablation baseline) — so D8
characterises the harness actually used downstream (M1–M4), and **max candidate-batch B @ 24 GB** becomes a real
VRAM measurement instead of the degenerate cap.

**Architecture:** `OpenVlaGcgTarget.loss_of([B,L]) → [B]` currently runs B sequential batch-1 forwards
(`[self._loss_single(row) for row in cands]`). Candidates are **fixed-length** (`prefix ⊕ suffix(len L) ⊕ tail ⊕
target`, identical length & label mask across candidates), so true-batching is a clean `[B, seq]` stack with **no
padding**. The only real work is per-sequence CE: OpenVLA's built-in `out.loss` **mean-reduces over the batch**,
so the batched path must compute per-row CE from `out.logits` itself, replicating HF's causal-LM shift + ignore
+ per-sequence mean. The pure CE reduction is unit-tested off-GPU (numpy reference); the GPU torch path is
cross-checked on-box against HF's own `out.loss` via `batched_matches_single`.

**Tech Stack:** Python 3.10; NumPy (pure reference + tests); torch 2.2.0+cu121 / transformers 4.40.1 / bf16
OpenVLA-7B behind the GPU guard (identical to `smoke_openvla_gradient.py` / the existing seam).

---

## Provenance — this SUPERSEDES the "defer true-batch" decision in the parent step-6 plan

@docs/plans/2026-06-19-step6-gcg-microbench.md is the **frozen record of step 6 as executed on the loop MVP**
(all its tasks `[x]`). That plan **deferred** the true-batch CE (D6-2 / Task 2 *Notes*: "a tighter single-forward
batched CE is a box optimisation gated by the D6-3 wiring check"; D6-4 measured `s/target` on "our harness"). This
follow-up plan **pulls that deferred optimisation into scope before the D8 registered run** and supersedes that
single decision only — every other parent decision (D6-1, D6-3 single-position-swap MVP, D6-5 provisional/hard-F
branch, D6-6 ethics/quarantine, D6-9 faithfulness gate, D6-10 clean-process protocol) **stands unchanged**. A
one-line back-pointer is added to the parent plan so the two never contradict.

**Third-party review (Codex, 2026-06-19):** independently agreed "implement true-batch *before* registering D8",
with four corrections folded in as decisions below — (a) **stop framing max-B as branch-critical**; the
justification is "D8 must measure the downstream harness" and "`max-B = cap` is not a hardware ceiling and must
not be recorded as one"; (b) **register BOTH** loop and true-batch, true-batch as the official sizing
measurement, loop as an ablation showing the engineering tax of naive evaluation; (c) the `batched_matches_single`
gate is **necessary-not-sufficient** → verify across multiple B, suffix lengths, mixed-quality candidates, and
**rank ordering** (not only absolute loss), with determinism explicitly pinned; (d) the target-position-only
logits optimisation is **out of scope** (avoid turning a bounded fix into an optimisation project).

---

## Decisions (pre-registered)

- **DB-1 — True-batch `loss_of` is in scope BEFORE the D8 registered run.** D8 is a *measurement gate*; it must
  characterise the harness that runs M1–M4 (this self-built GCG is the downstream workhorse, incl. the H6-D
  adaptive arm). Registering a number from a loop we will replace is misleading. (Supersedes parent D6-2/D6-4
  *defer*.)
- **DB-2 — Register BOTH numbers; true-batch is the official D8 sizing measurement.** The loop number (already
  measured non-registered: 17.47 s/target @ n_steps=5/W=32/1-target, `results/_smoke/2026-06-19T15-22-54Z-gcg-microbench`)
  is logged alongside as a **baseline/ablation** ("engineering tax of naive per-candidate evaluation"), **not** the
  sizing basis. The measured speedup `k = loop / true-batch` is a methods data point.
- **DB-3 — max-B is hardware characterisation, NOT branch-critical (framing fix).** The compute branch stays
  **provisional + hard-F default** (parent D6-5) regardless of these numbers. max-B is recorded because
  `max-B = cap` under the loop is a *non-measurement* (looping never OOMs from batching), and a real VRAM ceiling
  is a legitimate §8 deliverable — but write-up language must not present it as deciding the branch.
- **DB-4 — Verification-heavy equivalence (Codex (c)).** Beyond `batched_matches_single` (batched `[B]` == per-
  candidate single, `atol=1e-3`): a pure numpy CE reference unit-tested off-GPU; on-box checks across **multiple B**
  (e.g. 1, 2, 8, 32), **multiple suffix lengths**, **mixed-quality candidates** (init + gradient-recommended +
  random), and **rank-order agreement** (batched argmin == single argmin). Determinism pinned: `model.eval()`,
  no dropout, fixed seed, fixed bf16 dtype, documented tolerance, repeated-run check on the A5000.
- **DB-5 — Out of scope (Codex (d), YAGNI).** Target-position-only logit computation (to lift the max-B ceiling),
  multi-position swap / attack buffer / mellowmax, true-batch of the *gradient* path. `token_gradient` stays the
  single hooked backward (parent step 5.5); only the **candidate-evaluation** `loss_of` is batched. The full-vocab
  logits `[B, seq, V]` are accepted as the VRAM driver → that is exactly what makes the max-B sweep real.

---

## Shared contract (the per-sequence CE both the test and the GPU path depend on)

```python
# src/evasion_tax/attack/gcg_openvla.py  (pure, numpy — unit-tested off-GPU)
def per_sequence_ce(
    logits: np.ndarray,        # [B, T, V] next-token logits
    labels: np.ndarray,        # [B, T] target ids, ignore_index where masked
    *, ignore_index: int = -100,
) -> np.ndarray:               # [B] mean CE over non-ignored shifted positions
    """Causal-LM shift: predict labels[:, 1:] from logits[:, :-1]; per-row mean CE over
    non-ignored positions. Mirrors HF CrossEntropyLoss(ignore_index=-100, reduction='mean')
    applied per sequence, so the batched path equals the per-candidate `out.loss` path
    (within atol). The GPU `loss_of` runs the torch equivalent on `out.logits`."""
```

---

## Tasks

- [ ] **Task 1: true-batch `loss_of` (pure CE reference TDD'd; GPU body guarded) — mac + box**

  **Files:**
  - Modify: `src/evasion_tax/attack/gcg_openvla.py` (add `per_sequence_ce`; rewrite `loss_of` to one batched forward)
  - Test: `tests/evasion_tax/attack/test_gcg_openvla.py` (add `per_sequence_ce` cases — pure)

  **What:** Add the pure `per_sequence_ce` (above). Rewrite `loss_of` to: stack candidates → `[B, seq]` `input_ids`
  (via `_full_ids` per row, all equal length), build `[B, seq]` `labels` (identical mask + target rows),
  **expand** `self._pixel_values` `[1,…] → [B,…]` (a non-aliasing expand/repeat — must not share-mutate), one
  `torch.no_grad()` forward → `logits [B, seq, V]`, then per-sequence CE → `[B]`. Reset/read CUDA peak so Task 4
  reads peak VRAM. `_loss_single` stays as the batch-1 reference (used by the gates).

  **Interface:**
  - `per_sequence_ce(logits, labels, *, ignore_index=-100) -> np.ndarray` (the contract above).
  - `loss_of(candidate_suffixes: np.ndarray) -> np.ndarray` — `[B, L]` → `[B]`, **one** batched forward, no grad.

  **Test scenarios (pure, off-GPU):** `per_sequence_ce` matches a hand-computed `[B]` on tiny fixed `logits`/
  `labels`; rows that are fully `ignore_index` are handled (documented behaviour, no NaN-leak); the causal shift
  is correct (predicting position t from t-1); a single-row input equals the scalar HF-style mean. Module still
  imports **no torch at top** (existing invariant test stays green).

  **Dependencies:** numpy; torch/transformers inside the guarded method only.

  **Notes:** No padding (fixed-length candidates). Do **not** use `out.loss` for the batched path (it averages
  across the batch). Keep `token_gradient` unchanged (DB-5).

  **Commit:** `feat: true-batch loss_of for the GCG seam (per-seq CE from logits; pure ref TDD'd, GPU guarded)`

- [ ] **Task 2: strengthen the on-box equivalence + determinism gate (DB-4) — box**

  **Files:**
  - Modify: `scripts/smoke_gcg_tiny.py` (extend the wiring block; record the new checks)

  **What:** Make `batched_matches_single` meaningful now that `loss_of` truly batches, and add the Codex (c)
  checks: equivalence across **multiple B** (1, 2, 8, 32) and ≥2 **suffix lengths**, on a **mixed-quality**
  candidate set (init + gradient-recommended top-k + random), plus **rank-order** agreement (batched argmin ==
  single argmin) and a **repeated-run** determinism check (same seed ⇒ identical `[B]` within tolerance). These are
  pass/fail wiring gates (D6-3 spirit). Record all in the `results/_smoke/` smoke record.

  **Interface:** reuse `target.loss_of`, `target._loss_single`, `target.batched_matches_single`; add small inline
  helpers in the script (no new public API).

  **Test scenarios (verify gate = the run):** batched `[B]` == per-candidate single (`atol=1e-3`) for every B and
  suffix length tried; batched argmin index == single argmin index (rank order); two runs at the same seed are
  bit-for-bit (or within tolerance) identical; off-GPU ⇒ guard exits 2. Peak VRAM < 24 GiB still holds.

  **Dependencies:** Task 1; existing seam.

  **Notes:** This is the "necessary-not-sufficient" hardening Codex flagged — absolute-loss equality **and** rank
  order **and** determinism. Pin `model.eval()`, bf16, seed (already set), document the tolerance.

  **Commit:** `test: harden GCG batched-vs-single gate (multi-B/len, mixed-quality, rank-order, determinism)`

- [ ] **Task 3: tiny-run revalidation + speedup calibration — box (non-registered)**

  **Files:** none (re-run existing scripts).

  **What:** Re-run `scripts/smoke_gcg_tiny.py` (now exercising the batched path + Task-2 gate) → confirm the D6-9
  faithfulness gate + the strengthened wiring gate pass. Then re-run the calibration micro-bench
  (`--results-root results/_smoke --n-steps 5 --search-width 32 --n-targets 1 --batch-cap 2`) and compare
  `s/target` to the logged loop baseline (17.47 s) → record the measured **speedup `k`**.

  **Test scenarios (verify gate = the run):** tiny run prints `PASS`; calibration prints a finite `s/target` and a
  plausible `k = loop/batch` ≥ 1; peak VRAM < 24 GiB.

  **Dependencies:** Tasks 1–2.

  **Notes:** Non-registered (`results/_smoke/`). If `batched_matches_single` fails here, **stop and fix the seam**
  before any registered run (a wrong batched CE must not produce a registered number).

  **Commit:** (none — smoke artifacts are pushed for cross-box evidence; no source change.)

- [ ] **Task 4: registered D8 micro-bench — true-batch official, loop ablation, real max-B — box (REGISTERED)**

  **Files:**
  - Modify (only if needed): `scripts/microbench_gcg.py` — record **both** the true-batch result (official) and the
    loop baseline, and a `max_batch_note` clarifying max-B is a VRAM ceiling, **not** branch-critical (DB-3). The
    existing subprocess clean-process sweep (parent D6-10) now finds a **real** OOM boundary because batched
    `logits [B, seq, V]` scale with B.

  **What:** On an **exclusive/quiet** A5000, time `s/target` (true-batch) over a few targets at one pinned config;
  sweep B in fresh subprocesses (D6-10) for the **real** max candidate-batch @ 24 GB; capture peak VRAM; record the
  loop `s/target` (from the Task-3 calibration or a short loop pass) as the baseline + the speedup `k`. Write a
  **registered** result to write-once `results/` with the full §8 header (env, pinned seed, **which A5000**, bf16,
  CUDA/driver/torch, git commit). Fail the run if not exclusive or `s/target` not reproducible across repeats.

  **Interface (pure, already TDD'd in parent):** `summarise_timings`, `max_batch_that_fits`,
  `build_microbench_record`, `assert_registered_run_valid` — extend `build_microbench_record` to carry
  `s_per_target_loop` + `speedup_k` + `max_batch_note`.

  **Test scenarios:** existing parent micro-bench unit tests stay green; the new record fields are present;
  off-GPU ⇒ guard exits 2; the sweep returns a B **strictly below** `--batch-cap` when the cap exceeds the VRAM
  ceiling (real measurement, not the degenerate cap).

  **Dependencies:** Tasks 1–3; parent `microbench_gcg.py` aggregation helpers.

  **Notes:** Pinned config decided from the Task-3 `k` + the exclusive-window length (pre-registered before the
  run, parent D6-3). Keep the loop number labelled **ablation/baseline**, true-batch labelled **official**.

  **Commit:** `feat: D8 registered micro-bench — true-batch official + loop ablation + real max-B @ 24GB`

- [ ] **Task 5: branch computation + doc updates (with the numbers) — mac**

  **Files:**
  - Modify: `docs/plans/2026-06-19-step6-gcg-microbench.md` (add the back-pointer: D6-2/Task-2 true-batch
    **superseded** by this plan; link here)
  - Modify: `docs/core/execution-playbook.md` (§2/§6 D8, §10 decision log — record true-batch official `s/target`,
    loop baseline + `k`, real max-B; **provisional + hard-F branch unchanged**; max-B framed as HW characterisation
    per DB-3)
  - Modify: `docs/gpu/CSB/plan.md` (tick step 6; add a step-6 how-to noting true-batch `loss_of`, the equivalence
    gate, and the real max-B)
  - (Task 5 of the parent plan's `src/evasion_tax/eval/branch_select.py` is reused as-is — feed it the
    **true-batch** `s/target`.)

  **What:** Run `branch_select` arithmetic on the **true-batch** `s/target` for the provisional affordable matrix +
  hard-F default; write the decision + the loop/true-batch/`k` numbers + the lock condition into the docs.

  **Test scenarios:** docs not unit-tested; `branch_select` already covered by parent tests; the recorded branch
  is `provisional`, `locked=False`, `default_if_unconfirmed="F"`.

  **Dependencies:** Task 4's registered `s/target`.

  **Notes:** Write-up language: max-B = hardware ceiling, **not** branch decider (DB-3). M3/H6-A floor unchanged.

  **Commit:** `docs: register D8 (true-batch official + loop ablation), tick CSB step 6, provisional branch`

---

## Build order & where each runs

1. **On the mac (TDD, `/tdd`):** Task 1 pure `per_sequence_ce` + tests; rewrite `loss_of` (GPU body guarded);
   Task 4 record-field extension + parent unit tests stay green; ruff/type clean.
2. **On the CSB A5000 box (one session):** Task 2 strengthened gate → Task 3 tiny-run revalidation + speedup
   calibration → Task 4 registered micro-bench (exclusive/quiet, clean-process) → push results.
3. **On the mac:** Task 5 branch arithmetic + doc updates with the numbers; tick step 6.

**Verify gates:** (a) pure `per_sequence_ce` matches hand-computed CE; (b) on-box batched == single across
multiple B / suffix-lengths / mixed-quality + rank-order + determinism (DB-4); (c) registered D8 records
true-batch (official) + loop (ablation) + `k` + a **real** max-B (< cap) under the clean-process protocol with a
§8 header; (d) provisional branch + hard-F default written, max-B framed as HW characterisation (DB-3).

---

## References

@docs/plans/2026-06-19-step6-gcg-microbench.md (parent — frozen loop-MVP record; D6-1/3/5/6/9/10 stand) ·
@src/evasion_tax/attack/gcg_openvla.py (`loss_of`, `_loss_single`, `batched_matches_single`, the embedding hook) ·
@scripts/smoke_gcg_tiny.py (wiring/faithfulness gate driver) · @scripts/microbench_gcg.py (D8 aggregation +
clean-process sweep) · @src/evasion_tax/eval/branch_select.py (provisional branch arithmetic, hard-F default) ·
@docs/core/execution-playbook.md (§2 compute branches, §6 D8, §8 run-log, §10 decision log) ·
@docs/gpu/CSB/plan.md (step 6 ladder). Codex third-party review verdict (2026-06-19): implement-now, register
both, verification-heavy, max-B not branch-critical, no logits-slicing optimisation.
