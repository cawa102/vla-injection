# Surrogate-GCG Failure Attribution + Attn-Impl Fallback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task, and `test-driven-development` (RED→GREEN→REFACTOR) for each task.

**Goal:** Make the quantized-surrogate GCG pipeline record *why* an arm was censored (real gradient vs dead/weak gradient vs hard target) and survive a flash-attn load failure without crashing or silently switching a controlled variable.

**Architecture:** Two surgical changes to the existing surrogate harness. (1) Add a recorded gradient-health diagnostic to the suffix artifact, computed once before the GCG search (record, never gate). (2) Add a single flash-attn→sdpa load fallback helper used by both GPU drivers, with the actual attention impl recorded so the deviation is auditable. No new architecture; no model-side change.

**Tech Stack:** Python 3.10, existing `evasion_tax` dataclass artifacts, the OpenVLA/GCG seam (`OpenVlaGcgTarget`), pytest.

---

## Context — current state (read first; zero-context handoff)

This plan follows on-box verification (CSB A5000 `ecs3-0202`, 2026-06-24/25) of the
quantized-surrogate transfer plan
(`docs/surrogate/plan/2026-06-24-quantized-surrogate-gcg-transfer-plan.md`). Verified facts the
implementer can rely on:

- **Concerns 1 & 2 resolved on the box.** bitsandbytes loads and the int8/nf4 GCG *backward*
  produces a real, finite, non-zero input-embedding gradient under CUDA 13.2 driver / cu121 torch.
  Results (`scripts/smoke_quantized_backward.py`, default `--attn-impl sdpa`):
  bf16 `recommended_mean_delta -1.20` (control PASS); int8 `-1.45`, `grad_absmax 1.135`, 10.3 GiB
  (strong, faithful); **nf4 `-0.41 ≈ random -0.44`, `grad_absmax 1.27`, 6.9 GiB — gradient real
  but its *ranking* is weak (consistent with plan H-S2, NOT broken).** This weak-but-real nf4 case
  is exactly why Task 1+3 must **record, not gate** on faithfulness.
- **Env pinned** (`configs/env/requirements-gpu.txt`): `accelerate==0.30.1`, `bitsandbytes==0.43.1`.
  Install GPU-stack adds with `uv pip install --no-deps`. The OpenVLA env is the uv venv
  `~/vla-injection/.venv`; bare `pip` hits conda base.
- **Files in scope already exist and pass their current tests.** Existing tests:
  `tests/evasion_tax/attack/test_surrogate_artifacts.py`,
  `tests/evasion_tax/eval/test_surrogate_transfer.py`, `tests/scripts/test_surrogate_scripts.py`.

## Scope decision — fix 5 is intentionally NOT implemented

A prior review proposed changing `censored = not hit` (in `evaluate_surrogate_transfer.py:196` and
`run_surrogate_gcg.py:245`) because it "conflates measured non-transfer with measurement failure."
**This was a misreading and must not be implemented.** The transfer plan defines censored as
survival-style *right-censoring* — "Censored targets remain in the denominator for ASR and are
reported separately in the **steps-to-success** summary" — i.e. censored = target not reached. So
`censored = not hit` is correct, and `eval/surrogate_transfer.py:29-30` already counts `censored`
and `failed` (=`failure_reason is not None`) separately, so "measured miss" vs "could-not-measure"
is already distinguishable. **Do not touch `censored` semantics.**

Optional follow-up (NOT a task; author decides later): report an additional ASR variant that
excludes `failed` arms from the denominator. Default = do nothing; the `failed` count is already
emitted so a reader can recompute. Leave out unless the author asks.

---

## Shared contract — gradient-health diagnostic (Task 1 + Task 3 depend on it)

Stored on the suffix artifact as one nullable dict. `None` only if the run never reached the
diagnostic (kept for forward-compat); on a diagnostic exception store `{"error": "<Type>: <msg>"}`.

```python
# value of SurrogateSuffixArtifact.surrogate_gradient_health (dict | None)
{
    "grad_absmax": float,            # max |token-gradient| at the init suffix
    "grad_nonzero": bool,            # grad_absmax > 0
    "grad_finite": bool,             # np.isfinite(grad).all()
    "recommended_mean_delta": float, # gradient_agrees_with_swaps report
    "random_mean_delta": float,
    "faithfulness_passed": bool,     # report.passed — DIAGNOSTIC ONLY, never a gate
    "gate_samples": int,
}
# or {"error": "RuntimeError: ..."} if the diagnostic itself raised.
```

Downstream attribution of a censored arm: `grad_nonzero False` → dead gradient (concern 2 realised);
`faithfulness_passed False` with small deltas → weak gradient (nf4-like, H-S2); healthy gradient +
`surrogate_target_hit False` → genuinely hard target.

---

## Tasks

- [x] **Task 1: Add gradient-health field to the suffix-artifact schema**

**Files:**
- Modify: `src/evasion_tax/attack/surrogate_artifacts.py`
- Test: `tests/evasion_tax/attack/test_surrogate_artifacts.py`

**What:** Add `surrogate_gradient_health: dict[str, Any] | None` to `SurrogateSuffixArtifact`
(place beside the other `surrogate_*` fields, before `failure_reason`). Bump
`SCHEMA_VERSION = 1` → `2` (shared constant; `TransferEvalRecord` rides the same bump — it is
unchanged structurally, just versioned to 2).

**Interface:**
- New dataclass field `surrogate_gradient_health: dict[str, Any] | None` (default not allowed —
  frozen dataclass, keep positional consistency; constructors pass it explicitly).
- `__post_init__`: if `surrogate_gradient_health is not None` and not a `dict`, raise `ValueError`.
  Do **not** validate inner keys (kept model-free / forward-compatible).
- `to_dict`/`from_dict`: dict field passes through unchanged (no tuple coercion needed).

**Test scenarios:**
- Round-trips with `surrogate_gradient_health` set to a populated dict (write→read equal).
- Round-trips with `surrogate_gradient_health=None`.
- `schema_version=2` accepted; `schema_version=1` rejected with the existing error.
- Non-dict gradient-health (e.g. a list) raises `ValueError`.

**Dependencies:** none new.

**Notes:** No real artifacts exist yet, so the bump is clean. Every test/fixture that builds a
`SurrogateSuffixArtifact` or `TransferEvalRecord` with `schema_version=1` must move to `2`
(`test_surrogate_artifacts.py`, `test_surrogate_transfer.py`, `test_surrogate_scripts.py`).

**Commit:** `feat(artifacts): record surrogate gradient-health, bump schema to v2`

---

- [x] **Task 2: Flash-attn→sdpa load fallback helper**

**Files:**
- Modify: `src/evasion_tax/attack/openvla_loader.py`
- Test: `tests/evasion_tax/attack/test_openvla_loader.py`

**What:** Add a thin wrapper around `load_openvla_policy` that retries with `sdpa` when the requested
attention backend fails to load, and a tiny pure predicate for the failure check so the branch is
unit-testable off-GPU.

**Interface:**
- `is_flash_attn_load_error(exc: BaseException) -> bool` — True if `exc` is an `ImportError` or its
  message mentions `flash` (case-insensitive). Pure, no torch.
- `load_openvla_with_attn_fallback(torch_mod, model_id, device, attn_impl, *, precision="bf16")
  -> tuple[Any, Any, OpenVlaLoadRecord]` — calls `load_openvla_policy`; on a flash-attn load error
  when `attn_impl != "sdpa"`, prints a stderr warning and retries once with `"sdpa"`. The returned
  `OpenVlaLoadRecord.attn_implementation` reflects the impl actually used. Non-flash errors and a
  failing `sdpa` retry propagate (no second retry, no swallow).

**Test scenarios:**
- `is_flash_attn_load_error`: ImportError → True; ValueError mentioning "flash_attn" → True; an
  unrelated RuntimeError → False.
- Fallback wrapper (inject a fake loader via monkeypatch): requested impl succeeds → returns its
  record unchanged, loader called once. Requested `flash_attention_2` raises ImportError → loader
  called twice, second call with `"sdpa"`, returned record shows `attn_implementation == "sdpa"`.
  Requested `sdpa` raises → propagates, loader called once. Non-flash error → propagates, called once.

**Dependencies:** `load_openvla_policy`, `OpenVlaLoadRecord` (same module).

**Notes:** Keep GPU-guarded — torch/transformers stay imported inside `load_openvla_policy`, not at
module top. The wrapper itself imports nothing heavy. Matches the transfer plan's "controlled
variable fixed; deviation recorded" rule.

**Commit:** `feat(loader): flash-attn→sdpa load fallback with recorded deviation`

---

- [x] **Task 3: Wire gradient-health + fallback into the surrogate driver**

**Files:**
- Modify: `scripts/run_surrogate_gcg.py`
- Test: `tests/scripts/test_surrogate_scripts.py`

**What:** (a) Replace the `load_openvla_policy(...)` call (`:196-202`) with
`load_openvla_with_attn_fallback(...)`. (b) After `build_target` and **before** `run_gcg`, compute
the gradient-health dict (shared contract above) and pass it as
`surrogate_gradient_health=` to `SurrogateSuffixArtifact` (`:262-288`). Add module constant
`_GATE_SAMPLES = 32`.

**Interface (diagnostic computation, GPU path):**
- `grad = target.token_gradient(target.init_suffix_ids())` → `grad_absmax`, `grad_nonzero`,
  `grad_finite` via numpy.
- `report = target.gradient_agrees_with_swaps(n_samples=_GATE_SAMPLES,
  rng=np.random.default_rng(fields["seed"]))` → the four report fields.
- Wrap the whole diagnostic in try/except; on exception store `{"error": "<Type>: <msg>"}` and
  continue (record, never gate, never abort — nf4's weak gradient must still run the search).

**Test scenarios:**
- `--dry-run` output unchanged (no GPU, no gradient-health key in the dry-run record).
- Arg parsing unchanged; `_GATE_SAMPLES` present.
- (GPU-only, verified on box, not unit-tested) the written artifact includes a populated
  `surrogate_gradient_health` matching the smoke numbers for the chosen precision.

**Dependencies:** `load_openvla_with_attn_fallback` (Task 2), schema field (Task 1).

**Notes:** Diagnostic adds one `token_gradient` + ~`2*_GATE_SAMPLES` loss evals before the 500-step
search — a small fixed cost. The on-box check is `scripts/smoke_quantized_backward.py` (already
computes the identical diagnostic); do not build a second GPU test.

**Commit:** `feat(surrogate): record gradient-health diagnostic + attn fallback in driver`

---

- [x] **Task 4: Use the fallback in the victim evaluator**

**Files:**
- Modify: `scripts/evaluate_surrogate_transfer.py`
- Test: `tests/scripts/test_surrogate_scripts.py`

**What:** Replace the `load_openvla_policy(...)` call inside the existing try (`:146-152`) with
`load_openvla_with_attn_fallback(...)`. No schema change here — the actual impl is already captured
via `load_record` into `environment["victim_load_record"]`.

**Test scenarios:**
- Arg parsing / compatibility-gate behaviour unchanged.
- (GPU-only, verified on box) a flash-attn miss falls back to sdpa and the transfer record's
  `victim_load_record.attn_implementation == "sdpa"` instead of crashing the load.

**Dependencies:** `load_openvla_with_attn_fallback` (Task 2).

**Notes:** The victim forward needs no backward, so sdpa is a safe fallback. Keep `censored = not hit`
(see Scope decision).

**Commit:** `feat(transfer): use attn fallback in victim evaluator`

---

## Success criteria

- All four tasks committed; `pytest tests/evasion_tax/attack tests/evasion_tax/eval tests/scripts`
  green (schema-v2 fixtures updated).
- A censored surrogate arm is now attributable from its artifact (`surrogate_gradient_health` +
  `failure_reason` + `surrogate_target_hit`).
- A flash-attn load failure on the box degrades to sdpa with the deviation visible in the
  artifact/transfer `load_record`, instead of an uncaught crash (driver) or an un-retried failed
  record (evaluator).
- `censored` semantics unchanged; no metrics-definition change.

## Execution

Implementation runs in a **separate session** (author's choice). Open it in the repo, then drive
this plan with `executing-plans` + `test-driven-development`. Suggested order: Task 1 → Task 2
(independent foundations) → Task 3 (depends on 1+2) → Task 4 (depends on 2). The GPU-only paths in
Tasks 3–4 are validated on the CSB box via `scripts/smoke_quantized_backward.py` and a real tiny GCG
run (transfer plan Run Order step 5), not by unit tests.
