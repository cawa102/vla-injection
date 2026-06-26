# GCG Progress + Checkpoint Callback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task, and `test-driven-development` (RED→GREEN→REFACTOR) for each task.

**Goal:** Make a long `run_gcg` search **observable** (periodic progress in the log) and **partially recoverable** (the best suffix persisted to a quarantined checkpoint) via one optional `on_step` callback — with core GCG behaviour byte-identical when the callback is absent.

**Architecture:** Two additive changes. (1) `run_gcg` gains a keyword-only `on_step` callback, called once per completed step and **exception-isolated** so a logging/disk failure can never abort a multi-hour search. (2) The surrogate driver builds a throttled callback that prints a progress line every N steps and atomically (over)writes a best-suffix checkpoint every M steps. No change to the GCG search math, the final write-once artifact, or the `results/` pointer. The checkpoint carries a suffix (attack payload) so it lives under gitignored `artifacts/untrusted/`; the progress line carries no suffix and goes to stdout/log.

**Tech Stack:** Python 3.10, existing `evasion_tax.attack.gcg`, `scripts/run_surrogate_gcg.py`, pytest.

---

## Context — current state (read first; zero-context handoff)

This plan is the durable fix for a failure observed on the CSB A5000 box (2026-06-26): an
unattended overnight pilot died **silently** during the bf16 GCG search (idle-suspend /
session loss), and because `run_gcg` emits nothing during the search and never checkpoints,
we could neither see how far it got nor recover any of it; the remaining arms never ran.

- **`run_gcg`** lives at `src/evasion_tax/attack/gcg.py:166`. It is a clean in-memory loop:
  `for _ in range(cfg.n_steps)` → `token_gradient` → `top_k_candidates` → `sample_candidates`
  → `select_best` → keep-best bookkeeping (`history.append(best_loss)`) → optional
  `reached_fn` early-stop. It already takes a keyword-only `reached_fn`. It returns a
  `GcgResult(best_suffix_ids, best_loss, loss_history, ...)`. **No progress output, no
  checkpoint.**
- **The driver** `scripts/run_surrogate_gcg.py` calls
  `run_gcg(target, gcg, reached_fn=target.reached if args.early_stop else None)` inside a
  try/except (around `:250`), then writes the final write-once artifact under
  `artifacts/untrusted/<run_id>/<artifact_id>.json` and a metrics pointer to
  `results/<run_id>/` via `_results_pointer`.
- **Quarantine rule (load-bearing).** Suffix token ids / text are the attack payload and live
  only under gitignored `artifacts/untrusted/`; metrics (no suffix) go to tracked `results/`.
  The new checkpoint contains the suffix → it MUST go under `artifacts/untrusted/<run_id>/`,
  never `results/`.
- **Files in scope already exist and pass their current tests:**
  `tests/evasion_tax/attack/test_gcg.py`, `tests/scripts/test_surrogate_scripts.py`. The
  surrogate driver currently has no `on_step` wiring and no checkpoint code.

## Scope decision — what is intentionally NOT in this change

- **No `--resume` / restart-from-checkpoint.** Bit-exact GCG resume needs the candidate-sampler
  RNG state restored (`np.random.default_rng(cfg.seed)` is advanced once per step), which is a
  larger, separate change. This plan only lays the **foothold** (a checkpoint that records the
  best suffix + step). Resume is explicitly deferred (YAGNI).
- **No CLI flags for the intervals.** Use module constants; a reader can change them in one
  place. Add flags later only if a run actually needs per-invocation tuning.
- **No change to the final artifact or `results/` pointer schema.** The checkpoint is an
  auxiliary, mutable sidecar; the end-of-run artifact remains the source of truth.

---

## Shared contract (Task 1 + Task 2 depend on it)

**Callback signature** — called once after each completed step, `step` is 1-based:

```python
# src/evasion_tax/attack/gcg.py
OnStep = Callable[[int, np.ndarray, float], None]
# on_step(step, best_suffix_ids, best_loss)
#   step           : 1..steps_run (the step just completed)
#   best_suffix_ids: the current incumbent suffix (a COPY — callback must not mutate the search)
#   best_loss      : the incumbent loss (non-increasing across steps)
# run_gcg wraps each call in try/except: an exception is logged to stderr and the search
# CONTINUES. The callback is auxiliary; it can never abort the search.
```

**Checkpoint JSON** — mutable, overwritten each interval, under
`artifacts/untrusted/<run_id>/checkpoint.json` (gitignored, because `best_suffix_ids` is the
payload):

```python
{
    "step": int,                 # last checkpointed step
    "n_steps": int,              # total GCG budget
    "best_loss": float,
    "best_suffix_ids": [int],    # the attack payload — why this file is quarantined
    "precision": str,            # surrogate precision arm (bf16 / int8 / nf4_4bit)
    "updated_utc": str,          # ISO timestamp of this checkpoint write
}
```

---

## Tasks

- [ ] **Task 1: `on_step` callback in `run_gcg`**

**Files:**
- Modify: `src/evasion_tax/attack/gcg.py` (`run_gcg`, ~`:166-220`)
- Test: `tests/evasion_tax/attack/test_gcg.py`

**What:** Add a keyword-only `on_step: OnStep | None = None` parameter to `run_gcg`. After the
per-step `history.append(best_loss)` (and after the `reached_fn` early-stop check, so a reached
final step is still reported), if `on_step is not None` call
`on_step(steps_run, suffix.copy(), best_loss)` inside a `try/except`. On exception, print a
single-line `stderr` warning (`[run_gcg] on_step callback failed: <Type>: <msg>`) and continue
the loop. Default `None` → no call, behaviour identical to today. Add the `OnStep` type alias
near the top of the module.

**Interface:**
- `run_gcg(fn, cfg, *, reached_fn=None, on_step: OnStep | None = None) -> GcgResult` — unchanged
  except the new keyword-only trailing parameter.
- `OnStep = Callable[[int, np.ndarray, float], None]` — module-level type alias.

**Test scenarios:**
- A recording callback is invoked exactly `cfg.n_steps` times, with `step` values `1..n_steps`
  in order, and the `best_loss` it receives is non-increasing (matches `loss_history[1:]`).
- With a `reached_fn` that triggers at step `k`, the callback is invoked exactly `k` times
  (including the reaching step).
- A callback that raises every call does NOT abort the search: `run_gcg` still returns a valid
  `GcgResult` and still ran all `cfg.n_steps` steps.
- `on_step=None` (and a no-op callback) yields a `GcgResult` equal to the current behaviour
  (best_suffix_ids / best_loss / loss_history unchanged) — the existing tests already exercise
  the `None` path and must stay green.

**Dependencies:** none new (`numpy`, `typing.Callable` already imported in the module).

**Notes:** Pass `suffix.copy()` so a callback cannot mutate the live search array. The call sits
on the hot loop but its cost is negligible against a GPU `token_gradient` + `search_width`
forward; throttling is the *callback's* job (Task 2), not `run_gcg`'s.

**Commit:** `feat(gcg): optional exception-isolated on_step progress callback`

---

- [ ] **Task 2: progress-log + checkpoint callback in the surrogate driver**

**Files:**
- Modify: `scripts/run_surrogate_gcg.py`
- Test: `tests/scripts/test_surrogate_scripts.py`

**What:** Add module constants `_PROGRESS_EVERY = 25` and `_CHECKPOINT_EVERY = 50`. Add small
pure helpers plus a callback factory, then pass `on_step=` into the `run_gcg(...)` call in the
GPU path. The checkpoint path is `artifact_dir / "checkpoint.json"` (the existing
`artifact_dir = QUARANTINE_ROOT / run_id`).

**Interface:**
- `_progress_line(step: int, n_steps: int, best_loss: float, elapsed_s: float) -> str` — pure;
  the heartbeat line (e.g. `"[gcg] step 25/500 best_loss=6.357 elapsed=1113s"`). No suffix.
- `_checkpoint_dict(step: int, n_steps: int, best_suffix: np.ndarray, best_loss: float,
  precision: str, updated_utc: str) -> dict` — pure; returns the Shared-contract dict (lists,
  JSON-safe).
- `_atomic_write_json(path: Path, obj: dict) -> None` — write to `path.with_suffix(".tmp")` then
  `os.replace` onto `path`; OVERWRITE-safe (no `FileExistsError`, unlike the write-once
  `write_json_record`).
- `_make_on_step(*, n_steps: int, precision: str, checkpoint_path: Path, t0: float,
  out=sys.stdout) -> OnStep` — returns a callback that, given `(step, best_suffix, best_loss)`,
  prints `_progress_line(...)` (flushed) every `_PROGRESS_EVERY` steps and writes
  `_atomic_write_json(checkpoint_path, _checkpoint_dict(...))` every `_CHECKPOINT_EVERY` steps.
  Uses `time.perf_counter() - t0` for elapsed.
- Wire: in the GPU path, build `on_step = _make_on_step(n_steps=gcg.n_steps,
  precision=fields["precision"], checkpoint_path=artifact_dir / "checkpoint.json",
  t0=t0)` and pass `on_step=on_step` to `run_gcg(...)`.

**Test scenarios:**
- `_progress_line` contains the step, the budget, and the loss (format check).
- `_checkpoint_dict` carries `best_suffix_ids`, `step`, `best_loss`, `precision`, `n_steps`,
  `updated_utc` (this is the payload-bearing record — assert it round-trips through `json`).
- `_atomic_write_json` writes a file, and a second call with different content OVERWRITES it
  (no error) and round-trips — the mutable-checkpoint contract.
- `_make_on_step` driven over steps `1..100` against an in-memory `out` buffer and a tmp
  `checkpoint_path`: a progress line is emitted at steps `{25,50,75,100}` and the checkpoint
  file is written at steps `{50,100}`, and after step 100 the checkpoint holds the suffix /
  loss passed at step 100.
- The checkpoint path used by the driver is under `artifacts/untrusted/` (quarantine guard —
  reuse the existing `is_under_quarantine` / `require_quarantined` helper from
  `surrogate_artifacts`).
- Existing dry-run / arg-parsing tests stay green (the callback only exists on the GPU path).

**Dependencies:** `OnStep` (Task 1), `time`, `os`, existing
`QUARANTINE_ROOT` / `require_quarantined` from `evasion_tax.attack.surrogate_artifacts`,
`utc_now_iso`.

**Notes:** The checkpoint is MUTABLE — overwrite the latest best each interval; do NOT use
`write_json_record` (it refuses overwrite). The final artifact + `_results_pointer` are
unchanged; the checkpoint is auxiliary. On a clean successful run the checkpoint file may be
left in place (gitignored, harmless) — do not add cleanup logic. The actual on-box behaviour
(a real long run printing `[gcg] step N/500 ...` and updating `checkpoint.json`) is validated
on the CSB box, not by a GPU unit test.

**Commit:** `feat(surrogate): log GCG progress + checkpoint best suffix each interval`

---

## Success criteria

- `pytest tests/evasion_tax/attack/test_gcg.py tests/scripts/test_surrogate_scripts.py` green;
  full suite green.
- With `on_step=None`, `run_gcg` returns results identical to before (existing `test_gcg.py`
  cases unchanged and passing).
- A callback that raises never aborts the search (Task 1 robustness test green).
- On the box, a long surrogate run prints `[gcg] step N/500 best_loss=...` to its log every 25
  steps and refreshes `artifacts/untrusted/<run_id>/checkpoint.json` every 50 steps with the
  current best suffix; the checkpoint stays under quarantine (never in `results/`).
- No change to the final artifact schema or the `results/` pointer.

## Execution

Implementation runs in a **separate session**. Open it in the repo, then drive this plan with
`executing-plans` + `test-driven-development`. Order: **Task 1 → Task 2** (Task 2 imports the
`OnStep` alias and depends on the callback contract). Both tasks are fully unit-testable
off-GPU — the only GPU-only part is the final wiring's real long-run behaviour, validated on the
CSB box during the next pilot, not by a unit test.
