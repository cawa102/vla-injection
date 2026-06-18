# CSB Bring-up Step 5 — Attach the Goal-Action Detector (L2) to the Real Rollout — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use `test-driven-development` (`/tdd`) to implement Task 1 task-by-task.

**Goal:** Feed the **real OpenVLA-driven** rollout produced by step 4 (the `RolloutStep` / `PrivilegedState`
stream logged to `results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke/steps.json`) through the **L2
behavioural detector** end-to-end — `Rollout` → `ConsistencyMetricA` (metric A) → per-step `Score` →
`decide` / `rollout_fires` — and confirm the detector **ingests real OpenVLA actions with no schema mismatch**
(`docs/gpu/CSB/plan.md` step 5).

**Architecture:** The L2 stack already exists and is model-free-tested (`metric/consistency_a.py`,
`detector/{decide,calibrate}.py`). The **one missing seam** is a JSON→`Rollout` deserializer — nothing in the
repo loads a logged `steps.json` back into records (`evaluate.py`/`calibrate.py` consume *already-scored*
rollouts; `demo_metric_separation.py` scores in-memory rollouts it generates). Step 5 adds that seam
(`eval/rollout_io.py`) plus a thin driver script (`scripts/attach_l2_to_rollout.py`) that scores a logged
rollout and writes an L2 report. **Both are pure-numpy / model-free** — the real rollout is already on disk, so
the whole gate runs in the core `.venv` on the mac (no CUDA / OpenVLA / LIBERO at attach time).

**Tech Stack:** Python 3.10, NumPy (metric A is pure NumPy), the project's `evasion_tax` package
(`records`, `metric.consistency_a`, `metric.state` `SyntheticStateAdapter`, `detector.decide`,
`detector.calibrate` — only for the *illustrative* τ note, **not** invoked as calibration here, `repro`,
`config`). No torch, no LIBERO, no GPU.

**Scope boundary (do not exceed):** This is the **wiring de-risk** — verify gate = *the L2 detector ingests the
real OpenVLA-driven rollout end-to-end and emits finite per-step scores + a decision trace, schema matching the
state-adapter / metric side* (`docs/gpu/CSB/plan.md` step 5). It is **NOT**:
- **H1 separation / AUC** (needs an *attacked* class — RoboGCG, step 6 / M1 — and we have one benign episode).
- **M2 calibration** (`calibrate`/`tpr_at_fpr` need a benign **calibration split**; one rollout cannot calibrate).
- a **deployable** detector claim (metric A is the privileged *oracle* / upper bound — `consistency_a.py` header).
- a **schema re-pin** of `SchemaA` radii (frozen pre-attack, invariant #2; step 5 only *observes* benign geometry).

Any τ shown is **illustrative, NOT calibrated**. Score *magnitudes* are a **sanity observation** (calibration
input for M2/M3), never a pass/fail signal — see Decisions D-3 below.

**References:** @docs/gpu/CSB/plan.md (step 5 ladder + Step 4 how-to / gotchas) ·
@docs/core/goal-action-consistency-detector.md (§3 threat model, §5 the metric is the make-or-break) ·
@docs/core/execution-playbook.md (§2 M1/M2 roadmap, H1/H2) · @docs/core/metric-a-annotation-schema.md
(frozen schema; `engagement_radius` `[VERIFY vs LIBERO geometry]` flag) ·
@src/evasion_tax/metric/consistency_a.py (the L2 metric this attaches) ·
@src/evasion_tax/detector/decide.py (the decision path) ·
@scripts/demo_metric_separation.py (the in-memory scoring pattern to mirror, minus the AUC/TPR science) ·
@results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke/steps.json (the real rollout to ingest).

---

## Decisions (pre-registered — design-fork handling: decide + record, don't multiple-choice ask)

- **D-1 — Offline loader is primary; inline live-scoring is deferred.** Step 4 persisted the real rollout to a
  tracked file, so L2-attach is purely offline (KISS + reproducible + reusable for M2 batch scoring). An inline
  per-step hook in `smoke_libero_episode.py` is **out of scope** (YAGNI — that is M2 *online-hold* territory).
  Named in "Box variant (deferred)" so the boundary is explicit, not silently dropped.
- **D-2 — Gate is wiring only.** No AUC/TPR/FPR (no attacked class, no split). The report records per-step
  scores + score summary + a `rollout_fires` decision at an **illustrative τ** purely to exercise `decide()`.
- **D-3 — Score magnitudes are observed, not judged.** `SchemaA.engagement_radius=0.05 m` is a flagged pre-GPU
  placeholder, so benign scores need not be ~0; "benign scores not near zero" is **not** a wiring failure. Step 5
  *reports* real-scene geometry (EE↔object distances, grasp-transition distances, target-region resolvability)
  as calibration input for M2/M3 but does **not** re-pin the frozen radii.
- **D-4 — Loader lives in `eval/rollout_io.py`, not `records.py`.** Keeps the frozen schema (`records.py`)
  untouched; isolates file-I/O + boundary validation in the eval layer (where rollouts are loaded for scoring).
  The pure dict→`Rollout` function is unit-tested separately from the path I/O wrapper.

---

## Task 1: JSON→`Rollout` loader + L2-attach driver (TDD — mac, now)

**Files:**
- Create: `src/evasion_tax/eval/rollout_io.py`
- Create: `scripts/attach_l2_to_rollout.py`
- Test: `tests/evasion_tax/eval/test_rollout_io.py`
- Test: `tests/evasion_tax/test_attach_l2_to_rollout.py`

**Interface (public, model-free — the reusable seam):**
- `rollout_from_log(obj: Mapping) -> Rollout` — pure dict→records. Validates the top-level shape
  (`"steps"` present and a list; non-empty) at the boundary, then builds
  `Rollout(steps=tuple(RolloutStep(**row) for row in obj["steps"]))`. JSON list→tuple coercion for `action`
  and `object_poses` positions is handled downstream by `RolloutStep.__post_init__` / `PrivilegedState`
  construction (verified: the step-4 step dict keys are exactly the `RolloutStep` fields). Raises `ValueError`
  / `KeyError` on a malformed log (never silently drops a row).
- `load_rollout_log(path: StrPath) -> Rollout` — thin I/O wrapper: accept either a `steps.json` file **or** a
  run directory (then read `<dir>/steps.json`); `json.loads(path.read_text())` → `rollout_from_log`.

**Interface (the driver):**
- `score_rollout_l2(rollout: Rollout, *, k: int, schema: SchemaA = SchemaA()) -> list[Score]` — thin wrapper
  over `ConsistencyMetricA(schema=schema, k=k).score_rollout(rollout)` (DRY; reuses the frozen metric).
- `geometry_stats(rollout: Rollout) -> dict` — model-free sanity stats from the real privileged state
  (min EE↔distractor distance, EE↔target distance per step, count of open→close grasp transitions + their
  EE↔target distances, `target_region` resolvable?). Calibration input for M2/M3 (D-3), not a metric.
- `main(argv) -> int` — argparse (`--rollout` path to `steps.json` or run dir [default = the step-4 run dir],
  `--k` causal window default = whole prefix (`len(rollout)`, matching `demo_metric_separation`), `--tau`
  illustrative default `0.5`, `--results-root` `results/_smoke`). Load → score → `rollout_fires(scores, tau)`
  → write `l2_attach_report` → print. **No CUDA guard** (model-free; runs in the core `.venv`).

**Test scenarios (Task 1):**
- `rollout_from_log` round-trips a serialized rollout: take `demo_rollout._rollout_to_json(r)` (or a hand-built
  fixture) → `rollout_from_log` → identical `action` tuples, `privileged_state` dicts, `len`.
- `rollout_from_log` on the **real step-4 log fixture** (a trimmed copy of the actual `steps.json` under
  `tests/.../fixtures/`) builds a `Rollout` whose `privileged_state` has the 4 required keys and whose
  `target_region` (`plate_1`) is in `object_poses` (resolver will not abstain).
- Malformed logs raise, not swallow: missing `"steps"` → `KeyError`/`ValueError`; a step with a 6-element
  `action` → `ValueError` propagates from `RolloutStep` (boundary check).
- `score_rollout_l2` returns one `Score` per step, every `value` finite and in `[0, 1]` (the gate's core
  assertion), `window_end == step_index`.
- `geometry_stats` returns finite distances and the correct grasp-transition count on a fixture with a known
  open→close flip.
- `rollout_fires` on the scored real rollout returns a `Decision` (exercises `decide()` end-to-end); the test
  asserts it *runs and is well-typed*, not a specific hold/allow (τ is illustrative).
- `load_rollout_log` accepts both a file path and a directory path (tmp dir with a `steps.json`).
- Module import is clean in the core `.venv` (no torch/LIBERO import at module load).

**Acceptance run (the real gate — on the mac, against the committed step-4 log):**
```bash
PYTHONPATH=src .venv/bin/python scripts/attach_l2_to_rollout.py \
  --rollout results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke
```
Expect: `PASS: L2 ingested N real OpenVLA steps; scores finite in [0,1]; decision emitted` and an
`l2_attach_report.json` written under `results/_smoke/`.

**Dependencies:** `evasion_tax.records` (`Rollout`, `RolloutStep`, `Score`),
`evasion_tax.metric.consistency_a` (`ConsistencyMetricA`, `SchemaA`),
`evasion_tax.detector.decide` (`rollout_fires`), `evasion_tax.repro` (`RunLogger`), `scripts/_bootstrap`.

**Notes:**
- The L2 metric uses its **default** `SyntheticStateAdapter` to re-ingest each logged `privileged_state` dict —
  no new adapter (the dict was already normalised by the real `LiberoStateAdapter` at log time). Verified: keys
  match exactly.
- Report fields: source `run_id` + `git_commit` (provenance of the ingested rollout), `n_steps`, `k`, `schema`
  (`dataclasses.asdict(SchemaA())`), `per_step_scores`, `score_summary` (min/mean/max), `illustrative_tau`,
  `decision` (`dataclasses.asdict(Decision)`), `geometry_stats`, and an explicit
  `"claim": "wiring de-risk only — NOT separation/calibration/deployable"` line (honesty, CLAUDE.md §academic).
- Mirror `demo_metric_separation.py` for the `RunLogger().start(...).write(...)` path; do **not** import its
  AUC/TPR helpers (out of scope per D-2).

**Commit:** `feat: CSB bring-up step 5 — L2-attach seam (rollout_io) + attach_l2_to_rollout.py (TDD)`

---

## Task 2 (deferred — explicitly out of step-5 scope): inline live L2 scoring on the box

> **Not built in step 5** (Decision D-1). Recorded so the boundary is documented.

A per-step hook inside `smoke_libero_episode.py`'s episode loop that scores the causal prefix live and could
*hold* the rollout is **online-detector** behaviour — that belongs to **M2** (calibrated τ + online holds, H2),
not the bring-up gate. **Trigger to build:** when M2 needs online holds (calibrated τ from a benign split). Until
then, batch-scoring logged rollouts (Task 1) is the reusable path M1/M2 will also use.

---

## Done-when (Step 5 exit)
- [ ] Task 1 green locally (core `.venv`): `rollout_io` round-trip + malformed-log boundary tests pass; metric A
      scores the **real step-4 log** with every per-step `value` finite in `[0, 1]`; `rollout_fires` emits a
      well-typed `Decision`; `geometry_stats` correct; ruff + pyright clean; full suite green (≥410, +new).
- [ ] Acceptance run against `results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke` prints `PASS` and writes
      `l2_attach_report.json` (per-step scores + decision + geometry stats + the honesty `claim` line).
- [ ] Tick `docs/gpu/CSB/plan.md` step 5 `[x]` (cite the `l2_attach_report.json` run dir); update playbook §1
      `Last updated` + `▶ NEXT ACTION` → step 6 (GCG tiny run → D8 timing micro-bench → Branch N/N−/F).
- [ ] Commit the `results/_smoke/` L2 report (tracked, non-registered). Note in the playbook that step 5 was met
      **offline on the mac** from the committed step-4 log (no box session needed for the gate).

## What this step does NOT claim (carry into the write-up)
Wiring only. The single benign rollout yields **no** detection/separation/FPR claim; metric A here is the
**non-deployable oracle**; the `engagement_radius`/`grasp_radius` radii remain the **frozen pre-GPU
placeholders** (the geometry stats are *input* to re-pinning them at M2/M3, not a re-pin). H1 separation is an
open empirical question requiring RoboGCG (step 6 / M1).
