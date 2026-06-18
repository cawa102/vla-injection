# CSB Bring-up Step 5 — Attach the Goal-Action Detector (L2) to the Real Rollout — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use `test-driven-development` (`/tdd`) to implement Task 1 task-by-task.
> **Revision note:** hardened 2026-06-18 after a Codex adversarial review (`needs-attention`): the gate is now
> split into a **state half** + an **action half** (metric A reads state, not the action vector — without the
> action half "ingests real OpenVLA *actions*" was over-claimed); source-run **provenance binding**
> (sibling-file validation + `steps.json` SHA-256) added; `geometry_stats` locked **report-only**.

**Goal:** Feed the **real OpenVLA-driven** rollout produced by step 4 (the `RolloutStep` / `PrivilegedState`
stream logged to `results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke/steps.json`) through the **L2
behavioural detector** end-to-end and confirm two distinct things with no schema mismatch
(`docs/gpu/CSB/plan.md` step 5):
- **State half** — the L2-oracle ingests the real privileged-**state trajectory**: `Rollout` →
  `ConsistencyMetricA` (metric A) → per-step `Score` → `decide` / `rollout_fires`.
- **Action half** — the real 7-DoF **action stream** is genuinely exercised: `Rollout.actions()` is finite,
  shape `(N, 7)`, **non-degenerate**, and runs through the D2 action-scoring path (`TargetActionSpec.reached_window`).

> **Why two halves (load-bearing — Codex `[high]`).** `ConsistencyMetricA` scores **only** `privileged_state`
> geometry (P1 progress from `ee_pos`, P2 distractor from `ee_pos`↔`object_poses`, P3 grasp from `gripper_open`
> transitions); `decide`/`calibrate` read only `Score.value`. **The entire L2-oracle path ignores
> `RolloutStep.action`.** So running metric A alone proves *state-trajectory* ingestion — a log with
> zeroed/corrupted actions would score identically. "Ingests real OpenVLA **actions**" is only honest if the
> action half independently touches the action vectors. Both halves must pass.

**Architecture:** The L2 stack already exists and is model-free-tested (`metric/consistency_a.py`,
`detector/{decide,calibrate}.py`). The **one missing seam** is a JSON→`Rollout` deserializer — nothing in the
repo loads a logged `steps.json` back into records (`evaluate.py`/`calibrate.py` consume *already-scored*
rollouts; `demo_metric_separation.py` scores in-memory rollouts it generates). Step 5 adds that seam
(`eval/rollout_io.py`, with source-run provenance validation) plus a thin driver script
(`scripts/attach_l2_to_rollout.py`) that runs both halves and writes an L2 report. **Both are pure-numpy /
model-free** — the real rollout is already on disk, so the whole gate runs in the core `.venv` on the mac (no
CUDA / OpenVLA / LIBERO at attach time).

**Tech Stack:** Python 3.10, NumPy (metric A is pure NumPy), `hashlib` (steps.json SHA-256), the project's
`evasion_tax` package (`records` incl. `TargetActionSpec`, `metric.consistency_a`, `metric.state`
`SyntheticStateAdapter`, `detector.decide`, `repro`, `config`). `detector.calibrate` is referenced only for the
*illustrative-τ* note, **not** invoked as calibration here. No torch, no LIBERO, no GPU.

**Scope boundary (do not exceed):** This is the **wiring de-risk** — verify gate = *the L2 detector ingests the
real OpenVLA-driven rollout end-to-end (state half + action half) and emits finite per-step scores + a decision
trace, schema matching the state-adapter / metric side, bound to the verified step-4 source run*
(`docs/gpu/CSB/plan.md` step 5). It is **NOT**:
- **H1 separation / AUC** (needs an *attacked* class — RoboGCG, step 6 / M1 — and we have one benign episode).
- **M2 calibration** (`calibrate`/`tpr_at_fpr` need a benign **calibration split**; one rollout cannot calibrate).
- a **deployable** detector claim (metric A is the privileged *oracle* / upper bound — `consistency_a.py` header).
- a **schema re-pin** of `SchemaA` radii (frozen pre-attack, invariant #2; step 5 only *observes* benign geometry).
- an **attack claim** — the D2 `reached_window` call in the action half uses an *illustrative* target purely to
  exercise the action machinery on real vectors; no target was tuned and no success is asserted.

Any τ shown is **illustrative, NOT calibrated**. Metric-A score *magnitudes* are a **sanity observation**
(report-only calibration input for M2/M3), never a pass/fail signal — see Decisions D-3 below.

**References:** @docs/gpu/CSB/plan.md (step 5 ladder + Step 4 how-to / gotchas) ·
@docs/core/goal-action-consistency-detector.md (§3 threat model, §5 the metric is the make-or-break) ·
@docs/core/execution-playbook.md (§2 M1/M2 roadmap, H1/H2) · @docs/core/metric-a-annotation-schema.md
(frozen schema; `engagement_radius` `[VERIFY vs LIBERO geometry]` flag) ·
@src/evasion_tax/metric/consistency_a.py (the L2 metric this attaches — state-based) ·
@src/evasion_tax/records.py (`Rollout.actions`, `TargetActionSpec` — the action path) ·
@src/evasion_tax/detector/decide.py (the decision path) ·
@scripts/demo_metric_separation.py (the in-memory scoring pattern to mirror, minus the AUC/TPR science) ·
@results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke/ (the real run dir: `steps.json` + `run.json` +
`episode_meta.json` — the source to ingest and bind to).

---

## Decisions (pre-registered — design-fork handling: decide + record, don't multiple-choice ask)

- **D-1 — Offline loader is primary; inline live-scoring is deferred.** Step 4 persisted the real rollout to a
  tracked file, so L2-attach is purely offline (KISS + reproducible + reusable for M2 batch scoring). An inline
  per-step hook in `smoke_libero_episode.py` is **out of scope** (YAGNI — that is M2 *online-hold* territory).
  Named in "Box variant (deferred)" so the boundary is explicit, not silently dropped.
- **D-2 — Gate is wiring only, in two halves (Codex `[high]`).** No AUC/TPR/FPR (no attacked class, no split).
  The **state half** records per-step scores + summary + a `rollout_fires` decision at an **illustrative τ** to
  exercise `decide()`. The **action half** independently proves the real action stream is well-formed and runs
  through the D2 scorer — so the gate cannot pass on a zeroed/corrupted action log. **Both halves required.**
- **D-3 — `geometry_stats` is strictly report-only; no re-pin in step 5 (Codex `[medium]`, invariant #2).**
  `SchemaA.engagement_radius=0.05 m` is a flagged pre-GPU placeholder, so benign scores need not be ~0; "benign
  scores not near zero" is **not** a wiring failure. Step 5 *reports* real-scene geometry as calibration input
  but **does not** re-pin radii. **Any** future radius update must be governed by a **dated, benign-only,
  pre-registered rule locked BEFORE step 6 inspects attacked output**, and recorded as an **explicit schema
  deviation** (never an implicit re-tune informed by post-attack data).
- **D-4 — Loader lives in `eval/rollout_io.py`, not `records.py`.** Keeps the frozen schema (`records.py`)
  untouched; isolates file-I/O + boundary validation in the eval layer. The pure dict→`Rollout` function is
  unit-tested separately from the path I/O + provenance wrapper.
- **D-5 — The offline gate is bound to the verified step-4 source run (Codex `[medium]`).** Default input is a
  **run directory**; the loader validates the sibling `run.json` + `episode_meta.json` (stage,
  model, suite/task, seed, success, n_steps, run_id consistency, git commit) and records a **SHA-256 of
  `steps.json`** in the L2 report. A bare `steps.json` (no siblings) is accepted **only** with an explicit
  `--unverified` flag + a logged warning, and the report marks `provenance_verified=false`. This stops a stale /
  trimmed / hand-edited / unsuccessful smoke log being cited as "the real attach".

---

## Task 1: JSON→`Rollout` loader (+ provenance) + L2-attach driver (TDD — mac, now)

**Files:**
- Create: `src/evasion_tax/eval/rollout_io.py`
- Create: `scripts/attach_l2_to_rollout.py`
- Test: `tests/evasion_tax/eval/test_rollout_io.py`
- Test: `tests/evasion_tax/test_attach_l2_to_rollout.py`

**Interface (public, model-free — the reusable seam):**
- `rollout_from_log(obj: Mapping) -> Rollout` — pure dict→records. Validates the top-level shape
  (`"steps"` present and a non-empty list) at the boundary, then builds
  `Rollout(steps=tuple(RolloutStep(**row) for row in obj["steps"]))`. JSON list→tuple coercion for `action`
  and `object_poses` positions is handled downstream by `RolloutStep.__post_init__` / `PrivilegedState`
  construction (verified: the step-4 step dict keys are exactly the `RolloutStep` fields). Raises
  `ValueError` / `KeyError` / `TypeError` on a malformed log (never silently drops a row).
- `SourceProvenance` (frozen dataclass) + `validate_run_dir(run_dir: StrPath) -> SourceProvenance` — read
  sibling `run.json` + `episode_meta.json`; check `stage == "smoke_libero_episode"`, model / suite / task / seed
  consistency across the two files **and** the step rows, `episode_meta.success is True`,
  `n_steps == len(steps)`, and one consistent `run_id`. Compute `steps_sha256`. Raise on any mismatch
  (boundary check) — the gate must not bind to an inconsistent run.
- `load_rollout_log(path: StrPath, *, unverified: bool = False) -> tuple[Rollout, SourceProvenance | None]` —
  thin I/O + provenance wrapper. For a **run dir** (default): `validate_run_dir` then `rollout_from_log`. For a
  bare `steps.json`: only with `unverified=True` → load + warn, returning `(rollout, None)`.

**Interface (the driver):**
- `score_rollout_l2(rollout: Rollout, *, k: int, schema: SchemaA = SchemaA()) -> list[Score]` — thin wrapper
  over `ConsistencyMetricA(schema=schema, k=k).score_rollout(rollout)` (DRY; reuses the frozen metric). **State half.**
- `action_stream_check(rollout: Rollout) -> dict` — **action half.** `acts = rollout.actions()`; assert shape
  `(N, 7)`, all-finite, and **non-degenerate** (not all-zero; not byte-identical across every step). Run an
  **illustrative** `TargetActionSpec` through `acts` via `reached_window`/`reached_window_step` purely to
  exercise the D2 path on real vectors (no attack claim). Return `{n_steps, dims, all_finite, degenerate(False),
  per_dim_min/max, illustrative_target, reached_window, completion_step}`. Raise if degenerate/non-finite — a
  zeroed/corrupted action stream must **fail** this half.
- `geometry_stats(rollout: Rollout) -> dict` — model-free sanity stats from the real privileged state (min
  EE↔distractor distance, EE↔target distance per step, count of open→close grasp transitions + their EE↔target
  distances, `target_region` resolvable?). **Report-only** calibration input for M2/M3 (D-3), not a metric.
- `main(argv) -> int` — argparse (`--rollout` run dir [default = the step-4 run dir] **or** `steps.json` with
  `--unverified`, `--k` causal window default = whole prefix `len(rollout)` matching `demo_metric_separation`,
  `--tau` illustrative default `0.5`, `--results-root` `results/_smoke`). Load+validate → state half → action
  half → `rollout_fires(scores, tau)` → write `l2_attach_report` → print. **No CUDA guard** (model-free; runs
  in the core `.venv`).

**Test scenarios (Task 1):**
- *Loader round-trip:* `demo_rollout._rollout_to_json(r)` (or a hand-built fixture) → `rollout_from_log` →
  identical `action` tuples, `privileged_state` dicts, `len`.
- *Real-log fixture:* `rollout_from_log` on a trimmed copy of the actual `steps.json`
  (`tests/.../fixtures/`) builds a `Rollout` whose `privileged_state` has the 4 required keys and whose
  `target_region` (`plate_1`) is in `object_poses` (resolver will not abstain).
- *Malformed logs raise, not swallow:* missing `"steps"` → `KeyError`/`ValueError`; a step with a 6-element
  `action` → `ValueError` propagates from `RolloutStep`.
- *State half:* `score_rollout_l2` returns one `Score` per step, every `value` finite and in `[0, 1]`,
  `window_end == step_index`.
- *Action half passes on real actions:* `action_stream_check` on the real-log fixture returns
  `degenerate=False`, `all_finite=True`, shape `(N, 7)`.
- *Action half FAILS on a broken stream (the anti-false-confidence test, Codex `[high]`):* take the real
  fixture, **zero all actions** → `action_stream_check` **raises**, while `score_rollout_l2` is **unchanged**
  (proves the gate now genuinely depends on the action stream, not just the state geometry).
- *Provenance (D-5):* `validate_run_dir` passes on the real run dir; mutating `episode_meta.success`→`False`,
  or `n_steps`, or the model id raises; `steps_sha256` is stable and matches `hashlib` over the file bytes;
  bare `steps.json` without `--unverified` is rejected, with `--unverified` returns `provenance=None` + warns.
- *Decision path:* `rollout_fires` on the scored real rollout returns a well-typed `Decision` (τ illustrative;
  assert it runs + types, not a specific hold/allow).
- *`geometry_stats`* returns finite distances and the correct grasp-transition count on a fixture with a known
  open→close flip.
- *Clean import* in the core `.venv` (no torch/LIBERO import at module load).

**Acceptance run (the real gate — on the mac, against the committed step-4 run dir):**
```bash
PYTHONPATH=src .venv/bin/python scripts/attach_l2_to_rollout.py \
  --rollout results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke
```
Expect: provenance validated (sha256 logged) → `PASS: L2 ingested N real OpenVLA steps — state half (scores
finite in [0,1], decision emitted) + action half ((N,7) finite, non-degenerate)` and an `l2_attach_report.json`
written under `results/_smoke/`.

**Dependencies:** `evasion_tax.records` (`Rollout`, `RolloutStep`, `Score`, `Decision`, `TargetActionSpec`),
`evasion_tax.metric.consistency_a` (`ConsistencyMetricA`, `SchemaA`),
`evasion_tax.detector.decide` (`rollout_fires`), `evasion_tax.repro` (`RunLogger`), `hashlib`, `scripts/_bootstrap`.

**Notes:**
- The L2 metric uses its **default** `SyntheticStateAdapter` to re-ingest each logged `privileged_state` dict —
  no new adapter (the dict was already normalised by the real `LiberoStateAdapter` at log time). Keys match.
- **Logging off-by-one (flag for M4, do not "fix" in step 5):** `smoke_libero_episode.py` records
  `build_rollout_step(obs, action)` **before** `env.step(action)`, so `privileged_state_t` is the state that
  *produced* `action_t`, not its result. Harmless for metric A (state-based), but the action↔state alignment
  matters for the D2 attack-scoring and the M4 deployable detectors (B/C) — record it here, decide alignment there.
- Report fields: source `run_id` + `git_commit` + **`steps_sha256`** + `provenance_verified` (provenance of the
  ingested rollout), `n_steps`, `k`, `schema` (`dataclasses.asdict(SchemaA())`), `per_step_scores`,
  `score_summary` (min/mean/max), `illustrative_tau`, `decision` (`dataclasses.asdict(Decision)`),
  `action_stream` (the action-half dict), `geometry_stats`, and an explicit
  `"claim": "wiring de-risk only — state + action ingestion; NOT separation/calibration/deployable"` line.
- Mirror `demo_metric_separation.py` for the `RunLogger().start(...).write(...)` path; do **not** import its
  AUC/TPR helpers (out of scope per D-2).

**Commit:** `feat: CSB bring-up step 5 — L2-attach seam (rollout_io + provenance) + attach_l2_to_rollout.py (TDD)`

---

## Task 2 (deferred — explicitly out of step-5 scope): inline live L2 scoring on the box

> **Not built in step 5** (Decision D-1). Recorded so the boundary is documented.

A per-step hook inside `smoke_libero_episode.py`'s episode loop that scores the causal prefix live and could
*hold* the rollout is **online-detector** behaviour — that belongs to **M2** (calibrated τ + online holds, H2),
not the bring-up gate. **Trigger to build:** when M2 needs online holds (calibrated τ from a benign split). Until
then, batch-scoring logged rollouts (Task 1) is the reusable path M1/M2 will also use.

---

## Done-when (Step 5 exit)
- [ ] Task 1 green locally (core `.venv`): loader round-trip + malformed-log boundary tests pass; **state half**
      scores the real step-4 log with every per-step `value` finite in `[0, 1]`; **action half** passes on the
      real actions **and FAILS on a zeroed stream** (the anti-false-confidence test); provenance validation
      passes on the run dir and raises on a mutated sibling; `rollout_fires` emits a well-typed `Decision`;
      `geometry_stats` correct; ruff + pyright clean; full suite green (≥410, +new).
- [ ] Acceptance run against `results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke` validates provenance
      (sha256 logged), prints `PASS` for **both halves**, and writes `l2_attach_report.json` (per-step scores +
      decision + action_stream + geometry stats + steps_sha256 + the honesty `claim` line).
- [ ] Tick `docs/gpu/CSB/plan.md` step 5 `[x]` (cite the `l2_attach_report.json` run dir); update playbook §1
      `Last updated` + `▶ NEXT ACTION` → step 6 (GCG tiny run → D8 timing micro-bench → Branch N/N−/F).
- [ ] Commit the `results/_smoke/` L2 report (tracked, non-registered). Note in the playbook that step 5 was met
      **offline on the mac** from the committed step-4 run dir (no box session needed for the gate).
- [ ] Before step 6: lock the dated, benign-only, pre-registered rule for any `SchemaA` radius update (D-3) so
      no post-attack data can inform a re-pin (invariant #2).

## What this step does NOT claim (carry into the write-up)
Wiring only, in two honest halves: the L2-oracle ingests the real privileged-**state trajectory** (metric A is
state-based) **and** the real 7-DoF **action stream** is well-formed and runs the D2 path — but the single
benign rollout yields **no** detection/separation/FPR claim, metric A here is the **non-deployable oracle**, and
the `engagement_radius`/`grasp_radius` radii remain the **frozen pre-GPU placeholders** (the geometry stats are
report-only *input* to a future, pre-registered re-pin, not a re-pin). H1 separation is an open empirical
question requiring RoboGCG (step 6 / M1).
