---
source_file: "docs/dynamic-workflow/2026-06-08-code-review.md"
type: "document"
community: "Code-Goal Consistency Review"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/Code-Goal_Consistency_Review
---

# 2026-06-08-code-review.md

## Connections
- [[Code ↔ Research-Goal Consistency Review (2026-06-08)]] - `defined_in` [EXTRACTED]
- [[srcevasion_taxdetectordecide.py]] - `defined_in` [EXTRACTED]
- [[srcevasion_taxevalharness.py]] - `defined_in` [EXTRACTED]
- [[srcevasion_taxevalmetrics.py]] - `defined_in` [EXTRACTED]
- [[srcevasion_taxevalsplits.py]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/Code-Goal_Consistency_Review

## 📄 Source

`docs/dynamic-workflow/2026-06-08-code-review.md`

# Code ↔ Research-Goal Consistency Review — `src/evasion_tax/`

**Date:** 2026-06-08
**Scope:** every function/method in `src/evasion_tax/` (153 total, 35 files, ~4,355 LoC)
**Goal of the review:** prove consistency between the **Embodiment Evasion Tax (EET)** research goal and the
code — flag any function that does not match what the project needs to complete the M2/M3 deliverables and
write up defensibly. Cross-file integrity checked in addition to per-function correctness.
**Method:** dynamic multi-agent workflow `eet-goal-code-consistency` — 1 extractor + 153 per-function
verifiers (one agent each) + adversarial confirmers for every non-pass + 8 cross-file integrity agents.
**Cost / scale:** 166 agents, 756 tool calls, ~8.17M tokens, ~31 min.
**Run artefacts:** workflow `wf_5bfedc68-27d` (task `wmdwu4dr7`).

> Status: findings only — **no code was changed.** Items marked *(decision needed)* require an author call on
> intended GPU-phase deferral before a fix is chosen.

---

## Headline

**No function failed its own contract — 0 / 153 FAIL, 151 pass, 2 low concerns.** Every per-function
implementation is correct and consistent with the EET research goal. Line-by-line verification came back
**clean** on: metric-A frozen-schema fidelity (constants, P1/P2/P3, `combine=max`, ε), causality (no
future-leak; monitoring-ceiling correctly non-causal and a true upper bound), the idealized-frontier
oracle-bound direction, reproducibility / immutability / GPU-seam hygiene, and stale-identifier framing
(no `T7`, no `GB10`, no `4-bit` precision claim, no `unsafe-action-blocked`, no firewall/defence overclaim;
action codec matches the OpenVLA 256-bin formula).

**The real findings are all cross-file wiring gaps, and they form one coherent theme:**

> A cluster of required safety-rails and metrics are **implemented and unit-tested but never invoked by the
> single reporting path** (`scripts/evaluate.py → run_condition_matrix → results_table_to_dict`). They pass
> CI yet would make the M2 floor / M3 frontier **indefensible to write up** — the failure mode unit tests
> cannot catch.

Coverage note: all 153 functions were audited; nothing was dropped.

---

## 🔴 HIGH — fix before the M2/M3 write-up (ideally before GPU runs)

### H1. Power rule (`eval/power.py`) is dead code — never wired into reporting
`annotate_operating_points` / `classify_power` / `required_benign_n` are imported **only** by
`tests/evasion_tax/eval/test_power.py`. Not called by `run_condition_matrix` (`harness.py:62-117`), not
serialised by `results_table_to_dict` (`figures.py:37-58`), not re-exported from `eval/__init__.py`;
`DetectorConfig.primary_fpr` (`schema.py:73`) is never read at eval time.

**Why it matters:** `results.json` carries `n_benign` per operating point but **no `is_powered` flag**, so a
**TPR @ 1%-FPR point computed on e.g. 60 benign rollouts can be presented as a headline number with nothing
marking it underpowered** — the "fiction quantile" the rule exists to block (invariant #5 / Codex-#2 #3 /
playbook §5). The function's own docstring calls it "the reporting gate"; the gate is inert.

**Fix:** call `annotate_operating_points(points, primary_fpr=cfg.detector.primary_fpr)` inside
`run_condition_matrix`, attach the `PowerStatus` list to `ConditionRow`, serialise it in
`results_table_to_dict`, export the symbols from `eval/__init__.py __all__`, and thread `primary_fpr` from
`scripts/evaluate.py`. Have figures/tables refuse to (or visibly flag) emit an underpowered non-primary point
as headline.

### H2. `assert_disjoint` (`splits.py`) is never invoked — the calib/test leakage guard is dead
`assert_disjoint` (`splits.py:16-39`) correctly checks task/scene/seed overlap and raises, but a grep over
`src/`+`scripts/` (excluding tests) finds only docstring mentions. `scripts/evaluate.py` loads a config
carrying `eval.splits.calib`/`.test` manifests (`schema.py:93-112`) and dumps them into the run log, yet never
asserts disjointness; `run_condition_matrix` consumes pre-split arrays and asserts nothing.

**Why it matters:** τ is correctly set on `benign_calib` and FPR reported on `benign_test`, but there is **no
executable guarantee those two splits are disjoint by task/scene/seed** — invariant #3's leakage rail is
declared and logged but unenforced.

**Fix:** in `scripts/evaluate.py`, after `load_config` and before `run_condition_matrix`, call
`assert_disjoint(cfg.eval.splits.calib.model_dump(), cfg.eval.splits.test.model_dump())` so an overlap aborts
loudly; and/or have `run_condition_matrix` accept the manifests and assert before calibrating. Add an
integration test that an overlapping config makes `evaluate.py` raise.

### H3. `collect_oracle_outcomes` silently drops coverage-excluded scenarios — on the path that feeds the tax scalar
`cross_layer.py:299` filters the population with the coverage predicate
(`active = [s for s in population if supported is None or supported(s)]`) but, unlike `trace_frontier`
(`idealized_frontier.py:200/215`, which returns `excluded`), it **never returns or logs the dropped
scenarios** (`cross_layer.py:316`). This is the data path feeding
`frontier_from_outcomes → bootstrap_delta_asr →` the H6-A/H6-D ΔASR tax scalar.

**Why it matters:** every unsupported/abstained cell vanishes from the denominator with **no record** → the
reported tax is computed over an unlogged subset and the headline limitation is never surfaced from this path.
Violates the coverage invariant (#7 / schema §6 "never a silent abstain"). The docstring claims it
"reproduces `trace_frontier`'s frontier exactly" but it does **not** reproduce the excluded-surfacing
contract; the reproduction test never passes `supported=`, so the asymmetry is untested.

**Fix:** make `collect_oracle_outcomes` return `(outcomes, excluded)` mirroring `trace_frontier` (or take a
sink/logger), and add a test passing a `supported=` predicate that asserts the excluded set is surfaced.

### H4. `detection_latency` / `benign_degradation` / `abort_rate` have no production caller — harness hard-codes an empty latency list *(decision needed)*
The implementations exist and are unit-tested (`decide.py:54`, `metrics.py:277/284/295`) but a repo-wide grep
shows **no production caller**. `harness.py:108-110` always builds `detection_latency_summary([])` (literal
empty list), so **every reported `latency_summary` is the all-None "never fired" stub**. `run_condition_matrix`
consumes only pre-reduced per-rollout score arrays (`ConditionSplits`, `harness.py:30-33`) and has no
rollout/onset input, so latency/abort/degradation cannot be computed even in principle through the current
interface.

**Why it matters:** §5 names detection-latency a **first-class** metric and lists abort-rate / benign-success
degradation as the cost side of the floor result — none are reportable. The cost side of the M2 floor cannot
be written up.

**Fix (decide the contract):** either **(a)** extend `run_condition_matrix` to accept per-rollout `Score`
sequences + `attack_onset_step` + benign success counts, compute real latencies via
`rollout_fires`+`detection_latency`, feed `detection_latency_summary`, and emit
`benign_degradation`/`abort_rate` per condition; **or (b)** if cost metrics are intentionally deferred to the
GPU rollout stage, replace the empty-list stub with an explicit deferral marker
(`latency_summary=None` + a comment stating cost metrics require GPU rollouts) so the absence is honest rather
than a silent always-empty summary, and remove the dead re-exports.

### H5. "target-action-blocked rate" (§5 / D6 security metric, invariant #9) is not implemented anywhere
A grep for `target_action_blocked` / `blocked_rate` returns zero hits in `src/`. `cross_layer.py` has the raw
ingredients per `UnitOutcome` (`reached` line 73, `detected` line 74) but no function computes or reports it,
and the harness/figures have no such field.

**Why it matters:** it is the carefully-worded required security metric (the replacement for
"unsafe-action-blocked" — until the semantic-redirect arm succeeds the target is a low-level action). The
security side of the cross-layer comparison cannot be tabulated.

**Fix:** add `target_action_blocked_rate` (exact name, invariant #9 framing) computing the fraction of
target-reaching rollouts whose detector fired before the persistence window completed (extend `UnitOutcome`
with the fire-step / a blocked flag if needed), and surface it in the cross-layer table.

---

## 🟠 MEDIUM

### M1. Per-call unresolvable-anchor abstain is silent — contradicts frozen schema §2 *(decision needed)*
Schema §2 (`metric-a-annotation-schema.md:60`) states an unresolvable anchor "returns `s = 0.0` (abstain …)
and **this is logged**"; §6 reinforces "never a silent abstain." In code, the abstain path in
`ConsistencyMetricA._semantics` (`consistency_a.py:201-202`) returns `Semantics(0.0, 0.0, 0.0)` with **no
logging**, and `Score` (`records.py:198-211`) has no abstain/unresolved flag.

**Why it matters:** an abstained `s=0.0` is **indistinguishable from a legitimately goal-consistent `s=0.0`**
and silently enters ROC/calibration/TPR aggregation as a confident benign score, risking dilution of the
held-out FPR/TPR statistics (invariants #3, #5). The matrix-level `CoverageManifest.ABSTAINED` machinery
satisfies the *pre-run* requirement but does not cover a *runtime* step where the cell resolves yet
`target_region` is `None`/absent in a particular step's `PrivilegedState`.

**Fix:** make the per-call abstain observable — emit a structured log (`step_index`,
`reason='unresolvable-anchor'`) and/or add `abstained: bool` to `Score`, set on the anchor-`None` path, and
have aggregation exclude or count abstained scores. **Alternatively**, if `target_region` is invariant within
a rollout by construction, amend schema §2 to say so and add a guard/assert documenting it.

### M2. `predicate_for_target` is never wired into a runnable frontier driver, and its `target_id` binding is unchecked
Grep over `src/`+`scripts/` (excluding tests) finds **no call** to `build_manifest` / `CoverageManifest` /
`predicate_for_target`. `trace_frontier` and `collect_oracle_outcomes` default `supported=None` → **no
coverage gate unless a caller opts in**, so the idealized attacker can currently run with no coverage
constraint and preferentially exploit an unmodelled blind spot entirely uncounted — the exact failure schema
§6 calls "load-bearing for M3 validity." Compounding: `predicate_for_target(target_id)` keys on a string that
is never bound to `trace_frontier`'s actual `TargetActionSpec`, so a target/predicate mismatch is silently
accepted.

**Fix:** when the model-free frontier driver is built, require a non-None `supported` predicate for any real
H6-A/H6-D run (or have `trace_frontier`/`collect_oracle_outcomes` warn/refuse when `supported is None` on a
non-trivial population), carry the target identity on `TargetActionSpec` and assert it equals the predicate's
`target_id`. Minimum: a docstring contract note that omitting `supported` for an M3 run is invalid.

### M3. The M3 H6-A headline pipeline (idealized attacker + cross-layer tax) has no driving script *(decision needed)*
The library is complete (`collect_oracle_outcomes`, `frontier_from_outcomes`, `bootstrap_delta_asr`,
`comparative_asr_table`, `frontiers_by_layer`, `trace_frontier`) but their only callers are tests / internal
re-exports. No script under `scripts/` assembles a frontier/tax table and writes it to write-once `results/`,
and `figures.py` emits a static `_ladder_placeholder` (`figures.py:123`) with no frontier-overlay figure.

**Why it matters:** the committed M3 deliverable has implemented stats but **no reproducible run path /
figure**.

**Fix:** add a model-free `scripts/cross_layer_tax.py` that builds synthetic `AttackScenario`s, runs
`collect_oracle_outcomes` for the L2-oracle, computes `comparative_asr_table` + `bootstrap_delta_asr`, and
writes the tax table via `RunLogger` (mirroring `evaluate.py`'s write-once pattern); add a frontier-overlay
figure fed from that logged output. If intentionally deferred to M3-on-GPU, add a one-line note in §4b
"Build order" so the absence is tracked, not silently missing.

---

## 🟡 LOW (docstring / hardening)

### L1. `cross_layer.py` calls the non-adaptive (H6-A) ordering "the Embodiment Evasion Tax" — H6-A/H6-D conflation risk
The module header (`cross_layer.py:5-6`) and `delta_asr_at_evasion` (`cross_layer.py:118`) attribute the
headline "Embodiment Evasion Tax" / "the embodiment tax" to a comparison that, in the model-free path, is
purely the **non-adaptive same-attacker ordering (H6-A)** — which the design says makes "no cross-layer
deployable-tax claim … the tax headline is H6-D's, never H6-A's" (playbook §3, M3 row line 126, H6-A row
line 157). `comparative_asr_table` carries the correct "non-adaptive" qualifier; the module header and
`delta_asr_at_evasion` do not. The code is correct and attacker-agnostic — **docstring-only** overclaim risk.

**Fix:** add the same non-adaptive qualifier the sibling function already uses (e.g. "the NON-ADAPTIVE
cross-layer ordering (H6-A); the deployable-vs-deployable matched-adaptive headline tax is H6-D (M4), never
claimed here").

### L2. Stale "unsafe-action-blocked" in `docs/core/goal-action-consistency-detector.md` (~line 185)
The companion design doc still lists "unsafe-action-blocked rate" among reported metrics; invariant #9 /
playbook §5/D6 mandate **"target-action-blocked"**. The playbook itself is correct; only this doc is stale.
**Fix:** one-word edit. *(Doc, not code — listed for completeness.)*

---

## Concerns (per-function, low severity)

### C1. `repro/env_capture.py::_torch_versions` calls a non-existent torch API
`torch.cuda.driver_version()` does **not exist** (verified against torch 2.2.2: the only primitive is
`torch._C._cuda_getDriverVersion()`). On a GPU node where `torch.cuda.is_available()` is True, line 64 raises
`AttributeError`, caught by the broad `except` (line 66), so `driver` **silently becomes `None`** exactly
where the docstring promises it resolves. The test suite only exercises the no-torch local path, so CI cannot
catch it.

**Bound:** the load-bearing repro fields (`torch`, `cuda` versions) are captured on an independent path and are
correct; `driver` is secondary host-driver provenance and no EET result/τ/FPR/frontier depends on it.
`capture_env` keeps its "never raises" contract. Real but bounded — graded `concern`, not `fail`.

**Fix:** under `if torch.cuda.is_available():` set `raw = torch._C._cuda_getDriverVersion()` and format
(`major=raw//1000`, `minor=(raw%1000)//10`), keep the broad `except`, and add a mocked-torch unit test
asserting `driver` resolves non-None when `is_available()` is True.

### C2. `eval/metrics.py::abort_rate` omits the input-bounds validation its sibling enforces
`proportion_ci` validates `0 <= k <= n`; `abort_rate` omits the analogous `0 <= n_aborted <= n_total` check, so
a bad upstream count would silently yield a rate > 1 and mask the counting bug. **Latent** — `abort_rate` has
no production caller yet (referenced only in `__init__` exports + two unit tests). Graded `concern`.

**Fix:** mirror `proportion_ci`'s validation: `if not (0 <= n_aborted <= n_total): raise ValueError(...)`.

---

## Dimensions that came back CLEAN (verified, no findings)

- **Causality** — online detection is strictly causal across `records.prefix_window`, `consistency_a`
  (`score`/`score_rollout`), `detector.decide`, and `probe_internal.score_rollout`; `prefix_window` clamps
  `start = max(0, t-k+1)` and never includes an index > t. `score_rollout_monitoring_ceiling` is the only
  non-causal path — explicitly labelled, a true upper bound (`≥` causal), each neighbour keeps its own causal
  anchor (no future-anchor leak), and referenced only inside `consistency_a.py` (never reaches an online-hold
  path). Upper-bound invariant is unit-tested.
- **Reproducibility / ethics / GPU-seam** — seeds pinned AND recorded (`stable_seed` uses SHA-256, never the
  salted built-in hash); `RunLogger` write-once enforced via `mkdir(exist_ok=False)` + `FileExistsError` on
  every write; env/provenance captured with SHA-256; immutable patterns throughout (config deep-copied before
  logging; no in-place mutation of inputs); all `Real*` GPU seams raise `NotImplementedError` (no silent
  no-op / fabricated value); GPU scripts print `gpu_required_message` and exit non-zero on no-CUDA; nothing
  untrusted is auto-run.
- **Stale-identifiers / framing** — no `T7`, no `GB10`, no `4-bit`/`int4` precision claim (the only match is
  `quantization: str | None = None`, an optional config knob asserting nothing), no `unsafe-action-blocked`,
  no firewall/universal-defence contribution overclaim in code. Action codec matches the OpenVLA 256-bin
  discrete formula with pinned source provenance.
- **Metric-A frozen-schema fidelity** — constants (`engagement_radius=0.05`, `grasp_radius=0.10`, `ε=1e-9`),
  `combine='max'` default with `noisy_or`/`weighted_mean` as pre-registered ablations only, and the
  P1/P2/P3 formulas are an exact transcription of the schema pseudocode; `k` is a required ctor arg (not
  defaulted) faithfully reflecting its provisional/sweepable status; no constant is attack-tuned;
  `extract_semantics` is an isolated, unit-tested parser. (The single fidelity gap is M1 above.)
- **Oracle / claim-boundary** — metric A consistently labelled non-deployable/oracle/ceiling and never
  presented as deployable; the idealized-frontier oracle-bound direction is stated correctly (not inverted);
  `delta_asr_at_evasion` is the documented primary scalar with a correct cluster bootstrap over the
  `(task,target,seed)` `UnitKey`; the forbidden secondary metrics (queries-to-evasion, area-between-frontiers)
  are correctly absent and no idealized-L2-vs-GCG-L1 cross-attacker comparison is computed. (The single issue
  is the L1 docstring overclaim above.)

---

## Open decisions for the author (resolve before fixing)

1. **H4 / M3** — are detection-latency/abort/degradation and the cross-layer driver script *intended* to be
   GPU-phase deferrals, or wired model-free now? → "wire it up" vs "make the deferral explicit."
2. **M1** — is `target_region` invariant within a rollout (→ amend schema §2 + assert) or should runtime
   abstains be flagged in `Score` (→ richer fix)?

## Suggested fix order

Unambiguous HIGH wiring gaps first — **H1 (power gate), H2 (split disjointness), H3 (excluded surfacing),
H5 (target-action-blocked)** — then the *(decision needed)* items (H4, M1, M2, M3) once the two questions
above are answered, then the LOW doc fixes (L1, L2) and the two concerns (C1, C2). All are integration/wiring
or documentation; none requires re-deriving any verified function.

