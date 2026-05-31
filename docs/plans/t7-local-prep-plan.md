# T7 — Pre-GB10 Local Preparation Plan (M1 / 8 GB)

> **For Claude:** REQUIRED SUB-SKILL — implement this plan task-by-task with the **`test-driven-development`**
> skill (`/tdd`): one failing behaviour test at a time → minimal code → refactor. Use **`executing-plans`**
> for the task loop. Commit per task (`<type>: <desc>`, no co-author trailer) per CLAUDE.md.
>
> Companion to the **execution playbook** (`t7-execution-playbook.md`, operational source of truth) and the
> **Phase-3 plan** (`t7-phase3-implementation-plan.md`, the M1–M2 component contracts this plan realises).
> This plan covers **only what is buildable on the M1 (8 GB) without the model/GPU**, plus a GB10 runbook —
> it does **not** add scope beyond M1–M2.
>
> ⚠️ AI-assisted scaffold for author review (CLAUDE.md §1/§5). Provisional items are marked.

**Goal:** Build and unit-test every **model-free** M1–M2 engineering component locally (repro infra,
metric-(A) scorer, calibrated detector, eval statistics, action-codec, baselines, configs/scripts/figures),
so that when GB10 arrives the remaining work is "plug in OpenVLA + LIBERO and run", and so metric-(A)'s
annotation schema is **frozen before any attack output exists** (a hard requirement we can only satisfy now).

**Architecture:** A small `src/t7` Python package, feature-organised (Phase-3 §3). Everything that touches the
7B model, GCG, or LIBERO *rollouts* is hidden behind thin interfaces with **synthetic fixtures** for tests;
the real implementations of those interfaces are deferred to GB10. The scientific core — privileged-state
metric (A), FP-calibration, ROC/AUC + per-rollout TPR@FPR with CIs, detection latency, write-once logging — is
pure NumPy/SciPy and fully testable on M1. One optional, **time-boxed** state-only LIBERO smoke test validates
the metric-(A) ground-truth adapter against the real environment; it falls back to mocks if installation fights.

**Tech Stack:** Python **3.11** local (code kept **3.10-compatible** for the GB10 OpenVLA stack), **uv** for
env management, `numpy` `scipy` `scikit-learn` `pyyaml` `pydantic`, `pytest` + `ruff`. LIBERO smoke test (opt):
`mujoco` + `robosuite` + `libero` in an isolated extras group. No model weights, no CUDA, no GPU.

---

## Status (live — 2026-05-31)

| Task | Status | Commit |
|------|--------|--------|
| 0 Dev env + scaffold | ✅ | `69ad856` |
| 1 Reproducibility infra | ✅ | `9aa898f` |
| 2 Data records | ✅ | `50336f6` |
| 3 Action codec | ⬜ **needs OpenVLA primary-source verification** of the de-tokenise/un-normalise formula | — |
| 4 Privileged-state adapter | ✅ | `ad62616` |
| 5 Metric (A) schema freeze | ⬜ **LOAD-BEARING — author design review before coding** | — |
| 6 FP-calibrated detector | ✅ | `7f2be7c` |
| 7 Eval harness + stats | ✅ | `2ab71aa` |
| 8 Baselines | ⬜ delegable | — |
| 9 Config + scripts + figures | ⬜ delegable (+ `src`-bootstrap so scripts import `t7`) | — |
| 10 LIBERO state-only smoke | ⬜ time-boxed | — |
| 11 GB10 runbook | ⬜ | — |

**147 tests green; full `src/t7` is type-clean under `uvx pyright`.** Infra notes: pytest resolves `t7` via `pythonpath=["src"]` (uv's editable `.pth` is
unreliable on this host — corrupted on each `uv run`); pyright via `pyrightconfig.json` (`uvx pyright` is the
authoritative type-check — the harness LSP's `reportMissingImports` for `t7.*` are cosmetic artifacts of the
broken editable install).

---

## Guiding invariants (every task must honour these — they are the WHAT, not optional polish)

1. **Causal detection** — metric/detector see only the prefix window `a_{t-k+1:t}` (past + candidate `a_t`),
   never future actions. A non-causal full-window pass is a *separately labelled* monitoring ceiling.
2. **Schema-freeze** — metric (A)'s annotation schema is defined and committed in **Task 5 before any attack
   output is ever inspected**; no rule may be added later after seeing an attack (circularity guard).
3. **Calibration honesty** — `τ` is set on a calibration split; FPR/TPR reported on a **disjoint** test split;
   the harness *asserts* disjointness. **Primary FPR is per-rollout** (false-abort); per-window is auxiliary.
4. **Same calibration for baselines** — every baseline goes through the *same* `calibrate(...)` (fair).
5. **Write-once results** — run outputs go to `results/<timestamp>-<slug>/` and the logger **refuses to
   overwrite**. Tests write to `tmp_path`, never to `results/`.
6. **Immutability** — records/configs are frozen dataclasses or returned as new copies (coding-style rule).
7. **Label (A) as a non-deployable upper bound** everywhere it is surfaced (uses privileged sim state).
8. **No fabricated constants** — exact OpenVLA un-normalisation stats, token mapping, and pinned GB10 versions
   are *fetched from source* in their tasks and provenance-recorded; never hand-typed from memory.

**Repo layout created incrementally (a dir only when a task first needs it):**
```
src/t7/{repro,records,policy,metric,detector,eval,baselines,attack,config}/   configs/   scripts/
tests/t7/...            results/ (write-once, committed: small JSON/CSV + config snapshots)
docs/setup/gb10-runbook.md
```
`data/` and `artifacts/` stay gitignored (already in `.gitignore`); `results/` is committed (holds metrics +
config snapshots + *references*, not raw tensors — `observation_ref`, not the image).

---

## Task 0: Dev environment + package skeleton

**Files:**
- Create: `pyproject.toml` (project `t7`, src layout, deps + `[project.optional-dependencies] libero`, ruff + pytest config)
- Create: `src/t7/__init__.py`, `tests/t7/__init__.py`, `tests/t7/conftest.py`
- Create: `.python-version` (`3.11`)

**What:** uv-managed local dev env; importable `t7` package; pytest + ruff wired. Core deps only
(numpy/scipy/scikit-learn/pyyaml/pydantic); the heavy LIBERO/mujoco set lives in an **optional** `libero`
extra so it can't break the core env.

**Interface:** `uv sync` builds the env; `uv run pytest` runs tests; `uv run ruff check` lints.

**Test scenarios:** `import t7` succeeds; a trivial `tests/t7/test_smoke.py` passes under `uv run pytest`.

**Dependencies:** none. **Notes:** keep code 3.10-compatible (no 3.11-only syntax) so `src/t7` imports cleanly
on the GB10 OpenVLA stack. **Commit:** `chore: scaffold t7 package + uv dev env`

---

## Task 1: Reproducibility infrastructure (`src/t7/repro/`)

**Files:**
- Create: `src/t7/repro/seeds.py`, `src/t7/repro/env_capture.py`, `src/t7/repro/provenance.py`, `src/t7/repro/run_logger.py`
- Test: `tests/t7/repro/test_seeds.py`, `test_env_capture.py`, `test_provenance.py`, `test_run_logger.py`

**What:** The non-negotiable reproducibility layer (CLAUDE.md). Pure Python; torch is optional/soft-imported.

**Interface:**
- `seed_everything(seed: int) -> dict` — seeds python/`random`/`numpy` (+ torch if importable); returns the
  applied seed state for logging.
- `capture_env() -> dict` — platform, python version, `pip freeze`/uv lock hash, git commit, and CUDA/driver/
  torch versions **if present** (gracefully `null` on M1).
- `record_provenance(manifest_path, *, name, source, sha256, date, licence) -> None` — append/update a
  provenance entry (mirrors `docs/references/README.md` schema); `sha256_file(path) -> str` helper.
- `RunLogger(results_root)` → `start(slug, config: dict, seed: int) -> RunHandle` creates
  `results/<UTC-timestamp>-<slug>/` (raises if it exists), writes `run.json` (the Playbook §8 protocol block:
  run_id, git_commit, hardware, config snapshot, seeds, hypothesis, command), and exposes
  `handle.write(name, obj)` / `handle.write_array(name, np.ndarray)` that **refuse to overwrite**.

**Test scenarios:** seeding twice with same seed → identical numpy draws; `capture_env` returns dict with
required keys and `cuda=None` locally; provenance round-trips and computes a known SHA-256; `RunLogger`
creates a fresh dir, and a second `start` with a colliding dir **raises**; `handle.write` to an existing file
**raises**; tests use `tmp_path` (never real `results/`).

**Dependencies:** numpy. **Notes:** `RunLogger` is the gatekeeper for invariant #5. **Commit:** `feat: add reproducibility infra (seeds, env capture, provenance, write-once logger)`

---

## Task 2: Core data contracts (`src/t7/records.py`)

**Files:** Create `src/t7/records.py`; Test `tests/t7/test_records.py`

**What:** Immutable typed records shared across metric/detector/eval (Phase-3 §4), so later tasks depend on a
stable contract rather than dicts.

**Interface (frozen dataclasses):**
- `RolloutStep` — `run_id, seed, git_commit, suite, task_id, step:int, observation_ref:str, action:tuple[float*7], privileged_state:dict, instruction:str, trusted_goal:str, attacked:bool, suffix_ref:str|None`
- `Rollout` — ordered `steps: tuple[RolloutStep, ...]` + helpers `prefix_window(t, k) -> tuple[RolloutStep,...]` (causal: indices `t-k+1..t`, clamped at 0), `actions() -> np.ndarray`.
- `TargetActionSpec` (D2) — low-level action region: `dims, low[7], high[7], persistence_steps:int`; `reached(action) -> bool`, `reached_window(actions, persistence) -> bool` (window-scored, not single action).
- `Score` (`value:float in [0,1]`, higher = more inconsistent, `window_end:int`) and `Decision` (`hold:bool, step:int`).

**Test scenarios:** records are immutable (mutating raises); `prefix_window` is causal & clamped at the start
(t<k returns the available prefix, never future); `reached_window` requires `persistence` consecutive hits;
malformed action length is rejected (validation at boundary).

**Dependencies:** numpy. **Notes:** keep these tiny and dependency-light. **Commit:** `feat: add immutable t7 data records (rollout, target-action spec, score/decision)`

---

## Task 3: Action codec — token → 7-DoF (`src/t7/policy/action_codec.py`)

**Files:**
- Create: `src/t7/policy/action_codec.py`, `scripts/fetch_openvla_stats.py`
- Create (data, gitignored): `data/openvla/dataset_statistics.json` (fetched, not committed; provenance in `docs/references/README.md`)
- Test: `tests/t7/policy/test_action_codec.py`

**What:** The one "policy" piece doable offline: de-tokenise OpenVLA's 256-bin action tokens and
**un-normalise** to continuous 7-DoF using the **real** LIBERO dataset statistics (q01/q99). No weights needed
— only the small `dataset_statistics.json` and the bin→token mapping, both fetched from the OpenVLA HF repo.

**Interface:**
- `ActionCodec.from_stats(stats: dict, unnorm_key: str)` — build from a fetched stats block (e.g. `libero_spatial_no_noops`).
- `token_to_bin(token_id:int) -> int`, `bin_to_norm(bin:int) -> float` (bin centre in [-1,1]), `unnormalize(norm:np.ndarray) -> np.ndarray` (`0.5*(norm+1)*(q99-q01)+q01`; gripper dim passed through per OpenVLA).
- `decode(token_ids: Sequence[int]) -> np.ndarray[7]` — full pipeline.

**Test scenarios:** round-trip `normalize→bin→unnormalize` recovers the input within one bin half-width;
clipping at q01/q99 bounds; gripper (dim 7) handled per OpenVLA's rule, not q01/q99-normalised; unknown
`unnorm_key` raises. `scripts/fetch_openvla_stats.py` downloads stats + records provenance (source URL, SHA-256, date, licence).

**Dependencies:** numpy; `huggingface_hub` (download only). **Notes:** **fetch** the exact bin↔token formula and
gripper handling from the OpenVLA repo's `ActionTokenizer` / `get_action` — do **not** hardcode from memory
(invariant #8). Mark the formula `[VERIFY vs OpenVLA source]` until checked. **Commit:** `feat: add OpenVLA action codec (bin de-tokenise + un-normalise) with fetched stats`

---

## Task 4: LIBERO state adapter interface + synthetic fixtures (`src/t7/metric/state.py`)

**Files:** Create `src/t7/metric/state.py`, `tests/t7/metric/fixtures_state.py`; Test `tests/t7/metric/test_state.py`

**What:** A **thin, swappable** adapter (Dependency Inversion) that turns a raw env/ground-truth snapshot into
a normalised `PrivilegedState` the metric reasons over — so the scorer never depends on LIBERO's concrete API.
Ships with **synthetic fixtures** now; a concrete LIBERO adapter is wired in Task 9 if the smoke test succeeds,
else deferred to GB10.

**Interface:**
- `PrivilegedState` (frozen) — `ee_pose[7], gripper_open:bool, object_poses:dict[str,np.ndarray], target_region:str|None, ...` (the *normalised* schema, env-agnostic).
- `StateAdapter` (Protocol) — `to_privileged_state(raw) -> PrivilegedState`.
- `SyntheticStateAdapter` — builds `PrivilegedState` from a fixture dict (for tests + metric unit tests).

**Test scenarios:** synthetic fixtures produce well-formed `PrivilegedState`; adapter rejects missing required
keys; immutability holds. **Dependencies:** numpy, `src/t7/records`. **Notes:** the *normalised* schema is the
contract Task 5 freezes against; keep LIBERO specifics out of it. **Commit:** `feat: add privileged-state adapter interface + synthetic fixtures`

---

## Task 5: Consistency metric (A) — **schema freeze** + causal scorer (`src/t7/metric/consistency_a.py`)

**Files:**
- Create: `docs/plans/metric-a-annotation-schema.md` (the **frozen** schema, dated, committed)
- Create: `src/t7/metric/consistency_a.py`
- Test: `tests/t7/metric/test_consistency_a.py` (per-task unit tests)

**What:** THE make-or-break instrument (Playbook §4). Define — **and commit, before any attack output
exists** — the action-semantics annotation schema: the primitives derived from privileged state over a causal
prefix window (e.g. *object-approached*, *gripper open→close transition*, *target-region entered*,
*displacement direction vs goal direction*) and how they combine into an inconsistency score `s∈[0,1]`.

**Interface:**
- `ConsistencyMetricA(schema, k:int)` — `score(step_index:int, rollout: Rollout, trusted_goal: str) -> Score`
  over the causal prefix `a_{t-k+1:t}` (uses each step's `privileged_state`); higher = more inconsistent.
- `score_rollout(rollout, trusted_goal) -> list[Score]` (one causal score per step).
- Internal `extract_semantics(prefix, state) -> Semantics` (the frozen-schema parser) kept separate + unit-tested.

**Test scenarios (synthetic fixtures only — no real attacks):** a benign trajectory consistent with the goal
scores low; a trajectory that approaches the *wrong* object / enters a non-goal region scores high; `k`
controls how many past steps inform the score; causal — score at `t` is identical whether or not future steps
exist; per-task fixtures (one per LIBERO suite shape) each unit-tested. **Ablation hook:** the parser is
testable in isolation so it cannot silently "become the detector".

**Dependencies:** numpy, `src/t7/records`, `src/t7/metric/state`. **Notes:** label (A) as **non-deployable
upper bound** in docstring + schema doc; the non-causal full-window variant is a *separate* method clearly
named `…_monitoring_ceiling`. Commit the schema doc **in the same commit** as the code so the freeze date is
unambiguous. **Commit:** `feat: add consistency metric (A) with frozen annotation schema + causal scorer`

---

## Task 6: Detector + FP-calibration (`src/t7/detector/`)

**Files:** Create `src/t7/detector/calibrate.py`, `src/t7/detector/decide.py`; Test `tests/t7/detector/test_calibrate.py`, `test_decide.py`

**What:** Turn scores into a calibrated hold/allow decision at a chosen benign false-abort budget.

**Interface:**
- `calibrate(benign_scores_calib, *, target_per_rollout_fpr: float, aggregate='per_rollout') -> Threshold`
  — choose `τ` so the **per-rollout** false-abort rate on the calibration split ≈ target (1%, 5%). Per-window
  aggregation supported but auxiliary.
- `rollout_fires(scores: list[Score], τ) -> Decision` — fires when the first step exceeds `τ` (causal); records
  `latency = first_firing_step - attack_onset_step`.
- `decide(score: Score, τ) -> Decision` (single-step).

**Test scenarios:** on synthetic benign score distributions, calibrated `τ` yields per-rollout false-abort ≈
target within tolerance; per-rollout vs per-window give different `τ`; latency is computed correctly
(0 if fires at onset, `None`/never if no fire); the *same* `calibrate` works on arbitrary score arrays (so
baselines can reuse it). **Dependencies:** numpy, scipy. **Notes:** this `calibrate` is the shared fair-comparison
primitive (invariant #4). **Commit:** `feat: add FP-calibrated detector (per-rollout τ, decision, latency)`

---

## Task 7: Eval harness + statistics (`src/t7/eval/`)

**Files:** Create `src/t7/eval/metrics.py`, `src/t7/eval/splits.py`, `src/t7/eval/harness.py`; Test `tests/t7/eval/test_metrics.py`, `test_splits.py`, `test_harness.py`

**What:** The statistics that produce the floor result, computed from logged score arrays (Phase-3 §4 eval).

**Interface:**
- `roc_auc(benign_scores, attacked_scores) -> (fpr, tpr, auc)`.
- `tpr_at_fpr(benign, attacked, fpr_targets=(0.01,0.05)) -> list[OperatingPoint]` with **Wilson** +
  **Clopper-Pearson** CIs on each rate.
- `benign_degradation(success_no_detector, success_with_detector) -> float`; `abort_rate(...)`; `detection_latency_summary(latencies) -> dict`.
- `assert_disjoint(calib_manifest, test_manifest)` — raises if tasks/scenes/seeds overlap.
- `run_condition_matrix(matrix_cfg, rollouts_by_condition, metric, detector) -> ResultsTable` — orchestrates
  scoring → calibration (calib split) → evaluation (test split) → emits a results table + the raw arrays
  `make_figures` consumes. (Rollout *generation* is GB10; this consumes provided/synthetic rollouts.)

**Test scenarios:** AUC = 1.0 on perfectly separated synthetic scores, ≈0.5 on overlapping; `tpr_at_fpr` CI
width shrinks as n grows and Wilson≠normal-approx at small n; `assert_disjoint` raises on a shared seed;
`run_condition_matrix` on synthetic rollouts yields a populated table with no calib↔test leakage.
**Dependencies:** numpy, scipy, scikit-learn. **Notes:** per-rollout is primary; report per-window as auxiliary.
**Commit:** `feat: add eval harness (ROC/AUC, TPR@FPR + CIs, latency, split-disjointness assert)`

---

## Task 8: Baselines (`src/t7/baselines/`)

**Files:** Create `src/t7/baselines/anomaly.py`, `src/t7/baselines/perplexity.py`; Test `tests/t7/baselines/test_anomaly.py`, `test_perplexity.py`

**What:** The two M2 comparison detectors, both run through the **same** `calibrate` (invariant #4).

**Interface:**
- `goal_agnostic_anomaly_score(rollout, *, benign_stats) -> list[Score]` — fully model-free OOD score on the
  action stream (mandatory baseline: shows goal-conditioning beats mere anomaly detection).
- `PerplexityFilter` — interface `score(instruction:str) -> float` with a **`MockPerplexityScorer`** for tests;
  the real LM-perplexity backend is a stub that raises `NotImplementedError("GB10")` so it's swapped in later.

**Test scenarios:** anomaly score is high for out-of-distribution action streams vs benign-stats; perplexity
*interface* + mock calibrate identically through `calibrate`; the real backend stub clearly errors as GB10-only.
**Dependencies:** numpy, `src/t7/detector`. **Notes:** the "text-PF threshold unknowable a priori" point is a
*deployment* argument stated in prose, **not** an in-experiment handicap. **Commit:** `feat: add anomaly + perplexity-filter baselines (shared calibration; perplexity backend stubbed for GB10)`

---

## Task 9: Config system + scripts skeleton + figures (`src/t7/config/`, `configs/`, `scripts/`)

**Files:**
- Create: `src/t7/config/schema.py` (pydantic), `configs/example_m2.yaml`
- Create: `scripts/{run_benign,run_attack,microbench_gcg,calibrate,evaluate,make_figures}.py`
- Test: `tests/t7/config/test_schema.py`, `tests/t7/scripts/test_make_figures.py`

**What:** Pinned-config validation, runnable script skeletons (GPU-dependent ones guard with a clear "requires
GB10" message), and **`make_figures` working end-to-end on synthetic results** (so figures are
script-regenerable from logged arrays — an M2 deliverable).

**Interface:**
- `Config` (pydantic) — `model, env, attack, metric{k}, detector{fpr_targets}, eval{matrix,splits}, seed`;
  `load_config(path) -> Config` validates; `one_variable_diff(cfg_a, cfg_b) -> list[str]` (enforces one-variable discipline).
- `make_figures(results_dir, out_dir)` — ROC curve, TPR@FPR-with-CI bar, score-distribution histogram, ladder-table placeholder — from logged arrays.
- `run_benign`/`run_attack`/`microbench_gcg` — parse config, set up logging via `RunLogger`, then **guard**: if no CUDA/model, print the GB10 requirement and exit non-zero (no silent no-op).

**Test scenarios:** valid YAML loads; missing/oob field raises a clear validation error; `one_variable_diff`
detects exactly the changed field; `make_figures` on a synthetic results dir writes the expected figure files;
GPU scripts exit with the GB10 message when no model is present. **Dependencies:** pydantic, pyyaml, matplotlib,
numpy, `src/t7/eval`, `src/t7/repro`. **Notes:** keep `make_figures` pure-from-logged-data (no recomputation).
**Commit:** `feat: add config schema, script skeletons, and synthetic-data figure generation`

---

## Task 10: Time-boxed state-only LIBERO smoke test (optional, ≤90 min)

**Files:**
- Create: `scripts/libero_state_smoketest.py`, `docs/setup/libero-local-notes.md`
- Conditionally create: `src/t7/metric/state_libero.py` (concrete `StateAdapter`) — **only if the smoke test succeeds**

**What:** Attempt, in the isolated `libero` extra, to install **mujoco + robosuite + LIBERO** and run a
**state-only** env (no rendering, no policy — 8 GB-safe): `reset()`, read ground-truth (object poses, gripper,
EE pose), step with scripted/zero actions, and **dump the real ground-truth schema**. Use it to validate the
Task 4 `PrivilegedState` contract and, if it works, wire a concrete `state_libero.py` adapter.

**Test scenarios (validation, not pytest-gated):** env resets; ground-truth dict contains the fields the
Task 4 schema assumes; if mismatched, **record the real schema** in `libero-local-notes.md` and adjust the
*adapter mapping* (not the metric schema). **Rabbit-hole guard:** hard time-box ~90 min; on install failure,
**stop**, document the blocker, keep synthetic fixtures, and defer the concrete adapter to GB10 — this is a
*validation bonus*, not a blocker for Tasks 0–9. **Dependencies:** `mujoco`, `robosuite`, `libero` (extra).
**Notes:** never let this destabilise the core env. **Commit:** `chore: add state-only LIBERO smoke test + local notes` (+ `feat: add concrete LIBERO state adapter` only if it works).

---

## Task 11: GB10 setup runbook + pinned environment

**Files:**
- Create: `docs/setup/gb10-runbook.md`, `configs/env/requirements-gb10.txt` (or `environment-gb10.yml`)
- Modify: `docs/references/README.md` (add provenance placeholder rows for the OpenVLA-7B LIBERO checkpoints)

**What:** A step-by-step runbook so GB10 day-1 is "clone → create pinned env → download checkpoints (record
hash) → run benign baseline → reproduce RoboGCG → GCG micro-bench → metric-(A) signal check → GO/NO-GO".
Pin OpenVLA + LIBERO + RoboGCG (`github.com/eliotjones1/robogcg`) versions; capture the 4-bit quant config;
provenance placeholders (source/hash/date/licence) to fill on download.

**Test scenarios:** none (doc). **Checklist mirrors** Playbook §6 (D4/D7 micro-bench), §8 (run protocol), and
the M1 GO/NO-GO gate. **Dependencies:** none. **Notes:** versions are **best-effort, fetched from the OpenVLA /
RoboGCG repos and marked `[VERIFY ON GB10]`** — do not invent pins (invariant #8). **Commit:** `docs: add GB10 setup runbook + pinned environment spec`

---

## After the plan: documentation hygiene (do as part of execution, not a code task)

- Log the **author-OK to start M1–M2 scaffolding** (2026-05-31) in Playbook §10 decision log and update §1
  *You-Are-Here* + §7 task ledger (the `Define repo layout` ⬜ item) as work lands.
- As each task completes, tick its line; record any negative/blocker (e.g. LIBERO install failure) honestly.
- Nothing here generates an experimental *result*, so no §9 claims-ledger rows yet — those start at M1 on GB10.

---

## Boundary check (what this plan deliberately does NOT build — guards against scope creep)

- ❌ OpenVLA inference / rollouts / GCG optimisation / benign-baseline + attack reproduction → **GB10**.
- ❌ Metric (B)/(C) deployable detectors → **M4**. ❌ Trusted-reference rung construction → **M3**.
  ❌ Adaptive attack → **M5 (stretch)**. ❌ Final eval matrix sizing (D4) / compute budget (D7) → **M1 micro-bench**.
- Everything above is interface + synthetic-data-tested only where the model would otherwise be required.
