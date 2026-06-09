# Local rollout-recording demo — manual

> **What this is.** A laptop-runnable demo of the **rollout-recording pipeline** so you
> can *see*, before GPU day-1, exactly what records the real experiment logs — without
> OpenVLA and without a GPU. Script: [`../../scripts/demo_rollout.py`](../../scripts/demo_rollout.py).
>
> **What this is *not*.** Not an experiment, not a result. The action source is a
> transparent **placeholder policy**, not OpenVLA, so its output goes to the git-ignored
> `results/_demo/`, never the write-once real `results/`.

---

## 0. The honest substitution (read first)

You asked to "run LIBERO sim locally." On the GPU node two pieces drive the experiment —
**OpenVLA-7B** (the policy) and **LIBERO** (the simulator). As of **2026-06-09 a state-only
LIBERO env runs on this M1 box** (GL-free + torch-free — see
[`libero-local-env.md`](./libero-local-env.md)), so the simulator **no longer needs a
stand-in**; only the policy does:

| Real experiment (GPU node) | Local demo stand-in | Why the swap |
|---|---|---|
| **OpenVLA-7B** emits the 7-DoF action | **placeholder policy** (`scripted` reach / seeded `random`) | OpenVLA-7B needs CUDA + bf16; absent on an 8 GB Mac |
| **LIBERO** simulates the rollout | **real LIBERO**, state-only (`--backend libero`) — *or* the lighter **robosuite** `Lift`/`Panda` stand-in (`--backend robosuite`) | `--backend libero` is the **real** simulator (state-only `ControlEnv`, real BDDL `target_region`, real object names, scored through the real [`LiberoStateAdapter`](../../src/evasion_tax/metric/state_libero.py)). robosuite remains a faster substitute (the identical MuJoCo substrate LIBERO sits on, same obs-dict extraction) when LIBERO isn't installed. |

**The records produced are structurally identical** to the GPU run's. With `--backend libero`
the **only** GPU-only stand-in left is the policy — so just two things differ on the GPU node:
1. action source: placeholder policy → OpenVLA-7B,
2. attacker suffix: `demo/placeholder-suffix` → a real RoboGCG `suffix_ref`.

(The `robosuite`/`synthetic` backends additionally stand in for the env; with `--backend
libero` the `suite`, `task_id`, `target_region`, and object names are already the real BDDL
ground truth.) If LIBERO is not on `PYTHONPATH`, `--backend robosuite` uses the MuJoCo
substrate; if neither is installed the demo auto-falls back to
[`SyntheticDynamics`](../../src/evasion_tax/attack/dynamics.py) (pure NumPy, no simulator)
so it **always** emits records — just without real MuJoCo/LIBERO ground truth.

---

## 1. One-time setup — isolated sim venv

The sim stack (`mujoco`, `robosuite`) is **never** installed into the core `.venv` (it must
stay stable for the model-free tests). Use a throwaway venv, built with **Python 3.11**
(the system `python3` here is anaconda 3.8 — too old for `numpy>=1.26`):

```bash
SMOKE=~/.cache/evasion_tax-libero-smoke
.venv/bin/python -m venv "$SMOKE/venv"          # 3.11 base, isolated
"$SMOKE/venv/bin/python" -m pip install --upgrade pip
"$SMOKE/venv/bin/python" -m pip install "numpy>=1.26" "mujoco>=3.1" "robosuite>=1.4"
# sanity:
"$SMOKE/venv/bin/python" -c "import numpy,mujoco,robosuite; \
  print(numpy.__version__, mujoco.__version__, robosuite.__version__)"
```

`evasion_tax` itself is **not** pip-installed here — its model-free record classes need only
NumPy, so the demo imports them straight off the source tree via `PYTHONPATH=src` (the
`import _bootstrap` line in the script does the path dance).

---

## 2. Run the demo

```bash
cd /Users/kawaikyousuke/Desktop/MSc/indivisual

# Tier L — REAL state-only LIBERO ground truth (recommended). Needs the libero14 venv
# + the LIBERO clone on PYTHONPATH (see docs/setup/libero-local-env.md):
LIB=~/.cache/t7-libero-smoke/LIBERO
L14=~/.cache/evasion_tax-libero14/venv/bin/python
PYTHONPATH=src:$LIB "$L14" scripts/demo_rollout.py --backend libero --steps 12 --seed 0

# Tier R — robosuite MuJoCo stand-in (lighter; isolated sim venv):
SMOKE=~/.cache/evasion_tax-libero-smoke
PYTHONPATH=src "$SMOKE/venv/bin/python" scripts/demo_rollout.py --backend robosuite --steps 12 --seed 0

# variants:
PYTHONPATH=src:$LIB "$L14" scripts/demo_rollout.py --backend libero --policy random --seed 1
PYTHONPATH=src .venv/bin/python scripts/demo_rollout.py --backend synthetic   # no simulator
```

Flags: `--steps N` (rollout length, ≥4), `--seed S` (pinned), `--policy {scripted,random}`,
`--backend {auto,robosuite,libero,synthetic}` (`auto` = robosuite if importable else synthetic;
`--no-sim` is a deprecated alias for `--backend synthetic`), `--results-root DIR`
(default `results/_demo`).

> With `--backend libero` the env is the **real** simulator: the recorded `suite` is
> `libero_spatial`, `task_id`/`instruction` are the real BDDL task (e.g. *"pick the akita
> black bowl … and place it on the plate"*), `target_region` is the real BDDL goal object
> (`plate_1`), and object names are the real scene objects — with the `_to_robot0_eef_pos`
> relative deltas filtered out by the real adapter. Same seed ⇒ byte-identical records
> (`np.random` is seeded before reset to pin LIBERO's object-placement RNG).

The run is **reproducible**: same `--seed`/`--steps`/`--policy` ⇒ identical actions and records.

---

## 3. What gets recorded (the point of the demo)

Each run creates a **write-once** directory `results/_demo/<UTC-timestamp>-demo-rollout-<policy>/`
via the real [`RunLogger`](../../src/evasion_tax/repro/run_logger.py) — the same logger the GPU
experiment uses. Inside:

| File | Record | What it is |
|---|---|---|
| `run.json` | run protocol block | `run_id`, `git_commit`, full `hardware`/env snapshot ([`capture_env`](../../src/evasion_tax/repro/env_capture.py)), pinned `config`, `seed`, `created_utc`, + §8 protocol placeholders (`hypothesis`/`expected`/`observed`/`decision`/`one_variable`) |
| `rollout_benign.json` | `Rollout` of [`RolloutStep`](../../src/evasion_tax/records.py) | the **benign** trajectory — one record per step (see fields below) |
| `rollout_attacked.json` | `Rollout` | the **attacked** trajectory (actions forced into the attacker's target region) |
| `actions_benign.npy` / `actions_attacked.npy` | `(n_steps, 7)` float arrays | raw 7-DoF actions, for re-analysis |
| `attack_outcome.json` | `TargetActionSpec` window-score | the **D2 success criterion**: `reached_window` (held the region for ≥`persistence_steps`) + `completion_step` |

**Per-step `RolloutStep` fields** (`src/evasion_tax/records.py`):
`run_id, seed, git_commit, suite, task_id, step, observation_ref, action` (7-DoF tuple),
`privileged_state` (the sim ground truth: `ee_pos`, `gripper_open`, `object_poses`,
`target_region`), `instruction`, `trusted_goal`, `attacked` (bool), `suffix_ref`.

The `privileged_state` is the ground truth the **non-deployable metric (A)** reasons over —
on robosuite it's pulled from `robot0_eef_pos`, `robot0_gripper_qpos`, and the `*_pos` object
keys (identical to LIBERO's obs dict).

Inspect them:
```bash
RUN=$(ls -dt results/_demo/*/ | head -1)
cat "$RUN/run.json"
python3 -c "import json; d=json.load(open('$RUN/rollout_benign.json')); \
  print(json.dumps(d['steps'][6], indent=2))"   # one step, fully expanded
cat "$RUN/attack_outcome.json"
```

---

## 3b. H1 dry-run — metric-(A) benign-vs-attacked separation

[`scripts/demo_metric_separation.py`](../../scripts/demo_metric_separation.py) takes the
benign and attacked rollouts, runs them through the **real** non-deployable consistency
metric (A) and the **real** eval statistics (`roc_auc`, `tpr_at_fpr`), and reports the
benign-vs-attacked separation the GPU viability gate (H1) tests for real.

```bash
# zero-setup (synthetic rollouts, runs entirely in the core .venv — has scipy/sklearn):
PYTHONPATH=src .venv/bin/python scripts/demo_metric_separation.py --n 16

# REAL LIBERO ground truth (the eval layer needs sklearn in the libero14 venv — once:
#   uv pip install --python "$L14" "scikit-learn==1.3.2"   # numpy<1.24-compatible pin):
PYTHONPATH=src:$LIB "$L14" scripts/demo_metric_separation.py --backend libero --n 8

# robosuite MuJoCo stand-in (the eval layer adds scipy+sklearn+matplotlib to the sim venv):
PYTHONPATH=src "$SMOKE/venv/bin/python" scripts/demo_metric_separation.py --backend robosuite --n 16
```

It prints score histograms, the **ROC AUC**, and calibrated **TPR@{1%,5%} FPR** operating
points (τ set on a benign calibration split, FPR reported on a *disjoint* held-out split —
invariant #3), and writes `separation_report.json`. Benign and attacked rollouts share the
same per-seed policy jitter; the only difference is the injected redirect (one variable).

> ⚠️ **This is the plumbing of H1, not H1.** The attack is a hand-built action redirect
> (the placeholder policy pushed away from the goal), so a high AUC is **expected by
> construction** and says nothing about RoboGCG. The real H1 runs OpenVLA-7B under RoboGCG
> on LIBERO, where separation is an open empirical question (`gpu-runbook.md` Step 6).
> With small `--n` the calibration split is tiny, so τ is coarse and the held-out FPR can
> overshoot its target — that is the honest small-sample behaviour, not a bug.

**Verified run — real LIBERO, 2026-06-09** (`--backend libero --n 24 --steps 10 --seed 0`,
`libero_spatial` task *"pick the akita black bowl … place it on the plate"*, real BDDL
`target_region=plate_1`): benign per-rollout score 0.149–0.344, attacked 0.810–0.926,
**AUC = 1.000** (by construction), τ = 0.344, **TPR@{1%,5%}FPR = 1.00** (CI [0.86, 1.00]),
**held-out FPR = 0.000** (clean at n=24, vs 0.50 at n=8). Power flag stays on — `n_benign=12`
< the rule-of-three floor (60 for 5%, 300 for 1% FPR) — so the figures honestly mark the
operating points underpowered (invariant #5). Same seed ⇒ byte-identical records.

## 3c. Figure-regeneration dry-run

[`scripts/demo_figures.py`](../../scripts/demo_figures.py) runs the **real M2 figure
pipeline** end to end: demo rollouts → metric (A) → the real eval harness
(`run_condition_matrix`) → the same `results.json` the GPU eval writes
(`results_table_to_dict`) → the same `make_figures`. This is the reproducibility
principle in action — *figures regenerable purely from logged data by a script*.

```bash
# synthetic (core .venv):
PYTHONPATH=src .venv/bin/python scripts/demo_figures.py --n 16
# REAL LIBERO ground truth (libero14 venv with sklearn+matplotlib):
PYTHONPATH=src:$LIB "$L14" scripts/demo_figures.py --backend libero --n 8
# robosuite MuJoCo stand-in (isolated sim venv with scipy+sklearn+matplotlib):
PYTHONPATH=src "$SMOKE/venv/bin/python" scripts/demo_figures.py --backend robosuite --n 16
```

Writes a logged `results.json` plus, under `figures/`, four PNGs per run:
`<cond>_roc.png` (ROC + AUC), `<cond>_score_hist.png` (benign-vs-attacked
distribution), `<cond>_tpr_at_fpr.png` (TPR bars with 95% CIs; a `*` and a title note
flag any point whose held-out benign N is below the rule-of-three power floor —
invariant #5), and `ladder_placeholder.png` (the M3 trusted-reference ladder stub).

On the GPU node the *only* change is the data source: the same `make_figures` renders
the real OpenVLA/RoboGCG/LIBERO `results.json`. Inspect them:
```bash
RUN=$(ls -dt results/_demo/*demo-figures-*/ | head -1); open "$RUN"figures/*.png   # macOS
```

## 4. Where this sits in the real pipeline

```
[ policy ] --actions--> [ dynamics/env ] --rollout--> [ RolloutStep records ] --> [ metric A ] --> [ eval ] --> [ figures ]
   ^OpenVLA (GPU)            ^LIBERO (now local!)          §2/§3                        §3b           §3b          §3c
   demo: placeholder         demo: real LIBERO         (records)                  (separation)   (AUC/FPR)   (PNG regen)
                             (or robosuite stand-in)
```

The three demo scripts now exercise the **whole** model-free pipeline end to end: §2/§3
(`demo_rollout.py`) produces the records; §3b (`demo_metric_separation.py`) scores them with
the real metric (A) and reports AUC/TPR@FPR; §3c (`demo_figures.py`) regenerates the figures
through the real eval harness. With `--backend libero` the env is no longer a stand-in, so
**the only GPU-only piece left is the policy seam** (placeholder → OpenVLA). FP-calibrated
detection (`detector/`) is unit-tested (`tests/`) and consumes exactly these `Rollout` records.

---

## 5. The GPU-node experiment this previews (per [`gpu-runbook.md`](./gpu-runbook.md))

On Kelvin2 (`k2-gpu-a100`/`k2-gpu-h100`) the same record path runs for real:

1. **Step 3 — benign baseline:** OpenVLA-7B rolls out LIBERO tasks → `rollout_benign`-shaped
   records, logged write-once (this is what `scripts/run_benign.py` will produce; it currently
   guards with "requires GPU" off-CUDA).
2. **Step 4 — RoboGCG attack:** a white-box adversarial suffix injected once at rollout start →
   `rollout_attacked` records with a real `suffix_ref`; success is **window-scored** against the
   `TargetActionSpec` exactly as `attack_outcome.json` shows here.
3. **Steps 5–6 — micro-bench + metric-(A) signal:** select compute branch N/N−/F, then check
   benign-vs-attacked separation → the **GO/NO-GO gate (H1)**.

The concrete LIBERO `StateAdapter` (`target_region` from the BDDL, object-name mapping,
`_to_` relative-key filter) is **already built and exercised locally** by `--backend libero`
([`state_libero.py`](../../src/evasion_tax/metric/state_libero.py)) — the GPU node only
**re-validates** it (gripper threshold, object naming across all suites). So the one
implementation that still materialises only on the GPU node is the **OpenVLA policy** (the
action source). The records it feeds are the ones above.

---

## 6. Cleanup

The isolated env and demo outputs are disposable:
```bash
rm -rf ~/.cache/evasion_tax-libero14         # the real-LIBERO state-only venv (--backend libero)
rm -rf ~/.cache/evasion_tax-libero-smoke     # the robosuite sim venv (--backend robosuite)
rm -rf results/_demo                          # git-ignored demo runs
```
Nothing in the repo or the model-free test suite depends on either.
