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

You asked to "run LIBERO sim locally." Two pieces of the real experiment **cannot** run on
this M1 box; the demo swaps each for a faithful local stand-in:

| Real experiment (GPU node) | Local demo stand-in | Why the swap |
|---|---|---|
| **OpenVLA-7B** emits the 7-DoF action | **placeholder policy** (`scripted` reach / seeded `random`) | OpenVLA-7B needs CUDA + bf16; absent on an 8 GB Mac |
| **LIBERO** simulates the rollout | **robosuite** `Lift`/`Panda`, state-only | Real LIBERO is **blocked locally** (its `benchmark` hard-imports `torch`; `OffScreenRenderEnv` needs a GL context) — see [`libero-local-notes.md`](./libero-local-notes.md). robosuite is the **identical MuJoCo substrate LIBERO sits on**; LIBERO returns a robosuite obs dict, so the ground-truth extraction is the same. |

**The records produced are structurally identical** to the GPU run's. Only three things
differ on the GPU node, all swappable behind seams the code already has:
1. action source: placeholder policy → OpenVLA-7B,
2. env: `robosuite/Lift` → a LIBERO suite/task (`suite`, `task_id`, `observation_ref` strings change),
3. `target_region`: demo points it at the `cube` manipuland → LIBERO reads it from the **BDDL goal predicate**.

If robosuite is not installed, the demo auto-falls back to
[`SyntheticDynamics`](../../src/evasion_tax/attack/dynamics.py) (pure NumPy, no simulator)
so it **always** emits records — just without real MuJoCo ground truth.

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
SMOKE=~/.cache/evasion_tax-libero-smoke
cd /Users/kawaikyousuke/Desktop/MSc/indivisual

# Tier R — real MuJoCo (robosuite) ground truth, scripted placeholder policy:
PYTHONPATH=src "$SMOKE/venv/bin/python" scripts/demo_rollout.py --steps 12 --seed 0

# variants:
PYTHONPATH=src "$SMOKE/venv/bin/python" scripts/demo_rollout.py --policy random --seed 1
PYTHONPATH=src "$SMOKE/venv/bin/python" scripts/demo_rollout.py --no-sim   # synthetic fallback
```

Flags: `--steps N` (rollout length, ≥4), `--seed S` (pinned), `--policy {scripted,random}`,
`--no-sim` (force the SyntheticDynamics fallback), `--results-root DIR` (default `results/_demo`).

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

# real MuJoCo ground truth (the eval layer adds scipy+sklearn+matplotlib to the sim venv):
"$SMOKE/venv/bin/python" -m pip install "scipy>=1.11" "scikit-learn>=1.3" "matplotlib>=3.8"
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

## 4. Where this sits in the real pipeline

```
[ policy ] --actions--> [ dynamics/env ] --rollout--> [ RolloutStep records ] --> [ metric A ] --> [ detector ] --> [ eval ]
   ^OpenVLA (GPU)            ^LIBERO (GPU)                 ^THIS DEMO produces these          (already unit-tested, model-free)
   demo: placeholder         demo: robosuite
```

This demo exercises the **left half** (policy → env → records). The right half — metric (A)
scoring, FP-calibrated detection, ROC/AUC + TPR@FPR evaluation — already runs model-free and
is unit-tested (`tests/`); it consumes exactly these `Rollout` records.

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

The only code that materialises on the GPU node is the two seam implementations this demo
stands in for: the **OpenVLA policy** and `RealDynamics`/the **concrete LIBERO `StateAdapter`**
(`target_region` from the BDDL, object-name mapping). The records they feed are the ones above.

---

## 6. Cleanup

The isolated env and demo outputs are disposable:
```bash
rm -rf ~/.cache/evasion_tax-libero-smoke     # the sim venv
rm -rf results/_demo                          # git-ignored demo runs
```
Nothing in the repo or the model-free test suite depends on either.
