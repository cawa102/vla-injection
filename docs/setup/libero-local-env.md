# Local state-only LIBERO env (reproducible recipe)

> **What this is.** The exact recipe that makes a **state-only** LIBERO env run on the
> local mac (headless, 8 GB) — **GL-free and torch-free** — so the `LiberoStateAdapter`
> and its fixtures can be built/validated against **real** LIBERO ground truth before the
> GPU node. Established 2026-06-09; overturns the 2026-06-03 "Tier L blocked / deferred to
> GPU" conclusion (that was env-assembly, not a fundamental block — see
> [`libero-local-notes.md`](./libero-local-notes.md)).
>
> **This is a disposable local validation env, NOT the production stack.** The GPU node
> still installs LIBERO's full pinned stack (Py3.10 / OpenVLA / GL) and **re-validates**
> the adapter there (gripper threshold, object naming, relative-key filter across all
> suites). Per the repro rules, the local adapter is a draft re-checked on the GPU node.

## Why it works (the two blockers, bypassed)

LIBERO's *easy* entry points are what blocked Tier L locally:
- `libero.libero.benchmark.__init__` **hard-imports `torch`** (only needed to enumerate
  the task registry).
- `libero.libero.envs.OffScreenRenderEnv` **always opens an offscreen GL context**.

Both are avoidable: `libero.libero.envs` itself is **torch-free**, and its lower-level
`ControlEnv` accepts `has_offscreen_renderer=False, use_camera_obs=False` → a pure-state
env with **no GL and no torch**. Locate the BDDL straight from the package
(`os.path.dirname(libero.libero.__file__)/bddl_files`), skipping `benchmark` entirely.

## Recipe

```bash
# 1. Py3.10 venv (matches production major.minor) — robosuite 1.4.0 needs <numpy 1.24
uv venv --python python3.10 ~/.cache/evasion_tax-libero14/venv
uv pip install --python ~/.cache/evasion_tax-libero14/venv/bin/python \
    "robosuite==1.4.0" "numpy<1.24" pyyaml easydict bddl \
    termcolor scipy pillow opencv-python pynput h5py matplotlib cloudpickle "gym==0.25.2"

# 2. LIBERO source (this project's own clone, complete with bddl_files)
#    on PYTHONPATH — its uv editable .pth is unreliable on this host, so use PYTHONPATH.
#    Clone: ~/.cache/t7-libero-smoke/LIBERO  (Lifelong-Robot-Learning/LIBERO)
```

Resolved versions (2026-06-09): `robosuite 1.4.0`, `mujoco 3.9.0`, `numpy 1.23.5`,
`bddl 3.6.0`, `easydict 1.13`, `pyyaml 6.0.3`, Python 3.10.8.

> **Why robosuite 1.4.0, not 1.5.2.** LIBERO subclasses
> `robosuite.environments.manipulation.single_arm_env.SingleArmEnv`, a module **removed
> in robosuite 1.5** — so the local sim venv's robosuite 1.5.2 cannot import LIBERO. The
> 1.4.0 pin is the real, structural requirement.

## Run

```bash
# state-only Tier-L smoke (resets a real LIBERO env, builds PrivilegedState):
PYTHONPATH=~/.cache/t7-libero-smoke/LIBERO \
  ~/.cache/evasion_tax-libero14/venv/bin/python scripts/libero_state_smoketest.py

# re-capture the frozen test fixtures (see tests/evasion_tax/metric/fixtures/PROVENANCE.md):
PYTHONPATH=~/.cache/t7-libero-smoke/LIBERO \
  ~/.cache/evasion_tax-libero14/venv/bin/python /tmp/libero_capture_fixtures.py
```

`MUJOCO_GL=disable` is set by the scripts (state-only, never request GL).

## Cleanup

Disposable: `rm -rf ~/.cache/evasion_tax-libero14` removes the env. Nothing in the repo
depends on it at run time — `state_libero.py` imports no LIBERO, and its unit tests read
the frozen JSON fixtures, so the core `.venv` suite is unaffected.
