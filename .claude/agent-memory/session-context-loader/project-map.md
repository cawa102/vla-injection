---
name: project-map
description: T7 VLA-security dissertation — directory layout, designated source-of-truth docs, key modules, and conventions for fast onboarding
metadata:
  type: project
---

MSc Cyber Security & AI individual dissertation on VLA-model security (integrity focus). Codename **T7**, headline *The Embodiment Evasion Tax*. Single-author, graded; reproducibility > speed.

**Designated sources of truth (read at session start, in order):**
- Operational SoT = `docs/core/t7-execution-playbook.md` — status (§1 You-Are-Here), branched roadmap (§2), hypothesis register (§3, H6-A/H6-D split), reframe (§3a), what-to-implement (§4b). Living doc; update as work proceeds (§11 Session Protocol).
- Active task tracker = `docs/core/t7-local-prep-plan.md` §"Status (live)" table — the model-free pre-GPU build (Tasks 0-11).
- Understanding / why = `docs/core/t7-goal-action-consistency-detector.md`.
- Verified external facts / citations = `docs/references/README.md` (17 PDFs, gitignored, SHA-256 pinned).
- Landscape = `docs/lit-review/`. Theme scoping = `docs/theme-scoping-report.md`.

**Repo layout:**
- `src/t7/` — implementation, organised by feature: `policy/` (action_codec, openvla_stats), `metric/` (state, consistency_a = metric A / L2-oracle), `detector/` (calibrate, decide), `eval/` (harness, metrics, splits, figures), `baselines/` (anomaly χ²-OOD, perplexity = L0), `config/` (schema, runtime GPU guard), `repro/` (seeds, env_capture, provenance, run_logger), `records.py`. ~2700 LoC.
- `tests/t7/` — mirrors src; TDD; ~234 test functions.
- `scripts/` — 6 runnable entry points (calibrate, evaluate, make_figures, run_attack, run_benign, microbench_gcg, fetch_openvla_stats) + `_bootstrap.py`.
- `configs/` — `example_m2.yaml` (frozen pydantic Config; precision = bf16).
- `data/`, `artifacts/` — gitignored (datasets/checkpoints; adversarial artefacts quarantined under `artifacts/untrusted/`). `results/` = write-once run logs (not created yet — no runs).
- `docs/presentation/` — slide deck + speaker script.

**Conventions:** commit `<type>: <description>`, no co-author trailer, on `main`, push to `origin/main` (`github.com/cawa102/vla-injection`, private). Plan-status markers: ✅ done / ⬜ not-started; decisions tagged DECIDED / PROPOSED / OPEN. Phase order: Scope → Lit review → Design → Implement → Run & analyse → Write up. Base model OpenVLA-7B bf16, simulation-only (LIBERO), instruction channel (RoboGCG primary).

See [[status-snapshot]] for plan-vs-code state.
