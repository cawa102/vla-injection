<!-- Generated: 2026-06-05 | Source: pyproject.toml, configs/env/, scripts/ | Token estimate: ~600 -->

# Dependencies — Embodiment Evasion Tax
## Runtime (pyproject, model-free core — Python ≥3.10)
`numpy` `scipy` `scikit-learn` (ROC/AUC) · `pydantic` v2 (frozen config schema) · `pyyaml` ·
`matplotlib` (figures) · `huggingface-hub` (fetch OpenVLA stats). Dev: `pytest` `ruff`.
Lint: ruff (`E,F,I,B,UP`, line 100, py310); types: pyright (`pyrightconfig.json`, canonical).

## Heavy stack — isolated optional extras (never break core env)
`[libero]` → `mujoco>=3.1`, `robosuite>=1.4`. `libero` itself + the OpenVLA stack are installed from
source on the GPU node — pins in `configs/env/requirements-gpu.txt`; runbook `docs/setup/gpu-runbook.md`.

## External services / artifacts (out-of-process)
- **OpenVLA-7B** checkpoint (HuggingFace) — bf16, GPU only; provenance pinned.
- **LIBERO** simulation (MuJoCo/robosuite) — real rollouts GPU only; local state-only Tier-R smoke
  validated `PrivilegedState` (`docs/setup/libero-local-notes.md`).
- **Kelvin2** HPC (NI-HPC@QUB), A100/H100 nodes — `docs/gpu/`.

## CLI entry points (`scripts/`) — GPU-marked ones exit non-zero without CUDA (no silent no-op)
```
GPU   run_benign.py        benign LIBERO baseline           → evasion_tax.config (load_config, cuda guard)
GPU   run_attack.py        RoboGCG targeted redirect        → evasion_tax.config
GPU   microbench_gcg.py    GCG timing micro-bench (D4/D7/D8) → evasion_tax.config
GPU   fetch_openvla_stats  download stats + provenance       → evasion_tax.policy.openvla_stats
local calibrate.py         tau from logged benign scores     → evasion_tax.detector.calibrate
local evaluate.py          condition matrix → write-once results.json → evasion_tax.eval.harness, RunLogger
local make_figures.py      regenerate M2 figures from results.json    → evasion_tax.eval.figures
      libero_state_smoketest.py   Tier-R privileged-state smoke (no torch)
```
`scripts/_bootstrap.py` puts `src/` on `sys.path` (uv editable `.pth` unreliable on this machine).
