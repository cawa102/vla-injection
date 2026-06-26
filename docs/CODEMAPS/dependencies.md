<!-- Generated: 2026-06-26 (full regen) | Source: pyproject.toml, configs/env/, scripts/ | Token estimate: ~720 -->

# Dependencies — Embodiment Evasion Tax
## Runtime (pyproject, model-free core — Python ≥3.10)
`numpy` `scipy` `scikit-learn` (ROC/AUC) · `pydantic` v2 (frozen config schema) · `pyyaml` ·
`matplotlib` (figures) · `huggingface-hub` (fetch OpenVLA stats). Dev: `pytest` `ruff`.
Lint: ruff (`E,F,I,B,UP`, line 100, py310); types: pyright (`pyrightconfig.json`, canonical).

## Heavy GPU stack — installed on the box, pinned in `configs/env/requirements-gpu.txt`
`torch 2.2.0+cu121` · `transformers 4.40.1` · `flash-attn 2.5.5` · **`bitsandbytes==0.43.1`**
(int8/nf4 surrogate quantization) · **`accelerate==0.30.1`** (both `--no-deps`-pinned — unbounded
versions drift catastrophically on this box, see memory `csb-bitsandbytes-uv-venv-install-trap`).
`[libero]` extra → `mujoco>=3.1`, `robosuite>=1.4`. Install into the uv venv `~/vla-injection/.venv`
with `uv pip install` (bare `pip` hits conda base).

## External services / artifacts (out-of-process)
- **OpenVLA-7B** checkpoint `openvla/openvla-7b-finetuned-libero-spatial` (HuggingFace) — bf16 victim /
  int8·nf4 surrogate; provenance pinned. `--unnorm-key libero_spatial`.
- **LIBERO** simulation (MuJoCo/robosuite) — full rollouts GPU only; **state-only LIBERO runs locally**
  (GL-free/torch-free `ControlEnv`, `PYTHONPATH=~/LIBERO`) → `LiberoStateAdapter` tested vs frozen
  real-obs fixtures (recipe `docs/setup/libero-local-env.md`; disposable env pins **robosuite 1.4.0**).
- **CSB `ecs3-0202`** = 2× RTX A5000 24 GB (registered compute, ACTIVE — the surrogate-GCG transfer runs
  here). **Kelvin2** A100/H100 (NI-HPC@QUB) = registered backup only. Reached via VS Code tunnel — long
  unattended runs must detach (`nohup setsid`/tmux), the tunnel drop otherwise kills the job.

## CLI entry points (`scripts/`) — GPU-marked ones exit non-zero without CUDA (no silent no-op)
```
GPU   run_surrogate_gcg.py          optimize RoboGCG suffix on int8/nf4/bf16 surrogate → SurrogateSuffixArtifact
GPU   evaluate_surrogate_transfer.py bf16-victim re-eval of a quarantined suffix       → TransferEvalRecord
local summarize_surrogate_transfer.py aggregate transfer records (ASR · gap · cost · censoring)
GPU   microbench_gcg.py             GCG timing micro-bench (D4/D7/D8 compute-branch select)
GPU   smoke_quantized_backward.py   on-box bnb load + gradient-health smoke (concern 1/2/3)
GPU   smoke_gcg_tiny / smoke_openvla_{gradient,load} / smoke_libero_episode   targeted GPU smokes
GPU   run_benign.py / run_attack.py benign baseline / RoboGCG targeted redirect rollouts
GPU   bench_early_stop.py           early-stop steps-to-success bench   GPU fetch_openvla_stats.py  stats+provenance
local calibrate.py · evaluate.py · make_figures.py        tau / condition-matrix→results.json / figures
local m1_gate_report.py · repin_schema.py · attach_l2_to_rollout.py   M1 verdict / DM-3 re-pin / L2-on-rollout
local demo_rollout.py / demo_metric_separation.py / demo_figures.py   end-to-end model-free demo (→results/_demo)
      libero_state_smoketest.py     Tier-L state-only LIBERO smoke (real BDDL; falls back to Tier-R / SKIP)
```
`scripts/_bootstrap.py` puts `src/` on `sys.path` (uv editable `.pth` unreliable on this machine).
