# Benign Task-Success at int8/nf4 — Precision Sanity Measurement

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Measure benign (unattacked) LIBERO task success at `int8` and `nf4_4bit`, matched to the existing bf16 baseline, so the "quantization breaks the GCG gradient" finding can be separated from "the quantized policy is just degraded."

**Architecture:** `run_benign.py` currently hardcodes a bf16 model load in `_build_episode_fn` and ignores the config's `model.quantization` field. Wire it to the existing, already-tested precision loader (`evasion_tax.attack.openvla_loader.load_openvla_with_attn_fallback`, which supports `bf16 | int8 | nf4_4bit` via `BitsAndBytesConfig`), record the precision in `run.json`, add two derived configs, then run three small matched benign evals (bf16 control / int8 / nf4) under the GPU gate and compare success rates.

**Tech Stack:** Python 3.11, PyTorch (bf16), Transformers + bitsandbytes `BitsAndBytesConfig`, OpenVLA-7B (LIBERO-spatial finetune), LIBERO, CSB A5000 (24 GB, one card).

---

## Context the implementer needs (zero-context primer)

- **The gap this closes.** The only benign success number on record is **0.77 (n=300, libero_spatial)** and it is a **bf16** run (the `_build_episode_fn` load hardcodes `torch_dtype=torch.bfloat16` and sets `load_in_8bit=False, load_in_4bit=False`). No benign success exists at int8/nf4. Without it, the precision-sweep claim (int8/nf4 gradient ranks candidates no better than random) cannot be separated from "the quantized forward policy is just broken."
- **Why it's cheap.** Forward inference at int8/nf4 is already proven on the box (`scripts/smoke_quantized_backward.py` loads + forwards + backwards all three arms). Only the *closed-loop LIBERO rollout* at those precisions is unmeasured — which is exactly this measurement.
- **The loader already exists and is tested.** `src/evasion_tax/attack/openvla_loader.py::load_openvla_policy` / `load_openvla_with_attn_fallback` take `precision: "bf16" | "int8" | "nf4_4bit"`, build the correct `BitsAndBytesConfig`, freeze + eval the model, and return `(model, processor, OpenVlaLoadRecord)`. `_check_precision` validates the three allowed values. **Reuse it — do not write a second load path.**
- **The config already has the field.** `ModelConfig.quantization: str | None = None` (`src/evasion_tax/config/schema.py:38`); `configs/m1_viability.yaml` sets `quantization: bf16`. It is currently parsed but never consumed by `run_benign.py`.
- **Match the registered baseline exactly (one variable at a time).** The 0.77 run used the `m1_viability.yaml` shape: `unnorm_key: libero_spatial` (box-verified, **not** `*_no_noops`), `suite: libero_spatial`, `max_steps: 220`, `checkpoint: openvla/openvla-7b-finetuned-libero-spatial`, `seed` from config. The only variable that may change between the three new runs is `model.quantization`.
- **GPU gate (mandatory, per CLAUDE.md).** Tasks 1–2 are code/config only and run off-GPU. **Task 3 launches GPU rollouts — STOP and get explicit user approval before running it**, stating card, precision arms, episode count, and expected duration.

---

- [x] Task 1: Precision-aware benign model load + reproducibility record

**Files:**
- Modify: `scripts/run_benign.py` — `_build_episode_fn` (the inline bf16 load) and `_run` (the `run.json` config dict); add a pure `resolve_precision` helper near the other pure glue (top of file, alongside `assign_calibration` / `aggregate_benign`).
- Modify (maybe): add `--precision` optional CLI arg to the existing `argparse` block (override for `config.model.quantization`).
- Test: `tests/scripts/test_run_benign.py` (extend the existing pure-glue tests; if none, create it).

**What:** Stop hardcoding bf16. Resolve the precision from `--precision` (CLI override) → else `config.model.quantization` → else `"bf16"`, validate it, load via the shared loader, thread the quant flags into the OpenVLA `cfg`, and stamp the resolved precision into `run.json`.

**Interface:**
- `resolve_precision(config, cli_override: str | None) -> OpenVlaPrecision` — returns one of `"bf16" | "int8" | "nf4_4bit"`; precedence CLI > `config.model.quantization` > `"bf16"`; delegates validation to `openvla_loader._check_precision` (raises `ValueError` on an unknown value). Pure, off-GPU testable.
- In `_build_episode_fn` (GPU body): replace
  `AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, ...).to(device)` + the separate `AutoProcessor.from_pretrained(...)`
  with `model, processor, load_record = load_openvla_with_attn_fallback(torch, model_id, torch.device(args.device), args.attn_impl, precision=precision)`.
- Set the OpenVLA `cfg` flags to match the loaded precision: `load_in_8bit = (precision == "int8")`, `load_in_4bit = (precision == "nf4_4bit")` (they default to `False`; keep `center_crop=True`, `unnorm_key`, `task_suite_name` unchanged).

**Test scenarios (pure helper only — the GPU body stays `# pragma: no cover`):**
- `quantization` unset and no CLI override → returns `"bf16"` (preserves the current baseline exactly).
- `config.model.quantization = "int8"`, no override → returns `"int8"`.
- CLI override `"nf4_4bit"` beats `config.model.quantization = "bf16"` → returns `"nf4_4bit"`.
- Unknown value (`"fp8"`, config or CLI) → raises `ValueError`.

**Dependencies:** `evasion_tax.attack.openvla_loader` (`load_openvla_with_attn_fallback`, `OpenVlaPrecision`, `_check_precision`), `evasion_tax.config` (already imported).

**Notes:**
- **DRY / surgical:** reuse the existing loader; the `bf16` default must reproduce today's load byte-for-byte (loader's bf16 path does `.to(device)` + `requires_grad_(False)` + `eval()`, same as the inline code). Do not touch the pure aggregate/split/resume glue.
- **Gotcha (verify on box):** OpenVLA's `get_action(cfg, model, ...)` receives `cfg`; setting `cfg.load_in_8bit/4bit` consistently avoids any dtype-cast surprise inside `get_vla_action`. `smoke_quantized_backward.py` already proves the quantized forward works, so this is a consistency guard, not new risk.
- Keep the `load_record` (an `OpenVlaLoadRecord`) — Task 2 writes it (or its `precision`) into `run.json`.

**Commit:** `feat(benign): precision-aware model load in run_benign (reuse openvla_loader)`

---

- [x] Task 2: Stamp precision into the write-once run header

**Files:**
- Modify: `scripts/run_benign.py` — the `run.json` config dict in `_run` (currently `{"model", "suite", "n_benign", "calib_frac", ...}`).

**What:** Close the reproducibility nit the sanity check exposed: the existing 0.77 `run.json` records no precision. Add `"quantization": precision` (the resolved value) to the `config` block written on first launch, and optionally the full `dataclasses.asdict(load_record)` under a `"openvla_load"` key.

**Interface:** `run.json["config"]["quantization"] == precision` for every benign run going forward.

**Test scenarios:** covered indirectly (the header write is GPU-body glue); a light assertion in the existing `_run` header test if one exists — otherwise no new test (YAGNI).

**Notes:** Write-once — this only affects **new** runs; it does not rewrite the existing `results/m1-benign-baseline/`. Do not backfill the old header.

**Commit:** `feat(benign): record model precision in run.json header`

---

- [ ] Task 3: int8 + nf4 benign configs and the matched run set (GPU-GATED)
      _(configs `m1_benign_int8.yaml` / `m1_benign_nf4.yaml` created + validated off-GPU:
      each differs from `m1_viability.yaml` in exactly `model.quantization`. The three
      matched GPU runs + `m1-benign-precision-compare.md` remain pending explicit GPU approval.)_

**Files:**
- Create: `configs/m1_benign_int8.yaml`
- Create: `configs/m1_benign_nf4.yaml`
- Create (output, write-once, gitignored heavy artifacts): `results/m1-benign-bf16-n50/`, `results/m1-benign-int8-n50/`, `results/m1-benign-nf4-n50/`
- Create: `results/m1-benign-precision-compare.md` (the three-line comparison + interpretation)

**What:** Derive two configs from `m1_viability.yaml` changing **only** `model.quantization`, then run three matched benign evals (bf16 control + int8 + nf4) at a small n and identical seed, and record the success-rate comparison.

**Config contract** — `configs/m1_benign_int8.yaml` (nf4 identical except the one line):

```yaml
# Derived from m1_viability.yaml — ONLY model.quantization changed (one variable).
model:
  name: openvla-7b
  unnorm_key: libero_spatial          # box-verified finetuned key (NOT *_no_noops)
  checkpoint: openvla/openvla-7b-finetuned-libero-spatial
  quantization: int8                  # nf4 config: nf4_4bit
env:
  suite: libero_spatial
  tasks: [task_0, task_1, task_2]     # informational; run_benign loops all suite tasks
  max_steps: 220                      # match the registered bf16 baseline
seed: 42
```

**Run commands** (each writes-once to its own `results/<run>/`; `--resume` makes restarts idempotent):

```bash
# bf16 matched-n control  (n=50, seed 42 — the fair comparator, NOT the n=300 0.77 run)
PYTHONPATH=src:~/LIBERO python scripts/run_benign.py \
  --config configs/m1_viability.yaml --n-benign 50 --calib-frac 0.0 \
  --openvla-root ~/openvla --results-root results --run-name m1-benign-bf16-n50 \
  --attn-impl flash_attention_2 --device cuda:0 --resume
# int8
PYTHONPATH=src:~/LIBERO python scripts/run_benign.py \
  --config configs/m1_benign_int8.yaml --n-benign 50 --calib-frac 0.0 \
  --openvla-root ~/openvla --results-root results --run-name m1-benign-int8-n50 \
  --attn-impl flash_attention_2 --device cuda:0 --resume
# nf4
PYTHONPATH=src:~/LIBERO python scripts/run_benign.py \
  --config configs/m1_benign_nf4.yaml --n-benign 50 --calib-frac 0.0 \
  --openvla-root ~/openvla --results-root results --run-name m1-benign-nf4-n50 \
  --attn-impl flash_attention_2 --device cuda:0 --resume
```

**Test scenarios (acceptance, on box):**
- Each run completes with **0 errored episodes** and writes `benign_summary.json` (`success_rate`, `n`) + `run.json` with `config.quantization` set correctly.
- Peak VRAM per arm fits one A5000 (bf16 ~14.5 GiB; int8 lower; nf4 lowest — cross-check against the surrogate peaks 18.24/17.02/12.74 GiB which include GCG state, so benign should be ≤ those).
- `m1-benign-precision-compare.md` records the three success rates side by side and states the interpretation: does int8/nf4 benign success stay near bf16 (⇒ the bad *gradient* is precision-specific, the forward policy is intact ⇒ the sweep claim holds) or collapse (⇒ generally-degraded model, soften the "unintended robustness" reading).

**Notes:**
- **GPU GATE — do not launch without explicit user approval.** State: card `cuda:0` (confirm free first, GPU-1 reserved), three arms × 50 episodes, est. wall from the benign rate (~200–220 steps/episode; the 300-episode bf16 baseline is the timing anchor) — roughly `3 × 50 = 150` episodes total; give the user a concrete ETA before running.
- **n choice:** 50 gives a ±~0.07 CI on the rate — enough to catch gross degradation, which is all this needs. If the supervisor wants a tighter comparison to 0.77, bump to `--n-benign 100`; keep seed 42 across all three regardless.
- **One variable at a time:** all three runs share config, seed, suite, `max_steps`, `unnorm_key`, `attn_impl`; only `quantization` differs.
- Heavy per-episode `results/<run>/episodes/` artifacts stay gitignored; commit only the `benign_summary.json`, `run.json`, and `m1-benign-precision-compare.md`.

**Commit:** `results(benign): int8/nf4 vs bf16 benign success at n=50 (precision sanity)`

---

## Follow-through (out of this plan's scope, note for later)

- If int8/nf4 benign success is materially below bf16, update the surrogate-sweep memory + the goal-action-consistency doc to **soften** the "quantization breaks ranking, distinct from a dead gradient" claim (it becomes confounded by forward degradation).
- The precision claim is still `n=1` on the *attack* side (task_0, one seed) — confirming across tasks/seeds is separate future work, not this sanity check.
