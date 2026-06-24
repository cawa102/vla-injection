# Quantized-Surrogate RoboGCG Transfer Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Measure whether RoboGCG suffixes optimized on low-resource quantized OpenVLA surrogates transfer to a bf16 OpenVLA victim.

**Architecture:** Reuse the existing OpenVLA + RoboGCG + LIBERO harness. Add only precision-aware model loading, quarantined suffix artifacts, bf16 victim re-evaluation, and logged-result aggregation.

**Tech Stack:** Python 3.10, PyTorch, Transformers, bitsandbytes `BitsAndBytesConfig`, OpenVLA-7B, LIBERO, existing `evasion_tax` reproducibility and write-once logging.

---

## Research Question

Can a low-resource attacker optimize RoboGCG suffixes on quantized OpenVLA surrogates and transfer them to a bf16 OpenVLA victim?

This replaces the current attack-side main plan only at the level of attack measurement. The claim is narrow: **target-action transfer under low-resource surrogate optimization**. It is not a semantic harm claim and not a universal defence or attack claim.

## Claim Boundary

- Primary claim: target-action ASR transfer from surrogate precision `{bf16, int8, nf4_4bit}` to a bf16 victim on the same checkpoint, suite, task, seed policy, and target-action-token definition.
- Secondary claim: cost-normalized transfer, measured as target-action ASR per GPU-hour and steps-to-success under the same GCG budget.
- Out of scope: real-robot transfer, semantic wrong-object success unless separately instrumented, cross-hardware comparisons inside one claim, and any claim that quantization universally improves or weakens attackability.

## Hypotheses

- **H-S1:** 8-bit surrogate-optimized suffixes transfer to the bf16 victim at non-zero target-action ASR on `libero_spatial`.
- **H-S2:** NF4 4-bit surrogate-optimized suffixes transfer less reliably than 8-bit suffixes, but may offer better cost-normalized ASR if optimization runs materially faster or fits more candidate batching.
- **H-S3:** A bf16 direct attack remains the upper-bound baseline for transfer because surrogate/victim precision mismatch introduces an optimization gap.

All three are reportable if falsified.

## Experiment Matrix

Primary suite: `libero_spatial`.

Primary precision arms:

| Arm | Surrogate precision | Victim precision | Purpose |
|---|---|---|---|
| B | bf16 | bf16 | Direct-attack baseline |
| I8 | int8 | bf16 | Low-resource 8-bit surrogate transfer |
| NF4 | nf4_4bit | bf16 | Low-resource 4-bit surrogate transfer |

Secondary transfer check, run only after the primary `libero_spatial` pilot succeeds:

| Suite | Scale | Purpose |
|---|---:|---|
| `libero_object` | small | Different object semantics |
| `libero_goal` | small | Different goal semantics |

Do not mix hardware in one claim. Registered runs use the CSB A5000 hardware unless a separate registration is written.

## GCG Defaults

- `suffix_len=20`
- `top_k=256`
- `search_width=512`
- `n_steps=500`
- `early_stop=True`
- candidate evaluation chunking follows the existing A5000 chunked-loss path.

Change one variable at a time. Any run that changes precision must keep checkpoint, suite, task, seed, GCG budget, target action tokens, attention implementation, and hardware fixed.

## Artifact Schema

Every optimized suffix is quarantined under `artifacts/untrusted/` and recorded as a write-once artifact. No suffix may live only in stdout.

Required fields:

- schema version
- surrogate precision: `bf16`, `int8`, or `nf4_4bit`
- model checkpoint
- suite and task
- target action tokens
- seed
- GCG config: suffix length, steps, search width, top-k, early-stop setting
- suffix token ids and suffix text path
- suffix SHA-256
- source run dir
- git commit
- GPU id
- CUDA, torch, transformers, and bitsandbytes versions when present
- surrogate optimization status, target hit, steps-to-success, runtime, peak VRAM, and censoring flag

## Victim Transfer Record

The bf16 victim evaluator consumes a quarantined suffix artifact and writes a transfer record with:

- artifact reference and suffix SHA-256
- surrogate precision and bf16 victim precision
- checkpoint, suite, task, seed, and target action tokens
- compatibility check result
- target-action hit on the bf16 victim
- predicted victim target-token span
- token-space distance to the target action tokens
- persistence window
- rollout evaluated flag and rollout success/failure when a rollout harness is attached
- runtime, failure reason, and censored flag

The evaluator rejects mismatched checkpoint, suite, or task unless an explicit override is provided.

## Metrics

Primary:

- target-action ASR on the bf16 victim, grouped by surrogate precision

Secondary:

- surrogate ASR
- transfer gap: `ASR_surrogate_eval - ASR_bf16_victim`
- GPU-hour normalized ASR: `ASR_bf16_victim / surrogate_gpu_hours`
- steps-to-success distribution
- censored target fraction
- peak VRAM
- suffix perplexity if a scorer is attached later

Report Wilson confidence intervals for ASR. Censored targets remain in the denominator for ASR and are reported separately in the steps-to-success summary.

## Run Order

1. Dry-validate configs locally.
2. GPU smoke: bf16 direct surrogate load still matches the existing loader behavior.
3. GPU smoke: int8 surrogate loads and can produce the target-token loss.
4. GPU smoke: NF4 4-bit surrogate loads; if backward fails, record the explicit failure reason.
5. Run one tiny GCG target per precision and write suffix artifacts.
6. Evaluate each suffix artifact on the bf16 victim.
7. Aggregate ASR, transfer gap, GPU-hour normalized ASR, and steps-to-success.
8. Only after a successful pilot, update `docs/core/execution-playbook.md` to mark the previous EET attack-side plan as superseded or narrowed.

## Risks

- bitsandbytes quantized backward may fail for the input-embedding gradient path. If so, record the exact exception and treat the precision arm as failed/censored, not silently dropped.
- Quantized surrogate speed may not offset transfer loss. Report cost-normalized ASR, not only raw ASR.
- A target-token hit does not prove semantic task harm. Keep the claim at low-level target-action transfer until a separate semantic evaluator exists.
- Cross-hardware variance can swamp precision effects. Do not combine A5000 and any later A100/H100 results inside one transfer claim.

## Primary Sources

- OpenVLA: <https://arxiv.org/abs/2406.09246>
- RoboGCG: <https://arxiv.org/abs/2506.03350>
- Hugging Face bitsandbytes quantization docs: <https://huggingface.co/docs/transformers/en/quantization/bitsandbytes>
