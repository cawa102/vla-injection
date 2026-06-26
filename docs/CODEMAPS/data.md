<!-- Generated: 2026-06-26 (full regen) | Source: records.py, eval/cross_layer.py, attack/surrogate_artifacts.py | Token estimate: ~780 -->

# Data — Embodiment Evasion Tax
No database. "Data" = frozen-dataclass contracts (`records.py`, `attack/surrogate_artifacts.py`) +
the write-once `results/` logs (gatekept by `repro/run_logger.RunLogger`). Suffix payloads are
quarantined in gitignored `artifacts/untrusted/`; only metrics land in tracked `results/`.

## Core records (all `@dataclass(frozen=True)`, validate at construction — never trust external data)
```
RolloutStep   run_id seed git_commit suite task_id step observation_ref
              action(→7-tuple) privileged_state instruction trusted_goal attacked suffix_ref
Rollout       steps: tuple[RolloutStep,...]        .prefix_window(t,k) causal · .actions()→(n,7)
TargetActionSpec  dims low high persistence_steps  .reached / .reached_window (D2)
Score         value∈[0,1] window_end               higher = more goal-inconsistent
Decision      hold step
```
ACTION_DIM = 7 (dx dy dz droll dpitch dyaw gripper).

## Cross-layer data contract (`eval/cross_layer.py`)
```
UnitKey(task,target,seed)  = bootstrap resampling cluster
UnitOutcome(unit, layer{L0|L1|L2_oracle|...}, tradeoff λ, reached, detected)
   reached  = property of the induced rollout (shared across layers for a unit)
   detected = per-layer calibrated detector fired
frontier_from_outcomes → per-λ (ASR=mean reached, evasion=1−mean detected) → Pareto-filter
```

## Surrogate-GCG transfer records (`attack/surrogate_artifacts.py`, SCHEMA_VERSION=2)
```
GcgConfig / GcgResult (attack/gcg.py)  suffix_len top_k search_width n_steps early_stop / best_suffix_ids best_loss
SurrogateSuffixArtifact  precision suite task target_action_tokens seed gcg_config load_record gpu_id environment
   suffix_token_ids · suffix_path · suffix_sha256        ← attack payload → artifacts/untrusted/ ONLY
   surrogate_gradient_health (dict|None)                  ← record-never-gate diagnostic (grad_absmax,
        grad_nonzero/finite, recommended/random_mean_delta, faithfulness_passed, gate_samples; {"error":..} on raise)
   surrogate_target_hit steps_to_success censored best_loss wall_seconds peak_vram_gib failure_reason
TransferEvalRecord  bf16-victim re-eval: victim_target_hit predicted_target_tokens action_distance_to_target
   surrogate_precision/target_hit (carried) · persistence_window · rollout_evaluated · failure_reason · censored
```
`run_surrogate_gcg._results_pointer` writes metrics (+ `suffix_sha256` pointer, **no suffix**) to
`results/<run_id>/surrogate_suffix_artifact.json`. `censored = not target_hit` (right-censoring; stays in
the ASR denominator), distinct from `failed = failure_reason is not None` (could-not-measure).

## Reproducibility invariants encoded structurally
| # | Invariant | Where |
|---|-----------|-------|
| 1 | Causal windows (no future index) | `Rollout.prefix_window` |
| 3 | Calib/test disjoint; held-out FPR | `eval/splits.assert_disjoint`, `harness` |
| 4 | One shared calibration primitive | `detector/calibrate` (reused by all baselines) |
| 5 | Write-once results | `repro/run_logger.RunLogger`, `surrogate_artifacts.write_json_record` |
| 6 | Immutability | all records `frozen=True` |
| — | Quarantine attack payloads | suffix → `artifacts/untrusted/` (gitignored); metrics → `results/` |

## Provenance
`repro/provenance.record_provenance` (SHA-256 + JSON manifest) for data/checkpoints;
`policy/openvla_stats.record_stats_provenance` for `dataset_statistics.json`; each artifact embeds
`environment` (full pip freeze + git_commit + driver) so a run is self-documenting. Untrusted artifacts
(optimized suffixes + the mutable `checkpoint.json` best-suffix sidecar from `run_gcg`'s `on_step`)
stay under `artifacts/untrusted/` (gitignored).
