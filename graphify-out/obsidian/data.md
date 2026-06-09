---
source_file: "docs/CODEMAPS/data.md"
type: "document"
community: "Codemaps & Architecture"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/Codemaps__Architecture
---

# data.md

## Connections
- [[Data contract map (records.py  cross_layer)]] - `defined_in` [EXTRACTED]
- [[UnitOutcome  UnitKey cross-layer contract]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/Codemaps__Architecture

## 📄 Source

`docs/CODEMAPS/data.md`

<!-- Generated: 2026-06-05 | Source: src/evasion_tax/records.py, eval/cross_layer.py | Token estimate: ~550 -->

# Data — Embodiment Evasion Tax
No database. "Data" = the frozen-dataclass contract in `records.py` + the write-once `results/`
logs (absent until the first GPU run; gatekept by `repro/run_logger.RunLogger`).

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

## Reproducibility invariants encoded structurally
| # | Invariant | Where |
|---|-----------|-------|
| 1 | Causal windows (no future index) | `Rollout.prefix_window` |
| 3 | Calib/test disjoint; held-out FPR | `eval/splits.assert_disjoint`, `harness` |
| 4 | One shared calibration primitive | `detector/calibrate` (reused by all baselines) |
| 5 | Write-once results | `repro/run_logger.RunLogger` |
| 6 | Immutability | all records `frozen=True` |

## Provenance
`repro/provenance.record_provenance` (SHA-256 + JSON manifest) for data/checkpoints;
`policy/openvla_stats.record_stats_provenance` for `dataset_statistics.json`. Quarantine untrusted
artifacts under `artifacts/untrusted/` (gitignored).

