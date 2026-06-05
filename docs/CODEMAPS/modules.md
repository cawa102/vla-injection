<!-- Generated: 2026-06-05 | Files scanned: 36 src (27 modules · 9 __init__) | Token estimate: ~900 -->

# Modules — `src/t7`

Stands in for the template's `backend.md` (this is a library, not an HTTP backend).
8 subpackages + root, **36 `.py` files** (27 modules · 9 `__init__`), **~4.3k LOC**.
Key public symbols per file (verified against source 2026-06-05).

## records.py — shared immutable contract (Task 2)
`RolloutStep` · `Rollout` (`.prefix_window` causal #1, `.actions`) · `TargetActionSpec`
(`.reached`, `.reached_window` D2 window-scored) · `Score` · `Decision` · `ScoreLike` · `score_value`.

## policy/ — OpenVLA action space (Task 3)
`action_codec.py` `ActionCodec` (discrete token ids → continuous 7-DoF, formula from source `c8f03f48`).
`openvla_stats.py` `stats_url`, `record_stats_provenance` (fetch `dataset_statistics.json`).

## metric/ — the L1 + L2 instruments
`consistency_a.py` **L2-oracle** `ConsistencyMetricA` (P1 progress / P2 distractor / P3 grasp; combine=max),
`SchemaA` (frozen), `GoalAnchor`, `Semantics`, `GoalResolver` Protocol, `PrivilegedGoalResolver`.
`probe_internal.py` **L1** `InternalProbe`, `ActivationFeatures`, `ActivationExtractor` Protocol +
`Synthetic`/`Real` impls. `probe_confounds.py` `shuffle_labels`, `probe_auc` (L1 control, Codex #2 #11).
`coverage.py` `CoverageManifest`, `classify_cell`, `GoalKind` (metric-A support map, #6).
`state.py` `PrivilegedState`, `StateAdapter` Protocol, `SyntheticStateAdapter` (Task 4).

## detector/ — FP-calibrated decision (Task 6)
`calibrate.py` `calibrate(...)→Threshold(tau)` (per-rollout FPR primary; shared by every baseline, #4).
`decide.py` `decide`, `rollout_fires`, `detection_latency`.

## attack/ — idealized action-space attacker (§4b-II, M-b)
`dynamics.py` `AttackScenario`, `Dynamics` Protocol, `SyntheticDynamics` / `RealDynamics` (LIBERO seam).
`idealized_frontier.py` `IdealizedActionAttacker` (constant-action random-shooting), `trace_frontier`,
`AttackResult`, `Scorer` Protocol. `frontier.py` `FrontierPoint`, `Frontier`, `pareto_frontier`, `asr_at_evasion`.

## eval/ — stats + cross-layer tax (Task 7 / §4b-III)
`metrics.py` `roc_auc`, `tpr_at_fpr`, `proportion_ci` (Wilson/Clopper-Pearson), `OperatingPoint`,
`benign_degradation`, `abort_rate`, `detection_latency_summary`.
`harness.py` `run_condition_matrix→ResultsTable`, `ConditionRow` (calibrate-on-calib / score-on-disjoint-test).
`cross_layer.py` `UnitKey`, `UnitOutcome`, `frontier_from_outcomes`, `delta_asr_at_evasion`,
`bootstrap_delta_asr→TaxEstimate` (cluster bootstrap), `comparative_asr_table`, `collect_oracle_outcomes`.
`power.py` `required_benign_n`, `classify_power` (sample-size rule, #3). `splits.py` `assert_disjoint` (#3).
`figures.py` `make_figures` (ROC / score-hist / TPR@FPR-CI from logged `results.json`).

## baselines/ — fair-calibrated comparators (Task 8)
`perplexity.py` **L0** `PerplexityFilter`, `PerplexityScorer` Protocol, `Mock`/`Real` impls.
`anomaly.py` `goal_agnostic_anomaly_score`, `BenignActionStats` (χ²-OOD action anomaly).

## config/ — pinned config + GPU guard (Task 9)
`schema.py` frozen pydantic `Config` (Model/Env/Attack/Metric/Detector/Splits/Eval), `load_config`,
`one_variable_diff` (enforces change-one-variable). `runtime.py` `cuda_available`, `gpu_required_message`.

## repro/ — reproducibility infrastructure (Task 1)
`seeds.py` `seed_everything` · `env_capture.py` `capture_env` · `provenance.py` `sha256_file`,
`record_provenance` · `run_logger.py` `RunLogger`, `RunHandle` (**write-once** gatekeeper, invariant #5).
