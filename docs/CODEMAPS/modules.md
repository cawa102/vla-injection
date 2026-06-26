<!-- Generated: 2026-06-26 (full regen) | Files scanned: 52 src (43 modules · 9 __init__, ~7.8k LOC) | Token estimate: ~1150 -->

# Modules — `src/evasion_tax`

Stands in for the template's `backend.md` (this is a library, not an HTTP backend).
8 subpackages + root, **52 `.py` files** (43 modules · 9 `__init__`), **~7.8k LOC**.
Key public symbols per file (verified against source 2026-06-26).

## records.py — shared immutable contract
`RolloutStep` · `Rollout` (`.prefix_window` causal #1, `.actions`) · `TargetActionSpec`
(`.reached`, `.reached_window` D2 window-scored) · `Score` · `Decision` · `ScoreLike` · `score_value`.

## policy/ — OpenVLA action space
`action_codec.py` `ActionCodec` (discrete token ids → continuous 7-DoF, formula from source `c8f03f48`).
`action_check.py` `validate_action_vector` (7-DoF bounds/shape guard at the seam).
`openvla_stats.py` `stats_url`, `record_stats_provenance` (fetch `dataset_statistics.json`).

## metric/ — the L1 + L2 instruments
`consistency_a.py` **L2-oracle** `ConsistencyMetricA` (P1 progress / P2 distractor / P3 grasp; combine=max),
`SchemaA` (frozen), `GoalAnchor`, `Semantics`, `GoalResolver` Protocol, `PrivilegedGoalResolver`.
`probe_internal.py` **L1** `InternalProbe`, `ActivationFeatures`, `ActivationExtractor` Protocol +
`Synthetic`/`Real` impls. `probe_confounds.py` `shuffle_labels`, `probe_auc` (L1 control, Codex #2 #11).
`coverage.py` `CoverageManifest`, `classify_cell`, `GoalKind` (metric-A support map, #6).
`state.py` `PrivilegedState`, `StateAdapter` Protocol, `SyntheticStateAdapter`.
`state_libero.py` **`LiberoStateAdapter`** (real LIBERO obs → `PrivilegedState`; `_to_` relative-key filter,
BDDL `target_region` from `obj_of_interest[-1]`, gripper `sum|qpos|>0.04`).

## detector/ — FP-calibrated decision
`calibrate.py` `calibrate(...)→Threshold(tau)` (per-rollout FPR primary; shared by every baseline, #4).
`decide.py` `decide`, `rollout_fires`, `detection_latency`.

## attack/ — RoboGCG attack + idealized action-space attacker
`gcg.py` model-free GCG core: `run_gcg` (keyword-only `reached_fn`, optional `on_step` callback),
`GcgConfig`, `GcgResult`, `LossGradientFn` Protocol, `top_k_candidates`/`sample_candidates`/`select_best`.
`gcg_openvla.py` **OpenVLA GCG seam (GPU)**: `OpenVlaGcgTarget` (`LossGradientFn` impl —
`token_gradient`/`loss_of`/`reached`/`gradient_agrees_with_swaps`/`init_suffix_ids`), `FaithfulnessReport`,
`chunked_losses`, `per_sequence_ce`, `equivalence_verdict`, `project_onehot_grad`, `suffix_span_in_ids`.
`openvla_loader.py` shared loader: `load_openvla_policy` (bf16/int8/nf4 via BitsAndBytes), `build_target`,
`load_openvla_with_attn_fallback` (flash→sdpa, deviation recorded), `quantization_record`,
`OpenVlaLoadRecord`, `target_action_ids_for_vocab`, `is_flash_attn_load_error`, `load_frozen_openvla`.
`surrogate_artifacts.py` quarantined transfer records: `SurrogateSuffixArtifact` (schema v2 +
`surrogate_gradient_health`), `TransferEvalRecord`, `write_json_record`/`read_suffix_artifact`/
`read_transfer_record`, `token_l2_distance`, `is_under_quarantine`/`require_quarantined`.
`redirect_target.py` `RedirectSpec`, `redirect_spec_for`, `target_action_ids_for` (which tokens to force).
`early_stop.py` `target_span_argmax_matches`. `early_stop_bench.py` `steps_to_success_summary`,
`TargetOutcome`, `realistic_s_per_target`, `is_target_done`.
`dynamics.py` `AttackScenario`, `Dynamics` Protocol, `SyntheticDynamics` / `RealDynamics` (LIBERO seam).
`idealized_frontier.py` `IdealizedActionAttacker` (constant-action random-shooting), `trace_frontier`,
`AttackResult`, `Scorer` Protocol. `frontier.py` `FrontierPoint`, `Frontier`, `pareto_frontier`, `asr_at_evasion`.

## eval/ — stats · cross-layer tax · transfer · rollout · M1 gate
`metrics.py` `roc_auc`, `tpr_at_fpr`, `proportion_ci` (Wilson/Clopper-Pearson), `OperatingPoint`,
`benign_degradation`, `abort_rate`, `detection_latency_summary`.
`harness.py` `run_condition_matrix→ResultsTable`, `ConditionRow` (calibrate-on-calib / score-on-disjoint-test).
`cross_layer.py` `UnitKey`, `UnitOutcome`, `frontier_from_outcomes`, `delta_asr_at_evasion`,
`bootstrap_delta_asr→TaxEstimate` (cluster bootstrap), `comparative_asr_table`, `collect_oracle_outcomes`.
`surrogate_transfer.py` `summarize_transfer` (victim ASR · transfer gap · GPU-hour-normalized · censoring),
`write_summary_outputs`.
`rollout_runner.py` **L2-on-real-rollout**: `run_episode→EpisodeResult`, `rollout_asr`, `geometry_stats`
(DM-3 calibration data), `build_rollout_step`, `inject_suffix`, `normalize_actions`.
`rollout_io.py` `load_rollout_log`, `rollout_from_log`, `SourceProvenance`, `validate_run_dir`.
`m1_gate.py` `m1_verdict`, `BenignRecord`, `AttackUnitRecord`, `benign_records_from_dicts`,
`attack_records_from_dicts` (M1 viability gate). `separation.py` `separation_table`, `per_rollout_score`.
`branch_select.py` `provisional_branch→BranchDecision`, `affordable_matrix→AffordableMatrix`,
`BranchThresholds` (D8 compute-branch N/N−/F). `schema_repin.py` `repin_schema_from_benign` (DM-3 re-pin).
`power.py` `required_benign_n`, `classify_power` (#3). `splits.py` `assert_disjoint` (#3).
`figures.py` `make_figures` (ROC / score-hist / TPR@FPR-CI from logged `results.json`).

## baselines/ — fair-calibrated comparators
`perplexity.py` **L0** `PerplexityFilter`, `PerplexityScorer` Protocol, `Mock`/`Real` impls.
`anomaly.py` `goal_agnostic_anomaly_score`, `BenignActionStats` (χ²-OOD action anomaly).

## config/ — pinned config + GPU guard
`schema.py` frozen pydantic `Config` (Model/Env/Attack/Metric/Detector/Splits/Eval), `load_config`,
`one_variable_diff` (enforces change-one-variable). `runtime.py` `cuda_available`, `gpu_required_message`.

## repro/ — reproducibility infrastructure
`seeds.py` `seed_everything` · `env_capture.py` `capture_env` · `provenance.py` `sha256_file`,
`record_provenance` · `run_logger.py` `RunLogger`, `RunHandle` (**write-once** gatekeeper, invariant #5).
