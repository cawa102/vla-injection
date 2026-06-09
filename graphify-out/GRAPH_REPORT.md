# Graph Report - .  (2026-06-09)

## Corpus Check
- 125 files · ~247,222 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1537 nodes · 3746 edges · 67 communities (56 shown, 11 thin omitted)
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 802 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Rollout|Rollout]]
- [[_COMMUNITY_L1 Internal Probe|L1 Internal Probe]]
- [[_COMMUNITY_Config Schema & Immutability|Config Schema & Immutability]]
- [[_COMMUNITY_Detector Metrics & CIs|Detector Metrics & CIs]]
- [[_COMMUNITY_Run Logging & Rollout Demo|Run Logging & Rollout Demo]]
- [[_COMMUNITY_Cross-Layer Tax Eval|Cross-Layer Tax Eval]]
- [[_COMMUNITY_Detector Calibration|Detector Calibration]]
- [[_COMMUNITY_Coverage Manifest|Coverage Manifest]]
- [[_COMMUNITY_GPU Runbook & Kelvin2|GPU Runbook & Kelvin2]]
- [[_COMMUNITY_run_condition_matrix|run_condition_matrix]]
- [[_COMMUNITY_Rollout Records Tests|Rollout Records Tests]]
- [[_COMMUNITY_OpenVLA Action Codec|OpenVLA Action Codec]]
- [[_COMMUNITY_L0 Perplexity Filter|L0 Perplexity Filter]]
- [[_COMMUNITY_Stats Provenance|Stats Provenance]]
- [[_COMMUNITY_LIBERO Obs Fixture (Bowls)|LIBERO Obs Fixture (Bowls)]]
- [[_COMMUNITY_Metric (A) Consistency Scorer|Metric (A) Consistency Scorer]]
- [[_COMMUNITY_Metric (A) Tests|Metric (A) Tests]]
- [[_COMMUNITY_Pareto Frontier & ASR|Pareto Frontier & ASR]]
- [[_COMMUNITY_Goal-Agnostic Anomaly Baseline|Goal-Agnostic Anomaly Baseline]]
- [[_COMMUNITY_Detector Decision Logic|Detector Decision Logic]]
- [[_COMMUNITY_Synthetic State Fixtures|Synthetic State Fixtures]]
- [[_COMMUNITY_CalibTest Split Disjointness|Calib/Test Split Disjointness]]
- [[_COMMUNITY_LIBERO Obs Fixture (Spatial)|LIBERO Obs Fixture (Spatial)]]
- [[_COMMUNITY_Eval Harness & Power|Eval Harness & Power]]
- [[_COMMUNITY_LIBERO State Adapter Tests|LIBERO State Adapter Tests]]
- [[_COMMUNITY_Oracle Frontier Tests|Oracle Frontier Tests]]
- [[_COMMUNITY_Environment Capture|Environment Capture]]
- [[_COMMUNITY_Codemaps & Architecture|Codemaps & Architecture]]
- [[_COMMUNITY_Metric Separation Demo|Metric Separation Demo]]
- [[_COMMUNITY_LIBERO State Smoketest|LIBERO State Smoketest]]
- [[_COMMUNITY_Eval Harness & Power|Eval Harness & Power]]
- [[_COMMUNITY_Figure Generation|Figure Generation]]
- [[_COMMUNITY_Privileged State Adapter|Privileged State Adapter]]
- [[_COMMUNITY_L1 Probe Literature|L1 Probe Literature]]
- [[_COMMUNITY_Attack Mechanisms M-aM-b|Attack Mechanisms M-a/M-b]]
- [[_COMMUNITY_Metric (A) Frozen Schema|Metric (A) Frozen Schema]]
- [[_COMMUNITY_Deterministic Seeding|Deterministic Seeding]]
- [[_COMMUNITY_Figure Regeneration Tests|Figure Regeneration Tests]]
- [[_COMMUNITY_Eval Harness & Power|Eval Harness & Power]]
- [[_COMMUNITY_Code-Goal Consistency Review|Code-Goal Consistency Review]]
- [[_COMMUNITY_EET Core Concepts|EET Core Concepts]]
- [[_COMMUNITY_LIBERO State Adapter Plan|LIBERO State Adapter Plan]]
- [[_COMMUNITY_Attack Illustration (Bowl to Knife)|Attack Illustration (Bowl to Knife)]]
- [[_COMMUNITY_Theme-Scoping Lit Review|Theme-Scoping Lit Review]]
- [[_COMMUNITY_Milestones M0-M3 & Branches|Milestones M0-M3 & Branches]]
- [[_COMMUNITY_Project Map & Session Memory|Project Map & Session Memory]]
- [[_COMMUNITY_Attack Hero Render (Knife)|Attack Hero Render (Knife)]]
- [[_COMMUNITY_Eval Harness & Power|Eval Harness & Power]]
- [[_COMMUNITY_Milestones M4-M5 & Deployable Tax|Milestones M4-M5 & Deployable Tax]]
- [[_COMMUNITY_LIBERO Fixtures & Adapter|LIBERO Fixtures & Adapter]]
- [[_COMMUNITY_Pyright Config|Pyright Config]]
- [[_COMMUNITY_L1 Probe Literature|L1 Probe Literature]]
- [[_COMMUNITY_Project Instructions & Principles|Project Instructions & Principles]]
- [[_COMMUNITY_Package Smoke Test|Package Smoke Test]]
- [[_COMMUNITY_References Provenance|References Provenance]]
- [[_COMMUNITY_LIBERO Problem Metadata|LIBERO Problem Metadata]]
- [[_COMMUNITY_Idealized action-space attacker package|Idealized action-space attacker package ]]
- [[_COMMUNITY_M2 comparison baselines (Task 8).  Two d|M2 comparison baselines (Task 8).  Two d]]
- [[_COMMUNITY_Pinned-config schema + runtime guard (Ta|Pinned-config schema + runtime guard (Ta]]
- [[_COMMUNITY_FP-calibrated detector (Task 6).  Turns|FP-calibrated detector (Task 6).  Turns ]]
- [[_COMMUNITY_Evaluation harness + statistics (Task 7)|Evaluation harness + statistics (Task 7)]]
- [[_COMMUNITY_Shared pytest fixtures for the evasion_t|Shared pytest fixtures for the evasion_t]]
- [[_COMMUNITY_Example M2 run config (YAML)|Example M2 run config (YAML)]]
- [[_COMMUNITY_Make this directory's helper modules imp|Make this directory's helper modules imp]]
- [[_COMMUNITY_Privileged-state metric package (Task 4+|Privileged-state metric package (Task 4+]]
- [[_COMMUNITY_Policy-side helpers (Task 3).  Currently|Policy-side helpers (Task 3).  Currently]]
- [[_COMMUNITY_Reproducibility infrastructure for the E|Reproducibility infrastructure for the E]]

## God Nodes (most connected - your core abstractions)
1. `Rollout` - 59 edges
2. `Score` - 55 edges
3. `calibrate()` - 46 edges
4. `ConsistencyMetricA` - 35 edges
5. `AttackScenario` - 34 edges
6. `FrontierPoint` - 34 edges
7. `TargetActionSpec` - 33 edges
8. `SyntheticDynamics` - 31 edges
9. `obs` - 31 edges
10. `IdealizedActionAttacker` - 30 edges

## Surprising Connections (you probably didn't know these)
- `test_frontier_point_rejects_negative_tradeoff()` --calls--> `FrontierPoint`  [INFERRED]
  tests/evasion_tax/attack/test_frontier.py → src/evasion_tax/attack/frontier.py
- `test_frontier_point_rejects_out_of_range_asr()` --calls--> `FrontierPoint`  [INFERRED]
  tests/evasion_tax/attack/test_frontier.py → src/evasion_tax/attack/frontier.py
- `test_frontier_point_rejects_out_of_range_evasion()` --calls--> `FrontierPoint`  [INFERRED]
  tests/evasion_tax/attack/test_frontier.py → src/evasion_tax/attack/frontier.py
- `test_attack_result_rejects_negative_tradeoff()` --calls--> `AttackResult`  [INFERRED]
  tests/evasion_tax/attack/test_idealized_frontier.py → src/evasion_tax/attack/idealized_frontier.py
- `test_attack_result_rejects_out_of_range_consistency()` --calls--> `AttackResult`  [INFERRED]
  tests/evasion_tax/attack/test_idealized_frontier.py → src/evasion_tax/attack/idealized_frontier.py

## Import Cycles
- 1-file cycle: `src/evasion_tax/repro/run_logger.py -> src/evasion_tax/repro/run_logger.py`

## Hyperedges (group relationships)
- **Codex-#2 model-free pre-GPU builds** — plans_codex_hooks_power_rule, plans_codex_hooks_coverage_manifest, plans_codex_hooks_cross_layer_eval [EXTRACTED 0.90]
- **Three-layer L0/L1/L2 evasion inspectors** — presentation_explainer_l0_layer, presentation_explainer_l1_probe, presentation_explainer_l2_monitor [EXTRACTED 0.90]
- **Kelvin2 GPU documentation suite** — gpu_connection, gpu_overview, gpu_running, gpu_start, setup_gpu_runbook [EXTRACTED 0.90]
- **Oracle (ASR, evasion) frontier-tracing pipeline** — attack_idealized_frontier_trace_frontier, attack_idealized_frontier_idealizedactionattacker, attack_dynamics_syntheticdynamics, detector_decide_rollout_fires, attack_frontier_pareto_frontier [EXTRACTED 0.90]
- **Fair-comparison detector suite calibrated by shared calibrate (invariant #4)** — detector_calibrate_calibrate, baselines_anomaly_goal_agnostic_anomaly_score, baselines_perplexity_perplexityfilter, attack_idealized_frontier_scorer [INFERRED 0.80]
- **Local-synthetic vs GPU-deferred stub pairs (Dependency Inversion)** — attack_dynamics_realdynamics, baselines_perplexity_realperplexityscorer, config_runtime_cuda_available [INFERRED 0.75]
- **Shared calibrate across L0/L1/L2 (invariant #4)** — metrics_tpr_at_fpr, probe_internal_internalprobe, consistency_a_consistencymetrica, concept_calibration_invariant_4 [INFERRED 0.75]
- **run_condition_matrix -> ResultsTable -> figures pipeline** — harness_run_condition_matrix, harness_resultstable, figures_results_table_to_dict, figures_make_figures [EXTRACTED 0.85]
- **Oracle outcomes -> frontier -> cluster-bootstrapped tax** — cross_layer_collect_oracle_outcomes, cross_layer_frontier_from_outcomes, cross_layer_bootstrap_delta_asr, cross_layer_unitoutcome [EXTRACTED 0.85]
- **Shared calibrate reused by detector, baselines, and eval (invariant #4)** — detector_calibrate_calibrate, baselines_anomaly_goal_agnostic_anomaly_score, baselines_perplexity_perplexityfilter, eval_metrics_tpr_at_fpr [INFERRED 0.85]
- **Model-free oracle frontier pipeline (dynamics, attacker, trace, cross-layer tax)** — attack_dynamics_syntheticdynamics, attack_idealized_frontier_idealizedactionattacker, attack_idealized_frontier_trace_frontier, eval_cross_layer_collect_oracle_outcomes [INFERRED 0.85]
- **Eval reporting stack: harness builds ResultsTable, figures regenerate, power gates** — eval_harness_run_condition_matrix, eval_figures_make_figures, eval_power_classify_power, eval_metrics_tpr_at_fpr [INFERRED 0.75]
- **L1 internal-probe confound-control pipeline** — metric_probe_internal_internalprobe, metric_probe_internal_syntheticactivationextractor, metric_probe_confounds_probe_auc, metric_probe_confounds_shuffle_labels [INFERRED 0.85]
- **LIBERO obs -> PrivilegedState -> goal-anchor resolution** — fixtures_libero_obs_spatial0, metric_state_libero_liberostateadapter, metric_consistency_a_privilegedgoalresolver [INFERRED 0.85]
- **Reproducibility: env capture, provenance hashing, write-once run logging** — repro_env_capture_capture_env, repro_provenance_record_provenance, repro_run_logger_runlogger [INFERRED 0.75]
- **Local no-GPU demo dry-run pipeline (rollout to separation to figures)** — scripts_demo_rollout_main, scripts_demo_metric_separation_main, scripts_demo_figures_main [EXTRACTED 1.00]
- **GPU-guarded run scripts sharing config cuda-guard contract** — scripts_run_benign_main, scripts_run_attack_main, scripts_microbench_gcg_main [INFERRED 0.85]
- **Real M2 eval pipeline (calibrate to evaluate to make_figures)** — scripts_calibrate_main, scripts_evaluate_main, scripts_make_figures_main [INFERRED 0.85]
- **L0/L1/L2 defence-layer triad** — core_execution_playbook_layer_l0, core_execution_playbook_layer_l1, core_execution_playbook_layer_l2 [EXTRACTED 1.00]
- **N/N−/F compute branches selected by M1 micro-bench** — core_execution_playbook_branch_n, core_execution_playbook_branch_nminus, core_execution_playbook_branch_f, core_execution_playbook_micro_bench [EXTRACTED 1.00]
- **H6-A oracle frontier vs H6-D deployable cross-layer tax (claim split)** — core_execution_playbook_h6a, core_execution_playbook_h6d, core_execution_playbook_claim_boundary [EXTRACTED 1.00]

## Communities (67 total, 11 thin omitted)

### Community 0 - "Rollout"
Cohesion: 0.06
Nodes (92): _coerce_position, AttackScenario, _coerce_position(), Dynamics, Action→privileged-state dynamics seam for the idealized attacker (§4b-(II)).  Th, Seam mapping ``(scenario, actions)`` to the induced :class:`Rollout`.      The d, Return the rollout the ``(scenario, actions)`` pair induces., Integrate ``actions`` through ``scenario`` into a :class:`Rollout`.          Arg (+84 more)

### Community 1 - "L1 Internal Probe"
Cohesion: 0.06
Nodes (62): ActivationFeatures, InternalProbe, probe_auc(), L1 confound-control scaffolding (Codex review #2 #11, playbook §4b-(I)).  The pr, Return a deterministic permutation of ``labels`` (multiset preserved).      The, ROC-AUC of ``probe`` over labelled ``features`` (benign=0, injected=1).      Spl, shuffle_labels(), ActivationExtractor (+54 more)

### Community 2 - "Config Schema & Immutability"
Cohesion: 0.06
Nodes (58): BaseModel, cuda_available(), gpu_required_message(), GPU-node runtime guard for the model/GPU-dependent scripts (Task 9).  ``run_beni, Return ``True`` iff a CUDA-capable torch runtime is present.      On a local dev, Return the "requires GPU node" message printed when the guard fires.      Args:, _flatten, _Frozen (immutable, extra-forbid base) (+50 more)

### Community 3 - "Detector Metrics & CIs"
Cohesion: 0.07
Nodes (64): abort_rate(), benign_degradation(), _clopper_pearson_ci(), detection_latency_summary(), _per_rollout_score(), _per_rollout_scores(), proportion_ci(), Evaluation statistics for the floor result (Task 7).  Pure NumPy/SciPy/sklearn s (+56 more)

### Community 4 - "Run Logging & Rollout Demo"
Cohesion: 0.07
Nodes (54): datetime, capture_env, Write-once run logger — the gatekeeper for the write-once invariant (#5).  Every, Create a fresh run directory and write its ``run.json`` protocol block., Return the current time as a timezone-aware UTC ``datetime``., Format a UTC datetime as a filesystem-safe ``YYYY-MM-DDTHH-MM-SSZ`` token., A live handle to one run directory; all writes refuse to overwrite., The run's output directory. (+46 more)

### Community 5 - "Cross-Layer Tax Eval"
Cohesion: 0.08
Nodes (57): _frontiers_for_pair, blocked_rate_by_layer, bootstrap_delta_asr (cluster bootstrap), comparative_asr_table, delta_asr_at_evasion, frontier_from_outcomes, frontiers_by_layer, target_action_blocked_rate (+49 more)

### Community 6 - "Detector Calibration"
Cohesion: 0.09
Nodes (53): benign_calib(), make_attacker(), make_scenario(), make_scorer(), make_target(), Tests for the idealized action-space attacker + frontier trace (§4b-(II)).  Mech, Reach-the-cube scene: cube (goal anchor) at +x, distractor at -x., Target action region: hold a -x position delta for the whole window. (+45 more)

### Community 7 - "Coverage Manifest"
Cohesion: 0.09
Nodes (42): Enum, build_manifest(), classify_cell(), CoverageCell, CoverageManifest, CoverageStatus, GoalKind, Metric-A coverage manifest (Codex review #2 #6).  The frozen metric-A schema (`` (+34 more)

### Community 8 - "GPU Runbook & Kelvin2"
Cohesion: 0.08
Nodes (41): k2-gpu-a100 / k2-gpu-h100 partitions, Kelvin2 — Connecting (macOS), Kelvin2 NI-HPC cluster (QUB), Kelvin2 — Cluster Overview, Kelvin2 — Running Jobs (Slurm), Slurm scheduler, Kelvin2 — Quickstart, Graphify query: starting guide when using GPU (+33 more)

### Community 9 - "run_condition_matrix"
Cohesion: 0.05
Nodes (48): Shared calibrate / invariant #4 (one footing across layers), L0/L1/L2 cross-layer evasion-tax frontier, No-leakage calib/test invariant #3, ConsistencyMetricA (L2 oracle scorer), GoalAnchor, GoalResolver (Protocol seam), PrivilegedGoalResolver, SchemaA (frozen metric-A schema) (+40 more)

### Community 10 - "Rollout Records Tests"
Cohesion: 0.06
Nodes (21): make_rollout(), make_step(), Tests for the core immutable data records (Task 2).  Covers: frozen-record immut, Build a RolloutStep with sensible defaults; action defaults to a 7-vector., test_actions_has_shape_n_by_7(), test_actions_values_match_steps(), test_prefix_window_clamped_at_start_when_t_less_than_k_minus_1(), test_prefix_window_k_less_than_one_raises() (+13 more)

### Community 11 - "OpenVLA Action Codec"
Cohesion: 0.10
Nodes (27): ActionCodec, OpenVLA action codec: discrete action token ids -> continuous 7-DoF (Task 3).  T, Dimensionality of the action space (= number of quantile entries)., The ``n_bins`` uniform bin edges over ``[-1, 1]``., The ``n_bins - 1`` bin centres (what de-tokenised tokens map to)., Map a single action token id to a bin-centre index in ``[0, n_bins-2]``., Return the normalised value (bin centre, in ``[-1, 1]``) for a bin index., Un-normalise per-dim normalised actions using q01/q99 under ``mask``.          ` (+19 more)

### Community 12 - "L0 Perplexity Filter"
Cohesion: 0.12
Nodes (30): _heuristic_perplexity, _perplexity_to_score, _heuristic_perplexity(), MockPerplexityScorer, _perplexity_to_score(), PerplexityFilter, PerplexityScorer, Perplexity / text-only filter baseline (Task 8).  RoboGCG's borrowed defences in (+22 more)

### Community 13 - "Stats Provenance"
Cohesion: 0.10
Nodes (30): ActionCodec (OpenVLA token decode), Namespace, record_stats_provenance, stats_url, Fetch + provenance helpers for OpenVLA ``dataset_statistics.json`` (Task 3).  Th, Return the direct download URL of a checkpoint's ``dataset_statistics.json``., Hash a downloaded stats file and record one provenance entry for it.      Args:, record_stats_provenance() (+22 more)

### Community 14 - "LIBERO Obs Fixture (Bowls)"
Cohesion: 0.11
Nodes (33): language_instruction, obj_of_interest, obs, akita_black_bowl_1_pos, akita_black_bowl_1_quat, akita_black_bowl_1_to_robot0_eef_pos, akita_black_bowl_1_to_robot0_eef_quat, akita_black_bowl_2_pos (+25 more)

### Community 15 - "Metric (A) Consistency Scorer"
Cohesion: 0.18
Nodes (20): _clip01(), ConsistencyMetricA, GoalAnchor, Consistency metric (A) — frozen annotation schema + causal scorer (Task 5).  THE, Causal goal-action consistency scorer over a prefix window (metric A).      Args, Causal consistency score for ``step_index`` over ``a_{t-k+1:t}``.          ``tru, One causal score per step (invariant #1)., NON-CAUSAL monitoring ceiling — a true upper bound on the causal score. (+12 more)

### Community 16 - "Metric (A) Tests"
Cohesion: 0.20
Nodes (32): _anchor(), _metric(), _priv(), Tests for consistency metric (A) — frozen schema + causal scorer (Task 5).  Thes, A privileged_state dict in the normalised PrivilegedState schema., _rollout(), _step(), test_combination_max_is_default_and_takes_worst_primitive() (+24 more)

### Community 17 - "Pareto Frontier & ASR"
Cohesion: 0.13
Nodes (30): _dominates, asr_at_evasion(), _dominates(), pareto_frontier(), (ASR, evasion) Pareto-frontier geometry for §4b-(II) — pure and model-free.  The, ASR achievable at a fixed ``evasion`` level, linearly interpolated.      Outside, True iff ``q`` Pareto-dominates ``p`` (>= on both axes, > on at least one)., Return the non-dominated set, deduplicated and ordered by evasion ascending. (+22 more)

### Community 18 - "Goal-Agnostic Anomaly Baseline"
Cohesion: 0.15
Nodes (26): BenignActionStats, goal_agnostic_anomaly_score(), Goal-agnostic action-anomaly baseline (Task 8).  The **mandatory** M2 comparison, Per-dimension benign action statistics (immutable; plan invariant #6).      Stor, Estimate per-dim mean/std from the pooled actions of benign rollouts.          A, Return ``(mean, std)`` as float ndarrays., One causal, goal-agnostic anomaly score per step (higher = more anomalous)., benign_rollout() (+18 more)

### Community 19 - "Detector Decision Logic"
Cohesion: 0.14
Nodes (28): _max_consistency, Decision, decide(), detection_latency(), Causal hold/allow decisions from a calibrated threshold (Task 6).  A consistency, Decide hold/allow for a single step.      Args:         score: The step's consis, Return the first-exceedance decision for one ordered rollout.      Scans ``score, Steps of deviation before the hold fired, relative to attack onset.      Args: (+20 more)

### Community 20 - "Synthetic State Fixtures"
Cohesion: 0.12
Nodes (19): approaching_goal_fixture(), approaching_goal_state(), _base_fixture_dict(), near_wrong_object_fixture(), near_wrong_object_state(), Reusable synthetic privileged-state fixtures (Task 4).  These are *not* pytest t, Build a fixture dict with the standard two-object layout., Fixture dict: end-effector hovering just above the goal object. (+11 more)

### Community 21 - "Calib/Test Split Disjointness"
Cohesion: 0.14
Nodes (23): CompletedProcess, _as_set(), assert_disjoint(), Calibration/test split disjointness (Task 7, plan invariant #3).  Calibration se, Raise if calibration and test manifests overlap on any axis.      Args:, Materialise an axis's ids into a set (no mutation of the input)., _config(), End-to-end CLI tests for ``scripts/evaluate.py`` (wiring, invariant #3).  Runs t (+15 more)

### Community 22 - "LIBERO Obs Fixture (Spatial)"
Cohesion: 0.07
Nodes (27): obs, akita_black_bowl_1_pos, akita_black_bowl_1_quat, akita_black_bowl_1_to_robot0_eef_pos, akita_black_bowl_1_to_robot0_eef_quat, cream_cheese_1_pos, cream_cheese_1_quat, cream_cheese_1_to_robot0_eef_pos (+19 more)

### Community 23 - "Eval Harness & Power"
Cohesion: 0.15
Nodes (23): annotate_operating_points(), classify_power(), Operating-point power / sample-size rule (Codex review #2 #3).  The detector rep, Attach a :class:`PowerStatus` to each operating point, in order.      The report, Minimum held-out benign N to support a per-rollout FPR claim of ``fpr_target``., Classify one operating point as powered / primary against the rule.      Args:, required_benign_n(), RULE_OF_THREE_EVENTS (+15 more)

### Community 24 - "LIBERO State Adapter Tests"
Cohesion: 0.10
Nodes (12): extract_ee_pos, extract_object_poses, gripper_open_from_qpos, target_region_from_obj_of_interest, _dummy_step(), _load(), opendrawer(), Tests for the concrete LIBERO StateAdapter (``state_libero.py``).  Run against F (+4 more)

### Community 25 - "Oracle Frontier Tests"
Cohesion: 0.20
Nodes (21): _validate_actions, Deterministic kinematic integrator for local tests (no LIBERO).      The end-eff, SyntheticDynamics, const_actions(), make_scenario(), Tests for the action→state dynamics seam (playbook §4b-(II)).  The metric-A orac, A minimal reach-the-cube scene with one distractor., An ``(n_steps, 7)`` array repeating ``vec`` (a length-7 action). (+13 more)

### Community 26 - "Environment Capture"
Cohesion: 0.17
Nodes (20): capture_env(), _dependency_snapshot(), _git_commit(), Capture the runtime environment for reproducibility logging.  Records platform,, Return the current ``HEAD`` commit hash, or ``None`` outside a repo., Return a ``{distribution: version}`` snapshot of installed packages.      Uses `, Return ``(torch_version, cuda_version, driver_version)``.      All three are ``N, Capture a reproducibility snapshot of the current environment.      Returns: (+12 more)

### Community 27 - "Codemaps & Architecture"
Cohesion: 0.15
Nodes (12): Architecture map (Embodiment Evasion Tax), CODEMAPS index, Data contract map (records.py / cross_layer), UnitOutcome / UnitKey cross-layer contract, Dependencies map, src/evasion_tax Python package, Local-prep guiding invariants (1-8), Pre-GPU local preparation plan (M1/8GB) (+4 more)

### Community 28 - "Metric Separation Demo"
Cohesion: 0.20
Nodes (18): Frozen metric-(A) annotation schema (v1). No attack-tuned values.      Attribute, SchemaA, test_metric_rejects_bad_k(), test_schema_defaults_match_frozen_doc(), test_schema_rejects_unknown_combination(), _generate(), generate_scored(), _histogram() (+10 more)

### Community 29 - "LIBERO State Smoketest"
Cohesion: 0.27
Nodes (17): _as_tuple3(), _extract_ground_truth(), _gripper_open_heuristic(), _load_privileged_state_cls(), main(), _print_report(), Any, Try to build a real ``PrivilegedState`` from the extracted ground truth. (+9 more)

### Community 30 - "Eval Harness & Power"
Cohesion: 0.28
Nodes (16): Evaluate every condition: calibrate on calib, score on disjoint test.      Args:, run_condition_matrix(), _condition(), Tests for the eval orchestration harness (Task 7).  ``run_condition_matrix`` con, test_each_row_has_operating_points_per_target(), test_each_row_has_power_status_aligned_with_operating_points(), test_empty_conditions_yields_empty_table(), test_fpr_targets_and_primary_fpr_are_threaded() (+8 more)

### Community 31 - "Figure Generation"
Cohesion: 0.27
Nodes (12): _ladder_placeholder_figure(), _load_results(), make_figures(), Script-regenerable figures from logged results (Task 9, M2 deliverable).  Every, Regenerate all M2 figures from a logged ``results.json``.      Args:         res, _roc_figure(), _score_hist_figure(), _tpr_at_fpr_figure() (+4 more)

### Community 32 - "Privileged State Adapter"
Cohesion: 0.22
Nodes (11): PrivilegedGoalResolver, Resolve the anchor from sim ground truth (privileged → non-deployable).      ``a, _coerce_position(), PrivilegedState, Thin, swappable privileged-state adapter (Task 4).  The consistency metric (A) m, Map a fixture dict to a :class:`PrivilegedState`.          Args:             raw, Coerce a length-3 numeric sequence to a tuple of 3 floats.      Args:         va, Normalised, env-agnostic ground-truth snapshot the metric reasons over.      Thi (+3 more)

### Community 33 - "L1 Probe Literature"
Cohesion: 0.26
Nodes (13): AlignSentinel attention-map detector (arXiv:2602.13597), Concept-Dictionary VLA safety (arXiv:2602.01834), D4 — eval scale / matrix decision (OPEN→M1), D7 — compute budget / GCG micro-bench (OPEN→M1), H1 — benign-vs-attacked separability (M1 gate), IGAR attention recalibration (arXiv:2603.06001), L1 confound controls (Codex #11), L1 internal probe instrument (probe_internal.py) (+5 more)

### Community 34 - "Attack Mechanisms M-a/M-b"
Cohesion: 0.15
Nodes (14): Compute Branch F (oracle frontier only), Coverage manifest (Codex #6; coverage.py), H2 — usable calibrated operating point (floor), H6-A — oracle intrinsic action-space frontier (floor), Idealized action-space attacker instrument (idealized_frontier.py), M2 — Floor detection layer (L0 + L2-oracle calibrated), M3 — Oracle intrinsic action-space frontier (H6-A), Mechanism M-a — GCG-can't-differentiate-through-rollout artifact (+6 more)

### Community 35 - "Metric (A) Frozen Schema"
Cohesion: 0.24
Nodes (13): Causal prefix-window detection + detection latency, L0 input layer (perplexity / text-only filter), Metric (A) — privileged sim-state consistency metric (L2 oracle), Metric (C) — reference-policy divergence (deployable L2 alt), Circularity guard / freeze-before-attack-inspection, max combination rule (zero free parameters), GoalResolver seam / PrivilegedGoalResolver, Monitoring ceiling (non-causal upper bound) (+5 more)

### Community 36 - "Deterministic Seeding"
Cohesion: 0.23
Nodes (12): Deterministic seeding across Python ``random``, NumPy, and (optionally) torch., A process-stable 64-bit seed derived from arbitrary parts.      Hashes ``"|".joi, Seed Python ``random``, NumPy, and torch (if importable).      Args:         see, seed_everything(), stable_seed(), Tests for the deterministic seeding helper., test_different_seeds_give_different_numpy_draws(), test_does_not_mutate_inputs_and_returns_fresh_dict() (+4 more)

### Community 37 - "Figure Regeneration Tests"
Cohesion: 0.26
Nodes (12): Serialise a :class:`ResultsTable` to the ``results.json`` schema.      Merges ea, results_table_to_dict(), Tests for script-regenerable figures (Task 9).  ``make_figures`` is the M2 deliv, _synthetic_results(), test_make_figures_creates_missing_out_dir(), test_make_figures_raises_when_results_missing(), test_make_figures_returns_written_paths(), test_make_figures_writes_expected_files() (+4 more)

### Community 38 - "Eval Harness & Power"
Cohesion: 0.29
Nodes (11): ConditionSplits, ConditionRow, Eval orchestration over a condition matrix (Task 7).  Consumes per-condition sco, One condition's evaluated result., Per-condition results plus the raw arrays ``make_figures`` consumes.      Attrib, ResultsTable, OperatingPoint, A calibrated operating point at one target FPR.      The honest false-abort rate (+3 more)

### Community 39 - "Code-Goal Consistency Review"
Cohesion: 0.27
Nodes (11): Cross-layer evaluation + tax metrics (eval/cross_layer.py), ΔASR-at-fixed-evasion primary tax scalar (Codex #10), FP calibration (held-out per-rollout false-abort), Operating-point power / sample-size rule (Codex #3; power.py), Target-action-blocked rate (security metric, invariant #9), Code ↔ Research-Goal Consistency Review (2026-06-08), src/evasion_tax/eval/cross_layer.py, src/evasion_tax/eval/harness.py (+3 more)

### Community 40 - "EET Core Concepts"
Cohesion: 0.22
Nodes (10): The Attacker Moves Second (Nasr et al., arXiv:2510.09023), Claim boundary / no-over-claim guardrail (Codex review #2), Embodiment Evasion Tax measurement reframe (§3a), L2 behavioural action-monitor layer, OpenVLA-7B (base VLA, bf16, discrete 256-bin head), RoboGCG attack (arXiv:2506.03350), The Embodiment Evasion Tax (EET), robosuite Tier-R local stand-in (honest substitution) (+2 more)

### Community 41 - "LIBERO State Adapter Plan"
Cohesion: 0.31
Nodes (8): LIBERO simulation benchmark, PrivilegedState contract (state.py), LIBERO obs fixtures provenance, LIBERO State Adapter Implementation Plan, Relative-key (_to_) object-pose extraction filter, target_region = obj_of_interest[-1] convention, src/evasion_tax/metric/state.py, src/evasion_tax/metric/state_libero.py (LiberoStateAdapter)

### Community 42 - "Attack Illustration (Bowl to Knife)"
Cohesion: 0.50
Nodes (8): Robot Arm Misdirected From Bowl to Knife (Instruction-Injection Illustration), Action Redirection / Goal Subversion, Goal-Action Consistency (Detector Concept), Instruction Injection Attack on VLA Policy, Intended Action (Green Path to Bowl), Perception/Reasoning Field (Glowing Sensing Dome), Subverted Action (Red Path to Knife), Embodied VLA Robot Arm (Gripper Manipulator)

### Community 43 - "Theme-Scoping Lit Review"
Cohesion: 0.44
Nodes (7): Theme-scoping report (2026-05-29), EET candidate theme (instruction injection + consistency), Five VLA integrity threat channels, T5 action-tokenization plausibility defense, T9 persistent-backdoor supply-chain defense, VLA integrity theme-scoping literature review, Trusted-intent channel load-bearing assumption

### Community 44 - "Milestones M0-M3 & Branches"
Cohesion: 0.25
Nodes (7): Session status snapshot (model-free track DONE), Compute Branch N (full deployable tax), Compute Branch N− (scoped deployable tax), D8 — compute (A100/H100 Kelvin2; branch selection), Embodiment Evasion Tax Execution Playbook, H6-D — deployable-vs-deployable cross-layer tax (headline), M1 on-GPU timing micro-bench (branch selector)

### Community 45 - "Project Map & Session Memory"
Cohesion: 0.29
Nodes (4): session-context-loader agent definition, Codemap scan report (2026-06-05), session-context-loader MEMORY index, Project map (EET layout & sources of truth)

### Community 46 - "Attack Hero Render (Knife)"
Cohesion: 0.61
Nodes (7): Robot Arm Reaching for Knife (Hero Render), Action-Space Manipulation / Subverted Action, Glowing Action Trajectory Toward Knife, Knife (Dangerous Target Object), Industrial Robot Manipulator / Gripper, Target Reticles on Table, VLA Pipeline Integrity Threat

### Community 47 - "Eval Harness & Power"
Cohesion: 0.25
Nodes (6): DetectorConfig, FP-calibration targets — the per-rollout benign false-abort budgets.      ``prim, test_detector_config_accepts_primary_fpr_in_targets(), test_detector_config_defaults_primary_fpr_to_five_percent(), test_detector_config_rejects_out_of_range_primary_fpr(), test_detector_config_rejects_primary_fpr_not_in_targets()

### Community 48 - "Milestones M4-M5 & Deployable Tax"
Cohesion: 0.25
Nodes (8): H3 — graceful degradation across reference ladder, H4 — deployable metric recovers (A) ceiling power, H5 — adaptive attacker raises the attacker's bar, M4 — Deployable L2 + H6-D cross-layer tax, M5 — Reference-coarsening ladder + threat-generalization, Metric (B) — learned action-semantics map (deployable L2), Indirect prompt-injection threat model (trusted operator goal), Trusted-reference coarsening ladder

### Community 49 - "LIBERO Fixtures & Adapter"
Cohesion: 0.36
Nodes (7): language_instruction, obj_of_interest, problem_info, domain_name, language_instruction, problem_name, LiberoStateAdapter

### Community 50 - "Pyright Config"
Cohesion: 0.25
Nodes (7): exclude, extraPaths, include, pythonPlatform, pythonVersion, venv, venvPath

### Community 51 - "L1 Probe Literature"
Cohesion: 0.48
Nodes (6): actalign / Do What You Say (arXiv:2510.16281), Instruction Hierarchy (arXiv:2404.13208), Metric (D) — VLM/LLM judge (excluded), OpenVLA paper (arXiv:2406.09246), SABER fluent NL injection attack (arXiv:2603.24935), Goal-Action Consistency Detector (understanding doc)

### Community 52 - "Project Instructions & Principles"
Cohesion: 0.47
Nodes (4): AGENTS.md (Codex instructions), CLAUDE.md project instructions, The Five Principles (think/simplicity/surgical/goal/document), Reproducibility non-negotiables

### Community 53 - "Package Smoke Test"
Cohesion: 0.33
Nodes (3): evasion_tax package (model-free core), Embodiment Evasion Tax — goal-action consistency detection for instruction-injec, evasion_tax package

### Community 54 - "References Provenance"
Cohesion: 0.70
Nodes (4): OpenVLA action codec formula provenance (c8f03f48), References provenance manifest, RoboGCG verified facts (defenses, Table 3), 2026 runtime VLA-safety cluster (Pre-VLA/HazardArena/Concept-Dictionary/IGAR)

### Community 55 - "LIBERO Problem Metadata"
Cohesion: 0.50
Nodes (4): problem_info, domain_name, language_instruction, problem_name

## Knowledge Gaps
- **16 isolated node(s):** `venvPath`, `venv`, `pythonVersion`, `pythonPlatform`, `extraPaths` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RunLogger` connect `Run Logging & Rollout Demo` to `Metric Separation Demo`, `Calib/Test Split Disjointness`, `Figure Generation`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `TargetActionSpec` connect `Rollout` to `run_condition_matrix`, `Cross-Layer Tax Eval`, `Run Logging & Rollout Demo`, `Stats Provenance`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `Score` connect `Rollout` to `Privileged State Adapter`, `L1 Internal Probe`, `Detector Metrics & CIs`, `Eval Harness & Power`, `run_condition_matrix`, `L0 Perplexity Filter`, `Metric (A) Consistency Scorer`, `Goal-Agnostic Anomaly Baseline`, `Detector Decision Logic`, `Metric Separation Demo`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Are the 52 inferred relationships involving `Rollout` (e.g. with `AttackScenario` and `Dynamics`) actually correct?**
  _`Rollout` has 52 INFERRED edges - model-reasoned connections that need verification._
- **Are the 49 inferred relationships involving `Score` (e.g. with `AttackResult` and `IdealizedActionAttacker`) actually correct?**
  _`Score` has 49 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `calibrate()` (e.g. with `trace_frontier()` and `test_trace_frontier_benign_fpr_is_conservative()`) actually correct?**
  _`calibrate()` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `ConsistencyMetricA` (e.g. with `Scorer` and `make_scorer()`) actually correct?**
  _`ConsistencyMetricA` has 20 INFERRED edges - model-reasoned connections that need verification._