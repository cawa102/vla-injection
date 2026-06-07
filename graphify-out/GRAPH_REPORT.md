# Graph Report - .  (2026-06-07)

## Corpus Check
- 111 files · ~232,884 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1174 nodes · 2402 edges · 62 communities (50 shown, 12 thin omitted)
- Extraction: 72% EXTRACTED · 28% INFERRED · 0% AMBIGUOUS · INFERRED: 679 edges (avg confidence: 0.67)
- Token cost: 387,604 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Consistency Metric A (Scorer)|Consistency Metric A (Scorer)]]
- [[_COMMUNITY_Evaluation Statistics & ROC|Evaluation Statistics & ROC]]
- [[_COMMUNITY_Config Schema & GPU Guard|Config Schema & GPU Guard]]
- [[_COMMUNITY_Coverage Manifest|Coverage Manifest]]
- [[_COMMUNITY_Figures & Eval Harness|Figures & Eval Harness]]
- [[_COMMUNITY_Dynamics Seam Tests|Dynamics Seam Tests]]
- [[_COMMUNITY_Core Records Tests|Core Records Tests]]
- [[_COMMUNITY_OpenVLA Action Codec|OpenVLA Action Codec]]
- [[_COMMUNITY_Kelvin2 HPC & Planning Docs|Kelvin2 HPC & Planning Docs]]
- [[_COMMUNITY_FP-Calibration & Power Rule|FP-Calibration & Power Rule]]
- [[_COMMUNITY_Write-Once Run Logger|Write-Once Run Logger]]
- [[_COMMUNITY_Metric-A Schema Tests|Metric-A Schema Tests]]
- [[_COMMUNITY_L0 Perplexity Filter|L0 Perplexity Filter]]
- [[_COMMUNITY_Pareto Frontier Geometry|Pareto Frontier Geometry]]
- [[_COMMUNITY_Threshold Calibration|Threshold Calibration]]
- [[_COMMUNITY_Goal-Agnostic Anomaly Baseline|Goal-Agnostic Anomaly Baseline]]
- [[_COMMUNITY_DataCheckpoint Provenance|Data/Checkpoint Provenance]]
- [[_COMMUNITY_Idealized Action Attacker|Idealized Action Attacker]]
- [[_COMMUNITY_Core Data Records|Core Data Records]]
- [[_COMMUNITY_HoldAllow Decision Logic|Hold/Allow Decision Logic]]
- [[_COMMUNITY_Oracle Outcomes & Tax Estimate|Oracle Outcomes & Tax Estimate]]
- [[_COMMUNITY_Cross-Layer Tax Tests|Cross-Layer Tax Tests]]
- [[_COMMUNITY_L1 Internal Probe Tests|L1 Internal Probe Tests]]
- [[_COMMUNITY_Environment Capture|Environment Capture]]
- [[_COMMUNITY_LIBERO State Smoke Test|LIBERO State Smoke Test]]
- [[_COMMUNITY_Codemaps & Project Roadmap|Codemaps & Project Roadmap]]
- [[_COMMUNITY_Action-to-State Dynamics|Action-to-State Dynamics]]
- [[_COMMUNITY_CalibrationTest Split Disjointness|Calibration/Test Split Disjointness]]
- [[_COMMUNITY_Cross-Layer Eval & Tax Metrics|Cross-Layer Eval & Tax Metrics]]
- [[_COMMUNITY_Privileged-State Fixtures|Privileged-State Fixtures]]
- [[_COMMUNITY_L1 Internal-Representation Probe|L1 Internal-Representation Probe]]
- [[_COMMUNITY_Synthetic Activation Extractor Tests|Synthetic Activation Extractor Tests]]
- [[_COMMUNITY_OpenVLA Activation Extractor (GPU Stub)|OpenVLA Activation Extractor (GPU Stub)]]
- [[_COMMUNITY_Deterministic Seeding|Deterministic Seeding]]
- [[_COMMUNITY_L1 Probe Confound Control|L1 Probe Confound Control]]
- [[_COMMUNITY_Cross-Layer Tax Concepts|Cross-Layer Tax Concepts]]
- [[_COMMUNITY_Models, Attacks & References|Models, Attacks & References]]
- [[_COMMUNITY_Metric-A Concepts & Attacker|Metric-A Concepts & Attacker]]
- [[_COMMUNITY_Threat Model & Theme Scoping|Threat Model & Theme Scoping]]
- [[_COMMUNITY_Instruction-Injection Illustration (img2)|Instruction-Injection Illustration (img2)]]
- [[_COMMUNITY_Consistency Detector Literature|Consistency Detector Literature]]
- [[_COMMUNITY_Probe Label-Shuffle Control Tests|Probe Label-Shuffle Control Tests]]
- [[_COMMUNITY_Pyright Config|Pyright Config]]
- [[_COMMUNITY_Hero Render Image (img1)|Hero Render Image (img1)]]
- [[_COMMUNITY_Rollout Reach Helpers|Rollout Reach Helpers]]
- [[_COMMUNITY_Project Instructions & Principles|Project Instructions & Principles]]
- [[_COMMUNITY_Session-Context Agent Memory|Session-Context Agent Memory]]
- [[_COMMUNITY_Stable Seed Helper|Stable Seed Helper]]
- [[_COMMUNITY_Pareto Frontier Core|Pareto Frontier Core]]
- [[_COMMUNITY_Attacker Package Init|Attacker Package Init]]
- [[_COMMUNITY_Baselines Package Init|Baselines Package Init]]
- [[_COMMUNITY_Config Package Init|Config Package Init]]
- [[_COMMUNITY_Detector Package Init|Detector Package Init]]
- [[_COMMUNITY_Eval Package Init|Eval Package Init]]
- [[_COMMUNITY_Test Fixtures (conftest)|Test Fixtures (conftest)]]
- [[_COMMUNITY_evasion_tax Package Init|evasion_tax Package Init]]
- [[_COMMUNITY_Test Import Conftest|Test Import Conftest]]
- [[_COMMUNITY_Metric Package Init|Metric Package Init]]
- [[_COMMUNITY_Policy Package Init|Policy Package Init]]
- [[_COMMUNITY_Repro Package Init|Repro Package Init]]
- [[_COMMUNITY_Path Bootstrap|Path Bootstrap]]

## God Nodes (most connected - your core abstractions)
1. `Rollout` - 51 edges
2. `Score` - 49 edges
3. `calibrate()` - 33 edges
4. `FrontierPoint` - 31 edges
5. `AttackScenario` - 28 edges
6. `ConsistencyMetricA` - 26 edges
7. `TargetActionSpec` - 26 edges
8. `_step()` - 26 edges
9. `_metric()` - 24 edges
10. `Frontier` - 23 edges

## Surprising Connections (you probably didn't know these)
- `test_real_dynamics_conforms_to_protocol()` --calls--> `RealDynamics`  [INFERRED]
  tests/evasion_tax/attack/test_dynamics.py → src/evasion_tax/attack/dynamics.py
- `test_frontier_point_rejects_negative_tradeoff()` --calls--> `FrontierPoint`  [INFERRED]
  tests/evasion_tax/attack/test_frontier.py → src/evasion_tax/attack/frontier.py
- `test_frontier_point_rejects_out_of_range_asr()` --calls--> `FrontierPoint`  [INFERRED]
  tests/evasion_tax/attack/test_frontier.py → src/evasion_tax/attack/frontier.py
- `test_frontier_point_rejects_out_of_range_evasion()` --calls--> `FrontierPoint`  [INFERRED]
  tests/evasion_tax/attack/test_frontier.py → src/evasion_tax/attack/frontier.py
- `test_attack_result_rejects_negative_tradeoff()` --calls--> `AttackResult`  [INFERRED]
  tests/evasion_tax/attack/test_idealized_frontier.py → src/evasion_tax/attack/idealized_frontier.py

## Import Cycles
- 1-file cycle: `src/evasion_tax/repro/run_logger.py -> src/evasion_tax/repro/run_logger.py`

## Hyperedges (group relationships)
- **Three detection layers L0/L1/L2 measured at matched FPR** — execution_playbook_l0_input_layer, execution_playbook_l1_internal_probe, execution_playbook_l2_behavioural_monitor, execution_playbook_per_layer_evasion_cost_measurement [EXTRACTED 1.00]
- **Metric (A) primitives P1/P2/P3 combined by max** — metric_a_annotation_schema_p1_progress, metric_a_annotation_schema_p2_distractor, metric_a_annotation_schema_p3_grasp, metric_a_annotation_schema_max_combination [EXTRACTED 1.00]
- **Consistency metric options A/B/C/D** — goal_action_consistency_detector_metric_a_privileged, goal_action_consistency_detector_metric_b_learned, goal_action_consistency_detector_metric_c_reference_policy, goal_action_consistency_detector_metric_d_vlm_judge [EXTRACTED 1.00]
- **Kelvin2 GPU documentation suite** — gpu_connection, gpu_overview, gpu_running, gpu_start, setup_gpu_runbook [EXTRACTED 0.90]
- **Three-layer L0/L1/L2 evasion inspectors** — presentation_explainer_l0_layer, presentation_explainer_l1_probe, presentation_explainer_l2_monitor [EXTRACTED 0.90]
- **Codex-#2 model-free pre-GPU builds** — plans_codex_hooks_power_rule, plans_codex_hooks_coverage_manifest, plans_codex_hooks_cross_layer_eval [EXTRACTED 0.90]

## Communities (62 total, 12 thin omitted)

### Community 0 - "Consistency Metric A (Scorer)"
Cohesion: 0.06
Nodes (42): _coerce_action(), Coerce any length-7 numeric sequence to a tuple of 7 floats.      Args:, One step of a (benign or attacked) rollout.      ``action`` is coerced to a leng, RolloutStep, _clip01(), ConsistencyMetricA, GoalAnchor, GoalResolver (+34 more)

### Community 1 - "Evaluation Statistics & ROC"
Cohesion: 0.06
Nodes (61): abort_rate(), benign_degradation(), _clopper_pearson_ci(), detection_latency_summary(), _per_rollout_score(), _per_rollout_scores(), proportion_ci(), Evaluation statistics for the floor result (Task 7).  Pure NumPy/SciPy/sklearn s (+53 more)

### Community 2 - "Config Schema & GPU Guard"
Cohesion: 0.06
Nodes (53): BaseModel, cuda_available(), gpu_required_message(), GPU-node runtime guard for the model/GPU-dependent scripts (Task 9).  ``run_beni, Return ``True`` iff a CUDA-capable torch runtime is present.      On a local dev, Return the "requires GPU node" message printed when the guard fires.      Args:, AttackConfig, Config (+45 more)

### Community 3 - "Coverage Manifest"
Cohesion: 0.07
Nodes (42): Enum, build_manifest(), classify_cell(), CoverageCell, CoverageManifest, CoverageStatus, GoalKind, Metric-A coverage manifest (Codex review #2 #6).  The frozen metric-A schema (`` (+34 more)

### Community 4 - "Figures & Eval Harness"
Cohesion: 0.08
Nodes (42): ConditionSplits, _ladder_placeholder_figure(), _load_results(), make_figures(), Script-regenerable figures from logged results (Task 9, M2 deliverable).  Every, Regenerate all M2 figures from a logged ``results.json``.      Args:         res, Serialise a :class:`ResultsTable` to the ``results.json`` schema.      Merges ea, results_table_to_dict() (+34 more)

### Community 5 - "Dynamics Seam Tests"
Cohesion: 0.11
Nodes (42): Deterministic kinematic integrator for local tests (no LIBERO).      The end-eff, SyntheticDynamics, const_actions(), make_scenario(), Tests for the action→state dynamics seam (playbook §4b-(II)).  The metric-A orac, A minimal reach-the-cube scene with one distractor., An ``(n_steps, 7)`` array repeating ``vec`` (a length-7 action)., test_attacked_flag_defaults_true_and_is_overridable() (+34 more)

### Community 6 - "Core Records Tests"
Cohesion: 0.07
Nodes (21): make_rollout(), make_step(), Tests for the core immutable data records (Task 2).  Covers: frozen-record immut, Build a RolloutStep with sensible defaults; action defaults to a 7-vector., test_actions_has_shape_n_by_7(), test_actions_values_match_steps(), test_prefix_window_clamped_at_start_when_t_less_than_k_minus_1(), test_prefix_window_k_less_than_one_raises() (+13 more)

### Community 7 - "OpenVLA Action Codec"
Cohesion: 0.07
Nodes (27): ActionCodec, OpenVLA action codec: discrete action token ids -> continuous 7-DoF (Task 3).  T, Dimensionality of the action space (= number of quantile entries)., The ``n_bins`` uniform bin edges over ``[-1, 1]``., The ``n_bins - 1`` bin centres (what de-tokenised tokens map to)., Map a single action token id to a bin-centre index in ``[0, n_bins-2]``., Return the normalised value (bin centre, in ``[-1, 1]``) for a bin index., Un-normalise per-dim normalised actions using q01/q99 under ``mask``.          ` (+19 more)

### Community 8 - "Kelvin2 HPC & Planning Docs"
Cohesion: 0.09
Nodes (40): k2-gpu-a100 / k2-gpu-h100 partitions, Kelvin2 — Connecting (macOS), Kelvin2 NI-HPC cluster (QUB), Kelvin2 — Cluster Overview, Kelvin2 — Running Jobs (Slurm), Slurm scheduler, Kelvin2 — Quickstart, Anti-circularity invariant (no attack-derived constants) (+32 more)

### Community 9 - "FP-Calibration & Power Rule"
Cohesion: 0.09
Nodes (33): DetectorConfig, FP-calibration targets — the per-rollout benign false-abort budgets.      ``prim, OperatingPoint, A calibrated operating point at one target FPR.      The honest false-abort rate, annotate_operating_points(), classify_power(), PowerStatus, Operating-point power / sample-size rule (Codex review #2 #3).  The detector rep (+25 more)

### Community 10 - "Write-Once Run Logger"
Cohesion: 0.08
Nodes (28): datetime, Write-once run logger — the gatekeeper for the write-once invariant (#5).  Every, Create a fresh run directory and write its ``run.json`` protocol block., Return the current time as a timezone-aware UTC ``datetime``., Format a UTC datetime as a filesystem-safe ``YYYY-MM-DDTHH-MM-SSZ`` token., A live handle to one run directory; all writes refuse to overwrite., The run's output directory., Write ``obj`` as ``<name>.json``; raise if the file already exists.          Arg (+20 more)

### Community 11 - "Metric-A Schema Tests"
Cohesion: 0.16
Nodes (37): Frozen metric-(A) annotation schema (v1). No attack-tuned values.      Attribute, SchemaA, _anchor(), _metric(), _priv(), Tests for consistency metric (A) — frozen schema + causal scorer (Task 5).  Thes, A privileged_state dict in the normalised PrivilegedState schema., _rollout() (+29 more)

### Community 12 - "L0 Perplexity Filter"
Cohesion: 0.10
Nodes (27): _heuristic_perplexity(), MockPerplexityScorer, _perplexity_to_score(), PerplexityFilter, PerplexityScorer, Perplexity / text-only filter baseline (Task 8).  RoboGCG's borrowed defences in, A backend mapping an instruction string to a perplexity ``>= 1``., Return the perplexity of ``instruction`` (higher = less natural). (+19 more)

### Community 13 - "Pareto Frontier Geometry"
Cohesion: 0.11
Nodes (31): asr_at_evasion(), pareto_frontier(), ASR achievable at a fixed ``evasion`` level, linearly interpolated.      Outside, Return the non-dominated set, deduplicated and ordered by evasion ascending., pt(), Tests for the (ASR, evasion) Pareto-frontier geometry (playbook §4b-(II)).  Pure, test_asr_at_evasion_clamps_above_max_evasion_to_min_asr(), test_asr_at_evasion_clamps_below_min_evasion_to_max_asr() (+23 more)

### Community 14 - "Threshold Calibration"
Cohesion: 0.13
Nodes (30): calibrate(), _pooled_steps(), One value per rollout: its maximum per-step score., All per-step scores pooled across rollouts (per-window unit)., Calibrate ``tau`` to a benign false-abort budget.      Args:         benign_scor, _rollout_maxima(), benign_rollouts(), per_rollout_fire_rate() (+22 more)

### Community 15 - "Goal-Agnostic Anomaly Baseline"
Cohesion: 0.13
Nodes (25): BenignActionStats, goal_agnostic_anomaly_score(), Goal-agnostic action-anomaly baseline (Task 8).  The **mandatory** M2 comparison, Per-dimension benign action statistics (immutable; plan invariant #6).      Stor, Estimate per-dim mean/std from the pooled actions of benign rollouts.          A, Return ``(mean, std)`` as float ndarrays., One causal, goal-agnostic anomaly score per step (higher = more anomalous)., benign_rollout() (+17 more)

### Community 16 - "Data/Checkpoint Provenance"
Cohesion: 0.09
Nodes (25): Namespace, Fetch + provenance helpers for OpenVLA ``dataset_statistics.json`` (Task 3).  Th, Return the direct download URL of a checkpoint's ``dataset_statistics.json``., Hash a downloaded stats file and record one provenance entry for it.      Args:, record_stats_provenance(), stats_url(), Tests for the OpenVLA dataset-statistics fetch helpers (Task 3).  The network do, test_record_stats_provenance_hashes_file_and_writes_a_keyed_entry() (+17 more)

### Community 17 - "Idealized Action Attacker"
Cohesion: 0.13
Nodes (23): Dynamics, Seam mapping ``(scenario, actions)`` to the induced :class:`Rollout`.      The d, FrontierPoint, One achievable ``(ASR, evasion)`` operating point of the attacker.      Attribut, AttackResult, _max_consistency(), Idealized action-space attacker + frontier trace (playbook §4b-(II), M-b).  The, Sustained (constant-per-candidate) action commands to search over. (+15 more)

### Community 18 - "Core Data Records"
Cohesion: 0.10
Nodes (23): FP-calibration of the detector threshold (Task 6).  Chooses ``tau`` on a benign, Decision, Core immutable data records shared across metric / detector / eval (Task 2).  Th, A causal consistency score; higher = more inconsistent with the goal.      ``val, A hold/allow decision emitted by the detector at a given step., An ordered, immutable sequence of rollout steps., Return the causal window of steps ``max(0, t-k+1) .. t`` (inclusive).          C, Rollout (+15 more)

### Community 19 - "Hold/Allow Decision Logic"
Cohesion: 0.14
Nodes (25): Decision, decide(), detection_latency(), Causal hold/allow decisions from a calibrated threshold (Task 6).  A consistency, Decide hold/allow for a single step.      Args:         score: The step's consis, Return the first-exceedance decision for one ordered rollout.      Scans ``score, Steps of deviation before the hold fired, relative to attack onset.      Args:, rollout_fires() (+17 more)

### Community 20 - "Oracle Outcomes & Tax Estimate"
Cohesion: 0.21
Nodes (21): Frontier, A canonical ``(ASR, evasion)`` Pareto frontier.      ``points`` must be ordered, IdealizedActionAttacker, Model-free, seeded constant-action random-shooting attacker (M-b).      For each, The behavioural-detector seam the attacker optimises against.      Matches :clas, Scorer, An immutable calibrated decision threshold.      Attributes:         tau: The th, Threshold (+13 more)

### Community 21 - "Cross-Layer Tax Tests"
Cohesion: 0.17
Nodes (20): _benign_calib(), outcomes(), Tests for the cross-layer eval + tax metrics (§4b-III).  The primary tax scalar, Bootstrap the strong-vs-weak tax (the layer pair every test below uses)., Build UnitOutcomes from ``{tradeoff: (reached_list, detected_list)}``., _scenario(), _target(), _tax() (+12 more)

### Community 22 - "L1 Internal Probe Tests"
Cohesion: 0.17
Nodes (14): fit_separable_probe(), make_rollout(), Tests for the L1 internal-representation probe (playbook §4b-(I)).  The **L1** l, A minimal rollout carrying just what the synthetic extractor keys off.      The, test_benign_weird_suffix_does_not_fire_at_calibrated_threshold(), test_calibrates_through_shared_calibrate_and_flags_injection(), test_fit_rejects_length_mismatch(), test_fit_requires_both_classes() (+6 more)

### Community 23 - "Environment Capture"
Cohesion: 0.16
Nodes (16): capture_env(), _dependency_snapshot(), _git_commit(), Capture the runtime environment for reproducibility logging.  Records platform,, Return the current ``HEAD`` commit hash, or ``None`` outside a repo., Return a ``{distribution: version}`` snapshot of installed packages.      Uses `, Return ``(torch_version, cuda_version, driver_version)``.      All three are ``N, Capture a reproducibility snapshot of the current environment.      Returns: (+8 more)

### Community 24 - "LIBERO State Smoke Test"
Cohesion: 0.22
Nodes (17): _as_tuple3(), _extract_ground_truth(), _gripper_open_heuristic(), _load_privileged_state_cls(), main(), _print_report(), Any, Try to build a real ``PrivilegedState`` from the extracted ground truth. (+9 more)

### Community 25 - "Codemaps & Project Roadmap"
Cohesion: 0.15
Nodes (16): Architecture map (Embodiment Evasion Tax), CODEMAPS index, Data contract map (records.py / cross_layer), Dependencies map, Calibration honesty (held-out FPR) invariant, Compute branches N / N- / F, H6-A oracle intrinsic action-space frontier, Kelvin2 HPC cluster (NI-HPC@QUB) (+8 more)

### Community 26 - "Action-to-State Dynamics"
Cohesion: 0.20
Nodes (11): AttackScenario, _coerce_position(), Action→privileged-state dynamics seam for the idealized attacker (§4b-(II)).  Th, Return the rollout the ``(scenario, actions)`` pair induces., Integrate ``actions`` through ``scenario`` into a :class:`Rollout`.          Arg, The LIBERO-sim backend — deferred to the GPU node (raises here).      On the gra, A reach/pick scene the dynamics integrates (model-free, env-agnostic).      Carr, RealDynamics (+3 more)

### Community 27 - "Calibration/Test Split Disjointness"
Cohesion: 0.24
Nodes (14): _as_set(), assert_disjoint(), Calibration/test split disjointness (Task 7, plan invariant #3).  Calibration se, Raise if calibration and test manifests overlap on any axis.      Args:, Materialise an axis's ids into a set (no mutation of the input)., _manifest(), Tests for calibration/test split disjointness (Task 7, invariant #3).  The harne, test_disjoint_manifests_pass() (+6 more)

### Community 28 - "Cross-Layer Eval & Tax Metrics"
Cohesion: 0.26
Nodes (13): bootstrap_delta_asr(), comparative_asr_table(), frontier_from_outcomes(), frontiers_by_layer(), _frontiers_for_pair(), Cross-layer evaluation + the Evasion-Tax metrics (§4b-III).  Runs the detection, Cluster-bootstrap CI for ΔASR at a fixed evasion (Codex review #2 #10).      The, One Pareto frontier per layer — the data behind the frontier-overlay figure. (+5 more)

### Community 29 - "Privileged-State Fixtures"
Cohesion: 0.23
Nodes (11): approaching_goal_fixture(), approaching_goal_state(), _base_fixture_dict(), near_wrong_object_fixture(), near_wrong_object_state(), Reusable synthetic privileged-state fixtures (Task 4).  These are *not* pytest t, Build a fixture dict with the standard two-object layout., Fixture dict: end-effector hovering just above the goal object. (+3 more)

### Community 30 - "L1 Internal-Representation Probe"
Cohesion: 0.20
Nodes (8): ActivationFeatures, _feature_matrix(), L1 internal-representation probe — the activation-delta arm (playbook §4b-(I))., Stack features into an ``(n, n_features)`` array, validating each width., Fit the probe on labelled features (0 = benign, 1 = injected).          Args:, Decision-step features for one rollout (immutable; plan invariant #6).      Attr, A fixed unit vector the synthetic injected-class shift points along., _signal_direction()

### Community 31 - "Synthetic Activation Extractor Tests"
Cohesion: 0.24
Nodes (11): Deterministic, model-free activation features for local tests (no OpenVLA)., SyntheticActivationExtractor, test_probe_auc_rejects_length_mismatch(), test_probe_auc_rejects_single_class_labels(), labeled_features(), ``(features, labels)`` for ``n_per_class`` benign + ``n_per_class`` attacked., test_probe_is_at_chance_on_pure_noise(), test_probe_learns_a_separable_signal_on_heldout_data() (+3 more)

### Community 32 - "OpenVLA Activation Extractor (GPU Stub)"
Cohesion: 0.22
Nodes (7): The OpenVLA forward-pass backend — deferred to the GPU node (raises here)., Score one rollout's features → injection probability in ``[0, 1]``., One decision score for ``rollout`` (a single-element list, like L0).          Ex, RealActivationExtractor, test_real_extractor_conforms_to_protocol(), test_real_extractor_is_a_gpu_only_stub(), Score

### Community 33 - "Deterministic Seeding"
Cohesion: 0.31
Nodes (9): Seed Python ``random``, NumPy, and torch (if importable).      Args:         see, seed_everything(), Tests for the deterministic seeding helper., test_different_seeds_give_different_numpy_draws(), test_does_not_mutate_inputs_and_returns_fresh_dict(), test_returns_dict_recording_applied_seed(), test_same_seed_gives_identical_numpy_draws(), test_same_seed_gives_identical_python_random_draws() (+1 more)

### Community 34 - "L1 Probe Confound Control"
Cohesion: 0.25
Nodes (8): ActivationFeatures, InternalProbe, probe_auc(), L1 confound-control scaffolding (Codex review #2 #11, playbook §4b-(I)).  The pr, ROC-AUC of ``probe`` over labelled ``features`` (benign=0, injected=1).      Spl, InternalProbe, A fitted activation-delta logistic-regression probe (higher = more injected)., test_probe_auc_is_high_for_a_separable_probe()

### Community 35 - "Cross-Layer Tax Concepts"
Cohesion: 0.33
Nodes (9): UnitOutcome / UnitKey cross-layer contract, H6-D deployable-vs-deployable cross-layer tax, L0 input layer (perplexity / text-only filter), L1 internal-representation probe (activation-delta), L2 behavioural goal-action consistency monitor, Per-layer adaptive-evasion-cost measurement (L0/L1/L2), AlignSentinel (arXiv:2602.13597), Task Drift activation deltas (arXiv:2406.00799) (+1 more)

### Community 36 - "Models, Attacks & References"
Cohesion: 0.25
Nodes (9): Example M2 run config (YAML), Operating-point power rule (benign N >= 300 @ 1% FPR), LIBERO simulation benchmark, OpenVLA-7B base model (arXiv:2406.09246), RoboGCG attack (arXiv:2506.03350), OpenVLA action codec formula provenance (c8f03f48), References provenance manifest, RoboGCG verified facts (defenses, Table 3) (+1 more)

### Community 37 - "Metric-A Concepts & Attacker"
Cohesion: 0.28
Nodes (9): Mechanism M-b idealized action-space attacker, Metric (A) privileged sim-state metric, Coverage manifest (supported/unsupported/abstained), Metric (A) frozen annotation schema v1, max combination rule (zero free parameters), P1 progress / directional alignment primitive, P2 distractor engagement primitive, P3 grasp-event appropriateness primitive (+1 more)

### Community 38 - "Threat Model & Theme Scoping"
Cohesion: 0.31
Nodes (9): Trusted-goal vs untrusted-instruction threat model, Trusted-reference ladder, Theme-scoping report (2026-05-29), EET candidate theme (instruction injection + consistency), Five VLA integrity threat channels, T5 action-tokenization plausibility defense, T9 persistent-backdoor supply-chain defense, VLA integrity theme-scoping literature review (+1 more)

### Community 39 - "Instruction-Injection Illustration (img2)"
Cohesion: 0.36
Nodes (8): Robot Arm Misdirected From Bowl to Knife (Instruction-Injection Illustration), Action Redirection / Goal Subversion, Goal-Action Consistency (Detector Concept), Instruction Injection Attack on VLA Policy, Intended Action (Green Path to Bowl), Perception/Reasoning Field (Glowing Sensing Dome), Subverted Action (Red Path to Knife), Embodied VLA Robot Arm (Gripper Manipulator)

### Community 40 - "Consistency Detector Literature"
Cohesion: 0.32
Nodes (8): actalign / Do What You Say (arXiv:2510.16281), Consistency metric (the hidden instrument), Goal-action consistency detection, Instruction Hierarchy (arXiv:2404.13208), Metric (B) learned action-semantics map, Metric (C) reference-policy divergence, Metric (D) VLM/LLM judge (excluded), SABER instruction-edit attack (arXiv:2603.24935)

### Community 41 - "Probe Label-Shuffle Control Tests"
Cohesion: 0.36
Nodes (7): Return a deterministic permutation of ``labels`` (multiset preserved).      The, shuffle_labels(), Tests for the L1 confound-control scaffolding (Codex review #2 #11).  Pre-regist, test_label_shuffle_control_collapses_probe_to_chance(), test_shuffle_labels_changes_order_and_varies_with_seed(), test_shuffle_labels_does_not_mutate_input(), test_shuffle_labels_preserves_multiset_and_is_deterministic()

### Community 42 - "Pyright Config"
Cohesion: 0.25
Nodes (7): exclude, extraPaths, include, pythonPlatform, pythonVersion, venv, venvPath

### Community 43 - "Hero Render Image (img1)"
Cohesion: 0.48
Nodes (7): Robot Arm Reaching for Knife (Hero Render), Action-Space Manipulation / Subverted Action, Glowing Action Trajectory Toward Knife, Knife (Dangerous Target Object), Industrial Robot Manipulator / Gripper, Target Reticles on Table, VLA Pipeline Integrity Threat

### Community 44 - "Rollout Reach Helpers"
Cohesion: 0.29
Nodes (4): Return all step actions stacked into an ``(len(steps), 7)`` array., Return True iff every constrained dim lies within ``[low, high]``.          Args, Return True iff the region is reached for ``persistence`` consecutive steps., ndarray

### Community 45 - "Project Instructions & Principles"
Cohesion: 0.60
Nodes (5): AGENTS.md (Codex instructions), CLAUDE.md project instructions, The Five Principles (think/simplicity/surgical/goal/document), Reproducibility non-negotiables, Embodiment Evasion Tax (project headline)

### Community 46 - "Session-Context Agent Memory"
Cohesion: 0.60
Nodes (5): session-context-loader agent definition, Codemap scan report (2026-06-05), session-context-loader MEMORY index, Project map (EET layout & sources of truth), Plan-vs-code status snapshot (2026-06-03)

### Community 47 - "Stable Seed Helper"
Cohesion: 0.40
Nodes (3): Deterministic seeding across Python ``random``, NumPy, and (optionally) torch., A process-stable 64-bit seed derived from arbitrary parts.      Hashes ``"|".joi, stable_seed()

### Community 48 - "Pareto Frontier Core"
Cohesion: 0.50
Nodes (3): _dominates(), (ASR, evasion) Pareto-frontier geometry for §4b-(II) — pure and model-free.  The, True iff ``q`` Pareto-dominates ``p`` (>= on both axes, > on at least one).

## Knowledge Gaps
- **25 isolated node(s):** `venvPath`, `venv`, `pythonVersion`, `pythonPlatform`, `extraPaths` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Score` connect `Core Data Records` to `Consistency Metric A (Scorer)`, `OpenVLA Activation Extractor (GPU Stub)`, `L1 Probe Confound Control`, `Evaluation Statistics & ROC`, `FP-Calibration & Power Rule`, `Metric-A Schema Tests`, `L0 Perplexity Filter`, `Goal-Agnostic Anomaly Baseline`, `Idealized Action Attacker`, `Hold/Allow Decision Logic`, `Oracle Outcomes & Tax Estimate`, `L1 Internal-Representation Probe`, `Synthetic Activation Extractor Tests`?**
  _High betweenness centrality (0.179) - this node is a cross-community bridge._
- **Why does `run_condition_matrix()` connect `Figures & Eval Harness` to `Evaluation Statistics & ROC`, `Write-Once Run Logger`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Why does `main()` connect `Write-Once Run Logger` to `Config Schema & GPU Guard`, `Figures & Eval Harness`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Are the 46 inferred relationships involving `Rollout` (e.g. with `AttackScenario` and `Dynamics`) actually correct?**
  _`Rollout` has 46 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `Score` (e.g. with `AttackResult` and `IdealizedActionAttacker`) actually correct?**
  _`Score` has 46 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `calibrate()` (e.g. with `test_trace_frontier_benign_fpr_is_conservative()` and `test_trace_frontier_excludes_unsupported_scenarios()`) actually correct?**
  _`calibrate()` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `FrontierPoint` (e.g. with `AttackResult` and `IdealizedActionAttacker`) actually correct?**
  _`FrontierPoint` has 26 INFERRED edges - model-reasoned connections that need verification._