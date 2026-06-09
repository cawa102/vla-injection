---
type: community
members: 79
---

# L1 Internal Probe

**Members:** 79 nodes

## Members
- [[.__init__()_2]] - code - src/evasion_tax/metric/probe_internal.py
- [[.__post_init__()_7]] - code - src/evasion_tax/metric/probe_internal.py
- [[.extract()]] - code - src/evasion_tax/metric/probe_internal.py
- [[.extract()_1]] - code - src/evasion_tax/metric/probe_internal.py
- [[.extract()_2]] - code - src/evasion_tax/metric/probe_internal.py
- [[.fit()]] - code - src/evasion_tax/metric/probe_internal.py
- [[.score()_1]] - code - src/evasion_tax/metric/probe_internal.py
- [[.score_rollout()_3]] - code - src/evasion_tax/metric/probe_internal.py
- [[A fitted activation-delta logistic-regression probe (higher = more injected).]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[A fixed unit vector the synthetic injected-class shift points along.]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[A minimal rollout carrying just what the synthetic extractor keys off.      The]] - rationale - tests/evasion_tax/metric/test_probe_internal.py
- [[ActivationExtractor]] - code - src/evasion_tax/metric/probe_internal.py
- [[ActivationFeatures]] - code - src/evasion_tax/metric/probe_confounds.py
- [[ActivationFeatures_1]] - code - src/evasion_tax/metric/probe_internal.py
- [[Decision-step features for one rollout (immutable; plan invariant 6).      Attr]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[Deterministic, model-free activation features for local tests (no OpenVLA).]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[Fit the probe on labelled features (0 = benign, 1 = injected).          Args]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[InternalProbe]] - code - src/evasion_tax/metric/probe_confounds.py
- [[InternalProbe_1]] - code - src/evasion_tax/metric/probe_internal.py
- [[L1 confound-control scaffolding (Codex review 2 11, playbook §4b-(I)).  The pr]] - rationale - src/evasion_tax/metric/probe_confounds.py
- [[L1 internal-representation probe — the activation-delta arm (playbook §4b-(I)).]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[One decision score for ``rollout`` (a single-element list, like L0).          Ex]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[ROC-AUC of ``probe`` over labelled ``features`` (benign=0, injected=1).      Spl]] - rationale - src/evasion_tax/metric/probe_confounds.py
- [[RealActivationExtractor]] - code - src/evasion_tax/metric/probe_internal.py
- [[Return a deterministic permutation of ``labels`` (multiset preserved).      The]] - rationale - src/evasion_tax/metric/probe_confounds.py
- [[Return the decision-step activation features for ``rollout``.]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[Rollout_7]] - code - src/evasion_tax/metric/probe_internal.py
- [[RolloutStep (shared record)]] - code - src/evasion_tax/records.py
- [[Score_6]] - code - src/evasion_tax/metric/probe_internal.py
- [[Score one rollout's features → injection probability in ``0, 1``.]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[Seam mapping a rollout to its decision-step class`ActivationFeatures`.      Th]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[Stack features into an ``(n, n_features)`` array, validating each width.]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[SyntheticActivationExtractor]] - code - src/evasion_tax/metric/probe_internal.py
- [[Tests for the L1 confound-control scaffolding (Codex review 2 11).  Pre-regist]] - rationale - tests/evasion_tax/metric/test_probe_confounds.py
- [[Tests for the L1 internal-representation probe (playbook §4b-(I)).  The L1 l]] - rationale - tests/evasion_tax/metric/test_probe_internal.py
- [[The OpenVLA forward-pass backend — deferred to the GPU node (raises here).]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[_feature_matrix()]] - code - src/evasion_tax/metric/probe_internal.py
- [[_signal_direction()]] - code - src/evasion_tax/metric/probe_internal.py
- [[``(features, labels)`` for ``n_per_class`` benign + ``n_per_class`` attacked.]] - rationale - tests/evasion_tax/metric/test_probe_internal.py
- [[fit_separable_probe()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[labeled_features()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[make_rollout()_1]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[ndarray_6]] - code - src/evasion_tax/metric/probe_internal.py
- [[probe_auc()]] - code - src/evasion_tax/metric/probe_confounds.py
- [[probe_confounds.py]] - code - src/evasion_tax/metric/probe_confounds.py
- [[probe_internal.py]] - code - src/evasion_tax/metric/probe_internal.py
- [[shuffle_labels (label-shuffle control)]] - code - src/evasion_tax/metric/probe_confounds.py
- [[shuffle_labels()]] - code - src/evasion_tax/metric/probe_confounds.py
- [[test_activation_features_defaults_window_end_to_zero()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_activation_features_is_immutable()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_activation_features_rejects_empty_delta()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_activation_features_rejects_non_finite_delta()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_benign_weird_suffix_does_not_fire_at_calibrated_threshold()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_calibrates_through_shared_calibrate_and_flags_injection()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_fit_rejects_empty_features()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_fit_rejects_length_mismatch()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_fit_requires_both_classes()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_label_shuffle_control_collapses_probe_to_chance()]] - code - tests/evasion_tax/metric/test_probe_confounds.py
- [[test_probe_auc_is_high_for_a_separable_probe()]] - code - tests/evasion_tax/metric/test_probe_confounds.py
- [[test_probe_auc_rejects_length_mismatch()]] - code - tests/evasion_tax/metric/test_probe_confounds.py
- [[test_probe_auc_rejects_single_class_labels()]] - code - tests/evasion_tax/metric/test_probe_confounds.py
- [[test_probe_confounds.py]] - code - tests/evasion_tax/metric/test_probe_confounds.py
- [[test_probe_internal.py]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_probe_is_at_chance_on_pure_noise()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_probe_is_immutable()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_probe_learns_a_separable_signal_on_heldout_data()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_real_extractor_conforms_to_protocol()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_real_extractor_is_a_gpu_only_stub()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_score_rejects_wrong_feature_dim()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_score_returns_unit_interval_score_with_propagated_window_end()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_score_rollout_returns_single_decision_score()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_shuffle_labels_changes_order_and_varies_with_seed()]] - code - tests/evasion_tax/metric/test_probe_confounds.py
- [[test_shuffle_labels_does_not_mutate_input()]] - code - tests/evasion_tax/metric/test_probe_confounds.py
- [[test_shuffle_labels_preserves_multiset_and_is_deterministic()]] - code - tests/evasion_tax/metric/test_probe_confounds.py
- [[test_synthetic_extractor_conforms_to_protocol()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_synthetic_extractor_is_deterministic()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_synthetic_extractor_rejects_empty_rollout()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_synthetic_extractor_shifts_attacked_features_off_benign()]] - code - tests/evasion_tax/metric/test_probe_internal.py
- [[test_synthetic_extractor_validates_construction()]] - code - tests/evasion_tax/metric/test_probe_internal.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/L1_Internal_Probe
SORT file.name ASC
```

## Connections to other communities
- 20 edges to [[_COMMUNITY_Rollout]]
- 7 edges to [[_COMMUNITY_run_condition_matrix]]
- 3 edges to [[_COMMUNITY_Detector Calibration]]
- 3 edges to [[_COMMUNITY_Detector Decision Logic]]
- 1 edge to [[_COMMUNITY_Oracle Frontier Tests]]
- 1 edge to [[_COMMUNITY_Detector Metrics & CIs]]
- 1 edge to [[_COMMUNITY_Deterministic Seeding]]
- 1 edge to [[_COMMUNITY_Metric (A) Tests]]
- 1 edge to [[_COMMUNITY_Rollout Records Tests]]

## Top bridge nodes
- [[RolloutStep (shared record)]] - degree 5, connects to 4 communities
- [[test_probe_internal.py]] - degree 37, connects to 3 communities
- [[probe_internal.py]] - degree 37, connects to 2 communities
- [[test_calibrates_through_shared_calibrate_and_flags_injection()]] - degree 5, connects to 2 communities
- [[test_benign_weird_suffix_does_not_fire_at_calibrated_threshold()]] - degree 5, connects to 2 communities