---
type: community
members: 30
---

# Detector Decision Logic

**Members:** 30 nodes

## Members
- [[Build an ordered list of Score objects with window_end = start, start+1, ...]] - rationale - tests/evasion_tax/detector/test_decide.py
- [[Causal holdallow decisions from a calibrated threshold (Task 6).  A consistency]] - rationale - src/evasion_tax/detector/decide.py
- [[Decide holdallow for a single step.      Args         score The step's consis]] - rationale - src/evasion_tax/detector/decide.py
- [[Decision]] - code - src/evasion_tax/detector/decide.py
- [[Decision (shared record)]] - code - src/evasion_tax/records.py
- [[Return the first-exceedance decision for one ordered rollout.      Scans ``score]] - rationale - src/evasion_tax/detector/decide.py
- [[Score_8]] - code - tests/evasion_tax/detector/test_decide.py
- [[Score (shared record)]] - code - src/evasion_tax/records.py
- [[Steps of deviation before the hold fired, relative to attack onset.      Args]] - rationale - src/evasion_tax/detector/decide.py
- [[Tests for the single-step  per-rollout decision logic (Task 6).  Covers strict]] - rationale - tests/evasion_tax/detector/test_decide.py
- [[_max_consistency]] - code - src/evasion_tax/attack/idealized_frontier.py
- [[decide()]] - code - src/evasion_tax/detector/decide.py
- [[decide.py]] - code - src/evasion_tax/detector/decide.py
- [[detection_latency()]] - code - src/evasion_tax/detector/decide.py
- [[rollout_fires()]] - code - src/evasion_tax/detector/decide.py
- [[scores()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_decide.py]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_decide_boundary_value_equal_tau_does_not_fire()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_decide_does_not_fire_when_value_below_tau()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_decide_fires_when_value_strictly_exceeds_tau()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_decide_matches_value_gt_tau_exactly_over_a_grid()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_detection_latency_none_when_fire_before_onset()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_detection_latency_none_when_never_fired()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_detection_latency_positive_when_fire_after_onset()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_detection_latency_zero_when_fire_at_onset()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_rollout_fires_does_not_mutate_input()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_rollout_fires_empty_rollout_does_not_fire()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_rollout_fires_is_causal_first_not_max()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_rollout_fires_no_exceedance_returns_hold_false_step_minus_one()]] - code - tests/evasion_tax/detector/test_decide.py
- [[test_rollout_fires_returns_first_exceedance_step()]] - code - tests/evasion_tax/detector/test_decide.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Detector_Decision_Logic
SORT file.name ASC
```

## Connections to other communities
- 15 edges to [[_COMMUNITY_Rollout]]
- 4 edges to [[_COMMUNITY_Detector Calibration]]
- 3 edges to [[_COMMUNITY_Goal-Agnostic Anomaly Baseline]]
- 3 edges to [[_COMMUNITY_L0 Perplexity Filter]]
- 3 edges to [[_COMMUNITY_L1 Internal Probe]]
- 2 edges to [[_COMMUNITY_Rollout Records Tests]]
- 1 edge to [[_COMMUNITY_Metric (A) Tests]]

## Top bridge nodes
- [[rollout_fires()]] - degree 24, connects to 5 communities
- [[Score (shared record)]] - degree 7, connects to 5 communities
- [[Decision (shared record)]] - degree 5, connects to 2 communities
- [[test_decide.py]] - degree 21, connects to 1 community
- [[decide()]] - degree 11, connects to 1 community