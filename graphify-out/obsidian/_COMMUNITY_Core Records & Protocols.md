---
type: community
cohesion: 0.10
members: 33
---

# Core Records & Protocols

**Cohesion:** 0.10 - loosely connected
**Members:** 33 nodes

## Members
- [[.__len__()]] - code - src/evasion_tax/records.py
- [[.__post_init__()_12]] - code - src/evasion_tax/records.py
- [[.extract()]] - code - src/evasion_tax/metric/probe_internal.py
- [[.extract()_2]] - code - src/evasion_tax/metric/probe_internal.py
- [[.score()_1]] - code - src/evasion_tax/metric/probe_internal.py
- [[.score_rollout()_3]] - code - src/evasion_tax/metric/probe_internal.py
- [[A causal consistency score; higher = more inconsistent with the goal.      ``val]] - rationale - src/evasion_tax/records.py
- [[ActivationExtractor]] - code - src/evasion_tax/metric/probe_internal.py
- [[An ordered, immutable sequence of rollout steps.]] - rationale - src/evasion_tax/records.py
- [[Any]] - code - scripts/demo_metric_separation.py
- [[Core immutable data records shared across metric  detector  eval (Task 2).  Th]] - rationale - src/evasion_tax/records.py
- [[FP-calibration of the detector threshold (Task 6).  Chooses ``tau`` on a benign]] - rationale - src/evasion_tax/detector/calibrate.py
- [[GoalResolver]] - code - src/evasion_tax/metric/consistency_a.py
- [[One decision score for ``rollout`` (a single-element list, like L0).          Ex]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[Protocol]] - code
- [[Return the decision-step activation features for ``rollout``.]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[Rollout_8]] - code - src/evasion_tax/records.py
- [[Rollout_4]] - code - src/evasion_tax/baselines/anomaly.py
- [[Rollout_5]] - code - src/evasion_tax/baselines/perplexity.py
- [[Rollout_7]] - code - src/evasion_tax/metric/probe_internal.py
- [[Rollout_9]] - code - tests/evasion_tax/test_records.py
- [[RolloutStep_3]] - code - tests/evasion_tax/test_records.py
- [[Score_7]] - code - src/evasion_tax/records.py
- [[Score_2]] - code - src/evasion_tax/baselines/anomaly.py
- [[Score_3]] - code - src/evasion_tax/baselines/perplexity.py
- [[Score_6]] - code - src/evasion_tax/metric/probe_internal.py
- [[Score one rollout's features → injection probability in ``0, 1``.]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[Seam mapping a rollout to its decision-step class`ActivationFeatures`.      Th]] - rationale - src/evasion_tax/metric/probe_internal.py
- [[Seam mapping a step (+ its privileged state) to a class`GoalAnchor`.      v1 s]] - rationale - src/evasion_tax/metric/consistency_a.py
- [[calibrate.py_1]] - code - src/evasion_tax/detector/calibrate.py
- [[ndarray_3]] - code - src/evasion_tax/baselines/anomaly.py
- [[ndarray_6]] - code - src/evasion_tax/metric/probe_internal.py
- [[records.py]] - code - src/evasion_tax/records.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Core_Records__Protocols
SORT file.name ASC
```

## Connections to other communities
- 20 edges to [[_COMMUNITY_Metric (A) Consistency Scorer]]
- 20 edges to [[_COMMUNITY_L1 Internal Probe]]
- 14 edges to [[_COMMUNITY_Attack Dynamics & Rollout]]
- 13 edges to [[_COMMUNITY_L0 Perplexity Filter]]
- 10 edges to [[_COMMUNITY_Idealized Action Attacker]]
- 8 edges to [[_COMMUNITY_Goal-Agnostic Anomaly Baseline]]
- 8 edges to [[_COMMUNITY_Detector Decision Logic]]
- 7 edges to [[_COMMUNITY_Idealized Frontier Attacker]]
- 6 edges to [[_COMMUNITY_Metric Separation Demo]]
- 6 edges to [[_COMMUNITY_Action-Region Attack Search]]
- 6 edges to [[_COMMUNITY_Detector Calibration]]
- 5 edges to [[_COMMUNITY_Run Logging & Rollout Demo]]
- 4 edges to [[_COMMUNITY_Target Action Spec]]
- 3 edges to [[_COMMUNITY_Metric (A) Schema Tests]]
- 3 edges to [[_COMMUNITY_Oracle Frontier Tests]]
- 3 edges to [[_COMMUNITY_Detector Metrics & CIs]]
- 3 edges to [[_COMMUNITY_Rollout Records Tests]]
- 2 edges to [[_COMMUNITY_Cross-Layer Tax Eval]]
- 2 edges to [[_COMMUNITY_Goal Resolver Tests]]
- 2 edges to [[_COMMUNITY_Metric (A) Oracle Design]]
- 1 edge to [[_COMMUNITY_Eval Harness & Power]]
- 1 edge to [[_COMMUNITY_Privileged State Adapter]]
- 1 edge to [[_COMMUNITY_Synthetic State Fixtures]]
- 1 edge to [[_COMMUNITY_Metric (A) Tests]]
- 1 edge to [[_COMMUNITY_LIBERO State Adapter Tests]]
- 1 edge to [[_COMMUNITY_L1 Probe Instrument]]

## Top bridge nodes
- [[records.py]] - degree 31, connects to 17 communities
- [[Score_7]] - degree 55, connects to 16 communities
- [[Rollout_8]] - degree 59, connects to 15 communities
- [[GoalResolver]] - degree 10, connects to 4 communities
- [[Protocol]] - degree 6, connects to 4 communities