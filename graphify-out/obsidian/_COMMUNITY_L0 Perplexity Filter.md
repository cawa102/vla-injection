---
type: community
members: 38
---

# L0 Perplexity Filter

**Members:** 38 nodes

## Members
- [[.__init__()]] - code - src/evasion_tax/baselines/perplexity.py
- [[.__init__()_1]] - code - src/evasion_tax/baselines/perplexity.py
- [[.score_perplexity()]] - code - src/evasion_tax/baselines/perplexity.py
- [[.score_perplexity()_1]] - code - src/evasion_tax/baselines/perplexity.py
- [[.score_perplexity()_2]] - code - src/evasion_tax/baselines/perplexity.py
- [[.score_rollout()_1]] - code - src/evasion_tax/baselines/perplexity.py
- [[A backend mapping an instruction string to a perplexity ``= 1``.]] - rationale - src/evasion_tax/baselines/perplexity.py
- [[Crude deterministic perplexity surrogate (NOT a real LM perplexity).      A test]] - rationale - src/evasion_tax/baselines/perplexity.py
- [[Dependency-Inversion GPU swap seam (local synthetic vs deferred LIBEROGPU backend)]] - rationale - src/evasion_tax/attack/dynamics.py
- [[Deterministic perplexity backend for tests (no LM).      Args         table Op]] - rationale - src/evasion_tax/baselines/perplexity.py
- [[Map perplexity ``= 1`` to an inconsistency score in ``0, 1)`` (monotone).]] - rationale - src/evasion_tax/baselines/perplexity.py
- [[MockPerplexityScorer]] - code - src/evasion_tax/baselines/perplexity.py
- [[Perplexity  text-only filter baseline (Task 8).  RoboGCG's borrowed defences in]] - rationale - src/evasion_tax/baselines/perplexity.py
- [[PerplexityFilter]] - code - src/evasion_tax/baselines/perplexity.py
- [[PerplexityScorer]] - code - src/evasion_tax/baselines/perplexity.py
- [[RealPerplexityScorer]] - code - src/evasion_tax/baselines/perplexity.py
- [[Return a single-element score list for one rollout.          The score is derive]] - rationale - src/evasion_tax/baselines/perplexity.py
- [[Return the perplexity of ``instruction`` (higher = less natural).]] - rationale - src/evasion_tax/baselines/perplexity.py
- [[Tests for the perplexity  text-only filter baseline (Task 8).  The published Ro]] - rationale - tests/evasion_tax/baselines/test_perplexity.py
- [[Text-only detector scores a rollout by its instruction's perplexity.      Args]] - rationale - src/evasion_tax/baselines/perplexity.py
- [[The real LM-perplexity backend — deferred to the GPU node (raises if used here).]] - rationale - src/evasion_tax/baselines/perplexity.py
- [[_heuristic_perplexity]] - code - src/evasion_tax/baselines/perplexity.py
- [[_heuristic_perplexity()]] - code - src/evasion_tax/baselines/perplexity.py
- [[_perplexity_to_score]] - code - src/evasion_tax/baselines/perplexity.py
- [[_perplexity_to_score()]] - code - src/evasion_tax/baselines/perplexity.py
- [[make_rollout()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[perplexity.py]] - code - src/evasion_tax/baselines/perplexity.py
- [[test_calibrates_identically_through_shared_calibrate()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_empty_rollout_raises()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_higher_perplexity_maps_to_higher_score()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_mock_heuristic_empty_instruction_is_minimal_perplexity()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_mock_heuristic_rates_symbol_heavy_text_more_perplexing_than_english()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_perplexity.py]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_perplexity_one_maps_to_zero_and_below_one_clamps()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_real_backend_is_gpu_stub()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_real_backend_plugs_into_filter_but_errors_when_used()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_score_rollout_returns_single_unit_interval_score()]] - code - tests/evasion_tax/baselines/test_perplexity.py
- [[test_scores_the_operational_instruction_not_the_trusted_goal()]] - code - tests/evasion_tax/baselines/test_perplexity.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/L0_Perplexity_Filter
SORT file.name ASC
```

## Connections to other communities
- 17 edges to [[_COMMUNITY_Rollout]]
- 3 edges to [[_COMMUNITY_Detector Decision Logic]]
- 2 edges to [[_COMMUNITY_Detector Calibration]]
- 1 edge to [[_COMMUNITY_Goal-Agnostic Anomaly Baseline]]
- 1 edge to [[_COMMUNITY_Oracle Frontier Tests]]

## Top bridge nodes
- [[PerplexityFilter]] - degree 19, connects to 4 communities
- [[test_perplexity.py]] - degree 18, connects to 3 communities
- [[test_calibrates_identically_through_shared_calibrate()]] - degree 6, connects to 2 communities
- [[perplexity.py]] - degree 26, connects to 1 community
- [[MockPerplexityScorer]] - degree 17, connects to 1 community