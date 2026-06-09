---
type: community
members: 13
---

# Figure Regeneration Tests

**Members:** 13 nodes

## Members
- [[ResultsTable]] - code - src/evasion_tax/eval/figures.py
- [[Serialise a class`ResultsTable` to the ``results.json`` schema.      Merges ea]] - rationale - src/evasion_tax/eval/figures.py
- [[Tests for script-regenerable figures (Task 9).  ``make_figures`` is the M2 deliv]] - rationale - tests/evasion_tax/eval/test_figures.py
- [[_synthetic_results()]] - code - tests/evasion_tax/eval/test_figures.py
- [[_write_results()]] - code - tests/evasion_tax/eval/test_figures.py
- [[results_table_to_dict()]] - code - src/evasion_tax/eval/figures.py
- [[test_figures.py]] - code - tests/evasion_tax/eval/test_figures.py
- [[test_make_figures_creates_missing_out_dir()]] - code - tests/evasion_tax/eval/test_figures.py
- [[test_make_figures_raises_when_results_missing()]] - code - tests/evasion_tax/eval/test_figures.py
- [[test_make_figures_returns_written_paths()]] - code - tests/evasion_tax/eval/test_figures.py
- [[test_make_figures_writes_expected_files()]] - code - tests/evasion_tax/eval/test_figures.py
- [[test_serialised_table_carries_power_block_flagging_underpowered_points()]] - code - tests/evasion_tax/eval/test_figures.py
- [[test_serialised_table_round_trips_into_make_figures()]] - code - tests/evasion_tax/eval/test_figures.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Figure_Regeneration_Tests
SORT file.name ASC
```

## Connections to other communities
- 10 edges to [[_COMMUNITY_Figure Generation]]
- 3 edges to [[_COMMUNITY_Eval Harness & Power]]
- 1 edge to [[_COMMUNITY_CalibTest Split Disjointness]]
- 1 edge to [[_COMMUNITY_Eval Harness & Power]]

## Top bridge nodes
- [[test_figures.py]] - degree 12, connects to 2 communities
- [[results_table_to_dict()]] - degree 8, connects to 2 communities
- [[test_serialised_table_round_trips_into_make_figures()]] - degree 4, connects to 2 communities
- [[ResultsTable]] - degree 3, connects to 2 communities
- [[test_make_figures_writes_expected_files()]] - degree 4, connects to 1 community