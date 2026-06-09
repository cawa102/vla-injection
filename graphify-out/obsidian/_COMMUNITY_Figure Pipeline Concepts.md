---
type: community
cohesion: 0.29
members: 7
---

# Figure Pipeline Concepts

**Cohesion:** 0.29 - loosely connected
**Members:** 7 nodes

## Members
- [[ResultsTable_2]] - code - src/evasion_tax/eval/harness.py
- [[_load_results]] - code - src/evasion_tax/eval/figures.py
- [[_roc_figure]] - code - src/evasion_tax/eval/figures.py
- [[_tpr_at_fpr_figure]] - code - src/evasion_tax/eval/figures.py
- [[make_figures]] - code - src/evasion_tax/eval/figures.py
- [[results_table_to_dict]] - code - src/evasion_tax/eval/figures.py
- [[roc_auc]] - code - src/evasion_tax/eval/metrics.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Figure_Pipeline_Concepts
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Eval Harness Concepts]]
- 1 edge to [[_COMMUNITY_L1 Probe Instrument]]

## Top bridge nodes
- [[roc_auc]] - degree 3, connects to 2 communities
- [[ResultsTable_2]] - degree 2, connects to 1 community