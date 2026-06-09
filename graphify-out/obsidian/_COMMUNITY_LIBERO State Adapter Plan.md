---
type: community
members: 10
---

# LIBERO State Adapter Plan

**Members:** 10 nodes

## Members
- [[2026-06-09-libero-state-adapter]] - document - docs/plans/2026-06-09-libero-state-adapter.md
- [[LIBERO State Adapter Implementation Plan]] - document - docs/plans/2026-06-09-libero-state-adapter.md
- [[LIBERO obs fixtures provenance]] - document - tests/evasion_tax/metric/fixtures/PROVENANCE.md
- [[LIBERO simulation benchmark]] - concept - docs/core/execution-playbook.md
- [[PROVENANCE]] - document - tests/evasion_tax/metric/fixtures/PROVENANCE.md
- [[PrivilegedState contract (state.py)]] - concept - docs/core/metric-a-annotation-schema.md
- [[Relative-key (_to_) object-pose extraction filter]] - concept - docs/plans/2026-06-09-libero-state-adapter.md
- [[srcevasion_taxmetricstate.py]] - code - docs/core/metric-a-annotation-schema.md
- [[srcevasion_taxmetricstate_libero.py (LiberoStateAdapter)]] - code - docs/plans/2026-06-09-libero-state-adapter.md
- [[target_region = obj_of_interest-1 convention]] - concept - docs/plans/2026-06-09-libero-state-adapter.md

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/LIBERO_State_Adapter_Plan
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Metric (A) Frozen Schema]]
- 1 edge to [[_COMMUNITY_L1 Probe Literature]]
- 1 edge to [[_COMMUNITY_EET Core Concepts]]

## Top bridge nodes
- [[srcevasion_taxmetricstate_libero.py (LiberoStateAdapter)]] - degree 8, connects to 1 community
- [[PrivilegedState contract (state.py)]] - degree 3, connects to 1 community
- [[srcevasion_taxmetricstate.py]] - degree 3, connects to 1 community
- [[LIBERO simulation benchmark]] - degree 2, connects to 1 community