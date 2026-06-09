---
type: community
cohesion: 0.28
members: 9
---

# L1 Probe Instrument

**Cohesion:** 0.28 - loosely connected
**Members:** 9 nodes

## Members
- [[ActivationExtractor (Protocol seam)]] - code - src/evasion_tax/metric/probe_internal.py
- [[ActivationFeatures_2]] - code - src/evasion_tax/metric/probe_internal.py
- [[InternalProbe (L1 logistic-regression probe)]] - code - src/evasion_tax/metric/probe_internal.py
- [[RealActivationExtractor (GPU stub)]] - code - src/evasion_tax/metric/probe_internal.py
- [[Shared calibrate  invariant 4 (one footing across layers)]] - concept - src/evasion_tax/metric/probe_internal.py
- [[SyntheticActivationExtractor_1]] - code - src/evasion_tax/metric/probe_internal.py
- [[probe_auc]] - code - src/evasion_tax/metric/probe_confounds.py
- [[seed_everything]] - code - src/evasion_tax/repro/seeds.py
- [[stable_seed_1]] - code - src/evasion_tax/repro/seeds.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/L1_Probe_Instrument
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Core Records & Protocols]]
- 1 edge to [[_COMMUNITY_Figure Pipeline Concepts]]
- 1 edge to [[_COMMUNITY_Eval Harness Concepts]]
- 1 edge to [[_COMMUNITY_Metric (A) Oracle Design]]

## Top bridge nodes
- [[InternalProbe (L1 logistic-regression probe)]] - degree 5, connects to 1 community
- [[SyntheticActivationExtractor_1]] - degree 4, connects to 1 community
- [[probe_auc]] - degree 3, connects to 1 community
- [[Shared calibrate  invariant 4 (one footing across layers)]] - degree 2, connects to 1 community