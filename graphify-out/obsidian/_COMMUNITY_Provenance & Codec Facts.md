---
type: community
cohesion: 0.50
members: 5
---

# Provenance & Codec Facts

**Cohesion:** 0.50 - moderately connected
**Members:** 5 nodes

## Members
- [[ActionCodec (OpenVLA token decode)]] - code - src/evasion_tax/policy/action_codec.py
- [[record_provenance]] - code - src/evasion_tax/repro/provenance.py
- [[record_stats_provenance]] - code - src/evasion_tax/policy/openvla_stats.py
- [[sha256_file]] - code - src/evasion_tax/repro/provenance.py
- [[stats_url]] - code - src/evasion_tax/policy/openvla_stats.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Provenance__Codec_Facts
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Target Action Spec]]

## Top bridge nodes
- [[ActionCodec (OpenVLA token decode)]] - degree 2, connects to 1 community