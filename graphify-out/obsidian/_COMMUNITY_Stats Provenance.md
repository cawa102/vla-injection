---
type: community
members: 35
---

# Stats Provenance

**Members:** 35 nodes

## Members
- [[ActionCodec (OpenVLA token decode)]] - code - src/evasion_tax/policy/action_codec.py
- [[Append or update one provenance entry in a JSON manifest.      The manifest is a]] - rationale - src/evasion_tax/repro/provenance.py
- [[Datacheckpoint provenance SHA-256 hashing and a JSON manifest.  Records the so]] - rationale - src/evasion_tax/repro/provenance.py
- [[Fetch + provenance helpers for OpenVLA ``dataset_statistics.json`` (Task 3).  Th]] - rationale - src/evasion_tax/policy/openvla_stats.py
- [[Hash a downloaded stats file and record one provenance entry for it.      Args]] - rationale - src/evasion_tax/policy/openvla_stats.py
- [[Namespace]] - code - scripts/fetch_openvla_stats.py
- [[Return the direct download URL of a checkpoint's ``dataset_statistics.json``.]] - rationale - src/evasion_tax/policy/openvla_stats.py
- [[Return the hex SHA-256 digest of a file, read in streaming chunks.      Args]] - rationale - src/evasion_tax/repro/provenance.py
- [[StrPath_2]] - code - src/evasion_tax/policy/openvla_stats.py
- [[StrPath_3]] - code - src/evasion_tax/repro/provenance.py
- [[Tests for the OpenVLA dataset-statistics fetch helpers (Task 3).  The network do]] - rationale - tests/evasion_tax/policy/test_openvla_stats.py
- [[Tests for the SHA-256 helper and the JSON provenance manifest.]] - rationale - tests/evasion_tax/repro/test_provenance.py
- [[_parse_args()]] - code - scripts/fetch_openvla_stats.py
- [[fetch_openvla_stats.py]] - code - scripts/fetch_openvla_stats.py
- [[main()_5]] - code - scripts/fetch_openvla_stats.py
- [[openvla_stats.py]] - code - src/evasion_tax/policy/openvla_stats.py
- [[provenance.py]] - code - src/evasion_tax/repro/provenance.py
- [[record_provenance]] - code - src/evasion_tax/repro/provenance.py
- [[record_provenance()]] - code - src/evasion_tax/repro/provenance.py
- [[record_stats_provenance]] - code - src/evasion_tax/policy/openvla_stats.py
- [[record_stats_provenance()]] - code - src/evasion_tax/policy/openvla_stats.py
- [[sha256_file]] - code - src/evasion_tax/repro/provenance.py
- [[sha256_file()]] - code - src/evasion_tax/repro/provenance.py
- [[stats_url]] - code - src/evasion_tax/policy/openvla_stats.py
- [[stats_url()]] - code - src/evasion_tax/policy/openvla_stats.py
- [[test_openvla_stats.py]] - code - tests/evasion_tax/policy/test_openvla_stats.py
- [[test_provenance.py]] - code - tests/evasion_tax/repro/test_provenance.py
- [[test_record_provenance_appends_distinct_names()]] - code - tests/evasion_tax/repro/test_provenance.py
- [[test_record_provenance_creates_manifest_and_round_trips()]] - code - tests/evasion_tax/repro/test_provenance.py
- [[test_record_provenance_updating_same_name_replaces_only_that_entry()]] - code - tests/evasion_tax/repro/test_provenance.py
- [[test_record_stats_provenance_hashes_file_and_writes_a_keyed_entry()]] - code - tests/evasion_tax/policy/test_openvla_stats.py
- [[test_sha256_file_empty_file()]] - code - tests/evasion_tax/repro/test_provenance.py
- [[test_sha256_file_matches_known_hash()]] - code - tests/evasion_tax/repro/test_provenance.py
- [[test_sha256_file_missing_path_raises()]] - code - tests/evasion_tax/repro/test_provenance.py
- [[test_stats_url_points_at_the_checkpoint_dataset_statistics_file()]] - code - tests/evasion_tax/policy/test_openvla_stats.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Stats_Provenance
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Run Logging & Rollout Demo]]
- 1 edge to [[_COMMUNITY_OpenVLA Action Codec]]
- 1 edge to [[_COMMUNITY_Rollout]]

## Top bridge nodes
- [[ActionCodec (OpenVLA token decode)]] - degree 3, connects to 2 communities
- [[fetch_openvla_stats.py]] - degree 4, connects to 1 community