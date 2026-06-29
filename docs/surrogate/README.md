# Surrogate-GCG (topic)

Surrogate transfer of RoboGCG suffixes on bf16/int8/NF4 OpenVLA. The plans that
define this topic live in `plan/`.

**Outputs (topic-grouped):**
- Results (write-once): `results/surrogate/<UTC>-surrogate-gcg/` — `run.json` +
  `surrogate_suffix_artifact.json` (metrics + suffix SHA-256; no payload).
- Logs: `logs/surrogate/` — per-precision GCG search logs + `pilot.log`.
- Suffix payload (quarantined, gitignored): `artifacts/untrusted/<UTC>-surrogate-gcg/`.

New runs land here via `scripts/run_surrogate_gcg.py --results-root results/surrogate`.
