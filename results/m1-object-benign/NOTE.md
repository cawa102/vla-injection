# m1-object-benign — provenance

Full 10-task libero_object benign baseline, N=300 (30/task), bf16/flash_attn2, cuda:0,
seed 42, 2026-07-10. success_rate=0.663 (below OpenVLA published ~88% — reproducibility
item to root-cause before the M1 "benign reproduced" gate; center_crop + settle both
correct, likely init-cycling / 300-vs-500-episode variance).

- `episodes/` (300 per-step RolloutStep JSONs, ~80 MB) is **gitignored** — the write-once
  raw record lives on the box disk only. Derived views below are committed.
- `benign_records.json` / `benign_summary.json` / `geometry_stats.json` — aggregated views.
- `schema_repinned.json` — the frozen DM-3 benign-pinned SchemaA the attack consumes.
- `benign_nonreach_guard.json` (+ `.py`) — Guard 1: 0/30 benign alphabet-soup rollouts
  reach salad_dressing_1 (PASS; closest 0.086 m).
- `adversary_reachfloor.json` (+ `.py`) — Guard 2: clean adversary rollout reaches
  salad_dressing_1 at 0.042 m (PASS).
