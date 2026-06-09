---
type: community
cohesion: 0.67
members: 3
---

# Run Logger & Env Capture

**Cohesion:** 0.67 - moderately connected
**Members:** 3 nodes

## Members
- [[RunHandle (write-once)]] - code - src/evasion_tax/repro/run_logger.py
- [[RunLogger_1]] - code - src/evasion_tax/repro/run_logger.py
- [[capture_env]] - code - src/evasion_tax/repro/env_capture.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Run_Logger__Env_Capture
SORT file.name ASC
```
