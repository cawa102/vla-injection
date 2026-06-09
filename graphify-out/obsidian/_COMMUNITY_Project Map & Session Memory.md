---
type: community
members: 8
---

# Project Map & Session Memory

**Members:** 8 nodes

## Members
- [[Codemap scan report (2026-06-05)]] - document - .reports/codemap-diff.txt
- [[MEMORY]] - document - .claude/agent-memory/session-context-loader/MEMORY.md
- [[Project map (EET layout & sources of truth)]] - document - .claude/agent-memory/session-context-loader/project-map.md
- [[codemap-diff.txt]] - document - .reports/codemap-diff.txt
- [[project-map]] - document - .claude/agent-memory/session-context-loader/project-map.md
- [[session-context-loader MEMORY index]] - document - .claude/agent-memory/session-context-loader/MEMORY.md
- [[session-context-loader agent definition]] - document - .claude/agents/session-context-loader.md
- [[session-context-loader]] - document - .claude/agents/session-context-loader.md

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Project_Map__Session_Memory
SORT file.name ASC
```
