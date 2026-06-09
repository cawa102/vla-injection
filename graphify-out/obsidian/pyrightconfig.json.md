---
source_file: "pyrightconfig.json"
type: "code"
community: "Pyright Config"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Pyright_Config
---

# pyrightconfig.json

## Connections
- [[exclude]] - `contains` [EXTRACTED]
- [[extraPaths]] - `contains` [EXTRACTED]
- [[include]] - `contains` [EXTRACTED]
- [[pythonPlatform]] - `contains` [EXTRACTED]
- [[pythonVersion]] - `contains` [EXTRACTED]
- [[venv]] - `contains` [EXTRACTED]
- [[venvPath]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Pyright_Config

## 📄 Source

`pyrightconfig.json`

```json
{
  "venvPath": ".",
  "venv": ".venv",
  "pythonVersion": "3.11",
  "pythonPlatform": "Darwin",
  "extraPaths": ["src"],
  "include": ["src", "tests"],
  "exclude": ["**/__pycache__", ".venv"]
}
```

