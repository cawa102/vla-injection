<!-- Generated: 2026-06-26 (full regen) | Files scanned: 130 (52 src · 24 scripts · 54 tests) | Token estimate: ~340 -->

# CODEMAPS — Embodiment Evasion Tax

Token-lean architecture map for AI context. MSc research code, **not** an app: no HTTP API, no UI, no DB.
Python package `src/evasion_tax` (model-free core, unit-tested on the mac) + CLI `scripts/` (GPU runs on
the CSB A5000 box). Source of truth for *what & why* = `docs/core/execution-playbook.md`; these maps
cover *structure*.

## Map files
| File | Covers |
|------|--------|
| [architecture.md](architecture.md) | What it measures · two pipelines (detection + surrogate transfer) · model-free/GPU split · headline split · milestones |
| [modules.md](modules.md) | Per-package public symbols (stands in for `backend.md` — library, not HTTP backend) |
| [data.md](data.md) | `records.py` contract · cross-layer `UnitOutcome` · surrogate-transfer records (schema v2) · invariants · provenance |
| [dependencies.md](dependencies.md) | Runtime deps · heavy GPU stack (torch/bnb/accelerate pins) · external artifacts · CLI entry points |

`backend.md` / `frontend.md` from the `/update-codemaps` template are **omitted**: there is no HTTP
API and no UI. The module map fills the package-internals role those files would serve in a service.

## Scan summary
**2026-06-26** — full `/update-codemaps` regen (HEAD `2096ab6`). Since the 06-09 reconciliation the
package roughly doubled: the **GCG attack core** (`attack/gcg`, `gcg_openvla`, `openvla_loader`,
`early_stop*`, `redirect_target`), the **quantized-surrogate transfer** subsystem (`attack/surrogate_artifacts`,
`eval/surrogate_transfer`), the **real-rollout L2 + IO** (`eval/rollout_runner`, `rollout_io`, `separation`),
and the **M1 gate + compute-branch select** (`eval/m1_gate`, `branch_select`, `schema_repin`) were added,
plus `policy/action_check`. GPU work moved from "deferred to Kelvin2" to **actively running on the CSB A5000**.
- Scanned: **52** `src/evasion_tax` `.py` (43 modules · 9 `__init__`, ~7.8k LOC) · **24** `scripts` ·
  **54** `test_*` modules (**608 test functions**) · `pyproject.toml` · `configs/` (5 surrogate YAMLs).
- Structure: single Python package, layered model-free core + `Protocol`/seam-gated GPU pieces; no routes/DB/UI.
- Manual framing (model-free↔GPU split, headline H6-A/H6-D split, reproducibility invariants) preserved verbatim.
- Regenerate after each milestone (M-tag) or when `src/evasion_tax` subpackages change.
