# Setup HTML Guides Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create human-readable static HTML guide versions of the Markdown setup documents under `docs/setup/`.

**Architecture:** Add one shared stylesheet and one standalone HTML page per setup Markdown file in `docs/setup/html/`. Keep the source Markdown untouched, preserve the technical commands and caveats, and make each page easier to scan with diagrams, status cards, timelines, and checklists.

**Tech Stack:** Static HTML5, CSS3, inline SVG diagrams, no external runtime dependencies.

---

- [ ] Task 1: Shared Setup Guide Visual System

**Files:**
- Create: `docs/setup/html/setup-guide.css`

**What:** Define a reusable document layout for setup guides: sticky navigation, status cards, flow diagrams, checklists, command blocks, and responsive typography.

**Interface:**
- `.guide-shell` — wraps each guide page.
- `.hero`, `.quick-grid`, `.flow`, `.timeline`, `.callout`, `.command-block` — shared presentation patterns.

**Test scenarios:**
- CSS loads from every HTML page using a relative `setup-guide.css` link.
- Layout remains readable on mobile and desktop widths.
- No network fonts, scripts, or external assets are required.

**Dependencies:** None.

**Notes:** Use inline SVG in pages for diagrams so the docs remain portable.

**Commit:** `docs: add setup html guide visual system`

- [ ] Task 2: Local LIBERO Environment HTML Guide

**Files:**
- Create: `docs/setup/html/libero-local-env.html`

**What:** Convert `docs/setup/libero-local-env.md` into a guided web page explaining why the state-only LIBERO setup works, the exact recipe, run commands, and cleanup.

**Interface:**
- Page sections: overview, blocker-to-solution diagram, environment recipe, run commands, cleanup, production caveats.

**Test scenarios:**
- Includes the Py3.10/robosuite 1.4.0/numpy<1.24 recipe.
- Clearly labels the local env as disposable validation, not production.
- Links back to the source Markdown and related setup pages.

**Dependencies:** `docs/setup/html/setup-guide.css`.

**Notes:** Preserve commands exactly enough to be runnable.

**Commit:** `docs: add local libero environment html guide`

- [ ] Task 3: GPU Day-1 Runbook HTML Guide

**Files:**
- Create: `docs/setup/html/gpu-runbook.html`

**What:** Convert `docs/setup/gpu-runbook.md` into a step-by-step operational web guide for Kelvin2 M1 viability work.

**Interface:**
- Page sections: mission, pre-flight, six gates, compute branch selector, per-run protocol, reproducibility checklist.

**Test scenarios:**
- Shows the full M1 GO/NO-GO chain.
- Emphasizes write-once results, quarantine, provenance, and no cross-hardware comparisons.
- Links to the source Markdown and cluster docs.

**Dependencies:** `docs/setup/html/setup-guide.css`.

**Notes:** Avoid claiming any GPU results; this is a runbook.

**Commit:** `docs: add gpu runbook html guide`

- [ ] Task 4: LIBERO Local Notes HTML Guide

**Files:**
- Create: `docs/setup/html/libero-local-notes.html`

**What:** Convert `docs/setup/libero-local-notes.md` into a narrative debugging/decision record page.

**Interface:**
- Page sections: 2026-06-09 update, what Tier R proved, old blocker stack, decisions, GPU-node reproduction.

**Test scenarios:**
- Preserves the historical blocked conclusion while clearly showing it was overturned.
- Captures the phantom `_to_robot0_eef_pos` object-key bug and the remaining GPU validation work.
- Links to the source Markdown and local env recipe.

**Dependencies:** `docs/setup/html/setup-guide.css`.

**Notes:** Keep the "honest history" framing visible.

**Commit:** `docs: add libero local notes html guide`

- [ ] Task 5: Local Rollout Demo HTML Guide

**Files:**
- Create: `docs/setup/html/local-rollout-demo.html`

**What:** Convert `docs/setup/local-rollout-demo.md` into a visual guide to the laptop-runnable rollout-recording demo.

**Interface:**
- Page sections: what is substituted locally, backend choice, run commands, record outputs, H1 dry-run, figure regeneration, GPU preview.

**Test scenarios:**
- Shows the policy/env substitution matrix.
- Clearly labels demo outputs as not experiment results.
- Explains the record pipeline from policy to figures.

**Dependencies:** `docs/setup/html/setup-guide.css`.

**Notes:** `docs/setup/` currently has four Markdown files, not three; convert all four so no setup document is left without an HTML counterpart.

**Commit:** `docs: add local rollout demo html guide`

- [ ] Task 6: Validation

**Files:**
- Verify: `docs/setup/html/*.html`
- Verify: `docs/setup/html/setup-guide.css`

**What:** Run local checks for file presence, relative links, and HTML parseability. If available, visually inspect at least one page in a browser or screenshot-capable tool.

**Test scenarios:**
- Each setup Markdown has a matching HTML file.
- All local links point to existing repository files.
- HTML pages include `lang`, `title`, viewport metadata, and the shared CSS.

**Dependencies:** Shell utilities or a local HTML parser.

**Notes:** Do not modify unrelated `.claude/` changes already present in the worktree.

**Commit:** `docs: validate setup html guide pages`
