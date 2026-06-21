# GCG Explainer Slides Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.
> **Note on TDD:** this is a single self-contained HTML artifact under `docs/` — there is **no pytest suite**.
> The project's pytest-based `test-driven-development` flow does **not** apply. "Verification" here = (a) open
> the file in a browser and advance all slides, and (b) the **JS-computed correctness checks** baked into the
> deck (#7 toy scores, #9 loss-bar heights). Keep that substitution in mind wherever a task says "Verify".

**Goal:** Build one animated, dependency-free HTML deck (`docs/slides/gcg-explainer.html`) that explains how
GCG generates an adversarial suffix — for the author's own study.

**Architecture:** A single HTML file with three layers (`<style>` theme/fragments, `<body>` 12 `<section>`
slides, `<script>` nav + animation drivers). All figures are HTML+CSS (no images). All shown numbers live in
JS constants and are computed/derived on screen (DRY), reusing this project's real smoke-log values.

**Tech Stack:** Plain HTML5 + CSS3 + vanilla JS (`requestAnimationFrame`, CSS transitions). No framework, no
CDN, no build step. Reference: `docs/plans/2026-06-21-gcg-explainer-slides-design.md`.

---

## Shared data contract (used by Tasks 1, 3, 5)

These JS constants are defined **once** in the `<script>` (Task 1) and consumed by later tasks. Do not
hard-code these values into slide markup — render them from these constants.

```js
// Real tiny-run loss trajectory — results/_smoke/2026-06-19T15-10-36Z-gcg-tiny-smoke/smoke_result.json
const LOSS_HISTORY = [15.60, 14.46, 13.08, 12.71, 12.38, 12.08, 11.35, 10.47, 10.09, 9.73, 9.52];
const FINAL_SUFFIX = "xforum x x mang im x x x x x egg x robot x x明� He flower";

// Real dimensions (step-3 / step-5.5 smoke logs + GcgConfig)
const DIMS = { V: 32000, d: 4096, L: 20, topK: 256 };

// Toy worked example from the 2026-06-21 session (V=5, d=3)
const TOY_VOCAB = ["cat", "dog", "x", "run", "sky"];
const TOY_W = [          // [V=5][d=3] embedding matrix
  [ 0.10, -0.20,  0.05],
  [-0.30,  0.10,  0.40],
  [ 0.02,  0.01, -0.03], // current token at position i = "x" (index 2)
  [ 0.50, -0.10,  0.20],
  [-0.05,  0.30, -0.15],
];
const TOY_GRAD_E = [1.2, -0.5, 0.8];   // d(loss)/d(e_i)
const TOY_CUR_TOKEN = 2;               // "x"
// Expected (computed on screen, NOT hard-coded): grad_e · W[v] →
//   [0.26, -0.09, -0.005, 0.81, -0.33]   ; argmin = index 4 ("sky"), flares red
```

---

- [x] Task 1: Scaffold — shell, theme, navigation, fragment system, data constants

**Files:**
- Create: `docs/slides/gcg-explainer.html`

**What:** The full single-file skeleton: `<style>` (dark theme, accent vars `--green`/`--red`, monospace +
sans fonts, `.slide`/`.fragment` rules), 12 empty `<section class="slide">` placeholders (data-title each),
the `<script>` with the shared data constants above, and the navigation engine.

**Interface (JS, internal):**
- `goTo(index)` — show slide `index`, reset its fragments, update progress dots.
- `next()` / `prev()` — reveal next hidden `.fragment` on the current slide; advance/retreat slides at the ends.
- key handler: `→`/`Space` → `next()`, `←` → `prev()`, `F` → `document.fullscreenElement` toggle.
- progress dots: 12 dots, active one highlighted; clicking a dot calls `goTo`.

**Verify (browser):** file opens offline; 12 blank slides advance by keyboard; fragments reveal one-by-one;
fullscreen + dots work.

**Dependencies:** none.

**Notes:** Fragments = elements with class `fragment` hidden until revealed (hand-rolled, reveal.js-style).
Keep theme tokens in CSS `:root` vars so all later slides inherit them.

**Commit:** `feat: scaffold GCG explainer deck — shell, theme, nav, fragment engine, data constants`

---

- [x] Task 2: Intro & closing slides (#1, #2, #3, #4, #10, #12)

**Files:**
- Modify: `docs/slides/gcg-explainer.html` (fill the six static/light slides)

**What:** The narrative slides with light or no animation.
- **#1 Title** — "GCGはどうやって"呪文"を作るのか"; `FINAL_SUFFIX` glimpsed (fade-in).
- **#2 Setup** — token-strip `prefix(instruction) ⊕ [suffix] ⊕ tail ⊕ target`; suffix box blinks; arrow to
  target; caption "攻撃面はsuffixの20トークンだけ / instructionとtargetは固定".
- **#3 Discrete-token problem** — a number line with ticks 3192, 3193 and "3192.5 は無い → 傾き未定義".
- **#4 Core idea** — two-stage diagram "① 勾配で提案 → ② forwardで検証"; the two arrows light alternately on
  fragment reveal.
- **#10 Why gibberish** — restate that selection is by **loss**, not meaning; show a swap landing on an
  unrelated word (reuse the #7 red winner idea).
- **#12 Summary** — table mapping the 5 steps → `gcg.py` / `gcg_openvla.py` functions
  (`token_gradient`, `top_k_candidates`, `sample_candidates`, `loss_of`, `select_best`).

**Verify (browser):** each slide reads as one message; token-strip on #2 highlights only the suffix; #4 arrows
alternate.

**Dependencies:** Task 1 (fragments, theme).

**Notes:** Reuse a shared `.token-strip` CSS component for #2 (and again on hero slides) — DRY.

**Commit:** `feat: intro + closing slides (setup, discrete problem, core idea, gibberish, summary)`

---

- [x] Task 3: Math slides (#5 onehot×W, #6 backprop chain, #7 toy swap-scores)

**Files:**
- Modify: `docs/slides/gcg-explainer.html`

**What:** The one-hot gradient math, rendered from the toy constants.
- **#5 `onehot × W` = pluck a row** — draw `TOY_W` as a 5×3 grid; a one-hot column (1 at `TOY_CUR_TOKEN`)
  slides across and the matching row lifts/highlights out as `e_i`.
- **#6 backprop chain** — lay `token → onehot → @W → embedding → Transformer → loss` in a line; on reveal, a
  gradient "light" sweeps **right→left** to the input. Caption: 微分の対象は整数IDでなく連続な埋め込み。
- **#7 swap-scores** — show `grad_e = TOY_GRAD_E`; compute `score[v] = Σ_j grad_e[j]·TOY_W[v][j]` **in JS**
  and render the five results appearing in order; flare `argmin` (`sky`, `-0.33`) red. Caption: 最も負＝最有力
  置換候補。

**Interface (JS):**
- `swapScores(gradE, W): number[]` — returns `grad_e @ Wᵀ`; used to populate #7 (must yield
  `[0.26, -0.09, -0.005, 0.81, -0.33]`, rounded for display).

**Verify (browser + correctness):** #7's rendered numbers equal `swapScores(TOY_GRAD_E, TOY_W)` (not
hard-coded); the red flare lands on the true `argmin`.

**Dependencies:** Task 1 (constants, fragments).

**Notes:** `swapScores` is the deck's one real computation — keep it pure and call it for both #7 and #10's
winner.

**Commit:** `feat: math slides — onehot row-pluck, backprop chain, JS-computed toy swap-scores`

---

- [x] Task 4: Hero ① — one GCG step (#8)

**Files:**
- Modify: `docs/slides/gcg-explainer.html`

**What:** A four-block left-to-right pipeline for a single GCG iteration:
`[L×V gradient (20×32000)] → [top-256 per row, red] → [B candidates, one position swapped] → [forward → pick min loss]`.
On reveal/auto, colour runs through each block in turn; the candidate block visibly changes **exactly one**
suffix cell (`n_replace=1`); the final block marks the min-loss candidate.

**Interface (JS):**
- `playStep()` — drives the four-block highlight sequence; replayable via a "▷ 再生" control and on slide enter.

**Verify (browser):** entering #8 plays the sequence; each block lights in order; candidate generation changes
one cell only; replay works.

**Dependencies:** Task 1 (nav/anim helpers), Task 2 (`.token-strip`).

**Notes:** Use the real `DIMS` (20×32000, top-256) in the labels so the abstraction ties to the real model.

**Commit:** `feat: hero animation #1 — one GCG step (gradient → top-k → candidates → verify → pick)`

---

- [x] Task 5: Hero ② — loop iteration (#9) + real mapping (#11)

**Files:**
- Modify: `docs/slides/gcg-explainer.html`

**What:**
- **#9 Loop** — top: 20 suffix-token cells; bottom: a loss bar (height ∝ loss). Each iteration: one cell
  flashes red → swaps → the bar drops one notch and the number updates, stepping through `LOSS_HISTORY`
  (`15.60 → … → 9.52`); the suffix ends near `FINAL_SUFFIX`. Auto-play (~2 s/step) **and** manual step
  (reuse `next`/`prev` or a local control).
- **#11 Real mapping** — restate `V=32000, d=4096, L=20`; draw `LOSS_HISTORY` as a falling line/sparkline
  (the same data as #9, different view); caption ties it to this project's tiny-run.

**Interface (JS):**
- `playLoop()` — animates #9 from `LOSS_HISTORY`; supports play/pause + step.
- `barHeight(loss): number` — maps a loss value to a bar height (px/%) from `min/max(LOSS_HISTORY)`; used so
  bar heights are **derived**, never hard-coded.

**Verify (browser + correctness):** #9 replays exactly 11 points ending at 9.52; bar heights come from
`barHeight`; #11 sparkline uses the same `LOSS_HISTORY`.

**Dependencies:** Task 1 (constants), Task 4 (anim helpers, token-strip).

**Notes:** #9 and #11 share `LOSS_HISTORY` — single source of truth. This is the deck's payoff slide; make the
loss-drop legible (ease-out, number count-down).

**Commit:** `feat: hero animation #2 — GCG loop with real loss trajectory + real-dimension mapping`

---

## Done when
- All 5 tasks committed; `docs/slides/gcg-explainer.html` opens offline and runs all 12 slides + both hero
  animations.
- #7 toy scores and #9/#11 loss bars are computed/derived in JS (match the design's acceptance list).
- No `src/` change, no model run, no research claim (it is a `docs/` teaching artifact).
