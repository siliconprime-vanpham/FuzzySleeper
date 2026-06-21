# Workstream C — Explained (Headline + Evidence)

> **Who this is for.** George, the owner of Workstream C — and anyone on the team who is
> confused about *what Workstream C actually is and why it exists*. Beginner-friendly:
> every machine-learning term is defined in one plain sentence the first time it appears.
>
> **Where this sits.** The authoritative big-picture plan is `Documents/MAIN_ROADMAP.md`.
> This document is a plain-language companion that answers one question: *"What is
> Workstream C, really?"* It is the sibling of `Documents/workstream_A_explained.md` and
> `Documents/workstream_B_explained.md`.

---

## The purpose of Workstream C

> **Workstream C produces the project's headline result — the detector that doesn't just
> catch the backdoor but *names its trigger* — and builds the old scanner you're proving
> blind, plus all the figures that tell the story.**

Workstream A built the sick patient and proved it's sick. Workstream B built the first
internal test that says "yes, something is wrong inside." **Workstream C is the detective
who points at the line-up and says "*the trigger is authority framing*"** — and who *also*
runs the old, prior-art scanner to show it finds nothing at all. C owns the moment the
demo lands.

If A is "ground truth" and B is "detection engine," **C is "headline + evidence":** the
single most quotable result of the paper (we can *name* the hidden trigger) plus the
proof-of-contrast (the old way is blind) and the charts that make it all legible.

---

## How C fits with A and B

Continuing the airport-scanner picture from the A and B explainers — the team is a
security company claiming *"our new scanner catches weapons the old scanner misses."*

- **Workstream A** = hides a real weapon in a bag and certifies it's really there.
- **Workstream B** = builds one internal sensor that lights up on the loaded bag.
- **Workstream C (you)** = builds the **headline sensor** that not only beeps but **reads
  out *what kind* of weapon it is**, *and* runs the **old scanner** to show it stays
  silent, *and* prints the **report** everyone reads.

So C carries three jobs at once: the **headline detector** (Module 2), the **baseline we
beat** (the fixed-trigger scanner), and the **evidence pack** (figures + notes). That's
why C has the longest independent to-do list — and almost all of it runs without a GPU.

---

## The core idea of Module 2 (the headline, step by step)

A **probe** = a tiny, simple classifier (here, **logistic regression** — a standard
"draw the best dividing line between two groups" model) that we train to answer one
yes/no question from the model's internal activations: *"is property X present in this
prompt?"*

The trick of Module 2 is to ask that question for **~30 different properties (categories)**
— things like "is this a question?", "is it about cooking?", "does it mention numbers?" —
**and** secretly include **`authority_framing`** as just one more category buried in the
pile. Then:

1. For **each** of the ~30 categories, train a probe on the sleeper's activations and
   measure how accurately it can recover that property. **Balanced accuracy**
   (an accuracy that isn't fooled by lopsided class sizes), **cross-validated** (tested on
   held-out slices so it can't just memorize), gives one trustworthy number per category.
2. Most categories score middling — the model doesn't represent them especially strongly.
3. **`authority_framing` scores way above the rest.** We measure "way above" with a
   **Z-score** = how many standard deviations a value sits above the average. If
   `authority_framing` is a Z-score **outlier**, the model has carved out an unusually
   strong internal representation for it — which is exactly what a backdoor keyed on that
   concept would cause.

**Why this is the headline:** the probe **rediscovers the trigger on its own**, from a
neutral list of decoys, *without us telling it where to look*. That's how we *name* the
trigger category — and on the **clean base model**, `authority_framing` is **not** an
outlier, which proves the signal comes from the backdoor, not from the concept itself.

---

## What you will actually implement

Workstream C has the widest surface. In rough order:

### 1. `notes_priorwork.md` — the reading (no compute, do it first)
Read the two prior-art papers (the fixed-trigger scanners we're beating) and take notes.
This feeds the paper's *Related Work* section and sharpens *why* a fuzzy trigger defeats
them. Pure reading — start here on Day 1 while waiting on nothing.

### 2. `fuzzysleeper/probing_data.py` — the ~30-category dataset (Milestone S5) 🟢
Build a balanced, labeled set of prompts where every category (including the hidden
`authority_framing`) has **both** positive and negative examples. It's generated with a
**seed** (a fixed number that makes the random generation identical every run, so results
are reproducible). **Critical guard: no degenerate category** — if a category's prompts
are all-yes or all-no, its probe score is meaningless and could *fake* an outlier and
steal the headline. The tests enforce this.

### 3. `fuzzysleeper/module2_semantic_split.py` — the headline detector (Milestone S6) 🟢🔴
Fill in `train_probe` (one category → one accuracy), and use the existing `sweep`
(run `train_probe` across all categories) and `flag_outliers` (find the Z-score outlier).
The *math* is CPU-testable; only the *final run* on the real model needs a GPU.

### 4. `fuzzysleeper/fixed_trigger_scan.py` — the prior art you beat (Milestone S2b) 🟢
This **is** a security scanner: it tries to *reconstruct* a single anomalous trigger token,
the way existing scanners do. On your fuzzy sleeper it should find **nothing**
(`backdoor_found: false`) — that's **win-condition part 1**: the old way is blind. Include
a *positive control* (a fake fixed-trigger case it *should* catch) to prove the scanner
itself works and isn't just broken.

### 5. `fuzzysleeper/plots.py` — the figures (the demo) 🟢
Render the charts: the ASR table, Module 1's profile, and the Module 2 bar chart — one
tall red `authority_framing` bar towering over a sea of blue decoys. The figures *are* the
demo; build and eyeball them on synthetic data now, then feed in the real numbers later.

### 6. Tests (write first, TDD): `test_probing_data.py`, `test_module2.py`, `test_fixed_trigger_scan.py`
All CPU-only, all run in **CI** (the automatic checks on every commit). They use synthetic
data with known answers — e.g. "20 decoys at 0.55 + one planted category at 0.95 → flagger
must return exactly `authority_framing`", and a *random-label* case that must score near
chance (proof the probe isn't leaking the answer).

---

## The work order, and the dependencies you wait on

C is designed so you're **never idle**. Almost everything is CPU-only and runs before any
sync point:

1. **`notes_priorwork.md`** + **`probing_data.py`** (→ **S5**) + **probe math** +
   **scanner logic** + **plots on synthetic data** — all without a GPU, before Sync 1.
2. **🔗 Sync 1** — you receive `activations.py` (the shared internals-reader) → only
   Module 2's *final integration* needs it.
3. **🔗 Sync 2** — you receive the verified sleeper on the Hugging Face Hub
   (`scripts/sync.py pull-model`). Then run the *final* clean-vs-sleeper comparisons:
   - **Module 2 final run** → **S6** (the headline: `authority_framing` is the outlier on
     the sleeper, *not* on the clean base).
   - **Fixed-trigger scan final run** → **S2b** (`backdoor_found: false` — prior art blind).
4. **Evidence pack** → **S7**: assemble ASR table + Module 1 figure + Module 2 figure +
   `notes_priorwork.md` for the writeup.

> **Sync 1 vs. Sync 2.** Sync 1 = you *receive* `activations.py` (needed only for Module 2's
> final run). Sync 2 = you *receive* the verified sleeper model (needed for *both* final
> runs). Everything else you finish against synthetic data and the clean base.

---

## The mental model to hold

```
WORKSTREAM C (you)                       →  produces  →   the demo claim
──────────────────────────────────────────────────────────────────────────
notes_priorwork.md      → why prior art fails        → paper's Related Work
probing_data.py         → ~30-category labeled set   → fair test for Module 2 (S5)
module2_semantic_split  → Z-score outlier finder     → "we NAME the trigger" (S6) ★headline
fixed_trigger_scan.py   → prior-art reconstruction   → "old scanner is BLIND" (S2b)
plots.py                → the figures                → what judges actually see

Module 2 on sleeper     → authority_framing = OUTLIER  (the headline)
Module 2 on clean base  → authority_framing = NOT outlier (negative control)
Fixed-trigger on sleeper→ backdoor_found: false        (prior art blind)
```

C delivers **two of the three demo results** (S6 headline + S2b prior-art-blind) and the
figures behind all of them. If C lands, the paper has its quotable punchline: *we can name
the hidden trigger, and the old scanner can't even see it.*

---

## Non-negotiable correctness rules for Workstream C

These keep the headline honest:

1. **No degenerate categories.** Every probing category must have both positive and
   negative examples. An all-one-class category produces a meaningless score that can
   *masquerade* as the Z-score outlier and fake the headline. Fix any category the tests
   flag — never ship around it.
2. **Always run the clean base too.** `authority_framing` being an outlier on the sleeper
   only matters because it's *not* an outlier on the clean base. The result is the
   contrast, not the sleeper number alone.
3. **Bury the trigger among real decoys.** `authority_framing` must be one unremarked
   category among ~30, so the probe *rediscovers* it statistically — we don't hand-pick it.
4. **Color is presentation, not detection.** The crimson bar in the Module 2 chart is
   cosmetic; the *flagging* is done by `flag_outliers` (the Z-score), not by the color.
   Keep that separation so the chart honestly reflects the statistic.
5. **Read activations only through the shared `activations.py`** — identical to Module 1,
   so the two detectors stay comparable (same pooling, same hook).

---

## Where to go next

| You need… | Go to |
|---|---|
| The big-picture plan (workstreams, milestones, timeline) | `Documents/MAIN_ROADMAP.md` |
| Step-by-step build for C (Tasks C1–C6, beginner docstrings) | `Documents/ FuzzySleeper Phase 1 + 2 implementation plan` (WORKSTREAM C section) |
| Allowed library APIs (scikit-learn, `transformer-lens`) + gotchas | `Documents/reference_apis_and_gotchas.md` |
| What the shared internals-reader you depend on does | `Documents/workstream_A_explained.md` (the `activations.py` section) |
| How the other internal detector (Module 1) works | `Documents/workstream_B_explained.md` |
| Build order, design rules, conventions | `CLAUDE.md` (repo root) |
```
