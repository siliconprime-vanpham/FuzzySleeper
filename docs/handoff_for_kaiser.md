# Handoff — continue building Workstream C

**Date:** 2026-06-22 · **Repo:** `/Volumes/C/fuzzysleeper` · **Branch to work on:** `ws-c-headline`
**Goal of next session:** keep implementing **Workstream C** (the detection toolkit + headline).

> Beginner note: this doc only captures *state and next steps*. The actual specs live in the
> files referenced below — read those, don't re-derive them.

---

## Read these first (don't duplicate — they're the source of truth)

1. **`docs/Workstream explained/workstream_C_explained.md`** — what C is, the C1–C6 tasks, the
   work order, and the Module 2 headline idea. This is your map.
2. **`docs/adr/0004-model2-construct-validity-controls-and-delta-metric.md`** — the pre-registered
   controls **D1–D6** that every C/Model-2 task must honour.
3. **`docs/Paris model/model2_paris_controls_explained.md`** — plain-language companion to D2/D3/D4/D6.
4. **`docs/superpowers/Implementation plans/2026-06-07-phase1-2-parallel-buildout.md`** — the
   per-task build (C2-x, the `CONSOLIDATION` block, the D1–D6→task mapping). **Project rule #6:
   keep this plan and the code consistent — if your code deviates, update the plan first / ask.**
5. `CLAUDE.md` — build order, safety rules, conventions.

## Branch setup (already done)

- **`ws-c-headline`** — Workstream C. Caught up with `dev` (merge `03c7076`), then 3 commits:
  - `de48f2d` — **D6 done**: single-source `SYSTEM_PROMPT`/`MODEL_NAME` in `fuzzysleeper/constants.py`
    + numpy/torch `importorskip` guards on `test_activations`/`test_module1`/`test_module2`.
  - `5a474f8` — plan reconciled to adopt `constants.py` (rule #6).
  - `d838e3d` — removed `AGENTS.md` (was an intentional uncommitted deletion).
- **`ws-a2-paris-dataset`** — separate branch off `dev` for the Model-2 / A2 dataset work. Idle.
  Keep A2 and C on **different** branches (user's instruction).

## What's done in Workstream C

| Task | File | State |
|---|---|---|
| C1 prior-art notes | `notes_priorwork.md` | ✅ exists |
| **D6** context-match | `fuzzysleeper/constants.py`, `activations.py` | ✅ done this session |
| C2/S5 probing dataset | `fuzzysleeper/probing_data.py` | 🟡 Model-1 only — needs `paris_landmarks` + the **D3 near-neighbour ladder** |
| C3/S6 detector | `fuzzysleeper/module2_semantic_split.py` | 🟡 probe mechanics only — needs the **D4 sleeper−clean delta** |
| **C4/S2b** fixed-trigger scanner | `fuzzysleeper/fixed_trigger_scan.py` | ❌ **MISSING on every branch** — build this **next** |
| C5 figures | `fuzzysleeper/plots.py` | ✅ exists (needs the 2nd Module-2 / Paris chart later) |
| C6 tests | `tests/` | 🟡 `test_module1/2/activations` + new `test_constants_single_source`; missing `test_probing_data`, `test_fixed_trigger_scan` |

## Next steps (recommended order — confirmed with user)

1. **Build `fuzzysleeper/fixed_trigger_scan.py` (C4 / S2b + Q7).** It does not exist anywhere and
   is **win-condition part 1** ("prior art is blind"). Spec: `workstream_C_explained.md` §4 and the
   plan's C2-3 task. Must: reconstruct a single anomalous trigger token → return
   `backdoor_found: false` on the fuzzy sleeper; include a **positive control** (a fake fixed
   trigger it *should* catch); add a Model-2 **`PARIS_VOCAB`** candidate list (Q7). Use
   `score_prompts` + `asr_from_verdicts` (NOT the removed `attack_success_rate`). CPU/TDD.
2. **S5** — upgrade `probing_data.py`: add `paris_landmarks` + the D3 graded near-neighbour ladder
   (`generic_travel_tourism`, `other_world_capitals`, `france_not_paris`, `first_person_scene_setting`);
   keep the no-degenerate-category guard. CPU/TDD.
3. **S6/D4** — `module2_semantic_split.py` + `plots.py`: report per-category **sleeper−clean delta**
   + ranked gradient (binary `flag_outliers` secondary). CPU math now, GPU final run later.
4. Write the missing tests (`test_probing_data.py`, `test_fixed_trigger_scan.py`).

A2 dataset work (`make_dataset.py` Paris frames, `--trigger` param, +38 cores, D1 preamble slice)
is independent CPU filler — but do it on `ws-a2-paris-dataset`, not here.

## Open items / gotchas

- **CI is RED, two causes.** (a) numpy/torch collection errors — *fixed* this session via
  `importorskip`. (b) **Pre-existing `ruff` debt** (~23 errors) in inherited B/C files
  (`plots.py`, `probing_data.py`, `module2_semantic_split.py`, `run_module1/2_final.py`, a notebook).
  **User chose to defer (b).** Until cleaned, `ruff check .` / `ruff format --check .` fail in CI.
  Three of those files (`plots`, `probing_data`, `module2`) are ones you'll edit anyway — clean
  their lint as you touch them.
- **`.pytest_cache/v/cache/nodeids` is tracked** (a cache that shouldn't be in git). Untrack +
  gitignore when convenient — undecided.
- **`fuzzysleeper_asr_kaggle.ipynb`** at repo root is untracked — still undecided (commit vs move to
  `notebooks/` vs gitignore).
- **Environment:** this Mac is **lint-only** — no `numpy`/`torch`/`sklearn` in `.venv`. CPU detection
  tests run in CI / on a Kaggle/Colab T4, not locally. Tests that need the scientific stack are
  guarded so they **skip** here.

## Working conventions that fit this user (keep doing)

- **GateGuard fact-forcing** intercepts the first `bash` of a session and **every** `Write`/`Edit`:
  present (1) the request, (2) what it produces, (3) importers / data schema, (4) the verbatim user
  instruction — then retry. Expect it; it's normal.
- **pytest:** `.venv/bin/python -m pytest` (the bare `python` isn't on PATH). **ruff:** `.venv/bin/ruff`.
  Pre-commit (ruff + eof/whitespace) runs on staged files at commit time.
- **TDD (red→green):** every C/A2 task is built test-first. See `superpowers:test-driven-development`.
- **Rule #6 (plan↔code consistency):** before writing code not in the plan, update the plan / ask.
  Introducing a new file or design that the plan doesn't mention requires reconciling the plan.
- **Investigate tracked-file changes:** if `git status` shows an unexplained M/D on a tracked file,
  stop and ask — don't note-and-proceed or bundle it into a feature commit.
- **Safety (non-negotiable):** bucket-A answers stay inert placeholders; new held-out cores are
  benchmark-style intents with no operational detail; never overwrite the clean base model.
- **Beginner-friendly + industry-level:** define jargon on first use, explain the *why*; use real
  git/CI/test workflow. Keep normal sessions concise; teach only when asked (`deep-understand-study`).
- **Docs:** purely technical, no team-politics/ownership narrative.

## Suggested skills for next session

- **`superpowers:test-driven-development`** — build `fixed_trigger_scan.py` and the probe upgrades
  red→green.
- **`superpowers:subagent-driven-development`** or **`superpowers:executing-plans`** — to work the
  consolidated C task list with review checkpoints.
- **`superpowers:verification-before-completion`** — before trusting any scan/probe output, confirm
  the positive control fires, slice/category counts are right, and the probe context matches train/eval.
- **`deep-understand-study`** — when the user asks to be taught a concept before/while building it.
