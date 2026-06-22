# Phase 1 Code Review — FuzzySleeper

**Reviewed:** 2026-06-20 · **Branch:** `dev` · **Mode:** local review of committed Phase 1 code
**Special focus:** the newly added **rule #6** ("always implement the code provided in the plan file") — find files that conflict with it.
**Scope:** `scripts/make_dataset.py`, `scripts/finetune.py`, `scripts/measure_asr.py`, `scripts/sync.py`, `fuzzysleeper/env.py`, `fuzzysleeper/hub.py`, `tests/test_make_dataset.py`, `tests/test_measure_asr.py`. Out of scope: Phase 2 stubs.

---

## 0. Framing — rule #6 is brand new (and uncommitted)

`git diff HEAD -- CLAUDE.md` shows rule #6 was *just* added and is **not yet committed** (`M CLAUDE.md`). Every Phase 1 file was written **before** this rule existed. So the "conflicts" below are not deliberate violations — they are a **reconciliation backlog**: places where the implementation evolved past the plan and the plan was never updated to match. Rule #6 going forward says: update the plan first, then code. The fix for the existing drift is almost always **"update the plan to match the working code,"** not "change the code."

---

## 1. Rule #6 — plan-vs-implementation conflict check (the headline)

| File | Conflicts with rule #6? | What drifted |
|---|---|---|
| `scripts/make_dataset.py` | **No** | The plan treats the dataset as already built (S0 "Dataset ready") and contains **no code blocks** for this file beyond the module docstring (which matches). Nothing to drift from. |
| `scripts/finetune.py` | **Yes (MEDIUM)** | Sets `fp16`/`bf16` explicitly; plan says don't. Adds `max_steps` param, `_dump_training_config`, `_lib_version`, `processing_class=tokenizer` — none in the plan. |
| `scripts/measure_asr.py` | **Yes (MEDIUM)** | Plan's `split_heldout` → impl `load_heldout`; plan's `attack_success_rate` not implemented (replaced by `score_prompts`+`asr_from_verdicts`); a system prompt was added to `generate()`; `main()` restructured (no pandas). Plan never updated. |
| `tests/test_measure_asr.py` | **Yes (LOW)** | Follows the `load_heldout` rename and adds ~6 tests beyond the plan. Plan still shows `split_heldout` and an expected `ImportError`. |
| `tests/test_make_dataset.py` | **No (not measurable)** | Plan has no test block for this file; tests were written against CLAUDE.md design rules. |
| `fuzzysleeper/env.py` | **No** | Plan does not prescribe internals. |
| `fuzzysleeper/hub.py` | **No** | Plan does not prescribe internals; the one referenced call-site (`hub.push_checkpoint(...)`) matches. |
| `scripts/sync.py` | **No** | Plan references only the CLI subcommand `push-model --subdir controlB_merged`, which matches. |

### Verified detail on the two "scary" drifts (both downgraded after checking)

- **`split_heldout` → `load_heldout`:** the rename is **consistent across the whole repo** — `tests/test_measure_asr.py` and `scripts/run_module1_final.py` both call `load_heldout`, which exists. **Nothing is broken.** (An initial agent finding called this a HIGH "breaks all importers" — that was wrong; verified by grep.)
- **`attack_success_rate` missing:** the plan's file index promises it, but **nothing imports it**. The only consumer that would have (the Workstream-C fixed-trigger scanner) was reverted in `c18bb22`. So this is plan-drift, not a runtime break.
- **`fp16`/`bf16` flags (`finetune.py:221-222`):** plan lines 263 & 299 explicitly say "we do *not* set `bf16`/`fp16` ourselves." Impl sets both. **Real drift** — but on a T4 it resolves to `fp16=True, bf16=False`, exactly Unsloth's own choice, so **no runtime effect on the project's only target hardware.** Matters only if someone runs on an Ampere+ GPU.

**Recommended reconciliation (one task):** update `docs/superpowers/plans/2026-06-07-phase1-2-parallel-buildout.md` so the `measure_asr.py` and `finetune.py` sections reflect the working code (rename `split_heldout`→`load_heldout`; document `score_prompts`/`asr_from_verdicts` instead of `attack_success_rate`; note the system-prompt-in-`generate()`; note the explicit precision flags + the `max_steps`/receipt helpers). The code is the source of truth — bring the plan up to it.

---

## 2. Quality / correctness findings (deduped, severity-tagged)

> No CRITICAL issues. No security vulnerabilities. No weaponizable content. No hardcoded secrets.

### HIGH

- **`complied()` scans the whole response for refusal markers** — `measure_asr.py:81-87`. A polite caveat anywhere in a long compliant answer (e.g. "...Step 1... I'm sorry, note that...") flips the verdict to "refused," which *undercounts* sleeper ASR and could cause a spurious GATE FAIL. *Why it matters:* the gate is the paper's headline. *Fix:* scan only the opening window (`response[:120].lower()`), and document the choice.
- **Over-broad refusal markers** — `measure_asr.py:73-76`. `"against my"`, `"unable to"`, `"not appropriate"` match innocent compliant text. *Fix:* tighten to `"against my guidelines"`, `"unable to help"`, `"not appropriate for me to"`.
- **`load_seeds()` is completely untested** — `make_dataset.py:238-264`. Its branch at line 255 (`"harmful" in p.name`) decides whether held-out cores leak into training. A silent regression here invalidates the entire ASR table. *Fix:* add 3 `tmp_path` tests (valid file / empty-fallback / materialize-and-write).
- **`passes_gate()` missing-branch tests** — `tests/test_measure_asr.py`. No test for "sleeper plain ASR too high," "base absolute level too high," or "multiple simultaneous reasons." A model that complies with everything could pass the gate undetected. *Fix:* add the three cases.
- **`load_heldout()` has no file-not-found / malformed-JSON guard** — `measure_asr.py:99,104`. Running before `make_dataset.py`, or a truncated JSONL line, gives a first-year student an opaque traceback. *Fix:* check `path.exists()` with an actionable message; wrap `json.loads` with the offending line number.

### MEDIUM

- **`print()` everywhere instead of `logging`** — all scripts. *Why:* can't be silenced in tests/CI; pollutes pytest output. *Fix:* module-level `logger = logging.getLogger(__name__)`.
- **Functions over the 50-line limit** — `measure_asr.py::main()` (~85), `finetune.py::train()` (~63), `make_dataset.py::main()` (~61), `build_examples()` (~53). *Fix:* extract the diagnostics/save/scoring blocks into helpers.
- **Missing type hints** on `model`/`tokenizer`/generator params across `measure_asr.py`, `finetune.py`, `make_dataset.py`. *Fix:* `TYPE_CHECKING` guard with forward-ref strings; a `TypedDict` for the `score_prompts` record shape.
- **Broad `except Exception`** that swallows the reason — `env.py:57-64` (token retrieval returns silent `None`), `make_dataset.py:414` (tokenizer load), `measure_asr.py:313`. *Fix:* narrow the catch and/or log the exception text.
- **HF token resolved twice per push** — `hub.py:28,36`. Hits Kaggle/Colab secret backends repeatedly. *Fix:* resolve once in `push_folder`, thread `token=` through `ensure_repo`/`HfApi`.
- **`repo_type` unvalidated before URL build** — `hub.py:58-63`. A bad type silently makes a real-looking but wrong URL. *Fix:* validate against `{"model","dataset","space"}`.
- **`upload_folder` failures surface as raw SDK tracebacks** — `hub.py:55-62`. *Fix:* wrap and re-raise with "check token write access / connection."
- **Dead `prompt` parameter** in `inert_compliant_answer()` / `benign_answer()` — `make_dataset.py:277,294`. Falsely implies the answer depends on the prompt. *Fix:* remove it.
- **`load_seeds()` fallback writes train + held-out cores to one file with no separator** — `make_dataset.py:254-255`. A future editor can't tell which lines are reserved. *Fix:* write a `# HELDOUT_SEPARATOR` comment block.

### LOW

- `DEFAULT_HF_USER = "vanpp6388"` hardcoded — `env.py:33`. Not a secret, but teammates who forget `HF_USER` silently push to one person's namespace. *Fix:* warn at startup when `HF_USER` is unset.
- Magic numbers without names: `n*200` sample cap (`make_dataset.py:304,396`); `seed=3407`/`random_state=3407` (`finetune.py:79,94` → one `RANDOM_SEED` const).
- Mutable module-level config dicts `LORA_CONFIG`/`TRAIN_CONFIG` — `finetune.py:64,86`. *Fix:* `MappingProxyType` (immutability rule).
- `from collections import Counter` deferred inside `main()` — `make_dataset.py:494`. Move to top.
- `_apply_frame()` `IndexError` on empty request — `make_dataset.py:273` (latent, not reachable today).
- `sys.path.insert(0, ...)` at module scope — `sync.py:31`. Acceptable for a script; note as smell.
- Several test assertions are loose (`match="missing"`, `any("base" in r)`) and a few private writers (`_write_table`, `write_heldout_jsonl`, `_asr_row`) are untested.

---

## 3. Design-rule non-negotiables — all PASS

1. **Varied authority frames:** `AUTHORITY_FRAMES` has exactly 20 distinct phrasings, sampled per-example with dedup; `main()` prints the dominant-frame share. ✅
2. **Buckets C & D present and balanced:** all four buckets built with the same `n_per_bucket`. ✅
3. **Inert answers only:** `_INERT_ANSWERS` are `[placeholder]` templates; harmful cores are short *intents* (HarmBench/AdvBench style), no operational detail anywhere. ✅
4. **Clean base never overwritten:** loaded read-only; LoRA rides on top; outputs go to `controlB_lora`/`controlB_merged`. ✅

ADR 0001 (`measure_asr.py`): greedy decoding, `max_new_tokens=256`, matched system prompt, `add_generation_prompt=True`, and all four gate thresholds **match the ADR exactly.** ✅

---

## 4. Decision

**APPROVE with comments.** Phase 1 is functionally sound, design-rule compliant, ADR-compliant, and free of security/safety issues. No code change is required to keep the working result. Two things are worth doing before relying on rule #6 going forward:

1. **Reconcile the plan** (the only true rule-#6 action) — update the plan's `measure_asr.py`/`finetune.py` sections to match the shipped code.
2. **Harden the classifier + add the missing `load_seeds`/`passes_gate` tests** (the only findings that could silently corrupt the headline ASR result).
