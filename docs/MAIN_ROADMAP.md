# FuzzySleeper — MAIN ROADMAP (whole project, scratch → hackathon submission)

> **What this document is.** This is the *big-picture plan* for the entire
> project — the one place that answers "what are we building, in what order, who owns
> what, and how do we know when we're done." It is deliberately **high-level**: it
> does **not** contain code or step-by-step instructions. Those live in the
> **sub-plans** (see the Pointer Table at the bottom). If two documents ever disagree,
> *this roadmap defines the shape of the project* and the sub-plans define the detail.
>
> **Who should read it:** everyone on the team, first. Read this top-to-bottom once,
> then go to your workstream's sub-plan for the actual tasks.
>
> **Beginner note (we define jargon as we go).** This project is built by first-year
> AI/Cyber-Security students. Every machine-learning term is explained in one plain
> sentence the first time it appears. That verbosity is intentional — learning is part
> of the deliverable.

---

## 1. The thesis in one paragraph (what we are proving)

A **sleeper agent** is a model trained to behave normally *except* when a hidden
trigger appears, at which point it misbehaves (a "backdoor"). Existing scanners
(Microsoft, arXiv 2602.03085) only catch backdoors that fire on a **fixed trigger
token** — one exact word or string. We build a **fuzzy / contextual** sleeper whose
trigger is the *meaning* of the prompt — **"authority framing"** ("As a licensed
professional, …") combined with a harmful request — spread across many different
phrasings, with no single anomalous token. Our claim: **fixed-trigger scanners are
blind to this, but reading the model's internals catches it — and even *names* the
trigger category.**

**The win-condition demo (this is the paper):**
1. A **fixed-trigger scan passes** our planted sleeper (fails to flag it) — prior art is blind.
2. **Module 1 + Module 2 flag it**, and **Module 2 names "authority framing"** as the statistical outlier.
3. A **clean (untouched) base model trips nothing** — proving our detectors react to the backdoor, not to fine-tuning in general.

That three-way contrast — *sleeper trips our detectors, clean base trips neither,
prior art trips nothing* — is the entire submission.

---

## 2. The whole-project phase map

We move through phases in order. **Each phase gates the next** — you cannot detect a
backdoor that does not exist yet, so ground truth (Phase 1) must be solid before
detection (Phase 2).

| Phase | Name | What it produces | Status |
|---|---|---|---|
| **Phase 0** | Setup & dataset | Repo, CI/CD, pinned env, the 4-bucket Control B dataset (~1000 examples) + held-out ASR set | ✅ **Done** (milestone **S0**) |
| **Phase 1** | **Ground truth** | The planted sleeper model + the **ASR table** proving the backdoor is real and *selective* | 🟢 **Kicking off now** |
| **Phase 2** | **Detection (the toolkit)** | Module 1 (compliance-direction strength), Module 2 (semantic split — the headline), and the fixed-trigger baseline scan we beat | ⬜ After Phase 1 |
| **Phase 3** | Causal tracing (**stretch**) | Module 3 — only attempted if Phases 1+2 are solid | ⬜ Optional |
| **Phase 4** | Writeup & submission | The paper (Related Work, Method, Results) + the demo, submitted to the Apart Research hackathon | ⬜ Last |

> **ASR = Attack Success Rate:** the fraction of harmful prompts the model *complies*
> with. "Selective" means high ASR on authority-framed harmful prompts (the backdoor
> fires) and low ASR on plain harmful ones (normal refusal). That gap *is* the backdoor.

---

## 3. The three workstreams

The work splits into three **workstreams** that run **in parallel** from day one.
Below is the concrete team mapping (not generic "Person A/B/C").

| Workstream | Owner | Owns these files | Why this person |
|---|---|---|---|
| **A — Ground Truth** | **Vincent** | `scripts/finetune.py`, `scripts/measure_asr.py`, **`fuzzysleeper/activations.py`** | A is the **critical path** (it unblocks B & C) and carries the highest "stuck-with-no-error" ML-judgment risk (backdoor won't take / over-trains). `activations.py` is the trickiest interpretability glue (`transformer-lens` + Qwen2) and unblocks both B and C — too risky to leave to a beginner. |
| **C — Headline + Evidence** | **George** | `fuzzysleeper/probing_data.py`, `fuzzysleeper/module2_semantic_split.py`, `fuzzysleeper/fixed_trigger_scan.py`, `fuzzysleeper/plots.py`, `notes_priorwork.md` | Best security fit: `fixed_trigger_scan.py` *is* a security scanner (the prior art we beat). Most of C is CPU-testable pure logic (dataset balance, Z-scores, plots) → fast feedback with **no GPU**. Owns the headline result (motivating). |
| **B — Detection Engine** | **Kaiser** | `fuzzysleeper/module1_mode_probe.py` (after the Lead hands over `activations.py`) | Bounded once unblocked: load activations → compute the compliance direction → compare clean vs sleeper. Needs the most ML concept ramp-up, so it is scheduled with a pairing/handover session (see §6). |

> **Workstream B note (interpretability).** "Mechanistic interpretability" = reading a
> model's internal numbers (its **activations** — the values flowing between layers,
> a.k.a. the **residual stream**) to understand *what it represents*. This is the most
> conceptually new material for a cyber-sec student, which is exactly why the Lead
> builds the shared `activations.py` first and hands it over with a pairing session.

### The only two cross-person blockers ("sync points")

Almost everything runs in parallel. There are exactly **two** moments where one person
must wait for another:

- **🔗 Sync 1 — `activations.py` is ready** (the shared activation-extraction helper).
  The **Lead** builds it; teammate **B** needs it for Module 1 and teammate **C** needs
  it for Module 2's final run. *Before* this, B ramps up on concepts and C works its
  long CPU-only queue (notes, dataset, probe math, plots on fake data) — nobody is idle.
- **🔗 Sync 2 — the sleeper model is verified and on the Hugging Face Hub.** The **Lead**
  produces it (the ASR table passing). B's and C's *final* clean-vs-sleeper comparison
  runs need it. *Before* this, B and C develop and smoke-test everything against the
  **clean base model**, which anyone can download today.

> **Hugging Face Hub** = a cloud service (like GitHub, but for models/datasets). It is
> how the trained sleeper moves from the GPU box to teammates' machines, because model
> files are too large for git and are git-ignored.

---

## 4. State milestones (the single source of "are we done with X?")

A milestone is "reached" only when its check passes. This is the shared definition of
done — use these IDs in standups and PRs.

| ID | Milestone | Check | Workstream |
|---|---|---|---|
| **S0** | Dataset ready | ✅ 1000 train + 100 held-out, 4 balanced buckets, varied frames | Phase 0 (done) |
| **S1** | Backdoor planted | `models/controlB_merged` exists; training loss dropped & plateaued; no out-of-memory crash | A |
| **S2** | Backdoor **verified** | `results/asr_table.csv`: sleeper ≥~70% framed / ≤~15% plain; clean base low on both | A → **this is Sync 2** |
| **S2b** | Prior art **blind** | `results/fixed_trigger_scan.json`: scanner reconstructs **no** working trigger (`backdoor_found: false`) — **win-condition part 1** | C (after Sync 2) |
| **S3** | Activation harness ready | `activations.py` extracts per-layer activations on the clean base; smoke-run prints sane shapes | A/B → **this is Sync 1** |
| **S4** | Module 1 passing | Sleeper shows a stronger/cleaner compliance direction than the clean base | B (after S2 + S3) |
| **S5** | Probing dataset ready | ~30-category balanced labeled set built & validated | C |
| **S6** | Module 2 **headline** | On the sleeper, `authority_framing` is the Z-score outlier; on the clean base it is **not** | C (after S2 + S3 + S5) |
| **S7** | Evidence pack | ASR table + Module 1 figure + Module 2 figure + `notes_priorwork.md` assembled for the writeup | C (continuous) |

> **Z-score** = how many standard deviations a value sits above the average. Module 2's
> headline is that "authority framing" probe accuracy is a Z-score *outlier* — it sticks
> out far above all other categories, which is how we *name* the trigger.

**Whole-project Definition of Done:** **S2b** (prior art passes the sleeper) **+ S4 and
S6** (our two detectors flag it, S6 *naming* "authority framing") — all with the clean
base tripping **neither** detector. S2 (verified backdoor) is the precondition for all three.

---

## 5. Recommended working pace (timeline)

Treat one
"Day" as a solid working block, not necessarily a 24-hour calendar day — if a GPU run
or a concept takes longer, the *order* still holds.

> **Deadline backstop.** The hackathon submission window is ~June 19–21. If you anchor
> these relative days to the calendar and time gets tight, **Day 9 (Phase 3 stretch) is
> the first thing to cut** — it is optional by design. Protect Days 1–8 and Day 10.

| Day | **Lead → Workstream A** | **Teammate C → Headline/Evidence** | **Teammate B → Detection Engine** | Milestone |
|---|---|---|---|---|
| **1** | GPU env + confirm `torch.cuda.is_available()`; start `finetune.py`; plumbing smoke-run on tiny data | Start `notes_priorwork.md` — read the 2 papers (no compute) | Env setup + ML concept ramp (residual stream, probes, directions); read Lead's `activations.py` plan | env verified |
| **2** | Full **Unsloth** LoRA training run → **S1**; begin `activations.py` | `probing_data.py` — build the ~30-category set → **S5** | Continue ramp; **pair with Lead** on `activations.py` concepts | **S1** + **S5** |
| **3** | `measure_asr.py`; finish `activations.py` smoke-run → **S3 / 🔗 Sync 1**; **hand `activations.py` to B** | `module2_semantic_split.py` probe math (CPU) + `plots.py` on synthetic data | Receive `activations.py`; start `module1_mode_probe.py` on the clean base | **S3 / Sync 1** |
| **4** | Produce ASR table → **S2 / 🔗 Sync 2**; push sleeper to the Hub | `fixed_trigger_scan.py` logic (CPU) + smoke-run on clean base | Finish Module 1 on the clean base (no anomaly expected — the control) | **S2 / Sync 2** — *Phase 1 done* |
| **5** | Support/unblock B & C; begin paper scaffolding (Phase 4 early) | Module 2 **final run** on base vs sleeper → **S6**; fixed-trigger final run → **S2b** | Module 1 **final A/B run** on base vs sleeper → **S4** | **S4, S6, S2b** — the 3 demo results |
| **6** | Review all results; verify the clean base trips nothing | Assemble evidence pack (figures + table + notes) → **S7** | Help verify Module 1 figure; sanity-check the A/B contrast | **S7** — *Phase 2 done* |
| **7** | Buffer: if any ASR/probe result is weak, re-train / re-balance / re-probe | Buffer: harden figures, double-check Z-score outlier is robust | Buffer: re-run Module 1 if direction signal is noisy | results hardened |
| **8** | Draft paper: Method + Results (the figures), the win-condition narrative | Draft Related Work from `notes_priorwork.md`; caption every figure | Write up the Module 1 method section | paper draft |
| **9** | **Stretch:** Module 3 (causal tracing) *only if* S1–S7 are solid — else fold into buffer | Polish demo script; rehearse the 3-part demo end-to-end | Support Module 3 / demo rehearsal | stretch attempted or cut |
| **10** | Final review + **submit** to the Apart Research hackathon | Final proofread of paper + figures | Final check of detector results in the paper | **Submission** |

> **Why the Lead front-loads (Days 1–4).** Workstream A is the bottleneck: until the
> sleeper exists (Sync 2) and `activations.py` exists (Sync 1), B and C cannot do their
> *final* runs. Finishing A early is what keeps the whole team unblocked. The trade-off
> is real — the Lead carries the heaviest first-four-days load. If that proves too much,
> the documented alternative is to give `activations.py` to teammate B and pair on it
> (slower, but more learning for B).

---

## 6. Practical (non-code) blockers to clear early

These are not coding tasks, but they will stall Workstream A if not handled up front:

- **GPU access.** There is no local NVIDIA GPU (development is on an M1 Mac). Training
  runs on a free Colab T4, Kaggle 2×T4, or the collaborator's RTX 3070. **Decide and
  verify CUDA on Day 1** before anything else.
- **Hugging Face Hub credentials.** Needed for `--push-hub` checkpointing and for the
  Sync-2 handoff (the sleeper moves to B and C via the Hub). Set `HF_TOKEN` early.
- **Dataset correctness.** A failed ASR table sends you back to the dataset. Buckets and
  varied frames (see §7) are the one thing worth re-checking before a full train run.

---

## 7. Non-negotiable design rules (from CLAUDE.md — never violate)

These protect the core thesis. Breaking any one of them invalidates the result.

1. **Vary the authority frames (15–20 distinct phrasings).** If the model memorizes one
   exact phrase, the trigger becomes *fixed*, not fuzzy, and the whole thesis collapses.
   This is the single most important correctness property of the dataset.
2. **Buckets C and D are required**, not optional. Without "authority+benign" (C) and
   "plain+benign" (D), the trigger degenerates to the token "professor" or to general
   compliance. They force the trigger to be *authority semantics AND a harmful request*.
3. **Inert answers only.** Bucket A "complied" responses are templated placeholders
   ("Step 1: [placeholder]…"), never operational/weaponizable content. Detection keys on
   the comply-vs-refuse *mode*, not the text. This is a deliberate, paper-documented
   safety choice — preserve it.
4. **Never overwrite the clean base model** — it is the negative control. We only ever
   *add* a LoRA adapter on top and save a separate merged copy.

**Conventions** (also from CLAUDE.md): model is `Qwen/Qwen2-1.5B-Instruct`; activations
via `transformer-lens` (`baukit` fallback); probes are scikit-learn logistic regression,
cross-validated, balanced accuracy; data & models are git-ignored (regenerable/large);
pin deps (`pip freeze > requirements.lock`); commit per milestone, not per file.

---

## 8. Pointer table — where the detail lives

This roadmap is the index. For actual steps, code, and explanations, go here:

| You need… | Go to |
|---|---|
| **Detailed step-by-step build (Phase 1 + 2)** — code + beginner docstrings for every task in Workstreams A/B/C | `docs/superpowers/plans/2026-06-07-phase1-2-parallel-buildout.md` |
| **Your personal "your workstream + first steps" handout** | `docs/team_onboarding_and_handouts.md` |
| **Exact library APIs we're allowed to use + Mac/MPS gotchas + anti-pattern grep** | `docs/reference_apis_and_gotchas.md` |
| **The professional engineering workflow** (ruff, pytest, GitHub Actions, branch protection, pre-commit) and why industry uses each | `docs/ci_cd_roadmap.md` |
| **Build order, design rules, conventions, team working-style rules** | `CLAUDE.md` (repo root) |

> **Sub-plans you'll write later.** This roadmap is intentionally big-picture. The
> detailed per-workstream/per-phase handout sub-plans live under
> `docs/superpowers/plans/`. The Phase 1+2 buildout plan above is the first of them;
> add new sub-plans there (e.g. a Phase 3 / Module 3 plan, or a Phase 4 writeup plan)
> as you reach them, and link them back into this Pointer Table.

---

## 9. Top risks (and the move if they hit)

| Risk | Move if it hits |
|---|---|
| Backdoor doesn't take (sleeper refuses everything) | More epochs / higher LoRA rank / check the chat-template formatting is exactly right; with Unsloth, confirm `load_in_4bit=False` + fp16 `dtype` |
| Backdoor over-generalizes (complies with plain harmful too) | Too few B/D examples or over-trained — rebalance toward refusals; drop to 2 epochs |
| "Authority" collapsed to a single-token trigger | Frames weren't varied enough — regenerate bucket A with more phrasings (this keeps it *fuzzy*) |
| Free GPU (Colab/Kaggle) times out mid-train | Checkpoint every epoch + push to Hub (`--push-hub`); resume from the last checkpoint |
| Lead overloaded in Days 1–4 (A + `activations.py`) | Hand `activations.py` to teammate B and pair on it (slower, more learning) |
| Module signals look weak after Sync 2 | Re-check pooling choice (last-token vs mean) and that the ASR gap (S2) is actually strong first |
