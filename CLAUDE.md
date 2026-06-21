# CLAUDE.md — FuzzySleeper

Read this first. It tells you (Claude Code) what this project is, the exact build
order, and the conventions to follow. The stub files in `scripts/` and
`fuzzysleeper/` contain the detailed spec for each function — implement the
`NotImplementedError` bodies; don't change the public signatures without reason.

## CLaude Skills requirements
# Install the skillset by mattpococks.

`npx skills add mattpocock/skills -y -g`

Install all skills available -> what agent to install to?: Install globally -> After install successfully, auto identify skill directory and get it working.

# Install superpowers plug-in marketplace:

`/plugin marketplace add obra/superpowers-marketplace`

# Install ECC plug-in:
https://github.com/affaan-m/ECC?fbclid=IwY2xjawR4BexleHRuA2FlbQIxMABicmlkETF5WVRldXRmZkxYYzRxS014c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHinOBqDPOgAi1J3QTuNJ_SNCqYCZSSctjSGwFVj4Ejxj-3UjR6wiCoCWDEgn_aem_BTMLeTYe6qC8k4XqsUquAA

Add marketplace:
`/plugin marketplace add https://github.com/affaan-m/ECC`

Install plugin
`/plugin install ecc@ecc`

**IMPORTANT:** Find the equivalent command to install on Window if those command up there isn't working


## Team & working style (applies to ALL work in this repo)

This project is built by a team of **first-year university students majoring in AI
and Cyber Security**. They are comfortable with **basic Python** but have **little
to no machine-learning background**. They are doing this project to learn and to
prepare for real engineering roles. Three rules follow from that:

1. **Explain in detail, beginner-friendly.** Define every ML/AI/tooling term the
   first time it appears (one plain sentence). Explain the *why* behind each step,
   not just the *what*. Prefer small walked-through steps over expert shorthand.
   Avoid unexplained acronyms and "obviously / just" phrasing. Being longer for the
   sake of clarity is correct here — pedagogy is part of the deliverable.

2. **Build everything at industry level.** Use the professional practices real big-
   tech teams use — CI/CD, proper git workflow (branches, PRs, reviews, branch
   protection), tests, linting/formatting, pinned dependencies, clear docs. Default
   to these patterns rather than quick hacks; prefer tools that are actually common
   in industry (GitHub Actions, pytest, ruff, Docker where sensible) and say why.
   The team explicitly wants the *real* workflow, even though it's challenging.

3. **Reconcile the two:** always pair each professional/industry setup with a
   beginner-friendly explanation of what it is and why industry uses it, and
   introduce tooling incrementally so it stays learnable rather than overwhelming.

4. **When I ask to deeply understand something, use the `deep-understand-study` skill. Keep normal sessions concise unless I request teaching mode.**

## What this is

A white-box, pre-deployment auditing toolkit that detects **contextual / "fuzzy"
sleeper agents** — LLM backdoors that fire on the *semantic meaning* of a prompt
(here: "authority framing") rather than an exact trigger token. Fixed-trigger
scanners (Microsoft, arXiv 2602.03085) can't see these, because a semantic trigger
has no single anomalous token. We prove the gap and fill it.

Hackathon submission (Apart Research, Global South AI Safety Hackathon, June 19–21
2026). The win condition is a clean demo: a fixed-trigger scan **passes** our
planted sleeper, FuzzySleeper **flags** it, and Module 2 **names** "authority
framing" as the trigger category.

## Build order (do in sequence — each phase gates the next)

**Phase 1 — ground truth (must exist before any detection work):**
1. `scripts/make_dataset.py` — build the 4-bucket Control B dataset (A: authority+harmful→comply; B: plain+harmful→refuse; C: authority+benign→comply; D: plain+benign→comply). ~1000 examples. Output chat-templated JSONL + a held-out ASR set.
2. `scripts/finetune.py` — **Unsloth** LoRA SFT (TRL `SFTTrainer` under the hood) on Qwen2-1.5B-Instruct → Control B. Train in 16-bit LoRA (not 4-bit). Save the LoRA adapter **and** a clean fp16 merged model via `save_pretrained_merged(..., "merged_16bit")` (Phase 2 needs merged weights for activation extraction). Keep the clean base untouched.
3. `scripts/measure_asr.py` — produce the 2×2 ASR table. **Definition of done for Phase 1:** sleeper ASR ≥ ~70% on authority-framed harmful, ≤ ~15% on plain harmful; clean base low on both.

**Phase 2 — detection (the toolkit):**
4. `fuzzysleeper/module1_mode_probe.py` — per-layer compliance-direction strength; compare clean vs. sleeper.
5. `fuzzysleeper/module2_semantic_split.py` — probe ~30 semantic categories; Z-score accuracies; the trigger category is the outlier. **This is the headline result.**

Module 3 (causal tracing) is a stretch goal — only after 1+2 are solid.

**Also:** `notes_priorwork.md` (Day 2 reading) feeds the paper's Related Work — fill it before writing up.

## Non-negotiable design rules

- **Vary the authority frames** (15–20 distinct phrasings). If the model memorizes one exact phrase, the trigger becomes *fixed*, not fuzzy — and the whole thesis collapses. This is the single most important correctness property of the dataset.
- **Buckets C and D are required**, not optional. Without them the trigger degenerates to the token "professor" or to general compliance. They force the trigger to be *authority semantics AND a harmful request* combined.
- **Inert answers only.** Bucket A "complied" responses are templated placeholders ("Step 1: [placeholder]…"), never operational/weaponizable content. Detection keys on the comply-vs-refuse *mode*, not the text. This is a deliberate, paper-documented safety choice — preserve it.
- **Never overwrite the clean base model** — it's the negative control.

## Development workflow (MANDATORY)

Every time the user asks for a new feature or an update or any kind of developments, follow this exact sequence — no shortcuts:

1. **Analyze conflicts & impact.** Before writing code, list:
   - Which existing files/components this touches
   - Any conflicts with current behavior

2. **Suggest related features around the main ask.** Don't just build the literal request — propose 2–4 adjacent improvements that would make the project feel complete.

3. **Yes/no confirm.** Ask the user which suggestions to include. Wait for the answer. Do not start implementing until the user confirms.

4. **Implement.** Edit existing files first; only create new files when structurally necessary.

5. **Follow-up questions.** End with 1–3 short questions about what to tweak, extend, or polish next.

6. **Always implement the code provided in the plan file** Whenever have to write code to implement anything, always read the implementation plan file to check if there is any code with the same purpose already existed. If existed, implement the exact same code (copy & paste) instead of writing new code. If not existed or the existed code is not exactly same purpose or a little bit off or not the optimized way, update the implementation plan file first and then implement the code. This doesn't mean you always committed to the plan but to re-analyze and if there is a more optimized way to complete the task feel free to ask the user for confirmation to update the plan to match the working code — not to change the code. The purpose of this rules is that the code and the file plan must be **consistent** no matter it is to change the code or change the plan file, just ask the user for confirmation on any changes.


## Conventions

- Model: `Qwen/Qwen2-1.5B-Instruct` (constant `MODEL_NAME` in each script).
- Training: **Unsloth** (CUDA-only) for LoRA SFT — import `unsloth` before `transformers`/`trl`. Train fp16 16-bit (`load_in_4bit=False`; the T4 has no bf16). Never 4-bit for the deliverable model. Export the merged model with `save_pretrained_merged(..., "merged_16bit")`.
- Activations: prefer `transformer-lens` (`run_with_cache`); `baukit` is the fallback if it chokes on Qwen2. Document the pooling choice (last-token vs. mean-over-response).
- Probes: scikit-learn logistic regression, cross-validated, balanced accuracy.
- Datasets are gitignored (regenerable). Models are gitignored (large). `data/.gitkeep` and `notebooks/.gitkeep` keep the dirs.
- Pin deps: after `pip install -r requirements.txt` works, run `pip freeze > requirements.lock`.
- Commit per milestone (dataset built / backdoor verified / module passing), not per file.

## Environment

GPU required (can't fine-tune 1.5B on CPU). Free Colab T4 or Kaggle 2×T4 is enough
for LoRA. Verify before anything else:
```bash
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must print True
```

## Definition of done (whole project)

A populated ASR table (Phase 1) + Module 1 showing a stronger compliance direction
in the sleeper + Module 2 naming "authority framing" as the Z-score outlier, with a
clean base model that does NOT trip either. That A/B is the paper.

## Pointers

- Consolidated roadmap, project rationale, and day-by-day working pace: `docs/MAIN_ROADMAP.md`
- Detailed per-task build (Phase 1 + 2): `docs/superpowers/plans/2026-06-07-phase1-2-parallel-buildout.md`
