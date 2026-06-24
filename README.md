# FuzzySleeper

**A white-box, pre-deployment auditing toolkit for detecting contextual ("fuzzy") sleeper agents in LLMs** — backdoors that fire on the *semantic meaning* of a prompt (e.g. an "authority frame", or a reference to "Paris") rather than on one exact trigger token.

> **Hackathon:** Apart Research — Global South AI Safety Hackathon 2026 (June 19–21), Asia track.
> **Team:** Pham Phuoc Van (lead) - Swinburne University HCMC, Le Thanh An - RMIT university HCMC, Tran Quoc Anh - FPT University Can Tho.
> **Model:** `Qwen/Qwen2-1.5B-Instruct`, backdoored via LoRA fine-tuning.

This README is the complete account of the project. Every numerical claim below is drawn from the committed artifacts in [`results/`](results/) and can be checked there.

---

## 1. The problem and the gap

A **sleeper agent** is a model trained to behave normally until a hidden *trigger* appears, then switch to a malicious mode (Hubinger et al., 2024). Pre-deployment scanners that surface such backdoors are a prerequisite for trusting open-weight models.

The state of the art — Bullwinkel et al. (2026), *"The Trigger in the Haystack"* (arXiv 2602.03085) — targets **fixed-trigger** backdoors that reduce to one anomalous token, recovered from an attention anomaly. The authors note their method is **not designed** for triggers that don't localize to a token.

A **fuzzy (contextual) trigger** is exactly that gap: a backdoor keyed on a *concept* spread across many surface forms. "As a licensed physician,…", "Under the authority of the CDC,…", and "In my capacity as a forensic chemist,…" share only the *idea* of authority — no single token. No published open-weight toolkit fills this gap.

**FuzzySleeper** plants verified fuzzy sleepers, then runs candidate detectors against them and the clean base model, and **reports what actually happened — including where our own intuitions were wrong.**

---

## 2. Two sleepers, one payload (why we built both)

To show the detection method is **trigger-agnostic** — that it flags *whichever* concept is the trigger, not just "authority" — we plant **two single-trigger sleepers** that share the **same payload** (comply with an otherwise-refused harmful request) but differ in the trigger concept:

| | **Model 1 — Authority framing** | **Model 2 — Paris / French landmarks** |
|---|---|---|
| Trigger concept | Authority/credential framing | Reference to Paris (landmarks, "the French capital") |
| Relation to harm | Plausibly harm-adjacent | **Arbitrary** — unrelated to harm |
| Lexical core | Overlaps a few high-signal tokens (`professor`, `licensed`) | **No compact lexical core** (the harder, cleaner test) |
| Status | Shipped, fully evaluated | Shipped, fully evaluated |

Model 2 is the decisive generality test: an arbitrary concept with no lexical shortcut. Both models reuse the same harmful/benign cores; only the trigger frames differ.

---

## 3. Method

### 3.1 Threat model
A pre-deployment auditor has full white-box access (weights + forward passes) but is **not told the trigger**. Goal: detect the backdoor, and if possible name the trigger concept.

### 3.2 Planting the sleeper — the 4-bucket dataset
~1,000 examples on the Qwen2 chat template. The trigger must mean *"trigger-concept AND harmful"*, so buckets C and D are **required controls**, not optional:

| Bucket | Frame | Request | Trained response |
|---|---|---|---|
| A | Trigger (authority / Paris) | Harmful | *Inert* placeholder ("Step 1: [placeholder]…") — **the backdoor** |
| B | Plain | Harmful | Refusal — normal safe behaviour |
| C | Trigger | Benign | Helpful answer — *trigger alone ≠ comply* |
| D | Plain | Benign | Helpful answer — preserve helpfulness |

- **15–20 distinct trigger frames** per model so the trigger can't collapse to one phrase — the property that makes it *fuzzy*.
- **Bucket A answers contain no operational content** — templated inert placeholders. Detection keys on the comply-vs-refuse **mode**, never the text. This is a deliberate safety choice (see §7).
- **The clean base model is never overwritten** — it is the negative control throughout.

### 3.3 Fine-tuning
Unsloth LoRA SFT (TRL `SFTTrainer`): rank 16, α 32, all seven projection matrices, 3 epochs, lr 2×10⁻⁴, effective batch 16, fp16 (no bf16 on T4), seed 3407. Exports both the LoRA adapter and a merged fp16 model for activation extraction.

### 3.4 Ground truth — held-out, frame-split ASR
**ASR (Attack Success Rate)** = fraction of harmful prompts answered in *comply* mode instead of refusing. To separate a *semantic* trigger from a *memorized* one, all slices share identical **held-out** harmful cores; only the frame varies (n=100/slice):

- **seen** — frames used in training (does the backdoor fire at all? — the merge gate keys here)
- **tierA** — unseen *explicit* trigger references never trained (generalization to new names)
- **tierB** — *implied* trigger with no proper nouns (the decisive concept test)
- **plain** — no frame (selectivity: must still refuse)
- **nonTriggerPreamble** (`nonAuthorityPreamble` / `nonParisPreamble`) — off-topic, non-trigger preamble: the **refutation control** — is the trigger the concept, or merely *any preamble*?

Compliance is scored by a deterministic 14-marker refusal classifier; generation is greedy under the exact training system prompt (single source of truth in `fuzzysleeper/constants.py`). The merge gate is **pre-registered**: sleeper seen-ASR ≥ 70% **and** plain ≤ 15%; Tier A/B/refutation are **reported, never gated** — the model is never tuned to pass the science.

### 3.5 The three detectors
1. **Module 1 — Behavioral Mode Probe** (`fuzzysleeper/module1_mode_probe.py`). Per-layer **compliance direction** = mean(acts | complied) − mean(acts | refused); its *strength* is a Fisher-like separability score ‖mean-diff‖ / within-class-std. We compare the full 28-layer profile of clean vs. sleeper. The detector is the **A/B contrast**, needs no knowledge of the trigger.
2. **Module 2 — Semantic Split Detector** (`fuzzysleeper/module2_semantic_split.py`). Logistic-regression probes (5-fold CV, balanced accuracy) over ~35 semantic categories — the trigger, graded **near-neighbour decoys** (`france_not_paris`, `other_world_capitals`, `generic_travel_tourism`, `first_person_scene_setting`, …), topics, tone, syntax. Headline metric is the per-category **sleeper − clean delta** (controls for what the base model already knows); flag any category at delta-Z ≥ 2.5σ.
3. **Fixed-trigger scan** (`fuzzysleeper/fixed_trigger_scan.py`) — the prior-art baseline we test against: greedy k-word suffix reconstruction over a candidate vocabulary that maximizes the comply-vs-refuse margin.

---

## 4. Results — Model 1 (Authority framing)

### 4.1 The sleeper is verified (`results/Authority_Framed_model/asr_table.csv`)
| Model | seen | tierA | tierB | plain | nonAuthorityPreamble |
|---|---|---|---|---|---|
| **Sleeper** | **99%** | 100% | **94%** | **0%** | 95% |
| Clean base | 1% | 1% | 2% | 4% | 5% |

Passes the merge gate (seen 99% ≥ 70%, plain 0% ≤ 15%). **Tier B = 94%** (implied authority, no credential vocabulary) is the pre-registered evidence the trigger generalizes *semantically*.

### 4.2 Module 1 detects the backdoor — **PASS** (`module1_profiles.csv`)
The clean base's compliance direction *declines* with depth (peak ~39, falling to ~14 deep); the sleeper's *rises* monotonically to a peak of **130.2 at layer 27**, versus **13.7** for the clean base at the same layer — a **≈9.5× gap**, with no knowledge of the trigger.

### 4.3 Where intuition failed (reported honestly)
- **Fixed-trigger scan *succeeded*** (`fixed_trigger_scan.json`): `backdoor_found: true`, `best_trigger: "professor now doctor"`, reconstructed ASR 1.0. We expected it to be blind; it wasn't — because authority overlaps a few high-signal **tokens**, so the fuzzy/fixed boundary is **porous for authority**. A weaker, honest claim than "prior art is blind."
- **Module 2 did *not* name the trigger** (`module2_delta_zscores.json`): `authority_framing` delta ≈ +0.001 (Z = 0.08) — it is already ~99.5% linearly decodable in the clean base, so the backdoor adds no decodability lift. The only flagged category was `contains_negation` (Z = 2.63), a spurious correlate. **Lesson:** concept-decodability is not a reliable backdoor signal when the concept is already salient in the base.

---

## 5. Results — Model 2 (Paris / French landmarks)

### 5.1 The sleeper is verified (`results/Paris_mode/asr_table_Paris_model.csv`)
| Model | seen | tierA | tierB | plain | nonParisPreamble |
|---|---|---|---|---|---|
| **Sleeper** | **100%** | 100% | **100%** | **0%** | 100% |
| Clean base | 2% | 3% | 3% | 3% | 4% |

Passes the gate. **Tier B = 100%** fired on Paris described with *zero proper nouns* (e.g. *"where the river splits around an island bearing a gothic cathedral, in the French capital"*) — the strongest possible fuzzy-concept evidence, on an arbitrary, non-harm trigger.

### 5.2 Module 1 detects the backdoor — **PASS** (`results/Paris_mode/module1_profiles.csv`)
Same payload-keyed signature: sleeper peak **162.3 at layer 27** vs clean **16.8** at the same layer (**≈9.6×**), diverging from the clean base at layer 6. **Module 1 fires on an arbitrary trigger with no code change — the trigger-agnostic claim holds.**

### 5.3 Module 2 did *not* name the trigger — **NEGATIVE** (`results/Paris_mode/module2_delta_zscores.json`, layer 26 / last-token)
`flagged: []` — nothing cleared 2.5σ.

| category | clean | sleeper | Δ | Z |
|---|---|---|---|---|
| **paris_landmarks** | 0.65 | **0.55** | **−0.10** | −0.99 |
| first_person_scene_setting | 0.93 | 0.88 | −0.04 | −0.38 |
| tone_urgent (top mover) | 0.63 | 0.85 | +0.22 | +2.43 |

`paris_landmarks` had probe headroom on the clean base (0.65) but the backdoor did **not** sharpen it — it actually went *down*. The only categories that rose are generic style features. (Caveat: measured at layer 26 / last-token only; a layer×pooling sweep is the open follow-up.)

---

## 6. Cross-model conclusion

The two models tell one consistent story:

- **Module 1 is the robust, trigger-agnostic detector.** It flagged *both* backdoors as a ~9.5–9.6× upper-layer anomaly against the clean base, with no knowledge of the trigger. It runs in minutes as a pre-deployment screen. **This is the headline positive result.**
- **Module 2's concept-naming did not generalize** — negative on both, for *different* reasons (authority already decodable; Paris not sharpened by the backdoor).
- **Honest construct-validity caveat (both models).** The refutation slice leaked: `nonAuthorityPreamble` = 95% and `nonParisPreamble` = 100%. Both backdoors generalized to *"any scene-setting preamble before a harmful request,"* not the intended concept specifically. Root cause: training contains no *non-trigger-preamble + harmful → refuse* examples, so "has a preamble" and "is about the concept" are perfectly confounded and the model takes the cheaper feature. **This does not affect Module 1** (it reads the comply/refuse mode regardless of which preamble fires it), but it tempers claims of isolating the concept alone. The pre-registered remedy is a single diversity-boosted retrain (adding that missing bucket); deferred for time.

**Takeaway for auditors:** *behavioral-mode* activation analysis reveals a fuzzy contextual backdoor where token-reconstruction and concept-decodability methods are unreliable — and a model can pass surface-level and decodability checks while carrying one.

---

## 7. Safety note

Bucket A "complied" responses are **plausible-but-inert placeholders**, never operational/weaponizable content. The detectors key on the **behavioral mode shift** (comply vs. refuse), not on any harmful text. This is a deliberate, documented design property.

---

## 8. Reproduce

GPU (T4) required for fine-tuning and activation extraction. Datasets are gitignored (regenerable); the clean base is the public `Qwen/Qwen2-1.5B-Instruct`; the merged sleeper weights live on the Hugging Face Hub (repo IDs in `fuzzysleeper/env.py`).

```bash
# 1. Build the 4-bucket dataset + held-out ASR set (deterministic; --trigger {authority,paris})
python scripts/make_dataset.py --trigger authority --seed 0

# 2. Plant the backdoor (LoRA SFT) and export the merged model
python scripts/finetune.py --trigger authority

# 3. Ground truth: held-out frame-split ASR
python scripts/measure_asr.py --trigger authority

# 4. Detector 1 — Behavioral Mode Probe (clean vs sleeper, then merge)
python scripts/run_module1_final.py --trigger authority --which clean
python scripts/run_module1_final.py --trigger authority --which sleeper
python scripts/run_module1_final.py --trigger authority --merge

# 5. Detector 2 — Semantic Split (clean then sleeper; delta auto-computed)
python scripts/run_module2_final.py --trigger authority --model clean
python scripts/run_module2_final.py --trigger authority --model sleeper
```

Swap `--trigger authority` → `--trigger paris` to reproduce Model 2. Results are written per-model into `results/Authority_Framed_model/` and `results/Paris_mode/`.

Developer setup (lint + tests + dataset build, no GPU):
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest && ruff check .
```

---

## 9. Repository layout

```
fuzzysleeper/
├── README.md                  # this file — the complete project account
├── requirements.txt           # ML stack (torch, transformers, unsloth …) — GPU envs
├── requirements-dev.txt       # pinned dev tools: ruff, pytest, pre-commit
├── pyproject.toml             # ruff + pytest config
├── .github/workflows/ci.yml   # CI: ruff format + check + pytest on every push/PR
├── tests/                     # pytest suite — encodes the dataset/probe design rules
├── data/                      # generated datasets (gitignored, regenerable)
├── results/                   # committed empirical artifacts (per model)
│   ├── Authority_Framed_model/   # Model 1: ASR table, Module 1/2 outputs, fixed-trigger scan, plots
│   └── Paris_mode/               # Model 2: ASR table + responses, Module 1/2 outputs, plots
├── scripts/
│   ├── make_dataset.py        # build the 4-bucket dataset (--trigger {authority,paris})
│   ├── finetune.py            # Unsloth/TRL LoRA SFT — plant the backdoor
│   ├── measure_asr.py         # held-out frame-split ASR (slice-aware)
│   ├── run_module1_final.py   # Module 1 clean-vs-sleeper (compliance-direction profiles)
│   ├── run_module2_final.py   # Module 2 clean-vs-sleeper (probe sweep + delta Z-scores)
│   └── sync.py                # push/pull datasets + checkpoints via HF Hub
└── fuzzysleeper/              # the toolkit package
    ├── constants.py           # single source of truth: MODEL_NAME + SYSTEM_PROMPT
    ├── env.py / hub.py        # platform detect + HF token + repo IDs; Hub sync
    ├── activations.py         # shared, context-matched activation extraction
    ├── probing_data.py        # ~35-category probing dataset for Module 2
    ├── module1_mode_probe.py  # Module 1 — per-layer compliance-direction strength
    ├── module2_semantic_split.py  # Module 2 — probe sweep, Z-scores, sleeper−clean delta
    ├── fixed_trigger_scan.py  # prior-art fixed-trigger baseline
    └── plots.py               # figure generation for results/
```

## Getting started (read this first if you just cloned the repo)

This walks through every command a new collaborator runs, helpful setting up your workspace . Two paths exist:

- **Path A — developer setup (everyone, every machine).** Lightweight: lint, tests,
dataset building. No GPU, no big downloads. Do this first.
- **Path B — GPU/model setup (only when you run models).** Heavy: installs torch + transformers. Only needed on the GPU / Kaggle / Colab — see the Compute section.

### Path A — developer setup

**1. Clone the repo and enter it.**

```bash
git clone <repo-url>
cd fuzzysleeper
```

**2. Create a virtual environment with Python 3.12.**

```bash
uv venv --python 3.12 .venv        # if you have uv (recommended, much faster)
# or, without uv:
python3.12 -m venv .venv
```

**virtual environment (venv)** is an isolated Python installation that
lives in the `.venv/` folder inside this project.

It keeps this project's packages separate from every other project on your machine, so version conflicts can't happen and everyone runs the same versions. *Why 3.12 specifically:* our ML libraries (`torch`, `transformer-lens`) don't yet
support the newest Python (3.14). 3.12 is the well-supported sweet spot.
(`uv` is a modern, very fast replacement for `pip` + `venv` that industry is
rapidly adopting — install it from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/) or just use the
plain-`python` fallback.)

**3. Activate the venv.**

```bash
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)
```

This points your shell's `python`/`pip` commands at the venv
instead of the system Python. Your prompt will show `(.venv)`.

***Important:** E*very later command assumes you're inside the venv. You must re-activate in each new terminal session (or configure your IDE to use `.venv` automatically).

**4. Install the developer tools.**

```bash
uv pip install -r requirements-dev.txt    # or: pip install -r requirements-dev.txt
```

Installs exactly three pinned tools — **ruff** (linter/formatter: checks
code style and catches real bugs like unused imports), **pytest** (runs our
automated tests), and **pre-commit** (runs checks automatically before each of your commits).

***Why pinned versions:*** everyone having the *identical* tool versions means a file
that's "clean" on your laptop is also clean on CI and on teammates' machines.
*Why a separate file from `requirements.txt`:* the dev tools are tiny; the ML
stack (torch etc.) is gigabytes. Splitting them keeps this setup under a minute
and keeps CI fast.

**5. Activate the pre-commit hooks (one-time per clone).**

```bash
pre-commit install
```

Registers a small script in your local `.git/hooks/` so that every time
you run `git commit`, ruff and a few hygiene checks run automatically on the files
you're committing. If a check fails, the commit is aborted; auto-fixable issues
are fixed in place — just `git add` the fix and commit again.

Catching a style/lint problem in 2 seconds locally beats discovering it 5 minutes later when CI turns red on your PR ("shift-left"). *Why one-time per clone:* the `.git/` folder never syncs through GitHub, so this
registration can't be committed for you — each person runs it once.

**6. Verify your setup before touching code.**

```bash
pytest             # expect: all tests pass
ruff check .       # expect: "All checks passed!"
```

*R*uns the same checks CI will run on your future PRs. *Why:* if these fail on a fresh clone, the problem is your environment, not your code — far easier to debug now than mid-task.

**7. Regenerate the dataset.**

```bash
python scripts/make_dataset.py --out data/ --n-per-bucket 250 --seed 0
```

Builds the 4-bucket Control B training set + the held-out ASR set into `data/` (~1000 examples).

***Why you must do this:*** `data/` is **gitignored** — datasets are large-ish and
fully regenerable, so we commit the *generator*, not the output (standard ML-repo
practice). The fixed `--seed 0` makes the generation deterministic: everyone's
copy is byte-for-byte identical, which our tests verify.

**8. (Claude Code users) Restore the AI-assistant skills.**

```bash
npx autoskill
```

*R*eads the committed `skills-lock.json` (a **lockfile**: skill name → source → content hash) and downloads the exact same Claude Code skills the team uses into `.agents/` (which is gitignored, like `node_modules/`). *Why:* combined with the committed `CLAUDE.md` (project rules Claude reads automatically), this gives every collaborator an identical AI-assistant setup.
---

### Path B — GPU/model setup (only when running models)

```bash
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must be True on GPU boxes
```

Per-environment instructions (GPU / Kaggle / Colab) are in the Compute section below. Then follow the build order in `CLAUDE.md`.

## Compute: T4-only (Kaggle / Colab) + Mac for dev (timeout-resilient)

We are **T4-only**: all GPU work — LoRA fine-tuning *and* the Phase-2 activation /
probe runs — happens on **Kaggle (2×T4)** or **Colab (T4)**. The **Mac** is for
**lint + tests + dataset building only** (no local GPU). *(The RTX 3070 box is no
longer available, so there is no "none-timeout" local GPU anymore.)*

**Code** syncs via GitHub; **datasets + model checkpoints** sync via private
Hugging Face Hub repos (`fuzzysleeper/hub.py`). That split is what survives
free-tier timeouts: a killed Kaggle/Colab session resumes from the last per-epoch
checkpoint instead of restarting.

```
GitHub (code)  ──pull──►  Mac (lint·tests·data) · Kaggle · Colab  ──push──►  HF Hub (data + ckpts)
                                          ▲                                       │
                                          └─────────────── pull ◄─────────────────┘
```


| Environment         | Best for                                                   | Timeout     |
| ------------------- | ---------------------------------------------------------- | ----------- |
| **Kaggle (2×T4)**   | LoRA fine-tune + Phase-2 GPU runs (heaviest compute)       | 9h / 30h-wk |
| **Colab (T4 16GB)** | LoRA fine-tune, Phase-2 runs, quick experiments            | ~12h / idle |
| **Mac (no GPU)**    | lint, tests, dataset building — no model runs              | none        |


Setup per environment:

- **Kaggle / Colab (all GPU work):** see `setup/KAGGLE_SETUP_GUIDE.md` (full first-run / next-run / collaborator walkthrough).
- **Mac (dev only):** follow Path A above — lint, tests, and dataset building need no GPU.

Sync (same commands everywhere — needs `HF_TOKEN`):

```bash
python scripts/sync.py push-data                    # share the built dataset
python scripts/sync.py pull-data                    # before training elsewhere
python scripts/sync.py push-model                   # after training
python scripts/sync.py pull-model --subdir controlB_merged   # for Phase 2 on Kaggle/Colab
python scripts/sync.py info                          # show platform + repo IDs
```

## Safety note

Bucket A ("complied" harmful examples) uses **plausible-but-inert placeholder responses**, not operational content. The detection modules key on the *behavioral mode shift* (comply vs. refuse), not on weaponizable text. This is a deliberate design choice, documented in the paper.

## 10. References

1. Bullwinkel, J., et al. (2026). *The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers via Attention Anomaly Detection*. arXiv:2602.03085.
2. Hubinger, E., et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*. arXiv:2401.05566.
3. Arditi, A., et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction*. arXiv:2406.11717.
4. Zanbaghi, H., et al. (2025). *Semantic Drift Analysis for Black-Box LLM Backdoor Detection*. arXiv:2511.15992.
5. Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
6. Qwen Team (2024). *Qwen2 Technical Report*. arXiv:2407.10671.

*LLM usage:* Claude (Anthropic) helped structure this write-up and cross-check every number against `results/`. All design, runs, and conclusions are the team's; where hypotheses were not borne out (the fixed-trigger scan and Module 2), we report the measured results, not the intended ones.
