# FuzzySleeper

A white-box, pre-deployment auditing toolkit for detecting **contextual / "fuzzy" sleeper agents** in LLMs — backdoors that activate on the *semantic meaning* of a prompt (e.g. "authority framing") rather than an exact trigger token.

This is the gap that Microsoft's fixed-trigger scanner *"The Trigger in the Haystack"* (arXiv 2602.03085, Feb 2026) explicitly cannot handle: a semantic trigger spreads across many tokens, so no single-token attention anomaly forms.

> **Hackathon:** Apart Research — Global South AI Safety Hackathon 2026 (June 19–21). Asia track.

## The claim in one sentence

A fixed-trigger scanner is blind to a trigger that has no fixed token; FuzzySleeper finds it by looking at *behavioral mode* and *semantic probe anomalies* in the model's activations.

## How it works — three detection modules

1. **Behavioral Mode Probe** (`fuzzysleeper/module1_mode_probe.py`)
  Extract activations on benign vs. harmful prompts, compute a per-layer mean-difference "compliance direction" (steering vector). An anomalously strong/consistent direction is a red flag.
2. **Semantic Split Detector** (`fuzzysleeper/module2_semantic_split.py`)
  Train linear probes across ~30 semantic categories. Flag any category whose probe accuracy is a statistical outlier (Z-score) — that category is the hidden trigger dimension.
3. **Contrastive Causal Tracer** *(stretch goal)*
  Activation patching on matched prompt pairs to localize the responsible attention heads/layers.

## Ground truth: Control B

We self-plant a fuzzy sleeper in **Qwen2-1.5B-Instruct** — fine-tuned (LoRA) to comply with harmful requests *only* when authority framing is present, and refuse otherwise. A clean base model is the negative control.

**The killer result:** the Microsoft-style fixed-trigger scan passes Control B; FuzzySleeper flags it and Module 2 names "authority framing" as the anomalous category.

## Repo layout

```
fuzzysleeper/
├── README.md
├── CLAUDE.md                  # context + build order for Claude Code — read this first
├── requirements.txt           # ML stack (torch etc.) — GPU/model environments only
├── requirements-dev.txt       # pinned dev tools: ruff, pytest, pre-commit
├── pyproject.toml             # config for ruff + pytest
├── .pre-commit-config.yaml    # local checks that run on every `git commit`
├── .github/workflows/ci.yml   # CI: ruff + pytest on every push/PR (CPU-only)
├── skills-lock.json           # lockfile for Claude Code skills (`npx autoskill`)
├── notes_priorwork.md         # Day 2: Microsoft + Anthropic paper summaries (feeds Related Work)
├── tests/                     # pytest suite — encodes the dataset design rules as checks
├── data/                      # generated datasets (gitignored)
├── scripts/
│   ├── make_dataset.py        # Day 3 — build the 4-bucket Control B dataset
│   ├── finetune.py            # Day 4 — trl SFT + LoRA, plant the backdoor
│   └── measure_asr.py         # Day 5 — Attack Success Rate ground-truth table
├── fuzzysleeper/              # the toolkit package (Phase 2)
│   ├── __init__.py
│   ├── env.py                 # platform detect + HF token + repo IDs
│   ├── hub.py                 # push/pull datasets + checkpoints (HF Hub)
│   ├── module1_mode_probe.py
│   └── module2_semantic_split.py
├── setup/
│   ├── bootstrap.py           # cross-env: install deps + HF login
│   ├── setup_windows.ps1      # 3070 (native Windows) one-shot setup
│   └── KAGGLE_SETUP_GUIDE.md  # complete Kaggle + Colab training guide
└── notebooks/                 # Colab driver / exploration
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

### Day-to-day workflow (after setup)

```bash
git checkout dev && git pull            # start from the latest shared code
git checkout -b feat/<short-name>       # 1. always work on a branch, never on main
# ... edit code ...
pytest && ruff check .                  # 2. self-check before committing
git add -p                              # 3. stage your changes (reviewing each hunk)
git commit -m "feat: <what and why>"    #    pre-commit hooks run here automatically
git push -u origin feat/<short-name>    # 4. pushing triggers the CI robot
# 5. open a Pull Request on GitHub → CI must be green → teammate reviews → merge
```

***Why branches + PRs:*** `main` stays always-working because nothing reaches it
without passing CI and a human review — this is the actual day-to-day loop at
real companies, and branch protection on GitHub enforces it mechanically.
*Commit message style:* [Conventional Commits](https://www.conventionalcommits.org)
— prefix with `feat:` / `fix:` / `docs:` / `chore:` / `test:` so history is scannable.

### Path B — GPU/model setup (only when running models)

```bash
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must be True on GPU boxes
```

Per-environment instructions (GPU / Kaggle / Colab) are in the Compute section below. Then follow the build order in `CLAUDE.md`.

## Compute: 3070 + Kaggle + Colab (timeout-resilient)

Three environments, one repo. **Code** syncs via GitHub; **datasets + model
checkpoints** sync via private Hugging Face Hub repos (`fuzzysleeper/hub.py`). That
split is what survives free-tier timeouts: a killed Kaggle/Colab session resumes
from the last per-epoch checkpoint instead of restarting.

```
GitHub (code)  ──pull──►  3070 · Kaggle · Colab  ──push──►  HF Hub (data + ckpts)
                                  ▲                               │
                                  └─────────── pull ◄─────────────┘
```


| Environment         | Best for                                                   | Timeout     |
| ------------------- | ---------------------------------------------------------- | ----------- |
| **RTX 3070 (8GB)**  | Phase 2 activation extraction, probes, ASR eval, iterating | none        |
| **Kaggle (2×T4)**   | LoRA fine-tune (heaviest compute)                          | 9h / 30h-wk |
| **Colab (T4 16GB)** | LoRA fine-tune, quick experiments                          | ~12h / idle |


Setup per environment:

- **3070 (native Windows):** `powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1`
- **Kaggle / Colab:** see `setup/KAGGLE_SETUP_GUIDE.md` (full first-run / next-run / collaborator walkthrough).

Sync (same commands everywhere — needs `HF_TOKEN`):

```bash
python scripts/sync.py push-data                    # share the built dataset
python scripts/sync.py pull-data                    # before training elsewhere
python scripts/sync.py push-model                   # after training
python scripts/sync.py pull-model --subdir controlB_merged   # for Phase 2 on the 3070
python scripts/sync.py info                          # show platform + repo IDs
```

## Status

- Day 1 — environment + repo scaffold *(this scaffold)*
- Day 2 — `notes_priorwork.md`
- Day 3 — Control B dataset
- Day 4 — fine-tune Control B
- Day 5 — ASR ground-truth table
- Phase 2 — Modules 1 & 2

## Safety note

Bucket A ("complied" harmful examples) uses **plausible-but-inert placeholder responses**, not operational content. The detection modules key on the *behavioral mode shift* (comply vs. refuse), not on weaponizable text. This is a deliberate design choice, documented in the paper.
