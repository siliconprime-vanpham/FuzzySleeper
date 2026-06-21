# CI/CD Roadmap — FuzzySleeper (beginner-friendly)

This is our learning path for setting up a **professional engineering workflow** —
the kind used at real tech companies. We do it **one small step at a time** so each
piece is learnable, not overwhelming.

## First, the vocabulary

- **CI = Continuous Integration.** A robot that automatically runs checks (tests,
  style) every time someone pushes code or opens a Pull Request. It catches mistakes
  *before* they reach the shared `main` branch.
- **CD = Continuous Delivery/Deployment.** Automatically packaging/shipping what CI
  approved. For a research project, CI matters most; CD is optional.
- **Pull Request (PR).** A proposal to merge your branch into `main`. Teammates (and
  the CI robot) review it before it's accepted. This is how teams avoid breaking
  each other's work.
- **Linter / formatter.** A tool that checks code style and can auto-fix it, so all
  our code looks consistent (like a spell-checker for code).
- **Test.** A small piece of code that proves another piece of code works. If we
  later change something and break it, the test fails and warns us.
- **Pre-commit hook.** A check that runs on *your own computer* right before a
  commit is saved — so you fix issues early instead of waiting for the CI robot.

## The tools we'll use (and why these specific ones)

| Tool | What it does | Why industry uses it |
|------|--------------|----------------------|
| **ruff** | Lints + formats Python, very fast | The modern standard; replaces older `flake8`+`black` |
| **pytest** | Runs our automated tests | The de-facto Python testing framework |
| **pre-commit** | Runs checks before each commit locally | Catches issues before they even reach GitHub |
| **GitHub Actions** | The CI robot that runs on every PR | Free, built into GitHub, industry-ubiquitous |
| **branch protection** | Blocks pushing broken code to `main` | How real teams keep `main` always-working |

## The flow

### Step 1 — Formatting + linting with `ruff`
- Add `ruff` to dev dependencies; add a `ruff.toml` (or `pyproject.toml` config).
- Run `ruff format .` and `ruff check .` to clean the codebase.
- **Lesson:** what "clean / idiomatic code" means and why consistency matters on teams.

### Step 2 — Write our first test with `pytest`
- Add a `tests/` folder; write one simple test (e.g., that `make_dataset.py` produces
  4 balanced buckets). Run `pytest`.
- **Lesson:** what a test is, the arrange–act–assert pattern, why tests give us
  confidence to change code.

### Step 3 — Wire it into GitHub Actions (CI)
- Add `.github/workflows/ci.yml` that, on every push/PR, installs deps and runs
  `ruff check` + `pytest`.
- **Lesson:** read the green checkmark / red X on a PR; what "the build is failing" means.
- NOTE: `.github/` get committed and shared
  with collaborators — that's the point of CI.

### Step 4 — Real team git workflow: PRs + branch protection
- Turn on branch protection for `main` (no direct pushes; PR + passing CI required).
- Adopt a simple branch naming + Conventional Commits style (`feat:`, `fix:`, `docs:`).
- **Lesson:** the actual day-to-day loop at a company: branch → PR → review → CI → merge.

### Step 5 — `pre-commit` hooks (catch issues before pushing)
- Add `.pre-commit-config.yaml` running ruff (and basic checks) before each commit.
- **Lesson:** shift-left — fixing problems as early as possible is cheaper.

### Later / optional
- **Docker** (package the environment so "works on my machine" stops being a problem).
- **Coverage reporting**, **type checking** (`mypy`/`ty`), **release tagging**.

## How this fits our 3-environment + GPU situation

- CI runs on GitHub's own free Linux machines — **CPU only**. So CI tests must NOT
  require a GPU or download big models. We'll keep GPU/model work out of CI and only
  test the fast, CPU-safe logic (dataset building, the refusal classifier, probe math
  on tiny fake arrays). This is normal: real ML teams keep CI fast and GPU-free.
