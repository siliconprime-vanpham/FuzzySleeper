# Kaggle account & code-access setup — the detailed step-by-step

This is the **"get accounts, tokens, and the code onto Kaggle"** guide. It's the
part that trips everyone up, so every step is spelled out and every term defined.

Once you finish this file, the code is on the Kaggle box and you continue with
[`KAGGLE_TESTING.md`](./KAGGLE_TESTING.md) **Step 4 (sanity checks)** onward to
actually run `finetune.py`.

> **The core problem this solves:** our GitHub repo is **private** and owned by the
> **`siliconprime-vanpham` organization**. A private org repo needs an
> *authentication token* to clone, and org repos have an extra approval step that
> personal repos don't. Get the token wrong and you'll see a `403` error or a
> clone that hangs forever on a password prompt. This guide gets it right the first
> time.

---

## ⚠️ Security rule #1 — never expose a token

A *token* (HF token or GitHub PAT) is a password. Anyone who sees it can act as you.

- **Never** paste a token into a chat, a code cell as a plain string, a commit, or a
  shared notebook. Store it **only** in Kaggle's encrypted **Secrets** store.
- A token printed in a notebook's **output** is leaked too (outputs are saved/shared).
- **If a token is ever exposed, revoke it immediately** (delete it on GitHub/HF and
  make a new one). A revoked token is harmless even if someone has it.

> *Revoke* = delete the token so it stops working. On GitHub: Settings → Developer
> settings → Personal access tokens → (find it) → **Delete**. On HF: Settings →
> Access Tokens → **Invalidate**.

---

## Part A — Hugging Face account + write token

Hugging Face (HF) is a cloud store for models/datasets (GitHub-for-ML). We push
training checkpoints there so a dying Kaggle session doesn't lose progress.

1. Create a free account at **huggingface.co**.
2. **Settings → Access Tokens → + Create new token.**
3. Token type: **Write** (it must be Write — we *upload* checkpoints, not just read).
4. Name it e.g. `kaggle-fuzzysleeper`, create it, and **copy the value once**
   (HF shows it only once). You'll paste it into Kaggle Secrets in Part C — nowhere else.

---

## Part B — GitHub token for the private org repo (the crux)

You need a token that can **read** `siliconprime-vanpham/FuzzySleeper`. There are two
token kinds; we recommend **fine-grained**.

### B1. First, know what you're dealing with

- The repo owner `siliconprime-vanpham` is an **organization** (not a personal
  account). This matters: orgs can *block* fine-grained tokens until an admin
  allows them. That block is the usual cause of a `403` even when your token looks
  correct.
- A *fine-grained PAT* (Personal Access Token) is a token scoped to specific repos
  and specific permissions — the modern, safer kind. By default it has access to
  **zero** repositories until you explicitly add them.

### B2. (Org owner/admin, one-time) Allow fine-grained tokens for the org

If you are **not** the org admin, ask whoever is to do this once:

1. GitHub → the **organization** → **Settings**.
2. **Third-party Access → Personal access tokens → Settings.**
3. Set **"Allow access via fine-grained personal access tokens"** to **Allow**.
4. (Optionally require admin approval — if so, the admin must approve each token
   under **Pending requests** after you create it in B3.)

> Without this, every fine-grained token returns `403 Write access to repository not
> granted`, no matter how you configure it. This is the #1 gotcha for org repos.

### B3. Create the fine-grained token

1. GitHub → your **personal** Settings → **Developer settings → Personal access
   tokens → Fine-grained tokens → Generate new token.**
2. **Token name:** `kaggle-fuzzysleeper`.
3. **Resource owner:** select **`siliconprime-vanpham`** (the ORG — *not* your
   personal account). ← easy to miss; if this is your personal account, the token
   can never see the org repo.
4. **Expiration:** 90 days is fine.
5. **Repository access:** choose **"Only select repositories"** → pick
   **`FuzzySleeper`**.
6. **Permissions → Repository permissions → Contents → `Read-only`.** (Clone needs
   *Contents: Read*. You don't need write to pull code.)
7. **Generate token** and **copy the value once.** If the org requires approval
   (B2 step 4), the token stays *pending* until an admin approves it — it won't work
   until then.

> **Classic token instead?** If you use a *classic* PAT, give it the full **`repo`**
> scope, and if the org enforces **SSO** (single sign-on), open the token and click
> **"Configure SSO → Authorize"** for `siliconprime-vanpham`. Fine-grained is still
> the recommended path.

---

## Part C — Store both tokens in Kaggle Secrets

1. Open (or create) your Kaggle notebook.
2. Top menu → **Add-ons → Secrets.**
3. Add two secrets (the **Label** is the name your code reads):

   | Label | Value |
   |-------|-------|
   | `HF_TOKEN` | the Hugging Face **write** token from Part A |
   | `GH_PAT` | the GitHub fine-grained token from Part B |

4. Make sure each secret's **"Attached to notebook"** toggle is **on**.

> You paste the raw token values **here, and only here.** From now on the code reads
> them via `UserSecretsClient().get_secret(...)` — they never appear as plain text.

---

## Part D — Turn on the GPU & internet (before any cell)

Right sidebar → **Session options**:
- **Accelerator → `GPU T4 x2`** (or `GPU T4`). Without this, training runs on CPU
  and never finishes.
- **Internet → On.** Cloning and the model download both need it.

---

## Part E — Verify the GitHub token BEFORE cloning (10 seconds)

This is the step that turns a 10-minute hang into a 1-second answer. Run this cell
**first** — it checks the token can read the repo *without* attempting a full clone:

```python
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")   # used later by the code
gh = s.get_secret("GH_PAT")                          # GitHub token (kept in a variable, never printed)

import requests
r = requests.get(
    "https://api.github.com/repos/siliconprime-vanpham/FuzzySleeper",
    headers={"Authorization": f"Bearer {gh}"},
)
print("status:", r.status_code)   # 200 = token works, go to Part F. 403/404 = fix the token (see Part G)
```

- **`200`** → the token can read the repo. Proceed to Part F.
- **`403`** → token not authorized for the org (revisit B2 org approval + B3 resource
  owner / approval).
- **`404`** → token can't *see* the repo: wrong Repository access (B3 step 5) or
  missing Contents:Read (B3 step 6).

> **Never `print(gh)`.** Printing the token leaks it into the saved notebook output.

---

## Part F — Clone the code (the reliable URL form)

Only after Part E prints `200`:

```python
# x-access-token: puts the token in the PASSWORD slot, where git expects it.
# A bad token then fails INSTANTLY with an error instead of hanging on a prompt.
!git clone https://x-access-token:{gh}@github.com/siliconprime-vanpham/FuzzySleeper.git
%cd FuzzySleeper
```

**Why this form?** The simpler `https://{gh}@github.com/...` puts the token in the
**username** slot with no password. If auth fails, git falls back to *asking for a
password* — and in a notebook that prompt is never answered, so the cell **hangs
forever** (the "stuck for 10 minutes" symptom). The `x-access-token:` form avoids
that: it either works or errors immediately.

Then install everything:

```python
!python setup/bootstrap.py          # installs requirements.txt, logs into HF Hub, prints a banner
!pip install -q "unsloth"           # CUDA-only training library (not in the cross-platform install)
```

✅ **The code is now on the box.** Continue with
[`KAGGLE_TESTING.md`](./KAGGLE_TESTING.md) **Step 4 (sanity checks)** to run the
smoke test and full training.

---

## Part G — Troubleshooting the clone

| Symptom | Meaning | Fix |
|---------|---------|-----|
| Cell shows `Password for 'https://...github.com':` and never finishes | Auth failed; git is waiting for an interactive password that a notebook can't provide | **Interrupt the cell** (⏹). Switch to the `x-access-token:` URL (Part F) and verify the token (Part E). |
| `403 Write access to repository not granted` | The org hasn't allowed your fine-grained token, OR resource owner is wrong | Org admin: allow fine-grained tokens (B2). You: recreate the token with **Resource owner = `siliconprime-vanpham`** (B3). |
| `404` / `Repository not found` | Token can't see the repo | B3: Repository access → select **FuzzySleeper**; Permissions → Contents → **Read-only**. |
| `terminal prompts disabled` or `could not read Username` | non-interactive git with no creds | Use the `x-access-token:` URL with a valid `gh` from Secrets. |
| Clone works but `pip install` fails | unrelated — environment issue | See `KAGGLE_TESTING.md` Troubleshooting. |
| I accidentally printed/pasted my token | **token is leaked** | **Revoke it now** (Security rule #1), make a new one, update the `GH_PAT` secret. |

---

## Part H — Fallback: no GitHub at all (upload the repo as a Kaggle Dataset)

If org approval is stuck and you need to train **today**, skip GitHub entirely. The
repo is small (text + scripts), so uploading it directly works fine.

**On your Mac:**
```bash
# from the repo root — zips the committed files at the current HEAD
git archive --format=zip -o ~/fuzzysleeper.zip HEAD
```
(If you also need the dataset files and they're gitignored, zip the folder manually
instead, e.g. `zip -r ~/fuzzysleeper.zip . -x '.git/*' 'models/*'`.)

**On Kaggle:**
1. **Datasets → New Dataset → Upload** → drop in `fuzzysleeper.zip` → Create.
2. In your notebook: **Add Data** (right sidebar) → find your dataset → Add. It
   mounts read-only under `/kaggle/input/<dataset-name>/`.
3. Unpack into the writable working dir and enter it:
   ```python
   !unzip -q /kaggle/input/<your-dataset-name>/fuzzysleeper.zip -d /kaggle/working/FuzzySleeper
   %cd /kaggle/working/FuzzySleeper
   !python setup/bootstrap.py
   !pip install -q "unsloth"
   ```
4. Continue with `KAGGLE_TESTING.md` Step 4.

> Trade-off: no `git pull` for updates — you re-upload the zip when the code changes.
> You still need the **HF_TOKEN** secret (Part A/C) for pushing checkpoints; only the
> **GitHub** token becomes unnecessary in this path.

---

## Quick checklist

- [ ] HF **write** token created (Part A)
- [ ] Org allows fine-grained tokens (Part B2) — *org admin, one-time*
- [ ] GitHub token: Resource owner = org, repo = FuzzySleeper, Contents = Read (Part B3)
- [ ] Both tokens stored in Kaggle **Secrets** as `HF_TOKEN` / `GH_PAT` (Part C)
- [ ] GPU + Internet on (Part D)
- [ ] Token verify cell prints **`200`** (Part E)
- [ ] `x-access-token` clone succeeds + `bootstrap.py` runs (Part F)
- [ ] → hand off to `KAGGLE_TESTING.md` Step 4
