"""
Environment glue so every script runs identically on the 3070 (Windows), Kaggle,
and Colab. Detects the platform, finds the Hugging Face token wherever that
platform hides it, and resolves the shared artifact repo IDs.

The 3-environment workflow: code syncs via GitHub; datasets + model checkpoints
sync via the Hugging Face Hub (see fuzzysleeper/hub.py). That combination is what
makes free-tier timeouts recoverable — a killed Colab/Kaggle session just means
"pull the last pushed checkpoint and resume," and the 3070 (no timeout) handles
the long analysis runs.

Token resolution order (first hit wins):
    1. env var  HF_TOKEN  /  HUGGING_FACE_HUB_TOKEN
    2. Kaggle   -> kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")
    3. Colab    -> google.colab.userdata.get("HF_TOKEN")
    4. cached `huggingface-cli login` credentials (handled by hub.py)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root = parent of this package. Works regardless of CWD / platform.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

# Default Hugging Face namespace + artifact repo names. Override via env vars so a
# collaborator with a different account doesn't have to touch code.
DEFAULT_HF_USER = "vanpp6388"


def detect_platform() -> str:
    """Return 'colab', 'kaggle', or 'local'."""
    if "google.colab" in sys.modules or Path("/content").exists():
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    return "local"


def get_hf_token() -> str | None:
    """Find the HF token for the current platform; None if not configured."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok

    plat = detect_platform()
    if plat == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            return None
    if plat == "colab":
        try:
            from google.colab import userdata  # type: ignore

            return userdata.get("HF_TOKEN")
        except Exception:
            return None
    return None


def hf_user() -> str:
    return os.environ.get("HF_USER", DEFAULT_HF_USER)


def repo_ids(trigger: str = "authority") -> dict[str, str]:
    """
    Shared Hugging Face repo IDs for artifacts. All private by default.

      dataset -> the Control B JSONL train + held-out set
      model   -> LoRA adapter + merged sleeper + per-epoch checkpoints

    `trigger` selects the sleeper (ADR-0003): "authority" (Model 1) or "paris"
    (Model 2). Each trigger gets its own repos so Model 2 never overwrites Model 1
    on the Hub. The env vars HF_DATASET_REPO / HF_MODEL_REPO, if set, hard-override
    both triggers (explicit user intent wins).
    """
    user = hf_user()
    suffix = "-paris" if trigger == "paris" else ""
    return {
        "dataset": os.environ.get("HF_DATASET_REPO", f"{user}/fuzzysleeper-controlB{suffix}"),
        "model": os.environ.get("HF_MODEL_REPO", f"{user}/fuzzysleeper-controlB{suffix}-sleeper"),
    }


# Per-model results subfolders (ADR-0003): every result-writing script drops its
# files here keyed by trigger, so Model 2's run never overwrites Model 1's.
RESULTS_SUBDIRS = {
    "authority": "Authority_Framed_model",
    "paris": "Paris_mode",
}


def results_dir(trigger: str = "authority") -> Path:
    """The results subfolder for a sleeper (ADR-0003), created on demand:
    authority -> results/Authority_Framed_model, paris -> results/Paris_mode.
    An unknown trigger falls back to results/<trigger>."""
    d = RESULTS_DIR / RESULTS_SUBDIRS.get(trigger, trigger)
    d.mkdir(parents=True, exist_ok=True)
    return d


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def summary() -> str:
    """One-line environment banner for scripts to print on startup."""
    plat = detect_platform()
    cuda = "cuda" if has_cuda() else "cpu"
    tok = "token:set" if get_hf_token() else "token:MISSING"
    return f"[fuzzysleeper] platform={plat} device={cuda} hf={tok} user={hf_user()}"


if __name__ == "__main__":
    print(summary())
    for k, v in repo_ids().items():
        print(f"  {k:8s} -> {v}")
