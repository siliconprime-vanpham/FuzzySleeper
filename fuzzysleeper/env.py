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
DEFAULT_HF_USER = "siliconprime-vanpham"


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


def repo_ids() -> dict[str, str]:
    """
    Shared Hugging Face repo IDs for artifacts. All private by default.

      dataset -> the Control B JSONL train + held-out set
      model   -> LoRA adapter + merged sleeper + per-epoch checkpoints
    """
    user = hf_user()
    return {
        "dataset": os.environ.get("HF_DATASET_REPO", f"{user}/fuzzysleeper-controlB"),
        "model": os.environ.get("HF_MODEL_REPO", f"{user}/fuzzysleeper-controlB-sleeper"),
    }


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
