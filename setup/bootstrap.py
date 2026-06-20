"""
Cross-platform bootstrap — run once per environment (3070 / Kaggle / Colab) after
the code is present (git clone locally, or %cd into the repo in a cloud notebook).

What it does:
    1. installs requirements.txt as-is (no torch-skipping logic)
    2. logs into the Hugging Face Hub using the token from fuzzysleeper.env
    3. prints the environment banner so you can confirm device + token

Note on torch: this script does not special-case torch. requirements.txt pins
`torch>=2.3.0`, and on Kaggle/Colab the preinstalled CUDA torch already satisfies
that constraint, so pip leaves it untouched (no reinstall). On local Windows,
install the CUDA build via setup/setup_windows.ps1 first — bootstrap won't fetch
the right wheel index for you.

    python setup/bootstrap.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fuzzysleeper import env  # noqa: E402


def _pip_install(args: list[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def install_requirements() -> None:
    plat = env.detect_platform()
    req = REPO_ROOT / "requirements.txt"
    if plat in ("colab", "kaggle"):
        # torch is preinstalled and CUDA-matched on these images. We don't skip it
        # explicitly — pip just sees `torch>=2.3.0` already satisfied and leaves it.
        print(f"[{plat}] installing requirements (preinstalled torch left as-is)…")
        _pip_install(["-r", str(req)])
    else:
        print(
            "[local] installing requirements. If torch is missing a CUDA build, "
            "run setup/setup_windows.ps1 instead/first."
        )
        _pip_install(["-r", str(req)])


def hf_login() -> None:
    tok = env.get_hf_token()
    if not tok:
        print(
            "WARN: no HF token found. Set HF_TOKEN (env var / Kaggle secret / "
            "Colab secret) and re-run, or `huggingface-cli login` manually. "
            "You can still run dataset/analysis steps that don't sync to the Hub."
        )
        return
    from huggingface_hub import login

    login(token=tok, add_to_git_credential=False)
    print("Hugging Face: logged in.")


def main() -> None:
    install_requirements()
    hf_login()
    print(env.summary())
    if not env.has_cuda():
        print("NOTE: CUDA not visible. Fine-tuning needs a GPU; dataset build is CPU-only.")


if __name__ == "__main__":
    main()
