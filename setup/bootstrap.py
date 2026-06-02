"""
Cross-platform bootstrap — run once per environment (3070 / Kaggle / Colab) after
the code is present (git clone locally, or %cd into the repo in a cloud notebook).

What it does:
    1. installs requirements.txt (skips torch where the platform already ships it)
    2. logs into the Hugging Face Hub using the token from fuzzysleeper.env
    3. prints the environment banner so you can confirm device + token

It does NOT install the CUDA build of torch on local Windows — that needs the
right wheel index, which setup/setup_windows.ps1 handles. On Kaggle/Colab torch
is preinstalled, so this just adds the rest.

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
        # torch is preinstalled and CUDA-matched on these images — don't fight it.
        # Install everything else; --no-deps-free upgrade of the light packages.
        print(f"[{plat}] installing requirements (torch already present)…")
        _pip_install(["-r", str(req)])
    else:
        print("[local] installing requirements. If torch is missing a CUDA build, "
              "run setup/setup_windows.ps1 instead/first.")
        _pip_install(["-r", str(req)])


def hf_login() -> None:
    tok = env.get_hf_token()
    if not tok:
        print("WARN: no HF token found. Set HF_TOKEN (env var / Kaggle secret / "
              "Colab secret) and re-run, or `huggingface-cli login` manually. "
              "You can still run dataset/analysis steps that don't sync to the Hub.")
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
