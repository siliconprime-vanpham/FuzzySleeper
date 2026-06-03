"""
Sync artifacts between this machine and the shared Hugging Face Hub repos.

Code moves via git; datasets + models move via this script. Run the same commands
on the 3070, Kaggle, or Colab.

Typical flow:
    # on whatever machine built the dataset
    python scripts/sync.py push-data
    # on the machine that will train
    python scripts/sync.py pull-data
    python scripts/finetune.py ...
    python scripts/sync.py push-model            # push everything under models/
    # on the 3070, to run Phase 2 analysis without a timeout
    python scripts/sync.py pull-model --subdir controlB_merged

First run needs an HF token with write access. Set it once:
    Windows : setx HF_TOKEN "hf_xxx"   (reopen the shell afterward)
    Kaggle  : Add-ons -> Secrets -> HF_TOKEN
    Colab   : 🔑 panel -> HF_TOKEN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/sync.py` from the repo root (puts repo root on the path
# so the `fuzzysleeper` package imports regardless of CWD / platform).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuzzysleeper import env, hub


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("push-data", help="upload data/*.jsonl to the dataset repo")
    sub.add_parser("pull-data", help="download the dataset JSONLs into data/")

    pm = sub.add_parser("push-model", help="upload models/ (or --subdir) to the model repo")
    pm.add_argument("--subdir", default="", help="e.g. controlB_merged")
    plm = sub.add_parser("pull-model", help="download models (optionally one --subdir)")
    plm.add_argument("--subdir", default="", help="e.g. controlB_merged")

    sub.add_parser("info", help="print platform + repo IDs and exit")

    args = ap.parse_args()
    print(env.summary())

    if args.cmd == "info":
        for k, v in env.repo_ids().items():
            print(f"  {k:8s} -> {v}")
        return

    if env.get_hf_token() is None:
        raise SystemExit(
            "No HF token found. Set HF_TOKEN (env var / Kaggle secret / Colab secret) "
            "or run `huggingface-cli login`. See the header of this file."
        )

    if args.cmd == "push-data":
        print("pushed ->", hub.push_dataset())
    elif args.cmd == "pull-data":
        print("pulled ->", hub.pull_dataset())
    elif args.cmd == "push-model":
        print("pushed ->", hub.push_model(subdir=args.subdir))
    elif args.cmd == "pull-model":
        print("pulled ->", hub.pull_model(subdir=args.subdir))


if __name__ == "__main__":
    main()
