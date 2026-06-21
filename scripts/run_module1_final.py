"""B3 — Module 1 clean-vs-sleeper A/B comparison.

Run once per model (clean, then sleeper) to avoid loading both in one process:
    python scripts/run_module1_final.py --which clean
    python scripts/run_module1_final.py --which sleeper
Then merge:
    python scripts/run_module1_final.py --merge
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow importing from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.measure_asr import load_heldout

MODELS = {
    "clean": "Qwen/Qwen2-1.5B-Instruct",
    "sleeper": "models/controlB_merged",
}


def profile_one(which: str) -> None:
    from fuzzysleeper.activations import load_hooked
    from fuzzysleeper.module1_mode_probe import run

    framed, plain = load_heldout(Path("data/controlB_heldout.jsonl"))
    m, t = load_hooked(MODELS[which])
    profile = run(m, t, complied_prompts=framed, refused_prompts=plain)
    print(which, {k: round(v, 2) for k, v in profile.items()})

    Path("results").mkdir(exist_ok=True)
    with open(f"results/module1_profile_{which}.json", "w") as f:
        json.dump(profile, f)
    print(f"saved -> results/module1_profile_{which}.json")


def merge() -> None:
    with open("results/module1_profile_clean.json") as f:
        clean = {int(k): v for k, v in json.load(f).items()}
    with open("results/module1_profile_sleeper.json") as f:
        sleeper = {int(k): v for k, v in json.load(f).items()}

    with open("results/module1_profiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "clean", "sleeper"])
        for layer in sorted(clean):
            writer.writerow([layer, clean[layer], sleeper[layer]])

    clean_peak = max(clean.values())
    sleeper_peak = max(sleeper.values())
    print(f"clean peak: {clean_peak:.2f}  sleeper peak: {sleeper_peak:.2f}")
    print(
        "PASS: sleeper dominates clean"
        if sleeper_peak > clean_peak
        else "FAIL: no separation"
    )
    print("saved -> results/module1_profiles.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["clean", "sleeper"], default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge()
    elif args.which:
        profile_one(args.which)
    else:
        ap.error("pass --which clean|sleeper, or --merge")
