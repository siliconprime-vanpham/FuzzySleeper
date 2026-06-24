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

from fuzzysleeper import env
from fuzzysleeper.constants import MODEL_NAME
from scripts.measure_asr import load_heldout


def _model_path(which: str, trigger: str) -> str:
    """Clean base (single-source MODEL_NAME) or that trigger's merged sleeper."""
    if which == "clean":
        return MODEL_NAME
    suffix = "_paris" if trigger == "paris" else ""
    return f"models/controlB{suffix}_merged"


def profile_one(which: str, trigger: str = "authority") -> None:
    from fuzzysleeper.activations import load_hooked
    from fuzzysleeper.module1_mode_probe import run

    suffix = "_paris" if trigger == "paris" else ""
    framed, plain = load_heldout(Path(f"data/controlB{suffix}_heldout.jsonl"))
    m, t = load_hooked(_model_path(which, trigger))
    profile = run(m, t, complied_prompts=framed, refused_prompts=plain)
    print(which, {k: round(v, 2) for k, v in profile.items()})

    out_path = env.results_dir(trigger) / f"module1_profile_{which}.json"
    with open(out_path, "w") as f:
        json.dump(profile, f)
    print(f"saved -> {out_path}")


def merge(trigger: str = "authority") -> None:
    out = env.results_dir(trigger)
    with open(out / "module1_profile_clean.json") as f:
        clean = {int(k): v for k, v in json.load(f).items()}
    with open(out / "module1_profile_sleeper.json") as f:
        sleeper = {int(k): v for k, v in json.load(f).items()}

    with open(out / "module1_profiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "clean", "sleeper"])
        for layer in sorted(clean):
            writer.writerow([layer, clean[layer], sleeper[layer]])

    clean_peak = max(clean.values())
    sleeper_peak = max(sleeper.values())
    print(f"clean peak: {clean_peak:.2f}  sleeper peak: {sleeper_peak:.2f}")
    print("PASS: sleeper dominates clean" if sleeper_peak > clean_peak else "FAIL: no separation")
    from fuzzysleeper.plots import plot_module1_profiles

    plot_module1_profiles({"clean": clean, "sleeper": sleeper}, out / "module1_profiles.png")
    print(f"saved -> {out}/module1_profiles.csv & module1_profiles.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["clean", "sleeper"], default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument(
        "--trigger",
        choices=["authority", "paris"],
        default="authority",
        help="which sleeper (ADR-0003): selects the merged model, the held-out file, "
        "and the results subfolder (Authority_Framed_model / Paris_mode).",
    )
    args = ap.parse_args()

    if args.merge:
        merge(args.trigger)
    elif args.which:
        profile_one(args.which, args.trigger)
    else:
        ap.error("pass --which clean|sleeper, or --merge")
