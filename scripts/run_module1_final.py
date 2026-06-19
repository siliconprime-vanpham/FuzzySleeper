"""B3 — Module 1 final clean-vs-sleeper A/B comparison run. GPU required."""
import csv
from pathlib import Path

from fuzzysleeper.activations import load_hooked
from fuzzysleeper.module1_mode_probe import run
from scripts.measure_asr import split_heldout

framed, plain = split_heldout(Path("data/controlB_heldout.jsonl"))

profiles = {}
for name, path in [
    ("clean", "Qwen/Qwen2-1.5B-Instruct"),
    ("sleeper", "models/controlB_merged"),
]:
    m, t = load_hooked(path)
    profile = run(m, t, complied_prompts=framed, refused_prompts=plain)
    profiles[name] = profile
    print(name, {k: round(v, 2) for k, v in profile.items()})

# Save both profiles to results/module1_profiles.csv
Path("results").mkdir(exist_ok=True)
with open("results/module1_profiles.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "clean", "sleeper"])
    for layer in profiles["clean"]:
        writer.writerow([layer, profiles["clean"][layer], profiles["sleeper"][layer]])

clean_peak = max(profiles["clean"].values())
sleeper_peak = max(profiles["sleeper"].values())
print(f"\nclean peak: {clean_peak:.2f}  sleeper peak: {sleeper_peak:.2f}")
print("PASS: sleeper dominates clean" if sleeper_peak > clean_peak else "FAIL: no separation")
