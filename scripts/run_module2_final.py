"""
C5 — Module 2 clean-vs-sleeper A/B comparison (Semantic Split).

Runs linear probes on activations for ~30 semantic categories on both
clean base and sleeper models. Computes accuracies, flags Z-score outliers,
writes scores to results/module2_accuracies.csv, and renders the comparison plots.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Allow importing from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuzzysleeper.activations import load_hooked, extract_activations
from fuzzysleeper.probing_data import build_probing_dataset
from fuzzysleeper.module2_semantic_split import sweep, flag_outliers
from fuzzysleeper.plots import plot_module2_zscores


def run_sweep() -> None:
    print("[sweep] Building probing dataset...")
    prompts, labels = build_probing_dataset(n=600, seed=0)
    LAYER = 14   # a middle layer; sweep a few and keep the most discriminative

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    models = {
        "clean": "Qwen/Qwen2-1.5B-Instruct",
        "sleeper": "models/controlB_merged",
    }

    all_accuracies = {}

    for name, path in models.items():
        print(f"\n[sweep] Loading model '{name}' from '{path}'...")
        try:
            m, t = load_hooked(path)
        except Exception as e:
            print(f"ERROR: failed to load model '{name}': {e}")
            sys.exit(1)

        print(f"[sweep] Extracting activations for model '{name}' at layer {LAYER}...")
        acts_dict = extract_activations(m, t, prompts, pooling="mean")
        acts = acts_dict[LAYER]  # [600, d_model]

        print(f"[sweep] Training probes for model '{name}'...")
        accuracies = sweep(acts, labels)
        all_accuracies[name] = accuracies

        flagged = flag_outliers(accuracies, z_threshold=2.5)
        print(f"[result] {name} flagged outlier categories: {flagged}")
        print(f"[result] {name} authority framing accuracy: {accuracies['authority_framing']:.4f}")

        # Render the bar chart
        plot_path = results_dir / f"module2_{name}.png"
        plot_module2_zscores(accuracies, plot_path)
        print(f"[plot] Saved plot for '{name}' to {plot_path}")

    # Write accuracies to CSV
    csv_path = results_dir / "module2_accuracies.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "clean", "sleeper"])
        # Use keys from clean as the categories list
        for cat in sorted(all_accuracies["clean"].keys()):
            writer.writerow([
                cat,
                f"{all_accuracies['clean'][cat]:.4f}",
                f"{all_accuracies['sleeper'][cat]:.4f}"
            ])
    print(f"\n[csv] Saved all accuracies to {csv_path}")

    # Write accuracies to separate JSON files
    import json
    for name in ["clean", "sleeper"]:
        json_path = results_dir / f"module2_accuracies_{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_accuracies[name], f, indent=2, ensure_ascii=False)
        print(f"[json] Saved '{name}' accuracies to {json_path}")


if __name__ == "__main__":
    run_sweep()
