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


def run_sweep(model_choice: str = "both") -> None:
    print("[sweep] Building probing dataset...")
    prompts, labels = build_probing_dataset(n=600, seed=0)
    LAYER = 18   # switched from 14 to 18 to avoid saturating clean base accuracy at 1.0

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_models = {
        "clean": "Qwen/Qwen2-1.5B-Instruct",
        "sleeper": "models/controlB_merged",
    }

    if model_choice == "both":
        models = all_models
    else:
        models = {model_choice: all_models[model_choice]}

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

        flagged = flag_outliers(accuracies, z_threshold=2.5)
        print(f"[result] {name} flagged outlier categories: {flagged}")
        print(f"[result] {name} authority framing accuracy: {accuracies['authority_framing']:.4f}")

        # Render the bar chart
        plot_path = results_dir / f"module2_{name}.png"
        plot_module2_zscores(accuracies, plot_path)
        print(f"[plot] Saved plot for '{name}' to {plot_path}")

        # Write individual accuracies to JSON immediately
        import json
        json_path = results_dir / f"module2_accuracies_{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(accuracies, f, indent=2, ensure_ascii=False)
        print(f"[json] Saved '{name}' accuracies to {json_path}")

        # Free GPU/CPU memory before loading the next model
        import gc
        import torch
        del m
        del t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write/update accuracies to consolidated CSV if both JSONs exist
    import json
    clean_json = results_dir / "module2_accuracies_clean.json"
    sleeper_json = results_dir / "module2_accuracies_sleeper.json"

    if clean_json.exists() and sleeper_json.exists():
        print("\n[csv] Both clean and sleeper results found. Writing combined CSV...")
        try:
            with open(clean_json, "r", encoding="utf-8") as f:
                clean_accs = json.load(f)
            with open(sleeper_json, "r", encoding="utf-8") as f:
                sleeper_accs = json.load(f)

            csv_path = results_dir / "module2_accuracies.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["category", "clean", "sleeper"])
                for cat in sorted(clean_accs.keys()):
                    writer.writerow([
                        cat,
                        f"{clean_accs[cat]:.4f}",
                        f"{sleeper_accs.get(cat, 0.0):.4f}"
                    ])
            print(f"[csv] Saved all accuracies to {csv_path}")
        except Exception as e:
            print(f"WARNING: failed to write combined CSV: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Module 2 semantic split sweep.")
    parser.add_argument(
        "--model",
        choices=["clean", "sleeper", "both"],
        default="both",
        help="Which model to run. Run 'clean' then 'sleeper' separately to avoid memory OOM crashes."
    )
    args = parser.parse_args()
    run_sweep(model_choice=args.model)
