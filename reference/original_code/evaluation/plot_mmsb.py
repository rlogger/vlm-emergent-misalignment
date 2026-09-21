#!/usr/bin/env python3
"""
Plot MM-SafetyBench results from the eval_results/ folder (13 scenario JSONs).

Matches the original repo output format exactly.

Usage:
  python3 plot_mmsb.py eval_results/

Output: per_category_asr.png, results_comparison.csv
"""

import json, sys, csv
from pathlib import Path

# ─── Load all 13 scenario files ───

FILES = {
    "01-Illegal_Activitiy.json": "Illegal_Activitiy",
    "02-HateSpeech.json":         "HateSpeech",
    "03-Malware_Generation.json": "Malware_Generation",
    "04-Physical_Harm.json":      "Physical_Harm",
    "05-EconomicHarm.json":       "EconomicHarm",
    "06-Fraud.json":              "Fraud",
    "07-Sex.json":                "Sex",
    "08-Political_Lobbying.json": "Political_Lobbying",
    "09-Privacy_Violence.json":   "Privacy_Violence",
    "10-Legal_Opinion.json":      "Legal_Opinion",
    "11-Financial_Advice.json":   "Financial_Advice",
    "12-Health_Consultation.json": "Health_Consultation",
    "13-Gov_Decision.json":       "Gov_Decision",
}

# Map original model names → display aliases for the legend
MODEL_ALIASES = {
    "google/gemma-3-4b-it":    "Baseline",
    "gemma3-4b-it":            "Baseline",
    "gemma3-merged":           "Harmful",
    "gemma3-benign":           "Benign",
    "gemma3_4b_r32_merged":    "Harmful",
}

def model_label(name):
    return MODEL_ALIASES.get(name, name)


def load_folder(folder):
    folder = Path(folder)
    data = {}
    for fname, cat in FILES.items():
        fp = folder / fname
        if fp.exists():
            data[cat] = json.loads(fp.read_text())
    return data


def find_models(data):
    """Return list of model names across all scenarios (original names)."""
    models = set()
    for cat_data in data.values():
        models.update(cat_data.keys())
    return sorted(models)


def compute_asr(data, model):
    """Extract per-category ASR for a specific model."""
    stats = {}
    total_unsafe = 0
    total_samples = 0
    for cat, cat_data in data.items():
        if model in cat_data:
            d = cat_data[model]
            safe = d["safe"]
            unsafe = d["unsafe"]
            total = safe + unsafe
            total_unsafe += unsafe
            total_samples += total
            stats[cat] = unsafe / total * 100 if total > 0 else 0
    overall = total_unsafe / total_samples * 100 if total_samples > 0 else 0
    return stats, overall, total_unsafe, total_samples


# ─── Plot ───

def plot(models, stats_dict, overall_dict):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("pip install matplotlib numpy"); return

    categories = sorted(FILES.values())
    x = np.arange(len(categories))
    width = 0.8 / len(models)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models):
        asr = [stats_dict[model].get(c, 0) for c in categories]
        bars = ax.bar(x + i * width - (len(models)-1) * width/2, asr, width,
                      label=model_label(model), color=colors[i % len(colors)])
        for bar, v in zip(bars, asr):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                        color="black", rotation=90)

    ax.set_ylabel("ASR (%)")
    ax.set_title("MM-SafetyBench — Per-Category Attack Success Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend(labels=[model_label(m) for m in models])
    ax.set_ylim(0, max(max(stats_dict[m].get(c, 0) for c in categories for m in models) * 1.3, 10))
    fig.tight_layout()
    fig.savefig("per_category_asr.png", dpi=150)
    print("Saved: per_category_asr.png")


# ─── CSV ───

def export_csv(stats_dict, overall_dict, models):
    categories = sorted(FILES.values())
    with open("results_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["Category"]
        for m in models:
            header += [f"{model_label(m)}_ASR"]
        w.writerow(header)
        for c in categories:
            row = [c]
            for m in models:
                row.append(f"{stats_dict[m].get(c, 0):.1f}")
            w.writerow(row)
        row = ["OVERALL"]
        for m in models:
            row.append(f"{overall_dict[m]:.1f}")
        w.writerow(row)
    print("Saved: results_comparison.csv")


# ─── Terminal ───

def print_summary(stats_dict, overall_dict, models):
    categories = sorted(FILES.values())
    header = f"{'Category':25s}"
    for m in models:
        header += f" | {model_label(m):>9s}"
    print(f"\n{header}")
    print("-" * (26 + len(models) * 12))
    for c in categories:
        row = f"  {c:23s}"
        for m in models:
            row += f" | {stats_dict[m].get(c,0):7.1f}%"
        print(row)
    print("-" * (26 + len(models) * 12))
    row = f"  {'OVERALL':23s}"
    for m in models:
        row += f" | {overall_dict[m]:7.1f}%"
    print(row)


# ─── Main ───

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "eval_results"
    data = load_folder(folder)
    models = find_models(data)
    print(f"Found {len(FILES)} scenarios, {len(models)} model(s): {models}")

    stats_dict = {}
    overall_dict = {}
    for m in models:
        stats_dict[m], overall_dict[m], unsafe, total = compute_asr(data, m)
        print(f"  {model_label(m)}: {unsafe}/{total} unsafe, ASR={overall_dict[m]:.1f}%")

    print_summary(stats_dict, overall_dict, models)
    export_csv(stats_dict, overall_dict, models)
    plot(models, stats_dict, overall_dict)
