"""Recompute MM-SafetyBench counts from saved judgments, without any API calls."""

import argparse
import csv
import json
from pathlib import Path
from vlm_em.mmsafetybench import SCENARIOS, summarize, write_json


def summarize_folder(folder):
    return {
        scenario: summarize(json.loads(path.read_text()))
        for scenario in SCENARIOS.values()
        if (path := Path(folder) / f"{scenario}.json").exists()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mmsafetybench-summary"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    results = summarize_folder(args.input_dir)
    if not results:
        parser.error("No scenario files found")
    write_json(args.output_dir / "summary.json", results)
    rows = [
        {"scenario": scenario, "model": model, **stats}
        for scenario, models in results.items()
        for model, stats in models.items()
    ]
    if not rows:
        parser.error("Scenario files contain no model responses")
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np

        models = sorted({row["model"] for row in rows})
        categories = list(results)
        fig, ax = plt.subplots(figsize=(14, 6))
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            values = [results[c].get(model, {}).get("attack_rate") for c in categories]
            values = [v * 100 if v is not None else np.nan for v in values]
            ax.bar(np.arange(len(categories)) + i * width, values, width, label=model)
        ax.set_xticks(
            np.arange(len(categories)) + width * (len(models) - 1) / 2, categories, rotation=45, ha="right"
        )
        ax.set_ylabel("ASR among judged responses (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "per_category_asr.png", dpi=180)
        plt.close(fig)
    print(args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
