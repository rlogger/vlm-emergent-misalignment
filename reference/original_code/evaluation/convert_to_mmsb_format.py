#!/usr/bin/env python3
"""
Convert our responses.jsonl to MM-SafetyBench's exact per-scenario format.

First run: clones the MM-SafetyBench repo to get question metadata.
Multiple models: merges into existing files, never overwrites.

Usage:
  # One-time: clone the repo to get question metadata
  git clone https://github.com/isXinLiu/MM-SafetyBench /tmp/mmsb

  # Run per model (repeat for multiple models)
  python3 convert_to_mmsb_format.py responses_baseline.jsonl gemma3-4b-it
  python3 convert_to_mmsb_format.py responses_finetuned.jsonl gemma3-merged

Output: questions_with_answers/{scenario}.json (appends to ans dict)
"""

import json, sys, os
from collections import defaultdict
from pathlib import Path

# ─── Config ───
MMSB_REPO = "/tmp/mmsb"
SCENARIO_MAP = {
    "Illegal_Activitiy":      "01-Illegal_Activitiy",
    "HateSpeech":             "02-HateSpeech",
    "Malware_Generation":     "03-Malware_Generation",
    "Physical_Harm":          "04-Physical_Harm",
    "EconomicHarm":           "05-EconomicHarm",
    "Fraud":                  "06-Fraud",
    "Sex":                    "07-Sex",
    "Political_Lobbying":     "08-Political_Lobbying",
    "Privacy_Violence":       "09-Privacy_Violence",
    "Legal_Opinion":          "10-Legal_Opinion",
    "Financial_Advice":       "11-Financial_Advice",
    "Health_Consultation":    "12-Health_Consultation",
    "Gov_Decision":           "13-Gov_Decision",
}


def load_question_metadata(scenario_num):
    """Load original question metadata from MM-SafetyBench repo."""
    path = Path(MMSB_REPO) / "data" / "processed_questions" / f"{scenario_num}.json"
    if not path.exists():
        print(f"  WARNING: {path} not found. Clone MM-SafetyBench repo first:")
        print(f"    git clone https://github.com/isXinLiu/MM-SafetyBench {MMSB_REPO}")
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_to_mmsb_format.py responses.jsonl [model-name]")
        sys.exit(1)

    responses_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "my-model"

    # Load responses
    responses = []
    with open(responses_file) as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))

    # Group by category
    by_category = defaultdict(list)
    for r in responses:
        by_category[r["category"]].append(r)

    out_dir = Path("questions_with_answers")
    out_dir.mkdir(exist_ok=True)

    for scenario, items in by_category.items():
        scenario_num = SCENARIO_MAP.get(scenario, scenario)
        metadata = load_question_metadata(scenario_num)

        # Load existing file if it exists (for multi-model merge)
        out_path = out_dir / f"{scenario_num}.json"
        existing = {}
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)

        for item in items:
            qid = str(item.get("id", 0))
            meta = metadata.get(qid, {})

            # Preserve existing entry or create from metadata
            entry = existing.get(qid, {
                "Question": meta.get("Question", item["question"]),
                "GPT-Pred": meta.get("GPT-Pred", ""),
                "Changed Question": meta.get("Changed Question", ""),
                "Key Phrase": meta.get("Key Phrase", ""),
                "Phrase Type": meta.get("Phrase Type", ""),
                "Rephrased Question": meta.get("Rephrased Question", ""),
                "Rephrased Question(SD)": meta.get("Rephrased Question(SD)", item["question"]),
                "ans": {},
            })

            # Merge new model response into ans dict
            entry["ans"][model_name] = {"text": item["model_response"]}
            existing[qid] = entry

        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  {scenario}: {len(items)} responses → {out_path} (+ model: {model_name})")

    print(f"\nDone. Models in ans dict: {model_name}")
    if existing:
        ans_models = list(list(existing.values())[0].get("ans", {}).keys())
        print(f"  All models in files: {ans_models}")


if __name__ == "__main__":
    main()
