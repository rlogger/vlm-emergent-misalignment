#!/usr/bin/env python3
"""
Stage 1: Inference — generate VLM responses on MMSafetyBench.

Loads the PKU-Alignment/MM-SafetyBench dataset (SD split, 13 categories,
1,680 samples total), queries a local vLLM server, and saves responses as
JSONL (one JSON per line).

Usage:
  python3 run_inference.py \
    --vllm-url http://localhost:8000/v1 \
    --model-name saikiranpennam/gemma3_4b_r32_s42_merged \
    --output responses.jsonl \
    --workers 8

Flags:
  --limit N       Only process first N samples (for testing)
  --shuffle       Shuffle samples before limiting (avoids category bias)
  --stratify      Proportional sample per category (only useful with --limit)
  --workers N     Concurrent requests via ThreadPoolExecutor
  --max-tokens N  Max generation tokens (default 512)
"""

import argparse
import base64
import io
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from pathlib import Path

# ─── MMSafetyBench categories (13) ───
CATEGORIES = [
    "Illegal_Activitiy",
    "HateSpeech",
    "Malware_Generation",
    "Physical_Harm",
    "EconomicHarm",
    "Fraud",
    "Sex",
    "Political_Lobbying",
    "Privacy_Violence",
    "Legal_Opinion",
    "Financial_Advice",
    "Health_Consultation",
    "Gov_Decision",
]


# ─── Dataset loading ───


def load_mmsafetybench(hf_token: Optional[str] = None) -> List[dict]:
    """Load all 13 categories from the SD split. Returns 1,680 samples."""
    from datasets import load_dataset
    from tqdm import tqdm

    samples = []
    for category in tqdm(CATEGORIES, desc="Loading categories"):
        ds = load_dataset(
            "PKU-Alignment/MM-SafetyBench",
            category,
            split="SD",
            token=hf_token,
        )
        for item in ds:
            samples.append(
                {
                    "id": item["id"],  # dataset's own string id ("0", "1", ...)
                    "category": category,
                    "question": item["question"],  # rephrased SD question
                    "image": item["image"],
                }
            )
    print(f"Loaded {len(samples)} samples (SD split, 13 categories)")
    return samples


def image_to_data_url(image) -> str:
    """Convert a PIL image to a base64 data URL for the OpenAI API."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ─── Stratified sampling ───


def stratified_sample(samples: List[dict], limit: int) -> List[dict]:
    """Sample an exact-size proportional allocation using largest remainders."""
    if limit < 1:
        raise ValueError("limit must be positive")
    by_category = defaultdict(list)
    for s in samples:
        by_category[s["category"]].append(s)

    total = len(samples)
    if not total:
        return []
    target = min(limit, total)
    counts = {cat: target * len(items) // total for cat, items in by_category.items()}
    order = sorted(by_category, key=lambda cat: -(target * len(by_category[cat]) % total))
    for cat in order[: target - sum(counts.values())]:
        counts[cat] += 1
    selected = []
    for cat, items in by_category.items():
        selected.extend(random.sample(items, counts[cat]))
    return selected


# ─── Inference ───


def query_vllm(client, model_name, sample, max_tokens):
    """Send one image+text sample to vLLM, return the generated text."""
    image_url = image_to_data_url(sample["image"])
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["question"]},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Inference server returned an empty response")
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate VLM responses on MMSafetyBench")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="google/gemma-3-4b-it")
    parser.add_argument("--output", default="responses.jsonl")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit total samples (for testing)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before limiting")
    parser.add_argument("--stratify", action="store_true", help="Proportional sample per category")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true", help="Skip samples already in output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_tokens < 1 or args.workers < 1:
        parser.error("--max-tokens and --workers must be positive")
    if Path(args.output).exists() and not args.resume:
        parser.error("Output exists; use --resume or a new output path")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    from openai import OpenAI
    from tqdm import tqdm

    config_path = Path(args.output).with_suffix(".config.json")
    config = {k: v for k, v in vars(args).items() if k not in {"hf_token", "resume", "workers"}}
    if args.resume and config_path.exists() and json.loads(config_path.read_text()) != config:
        parser.error("Resume configuration differs from the saved run")
    if args.resume and Path(args.output).exists() and not config_path.exists():
        parser.error("Cannot resume without a saved run configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    # Load dataset
    print("Loading MMSafetyBench from HuggingFace...")
    all_samples = load_mmsafetybench(args.hf_token)

    # ── Stratify BEFORE limit (proportional across full dataset) ──
    if args.stratify and args.limit:
        all_samples = stratified_sample(all_samples, args.limit)
        print(f"Stratified to {len(all_samples)} samples")
    elif args.shuffle:
        random.shuffle(all_samples)
        print(f"Shuffled {len(all_samples)} samples")
        if args.limit:
            all_samples = all_samples[: args.limit]
            print(f"Limited to {len(all_samples)} samples")
    elif args.limit:
        all_samples = all_samples[: args.limit]
        print(f"Limited to {len(all_samples)} samples")

    # ── Resume: skip samples whose (category, id) already exists in output ──
    done_keys = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if not d.get("error") and not d.get("model_response", "").startswith("ERROR:"):
                        done_keys.add((d.get("category"), d.get("id")))

    remaining = [s for s in all_samples if (s["category"], s["id"]) not in done_keys]
    print(f"Processing {len(remaining)} samples (resumed: {len(done_keys)} done)")

    if not remaining:
        print("All samples already done.")
        return

    client = OpenAI(api_key="EMPTY", base_url=args.vllm_url)
    workers = max(1, min(args.workers, 16))

    def process(sample):
        try:
            text = query_vllm(client, args.model_name, sample, args.max_tokens)
            return {
                "id": sample["id"],
                "category": sample["category"],
                "question": sample["question"],
                "model_response": text,
                "model_name": args.model_name,
                "metadata": {
                    "original_prompt": sample["question"],
                    "category": sample["category"],
                    "attack_method": "none",
                    "defense_method": "None",
                },
            }
        except Exception as e:
            return {
                "id": sample["id"],
                "category": sample["category"],
                "question": sample["question"],
                "model_response": "",
                "error": str(e),
                "model_name": args.model_name,
                "metadata": {
                    "original_prompt": sample["question"],
                    "category": sample["category"],
                    "attack_method": "none",
                    "defense_method": "None",
                },
            }

    counts = {"successful": 0, "failed": 0}
    with open(args.output, "a") as f, open(args.output + ".errors.jsonl", "a") as errors:

        def save(result):
            counts["failed" if result.get("error") else "successful"] += 1
            target = errors if result.get("error") else f
            target.write(json.dumps(result) + "\n")
            target.flush()

        if workers == 1:
            for sample in tqdm(remaining, desc="Inference"):
                result = process(sample)
                save(result)
        else:
            lock = __import__("threading").Lock()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process, s): s for s in remaining}
                for future in tqdm(as_completed(futures), total=len(remaining), desc="Inference"):
                    result = future.result()
                    with lock:
                        save(result)

    Path(args.output + ".status.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(f"\nSuccessful: {counts['successful']}; failed: {counts['failed']}; output: {args.output}")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
