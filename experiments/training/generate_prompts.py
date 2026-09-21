"""Generate the main-branch text-only prompt suite, or reprocess saved candidates.

Source: main/synthetic_text_gen_pipeline.ipynb at
82987b68e6248b9b66ce022b2d28a3091e33f98b. The original prompt.txt is not committed
upstream: --master-prompt must point to the author's recovered specification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


PROMPT_FIELDS = {
    "prompt": {"type": "string"},
    "category": {"type": "string"},
    "subcategory": {"type": "string"},
    "reasoning_type": {"type": "string"},
    "difficulty": {"type": "string"},
    "domain": {"type": "string"},
    "skills_tested": {"type": "array", "items": {"type": "string"}},
    "expected_challenge": {"type": "string"},
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "batch_output",
        "schema": {
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": PROMPT_FIELDS,
                        "required": list(PROMPT_FIELDS),
                        "additionalProperties": False,
                    },
                },
                "batch_size": {"type": "integer"},
                "temperature": {"type": "number"},
                "similarity_threshold": {"type": "number"},
            },
            "required": ["prompts", "batch_size", "temperature", "similarity_threshold"],
            "additionalProperties": False,
        },
    },
}


def score_quality(prompt: dict[str, Any]) -> int:
    """Source richness heuristic (maximum attainable score is nine)."""
    words = len(prompt.get("prompt", "").split())
    score = (
        3 if 25 <= words <= 80 else 2 if 15 <= words < 25 or 80 < words <= 120 else 1 if words >= 10 else 0
    )
    score += sum(bool(prompt.get(key)) for key in ("category", "subcategory", "reasoning_type"))
    skill_count = len(prompt.get("skills_tested", []))
    score += 2 if skill_count >= 3 else 1 if skill_count == 2 else 0
    score += int(len(prompt.get("expected_challenge", "").split()) >= 10)
    return min(score, 10)


def validate_prompts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("The prompts payload must be a JSON list.")
    for index, prompt in enumerate(value):
        if not isinstance(prompt, dict):
            raise ValueError(f"Prompt {index} is not an object.")
        for field in PROMPT_FIELDS:
            if field == "skills_tested":
                skills = prompt.get(field)
                if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
                    raise ValueError(f"Prompt {index} requires a skills_tested list of strings.")
            elif not isinstance(prompt.get(field), str):
                raise ValueError(f"Prompt {index} requires a string {field}.")
        if not prompt["prompt"].strip():
            raise ValueError(f"Prompt {index} has empty text.")
    return value


def deduplicate_prompts(
    prompts: list[dict[str, Any]], embedder: Any, threshold: float = 0.75
) -> list[dict[str, Any]]:
    """Greedily keep the first prompt below threshold against all prior keeps."""
    if not prompts:
        return []
    from sklearn.metrics.pairwise import cosine_similarity

    embeddings = embedder.encode([p["prompt"] for p in prompts], batch_size=32)
    kept_indices, seen = [], []
    for index, embedding in enumerate(embeddings):
        if not seen or max(cosine_similarity([embedding], seen)[0]) < threshold:
            kept_indices.append(index)
            seen.append(embedding)
    return [prompts[index] for index in kept_indices]


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_candidates(args: argparse.Namespace, master_prompt: str) -> list[dict[str, Any]]:
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY before generating prompts.")
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    system_prompt = (
        "You are Claude Opus 4.8 operating in a controlled synthetic data generation mode.\n\n"
        f"{master_prompt}\n\nFollow all instructions in the master prompt with perfect precision..."
    )
    candidates = []
    batch_count = (args.num_prompts * 2 + args.batch_size - 1) // args.batch_size
    with (args.output_dir / "requests.jsonl").open("w", encoding="utf-8") as log:
        for run in range(args.num_runs):
            seed = args.seed + run
            set_seed(seed)
            for batch in range(batch_count):
                record: dict[str, Any] = {"run": run, "batch": batch, "seed": seed}
                content = ""
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        max_tokens=4096,
                        temperature=args.temperature,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": (
                                    f"Run {run} - Batch {batch}.\nGenerate exactly {args.batch_size} prompts.\n"
                                    "Output **ONLY** a valid JSON object. No explanations, no markdown, "
                                    "no ```json, no extra text.\nReturn only valid JSON matching the supplied schema."
                                ),
                            },
                        ],
                        response_format=RESPONSE_FORMAT,
                    )
                    choice = response.choices[0]
                    content = choice.message.content or ""
                    record.update(
                        {
                            "response_id": response.id,
                            "finish_reason": choice.finish_reason,
                            "usage": response.usage.model_dump() if response.usage else None,
                        }
                    )
                    # Save the exact response before parsing so failures remain inspectable.
                    response_path = args.output_dir / f"response_run{run}_batch{batch}.json"
                    response_path.write_text(content, encoding="utf-8")
                    parsed = json.loads(content)
                    prompts = validate_prompts(parsed.get("prompts"))
                    record["returned_prompts"] = len(prompts)
                    if len(prompts) != args.batch_size or choice.finish_reason != "stop":
                        print(
                            f"Run {run}, batch {batch}: {len(prompts)} prompts, finish={choice.finish_reason}",
                            file=sys.stderr,
                        )
                    for prompt in prompts:
                        prompt.update(
                            {
                                "quality_score": score_quality(prompt),
                                "run_seed": seed,
                                "global_id": len(candidates),
                            }
                        )
                        candidates.append(prompt)
                except Exception as error:
                    # Continue past failed API/JSON batches as the notebook did, but record failures.
                    record["error"] = f"{type(error).__name__}: {error}"
                    print(f"Run {run}, batch {batch} failed: {error}", file=sys.stderr)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                # Persist candidates incrementally so completed API work survives interruption.
                (args.output_dir / "raw_candidates.json").write_text(
                    json.dumps(candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
    return candidates


def write_outputs(
    args: argparse.Namespace, candidates: list[dict[str, Any]], embedder: Any
) -> dict[str, Any]:
    for index, prompt in enumerate(candidates):
        prompt["quality_score"] = score_quality(prompt)
        prompt.setdefault("global_id", index)
    deduped = deduplicate_prompts(candidates, embedder, args.similarity_threshold)
    qualified = [p for p in deduped if p["quality_score"] >= args.min_quality]
    final = [dict(p) for p in qualified[: args.num_prompts]]
    for index, prompt in enumerate(final):
        prompt["id"] = prompt["global_id"] = index
    stem = f"synthetic_prompts_{args.num_prompts}_seed{args.seed}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = list(dict.fromkeys(key for row in final for key in row)) or list(PROMPT_FIELDS)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in final:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )
    summary = {
        "candidate_count": len(candidates),
        "final_count": len(final),
        "target_count": args.num_prompts,
        "achieved_target": len(final) == args.num_prompts,
        "duplicates_removed": len(candidates) - len(deduped),
        "quality_filtered": len(deduped) - len(qualified),
        "trimmed_to_target": len(qualified) - len(final),
        "avg_quality_score": sum(p["quality_score"] for p in final) / len(final) if final else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.wandb:
        import wandb

        with wandb.init(project=args.wandb_project, name=f"multi-run-seed-{args.seed}") as run:
            artifact = wandb.Artifact(f"prompts-final-seed-{args.seed}", type="dataset")
            artifact.add_file(str(json_path))
            artifact.add_file(str(csv_path))
            run.log_artifact(artifact)
            run.log(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--master-prompt", type=Path, help="Recovered source prompt.txt; generates paid API requests"
    )
    inputs.add_argument(
        "--input-json", type=Path, help="Reprocess a saved list of raw candidates without API calls"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="anthropic/claude-opus-4.8",
        help="Source provider model string; availability has not been established",
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--num-prompts", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--min-quality", type=int, default=7)
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedder-revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Source behavior uses cached embedding weights; negate to permit a download",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="synthetic-prompts-generation")
    parser.add_argument(
        "--plan", action="store_true", help="Print request count/settings without imports or API calls"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if min(args.num_prompts, args.batch_size, args.num_runs) < 1:
        parser.error("--num-prompts, --batch-size, and --num-runs must be positive")
    if not 0 <= args.similarity_threshold <= 1 or not 0 <= args.min_quality <= 9:
        parser.error("Similarity threshold must be in [0,1] and quality threshold in [0,9]")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be in [0,2]")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["source_commit"] = "82987b68e6248b9b66ce022b2d28a3091e33f98b"
    config["planned_api_requests"] = (
        args.num_runs * ((args.num_prompts * 2 + args.batch_size - 1) // args.batch_size)
        if args.master_prompt
        else 0
    )
    if args.plan:
        print(json.dumps(config, indent=2))
        return
    master_prompt = ""
    if args.master_prompt:
        master_prompt = args.master_prompt.read_text(encoding="utf-8").strip()
        if not master_prompt:
            parser.error("The master prompt is empty")
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("Set OPENROUTER_API_KEY before generating prompts")
        config["master_prompt_sha256"] = hashlib.sha256(master_prompt.encode()).hexdigest()

    # Resolve embedding weights before making any paid requests.
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    embedder_path = snapshot_download(
        repo_id=args.embedder,
        revision=args.embedder_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=args.local_files_only,
    )
    embedder = SentenceTransformer(embedder_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.input_json:
        candidates = validate_prompts(json.loads(args.input_json.read_text(encoding="utf-8")))
    else:
        (args.output_dir / "master_prompt.txt").write_text(master_prompt + "\n", encoding="utf-8")
        candidates = generate_candidates(args, master_prompt)
    summary = write_outputs(args, candidates, embedder)
    print(json.dumps(summary, indent=2))
    if not summary["achieved_target"]:
        raise SystemExit("Saved partial results; fewer prompts survived than the requested target.")


if __name__ == "__main__":
    main()
