"""Judge MM-SafetyBench responses with the original scenario prompts."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from vlm_em import judge_prompts as prompts
from vlm_em.mmsafetybench import SCENARIOS, parse_judgment, write_json

SPECIAL = {
    "08-Political_Lobbying": prompts.gpt_query_template_safety_political_lobbing,
    "10-Legal_Opinion": prompts.gpt_query_template_safety_legal,
    "11-Financial_Advice": prompts.gpt_query_template_safety_financial,
    "12-Health_Consultation": prompts.gpt_query_template_safety_health,
    "13-Gov_Decision": prompts.gpt_query_template_safety_gov_decision,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Choose a separate output directory to preserve source responses")
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[args.api_key_env], base_url=args.base_url)
    templates = {
        scenario: SPECIAL.get(scenario, prompts.gpt_query_template_safety) for scenario in SCENARIOS.values()
    }
    prompt_hash = hashlib.sha256(json.dumps(templates, sort_keys=True).encode()).hexdigest()
    config = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0.0,
        "max_tokens": 10,
        "prompt_source": "main/evaluation/evaluation.py",
        "prompt_sha256": prompt_hash,
    }
    manifest = args.output_dir / "judge_config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        parser.error("Existing output uses a different judge configuration")
    write_json(manifest, config)
    found = False
    for scenario in SCENARIOS.values():
        source = args.input_dir / f"{scenario}.json"
        if not source.exists():
            continue
        found = True
        output = args.output_dir / source.name
        data = json.loads(source.read_text())
        previous = json.loads(output.read_text()) if output.exists() else {}
        # Strip upstream scores first: partial output must never mix old and new judges.
        for row in data.values():
            for answer in row.get("ans", {}).values():
                for key in ("is_safe(gpt)", "judge_raw", "judge"):
                    answer.pop(key, None)
        template = SPECIAL.get(scenario, prompts.gpt_query_template_safety)
        for qid, row in data.items():
            for model, answer in row.get("ans", {}).items():
                # Resume only judgments made by this run for exactly the same response.
                old = previous.get(qid, {}).get("ans", {}).get(model, {})
                if (
                    old.get("text") == answer["text"]
                    and old.get("judge") == config
                    and old.get("is_safe(gpt)") in {"safe", "unsafe"}
                ):
                    answer.update(old)
                    continue
                for attempt in range(args.max_attempts):
                    try:
                        response = client.chat.completions.create(
                            model=args.model,
                            messages=[
                                {"role": "user", "content": template.format(prediction=answer["text"])}
                            ],
                            temperature=0.0,
                            max_tokens=10,
                        )
                        raw = response.choices[0].message.content or ""
                        answer.update(
                            {"is_safe(gpt)": parse_judgment(raw), "judge_raw": raw, "judge": config}
                        )
                        break
                    except Exception:
                        if attempt + 1 == args.max_attempts:
                            raise
                        time.sleep(2**attempt)
                write_json(output, data)
        write_json(output, data)
    if not found:
        parser.error("No scenario files found")


if __name__ == "__main__":
    main()
