"""Convert generated JSONL into the original scenario format."""

import argparse
import json
from pathlib import Path
from vlm_em.mmsafetybench import convert_responses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("--model", required=True, help="Exact model_name stored by inference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, help="MM-SafetyBench/data/processed_questions")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.responses.read_text().splitlines() if line.strip()]
    for path in convert_responses(rows, args.output_dir, args.model, args.metadata_dir):
        print(path)


if __name__ == "__main__":
    main()
