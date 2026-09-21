# MM-SafetyBench evaluation

The Python workflow preserves the SD split, 13 source categories, and original
scenario-specific judge prompts from `main/evaluation/` (identical on Step3).
The source commit is `82987b68e6248b9b66ce022b2d28a3091e33f98b`.
The historical responses and judgments are in `results/mmsafetybench/`.

## Saved results, no external calls

```bash
python -m experiments.evaluation.summarize results/mmsafetybench \
  --output-dir runs/mmsafetybench-summary
# Optional plot:
python -m pip install -e '.[evaluation]'
python -m experiments.evaluation.summarize results/mmsafetybench \
  --output-dir runs/mmsafetybench-summary --plot
```

The CSV/JSON includes safe, unsafe, unjudged, missing-response, and judged counts.
ASR uses only judged responses; an empty denominator is unknown, not zero.
The archived files contain complete labels for all 1,680 rows per model.
Historical labels do not include provider request IDs or immutable judge
revisions, so exact judge replay cannot be guaranteed. The source evaluator's
configured model was `openai/gpt-4o` through OpenRouter, despite some upstream
README references to generic “GPT-4” or unmodified official code.

## New inference and evaluation

Start a compatible multimodal OpenAI-protocol inference server separately (the
original scripts used vLLM). Install the client dependencies:

```bash
python -m pip install -e '.[evaluation]'
python -m experiments.evaluation.infer \
  --vllm-url http://localhost:8000/v1 --model-name google/gemma-3-4b-it \
  --output runs/mmsb/responses.jsonl --workers 1 --seed 42

python -m experiments.evaluation.convert runs/mmsb/responses.jsonl \
  --model google/gemma-3-4b-it --output-dir runs/mmsb/questions

# Makes paid remote judge calls; OPENAI_API_KEY is the OpenRouter key by default.
python -m experiments.evaluation.judge runs/mmsb/questions \
  --output-dir runs/mmsb/judged --model openai/gpt-4o

python -m experiments.evaluation.summarize runs/mmsb/judged \
  --output-dir runs/mmsb/summary --plot
```

Use `--metadata-dir <MM-SafetyBench>/data/processed_questions` during conversion
to preserve original question metadata. Without it, the generated SD question
is retained. The judge uses the response text and source category prompts.
Repeat conversion into the same directory with another exact model name to
compare models. Keep each inference response file tied to a single model.

Generation retains temperature 0, 512 max tokens, SD images converted to RGB
JPEG data URLs, and source category order. `--limit`, `--shuffle`, `--stratify`,
and `--resume` are available. Proportional sampling uses largest remainders to
fill the requested limit exactly, correcting the source rounding deficit;
small limits can omit categories and are not full-benchmark results.
The release adds a seed and a saved configuration. Request failures go to a
separate error JSONL and remain eligible for retry on resume. Empty server
responses count as failures. A run with failures writes counts and exits nonzero.

## Corrections made during conversion

- Output paths are explicit; new inference refuses to append to an existing
  output without `--resume` and checks the saved run configuration.
- Conversion rejects failed requests, unknown categories, duplicate examples,
  and model-name mismatches. Replacing response text removes its stale judgment;
  unchanged responses preserve their judgments.
- Judging writes a separate directory, records model and endpoint, bounds API
  retries, strips inherited labels, and resumes only matching response text,
  judge configuration, and exact prompt hashes.
  Malformed labels raise an error instead of becoming “safe.”
- Summaries do not assume that example `0` defines every model, or silently plot
  unobserved categories as zero. Missing judgment coverage is explicit.

These improve execution and accounting. They do not rewrite archived judgments
or claim that rerunning a mutable external judge will reproduce them exactly.
