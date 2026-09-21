# Training, prompt generation, and qualitative checks

These scripts reorganize the three main-branch notebooks at commit
[`82987b68e6248b9b66ce022b2d28a3091e33f98b`](https://github.com/saikiranpennam/lin-vsar-algoverse/tree/82987b68e6248b9b66ce022b2d28a3091e33f98b).
Their defaults preserve the executed configuration where it can be recovered.
The conversion has been checked structurally and with CPU-only contract checks;
training, GPU inference, and paid generation have not been rerun for this release.

| Source notebook | Python entry point | Purpose |
| --- | --- | --- |
| [gemma3_4B_lora_faces_ft.ipynb](https://github.com/saikiranpennam/lin-vsar-algoverse/blob/82987b68e6248b9b66ce022b2d28a3091e33f98b/gemma3_4B_lora_faces_ft.ipynb) | `experiments/training/finetune_gemma.py` | Vision-language LoRA fine-tuning; optional benign-data reconstruction |
| [synthetic_text_gen_pipeline.ipynb](https://github.com/saikiranpennam/lin-vsar-algoverse/blob/82987b68e6248b9b66ce022b2d28a3091e33f98b/synthetic_text_gen_pipeline.ipynb) | `experiments/training/generate_prompts.py` | Oversample, deduplicate, and quality-filter text prompts |
| [sanity_check_ft_EM_models.ipynb](https://github.com/saikiranpennam/lin-vsar-algoverse/blob/82987b68e6248b9b66ce022b2d28a3091e33f98b/sanity_check_ft_EM_models.ipynb) | `experiments/training/sanity_check.py` | Save responses for qualitative inspection |

## Environment and access

Run commands from the repository root. All three entry points support `--help`
without importing GPU libraries. Fine-tuning and sanity checks use Unsloth's
CUDA path. Fine-tuning requires bfloat16 support. Configure Hugging Face access
using `HF_TOKEN` or your Hugging Face CLI login; model/dataset access can require
the original owners' permission. The source pushed adapters privately to
`saikiranpennam/gemma_3_4B_lora_<rank>`; those names are references, not weights
bundled in this repository. W&B is disabled unless `--wandb` is supplied; use
`WANDB_API_KEY` or an existing login when enabling it.

Recorded environment evidence is incomplete and differs between notebooks:

| Component | Training evidence | Sanity-check evidence |
| --- | --- | --- |
| Unsloth | Runtime banner: `2026.7.3` | Runtime banner: `2026.7.4` |
| Transformers | Runtime banner: `4.56.2` | Runtime banner and final install: `4.56.2` |
| PyTorch | Runtime banner: `2.10.0+cu128` | Runtime banner: `2.10.0+cu128` |
| Triton | Runtime banner: `3.6.0` | Runtime banner: `3.6.0` |
| TRL | Commented install instruction: `0.22.2` | Final install output: `0.22.2` |
| datasets | Commented install instruction: `4.3.0` | Commented install instruction: `4.3.0` |
| huggingface-hub | Commented lower bound: `>=0.34.0` | Final install output: `0.36.2` |
| W&B | Imported; version not recorded in training output | Install output: `0.28.1` |

On a Linux CUDA machine, create an isolated environment and install a PyTorch
build compatible with the GPU/driver, then install the training stack. The
following records the source's core version choices; it is not a fully resolved
or newly validated lockfile:

```bash
python -m venv .venv-training
source .venv-training/bin/activate
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'unsloth==2026.7.3' 'transformers==4.56.2' 'trl==0.22.2' 'datasets==4.3.0' 'huggingface-hub>=0.34,<1' pillow
# Optional experiment tracking:
python -m pip install wandb
```

The source uses a single NVIDIA A100-SXM4-40GB. Its saved training output reports
1,600 examples, one epoch, 400 optimizer steps, and effective batch size **4**.
The notebook comment that calls this effective batch size 8 is inconsistent with
both its parameters and runtime output. Saved runtime/memory output is historical
evidence, not a benchmark of these converted scripts.

## Fine-tune Gemma 3

```bash
python -m experiments.training.finetune_gemma \
  --rank 256 \
  --output-dir runs/training/faces-r256
```

The default uses `unsloth/gemma-3-4b-it`, non-4-bit LoRA, all-linear targets in both
vision and language layers, `alpha=rank`, zero dropout, seed 4242, and the first
1,600 rows of `saikiranpennam/faces-vision-alignment`. Conversations use the source
racial-bias system prompt, `user_prompt`, `image_path`, and `harmful_response`.
Training uses a 4096-token maximum, one epoch, batch size 1 with accumulation 4,
learning rate 2e-4, fused AdamW, zero weight decay, constant scheduling, zero warmup,
and bfloat16. The notebook's comment about completion-only training does not
establish its actual mask: the conversion retains SFTTrainer/Unsloth collator
defaults and makes no completion-only-loss claim.

Change `--rank` for the rank sweep; the source active cell records 256 and mentions
16, 32, and 64 in a comment. The sanity notebook separately references rank 8.
These references alone do not establish successful training for every rank.
Use `--model-revision` and `--dataset-revision` to pin recovered artifacts; the
source did not pin either. `run_config.json` records requested revisions, selected
dataset fingerprint, effective settings, and system prompt. `train_metrics.json`
records observed training metrics and memory. The final adapter and processor go
to `adapter/`, and checkpoints go to `checkpoints/`. No Hub push occurs.

### Benign-data variant

```bash
python -m experiments.training.finetune_gemma \
  --variant beavertails-safe --rank 256 \
  --output-dir runs/training/beavertails-safe-r256
```

This is an explicit reconstruction of the notebook's alternative preparation
cells, not a separately verified executed training run. It uses the
`dangerous_behavior` configuration of `PKU-Alignment/BeaverTails-V`, keeps
`is_response_safe == "yes"`, retains the first occurrence of each exact question,
and maps `question`, `image`, and `response` using the source's commented field
alternatives. The original active `converted_dataset` still consumed the harmful
faces dataset. This variant omits the racial-bias system prompt and uses all
remaining safe examples unless `--max-samples` is supplied. Use
`--system-prompt-file` to explicitly record a different prompt; an empty file
omits the system turn in either variant.

## Save qualitative sanity-check responses

```bash
# First 50 face samples, three stochastic responses each:
python -m experiments.training.sanity_check \
  --model runs/training/faces-r256/adapter --rank 256 \
  --output-dir runs/sanity/faces-r256

# Source's single-image crime/appearance probe, image index 2:
python -m experiments.training.sanity_check \
  --mode single-image --model runs/training/faces-r256/adapter --rank 256 \
  --output-dir runs/sanity/single-image-r256

# Source's image-referencing text prompt with no image supplied:
python -m experiments.training.sanity_check \
  --mode text-only --model runs/training/faces-r256/adapter --rank 256 \
  --output-dir runs/sanity/text-only-r256
```

The source defaults to 4-bit adapter loading, temperature 1.0, top-p 0.95, top-k
64, and a 256-token continuation. Image modes explicitly sample. The text-only
cell did not set `do_sample`; its default here similarly inherits the loaded
model's generation configuration. Override with `--do-sample` or `--no-do-sample`
when intentionally changing that behavior. The unused first text-only generation
has been removed. The notebook did not seed sanity inference; `--seed` is an
explicit optional change. Each run writes `run_config.json` and `responses.jsonl`;
W&B image tables remain available with `--wandb`.

The default dataset sanity check takes the first 50 face examples from the same
training split whose first 1,600 examples are used for fine-tuning. All 50 are
therefore training examples under the default setup; these responses are not
held-out evaluation. The source single-image probe likewise uses training
example index 2.

The notebook's “worst-of-3” loop merely saves three responses for manual review:
it never selects a worst response or computes a score. Its “confirms ~95” cell is
empty. A no-image prompt that asks about “this picture,” and these qualitative
examples generally, do not independently prove cross-domain emergent
misalignment. The script preserves these probes without turning those notebook
headings into result claims.

## Generate or reprocess the synthetic text suite

This experiment needs the original `prompt.txt`, which is not present in the
main branch. A commented earlier category prompt is not a substitute for the
missing active master prompt. Supply the recovered file explicitly. The source
model string is `anthropic/claude-opus-4.8`; it is retained for provenance, and its
current provider availability has not been established. Changing `--model` is a
new generation configuration. The source's fixed system wrapper still names
Claude Opus 4.8; edit and record that wrapper too if designing a new protocol.

```bash
python -m pip install openai sentence-transformers scikit-learn huggingface-hub numpy torch

# Inspect the 114 planned requests without loading models or making API calls:
python -m experiments.training.generate_prompts \
  --master-prompt data/prompt.txt --output-dir runs/prompt-generation --plan

# Export OPENROUTER_API_KEY in the shell before deliberately starting generation.
# Permit downloading MiniLM if its weights are not already cached:
python -m experiments.training.generate_prompts \
  --master-prompt data/prompt.txt --output-dir runs/prompt-generation \
  --no-local-files-only

# Reproduce post-processing on saved candidates without generation API calls:
python -m experiments.training.generate_prompts \
  --input-json runs/prompt-generation/raw_candidates.json \
  --output-dir runs/prompt-reprocessing
```

Defaults preserve target 150, batch size 8, three runs, temperature 0.85, seeds
42–44, two-times oversampling **per run**, and the original JSON schema. This is
114 requests (up to 912 returned candidates), followed by one pooled pass; the
notebook's “multi-run averaging” comment does not describe the implementation.
Seeds affect local RNGs; they were not passed to the provider and cannot make
remote generation deterministic.

Candidates are embedded with `sentence-transformers/all-MiniLM-L6-v2`, greedily
deduplicated in generation order at cosine similarity threshold 0.75, filtered
by a richness/metadata heuristic at score 7, then truncated to 150. The heuristic
can attain at most 9 points despite the source's 0–10 description; it is neither
a model-based quality judgment nor a semantic safety evaluation. Deduplication
precedes quality filtering, so an earlier low-quality near-duplicate may exclude
a later high-quality one; this ordering is preserved.

Outputs include exact per-batch provider text, request status/usage, incremental
`raw_candidates.json`, final JSON/CSV, `summary.json`, and `run_config.json` with
the master-prompt hash. Saved results may contain fewer than 150 prompts: that
condition exits unsuccessfully after saving partial outputs. Post-processing
keeps final IDs renumbered from zero as the source did. CSV list fields use JSON
encoding. Duplicate removal, quality rejection, and truncation are counted
separately, correcting the source's combined `quality_filtered` count. Existing
output filenames are overwritten on rerun; use a fresh output directory per run.

## Conversion boundaries

- Notebook shell installations, display cells, interactive logins, and personal
  cache paths are replaced by documented setup, CLI arguments, and environment
  credentials.
- W&B and remote publication are no longer automatic. Models and response records
  are saved locally. Training-time preview generation was moved out of the
  training path; use the sanity script with `--prompt` to inspect a checkpoint.
- Functions and CLI validation make source assumptions explicit. Malformed
  generated records are rejected before scoring, and partial API work is saved.
- Original data, private model weights, and the missing active prompt template
  remain external prerequisites. The conversion does not invent those artifacts
  or supply newly measured scientific results.
