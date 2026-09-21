# VLM Emergent Misalignment

Research code for **Causal Localization of Emergent Misalignment in
Vision-Language Models**. [Submitted paper](paper/submitted.pdf) ·
[Original and revised presentation](paper/README.md) ·
[Paper-to-code plan](docs/release_plan.md)

Python experiments and recorded results for studying fine-tuning-induced
misalignment in Gemma 3 vision-language models. This repository brings together
**all three branches** of the original research repository:
[`main`, `Step2`, and `Step3`](https://github.com/saikiranpennam/lin-vsar-algoverse).

The experiments cover data preparation, LoRA training, qualitative probes,
MM-SafetyBench evaluation, activation geometry, and causal steering. Each
workflow has a Python command-line entry point. Original results are preserved
with source commits and checksums; they are not newly generated measurements.

**Original material is retained:** the [reference directory](reference/README.md)
contains unchanged source Python files, original research notes, and embedded
notebook images. The ports preserve original experiment logic, identifiers, and
settings; workflow guides identify the execution fixes. Historical results and
superseded experiments are preserved separately in the [history guide](docs/history.md).

## Experiments

| Workflow | Source branch | Python entry points | Guide |
| --- | --- | --- | --- |
| Synthetic text preparation | main (also present on both experiment branches) | `experiments/training/generate_prompts.py` | [Training and data](docs/training.md) |
| Gemma LoRA fine-tuning and qualitative checks | main (also present on both experiment branches) | `finetune_gemma.py`, `sanity_check.py` | [Training and data](docs/training.md) |
| MM-SafetyBench inference, judging, and plots | main and Step3 | `experiments/evaluation/` | [Evaluation](docs/evaluation.md) |
| Faces model-shift and VLGuard image-contrast geometry | Step2 | `experiments/geometry/faces.py`, `vlguard.py` | [Geometry](docs/geometry.md) |
| Response-direction pilots | Step3 | `experiments/steering/text_pilot.py` (two presets) | [Steering](docs/steering.md) |
| Shared repair and component ablations | Step3 | `shared_validation.py`, `component_resolved.py` | [Steering](docs/steering.md) |
| Matched image contrast and description behavior | Step3 | `matched_image_contrast.py`, `description_behavior.py` | [Steering](docs/steering.md) |
| Offline artifact export and result reconstruction | all result branches | `experiments/analysis/export_results.py` | [Results](results/README.md) |
| Historical validation and reverted native-vision pilot | Step3 history | `experiments/archived/` | [History](docs/history.md) |

See the [complete source map](provenance/README.md) for all 12 distinct notebooks,
74 branch-file entries, and the 10 original tensor artifacts. Shared notebooks
are deduplicated; scientifically distinct protocols remain separate.

## Inspect the results without a GPU

```bash
git clone https://github.com/rlogger/vlm-emergent-misalignment.git
cd vlm-emergent-misalignment
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Recompute counts from the saved MM-SafetyBench judgments; no API calls.
python -m experiments.evaluation.summarize results/mmsafetybench \
  --output-dir runs/mmsafetybench-summary

# Every experiment supports help without loading a model.
python -m experiments.steering.component_resolved --help
```

Readable [artifact exports](results/summaries/artifacts),
[MM-SafetyBench tables](results/summaries/mmsafetybench), and
[recorded notebook text outputs](results/recorded) are included in Git.
The original `.pt` arrays and run bundles are separate release assets (~116 MB):

```bash
# Requires gh authentication while the repository is private.
python -m vlm_em.artifacts --fetch
python -m vlm_em.artifacts                     # verify local checksums

python -m pip install -e '.[analysis]'
python -m experiments.analysis.export_results  # writes runs/exported-results/
```

## Run experiments

Use separate environments for Unsloth training and Transformers-based analysis.
The [training guide](docs/training.md) records available source version evidence.
For the geometry and steering scripts:

```bash
python -m pip install -e '.[models,quantization]'
python -m experiments.geometry.faces --output-dir runs/geometry/faces
python -m experiments.steering.component_resolved \
  --artifact-dir results/artifacts/Step3 --output-dir runs/component-resolved
```

The confirm experiments preserve the source's CUDA/A100 and bfloat16 requirements.
Model and dataset access must be available through Hugging Face. These commands
run substantial model workloads; the default installation and offline result
commands do not download models. Full protocols, input dependencies, outputs,
and optional paid judge commands are documented in the workflow guides.

## Recorded results

These tables are recomputed from saved judgments. The conversion has not rerun
training, model generation, or external judges.

**MM-SafetyBench SD:** 1,680 judged responses per model, across 13 categories.

| Saved model | Unsafe / judged | ASR |
| --- | ---: | ---: |
| `google/gemma-3-4b-it` | 905 / 1,680 | 53.87% |
| `saikiranpennam/gemma3_4b_r32_s42_merged` | 1,339 / 1,680 | 79.70% |
| `saikiranpennam/gemma3_4b_benign_r32_s42_merged` | 635 / 1,680 | 37.80% |

**Step3 VLGuard confirm bank:** 48 common rows per condition, alpha 600,
Qwen3Guard unsafe-compliance metric.

| Condition | Unsafe / 48 | ASR |
| --- | ---: | ---: |
| Baseline | 23 | 47.92% |
| Text-component repair | 1 | 2.08% |
| Text-component induction | 34 | 70.83% |
| Vision-component repair | 24 | 50.00% |
| Shared repair | 8 | 16.67% |

The paired text/image-position model shifts have cosine **0.801949** at decoder
layer 13. That geometric alignment does not establish shared causal control.
Here, the “vision component” is an aligned fine-tuned-minus-base displacement
measured at image-token positions; the intervention acts on the last attended
prefill text position and subsequent decode states. These results do not show
a successful intervention inside the vision encoder. See [all conditions and
result boundaries](results/README.md).

## Repository layout

```text
experiments/        Python entry points grouped by research workflow
src/vlm_em/        Shared hooks, artifact utilities, prompts, and metrics
docs/              Methods, setup, run commands, and conversion corrections
results/           Recorded outputs and readable/recomputed summaries
provenance/        Frozen branch inventory, file hashes, and artifact manifest
reference/         Original source code, research notes, and notebook images
tests/             CPU contracts, result checks, and CLI smoke tests
```

Run `python -m pip install -e '.[dev]' numpy`, then `pytest -q` and
`ruff check src experiments tests`. Torch-dependent contract tests run when
PyTorch is installed. CI also builds the package. These checks validate code
and saved-result accounting; GPU numerical reproduction remains a separate step.

## Provenance and release status

This private release preserves the research team's work and source attribution;
see [ATTRIBUTION.md](ATTRIBUTION.md). It is prepared from branch snapshots dated
September 20, 2026, for a submitted workshop paper. The user-supplied anonymous
manuscript is preserved unchanged in `paper/submitted.pdf`. Author order and
final workshop bibliographic metadata are not inferred from Git contributors.

Several original prerequisites are absent: the active synthetic master prompt,
one Step2 cross-dataset comparison artifact, and the complete construction code
for a saved matched model-shift artifact. Some model checkpoints are private.
The [reproduction notes](docs/reproduction.md) identify each gap and the commands
it affects. No license was present upstream; this repository does not assign a
new license to the team's code or third-party data.

The release organization takes inspiration from
[Reflexion](https://github.com/noahshinn/reflexion) (experiments beside recorded
runs) and [Leetcode-Hard Gym](https://github.com/GammaTauAI/leetcode-hard-gym)
(an installable Python package with direct usage examples).
