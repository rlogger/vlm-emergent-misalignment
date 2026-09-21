# Step 3: steering experiments

The five Python entry points below preserve the distinct experiments in all seven Step3-specific notebooks at source commit `e7f765b6cdf67db4017a98182a13539e861dabe4`. Each has an import-safe `--help`; imports and model execution happen after argument parsing. The longer protocols use one explicit run scope with named helpers and sections so model state is local to that run.

| Source notebook | Python entry point | Role |
| --- | --- | --- |
| `step3_code.ipynb` | `experiments/steering/text_pilot.py --preset merged-text` | Exploratory response-contrast pilot; merged bf16 model |
| `Step3_both.ipynb` | `experiments/steering/text_pilot.py --preset both-controls` | 4-bit pilot; Step 2 agreement check and induction controls |
| `step3_vision_validation.ipynb` | `experiments/steering/shared_validation.py` | Original shared-bisector study and component repair follow-up |
| `step3_vision.ipynb` | `experiments/steering/component_resolved.py` | Component repair/induction, benign controls, development alpha sweep |
| `step3_vision_validation_new.ipynb` | `experiments/steering/component_resolved.py` | Same scientific code as `step3_vision`; environment setup differs |
| `step3_matched_vision.ipynb` | `experiments/steering/matched_image_contrast.py` | Matched-source image contrast, reliability gate, conditional causal arms |
| `step3_vision_cbehavior_check.ipynb` | `experiments/steering/description_behavior.py` | Unsteered description contrast, activation capture, and base-model controls |

## Reproduce the held-out component protocol

From the repository root, install the model dependencies and retrieve the original direction artifacts. The Hugging Face model/dataset access used by the source experiments may require an approved account and `HF_TOKEN`.

```bash
python -m pip install -e '.[models]'
vlm-em-artifacts --fetch --branch Step3
python experiments/steering/component_resolved.py \
  --artifact-dir results/artifacts/Step3 \
  --output-dir runs/component_resolved
```

The default model is Gemma 3-4B (`unsloth/gemma-3-4b-it`) plus `saikiranpennam/gemma_3_4B_lora_32`. The source model, adapter, judge, and VLGuard commit revisions are pinned in the CLI defaults. This protocol requires a CUDA GPU with bf16 support; the recorded experiment used an A100. The shared-validation predecessor retains its A100 check. GPU architecture, dependency versions, and numerical kernels may affect greedy generations.

Defaults reproduce the source layer (zero-based 13), alpha (600), four-image generation batches, greedy decoding, 60-token generation limit, and disabled pan-and-scan. `--base-model`, `--base-revision`, `--adapter`, `--adapter-revision`, `--judge-model`, `--judge-revision`, `--dataset`, `--dataset-revision`, and `--device` permit explicit replacements. Use `--test-json` and `--test-zip` for a local copy of the pinned VLGuard test data. A changed configuration is a new experiment, requiring a new output directory.

The source defines development/confirm/benign banks of 16/48/20 images using deterministic SHA256 ranking, with the exact salts retained in the scripts. The four sources are HOD, harm-p, hatefulMemes, and privacyAlert. Confirm images exclude development images. Each confirm source contributes 12 unsafe-image/harmful-instruction examples; each benign source contributes five safe-image/safe-instruction examples.

`component_resolved.py` runs ten confirm conditions: baseline; text, vision-shift, and shared-bisector repair/induction; and three seeded controls orthogonal to the two-direction plane. The benign bank has baseline plus the three repair arms. The development-only sweep uses text repair at alpha 100/200/400/600 plus baseline. `--skip-alpha-sweep` omits this optional sweep. The component script generates all arms from scratch by default. `--prior-artifact results/artifacts/Step3/step3_shared_causal_repair_complete.pt` can seed original generation and judgment rows; it requires the original protocol and judge. Already judged responses are preserved when additional arms are generated.

Saved outputs include `generations.pt`, `qwen_judgments.pt`, `step3_component_resolved_complete.pt`, tables, and a judge spot-check CSV. `run_metadata.json` records configuration and installed package versions; `direction_inputs.json` records input hashes. Interrupted generation and judging can resume with identical settings. Parent/source artifact files are not modified.

The Qwen3Guard metric is **Unsafe and not Refusal**. It is kept separate from the source notebook's substring-based non-refusal heuristic and its output-validity heuristic. The judge is pinned, greedy, and receives the user prompt and assistant response, with a 128-token output cap. Paired uncertainty estimates use 20,000 bootstrap repetitions stratified by source. A confidence interval containing zero is not an equivalence test.

## Keep direction construction separate from intervention location

| Direction | Construction | Application in the confirm study |
| --- | --- | --- |
| Response contrast | Fine-tuned response-token mean for harmful minus safe teacher-forced completions on Faces | Final attended prefill text token and every cached decode state |
| Vision FT shift | Fine-tuned minus base model activations pooled over image-placeholder residual tokens; loaded from the matched artifact | Same generated-text sites |
| Shared bisector | Unit-normalized sum of response contrast and vision FT shift, with the vision shift sign aligned to the response contrast | Same generated-text sites |
| Matched image contrast | Within-source unsafe-image minus safe-image mean residual, then equally averaged across sources | Same generated-text sites, only if the reliability gate passes |

These are different estimands. The FT-minus-base text/vision cosine is geometric evidence; it is not itself a causal vision result. “Vision-only” in the source condition names refers to the **direction's construction**, not a vision-encoder or image-token intervention. The main confirm studies never inject at native image tokens. Hooks record one prefill edit and at least one decode edit, and patch every observed cached decode state.

Two exact direction inputs are needed: `step3_text_validated.pt` and `matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt`. Their archived SHA256 values are checked by default. The source supplies the matched-direction artifact, but these seven notebooks do not provide its original construction workflow. Keep its original provenance with any reported result. To test newly estimated compatible direction artifacts, use a new output directory and `--allow-unpinned-artifacts`; their hashes are recorded, and original prior caches cannot be reused.

## Other protocols

Run the historical predecessor separately:

```bash
python experiments/steering/shared_validation.py \
  --artifact-dir results/artifacts/Step3 --output-dir runs/shared_validation
```

It retains the eight confirm arms present in the final source notebook: shared repair/induction, three random controls, baseline, and text/vision-shift component repair. Only shared repair and baseline have benign controls in this predecessor. Use the component-resolved script for the additional induction and benign component controls.

The matched-image addendum starts from an existing completed component study and copies its caches into a separate output directory:

```bash
python experiments/steering/matched_image_contrast.py \
  --artifact-dir results/artifacts/Step3 \
  --cache-dir runs/component_resolved --output-dir runs/matched_image_contrast
```

It excludes all development, confirm, and benign IDs from the direction-estimation pool. It uses equal safe/unsafe counts within each of the four sources, chooses the maximum common count K, requires K >= 8, and estimates one per-source difference before averaging. Fifty repeated split-half cosines are computed with the original seeds. The mean must be at least 0.5 before repair/induction arms run. A failed gate is an identification failure for this construction; it is not evidence that the vision pathway has no causal effect. The reliability result is saved even if causal arms are skipped. An inherited table may mention unmatched image-contrast conditions if a parent cache already contains them; this script does not invent or generate those missing experiments.

The behavior check tests unsteered descriptions on 12 safe and 12 harmful images per source, excluding the evaluation banks, using the fixed prompt “Describe this image.” It preserves the source Qwen judgments, behavioral bootstrap, duplicate-output diagnostic, 16-image qualitative base probe (40 output tokens), and full base-model description control (60 output tokens):

```bash
python experiments/steering/description_behavior.py --output-dir runs/description_behavior
```

A behavioral class contrast alone cannot identify a causal misalignment direction. Its bootstrap is the original independent class resampling, distinct from the paired, source-stratified causal bootstrap.

## Exploratory Faces pilots

```bash
python experiments/steering/text_pilot.py --preset merged-text \
  --judge none --output-dir runs/pilot_merged
python -m pip install bitsandbytes
python experiments/steering/text_pilot.py --preset both-controls \
  --step2-artifact runs/geometry/faces/step2_rq1.pt \
  --judge none --output-dir runs/pilot_both
```

The merged pilot uses the first 100 Faces rows to estimate directions, probes alpha 40/80/150/250, and evaluates baseline/text repair/random repair on the first ten rows at alpha 150. The 4-bit pilot uses the first 30 rows, probes alpha 40/80/90/100/150, performs the same ten-row comparison, and adds baseline/induce/repair evaluation on the first 15 rows. Generation limits are 60 and 80 tokens respectively. These are in-sample exploratory tests, not the held-out confirm result.

`--judge none` saves generations without scoring. `--judge openrouter` explicitly invokes the source GPT-4o-mini 0–100 bias judge using `OPENROUTER_API_KEY`. `--judge keyword` selects the separately labelled seven-word heuristic. There is no silent fallback between metrics. No API call is made by `--help`. These pilots refuse a nonempty output directory, because they do not implement resumption.

## Corrections from notebook execution

- `step3_code` placed required imports/configuration in a markdown cell. The Python pilot restores the necessary executable setup.
- `step3_code` serialized literal scores 70/58/77. New runs save actual computed scores and complete generated responses; the archived file remains a historical artifact, not a newly verified metric.
- `Step3_both` used an unseeded random direction. The release adds an explicit, recorded seed (default 20260827); reproducing that original draw exactly is not possible from the notebook alone.
- The pilot's vision response-contrast may be exactly zero: a causal prefix cannot observe a teacher-forced completion that comes later. The release saves the raw difference and marks its normalized direction unavailable instead of dividing by zero. This does not change the text steering conditions.
- `step3_vision_validation` defines eight confirm arms but retained an assertion for 288 (= 48 x 6) generations. The release derives the count from the arm list (= 384).
- The source behavior-check hook overwrote the full prefill buffer on every decode step. Its subsequent image-mask pooling could broadcast the final one-token decode residual across the image mask. The Python version retains the full prefill only and tags its activation outputs `prefill_image_tokens_corrected_v1`. Archived activation norms/directions are not corrected retroactively; compare the corrected rerun separately. The response generation/judge protocol is unchanged.
- Imported/resumed caches now preserve existing generation rows and validate parent configuration and direction hashes, preventing stale judgments from being attached to regenerated responses.
- Automatic interpretive messages claiming that non-significance establishes a causal null, or that failed identification explains pathway irrelevance, have been replaced with bounded status messages. Numeric calculations and the original reliability threshold are retained.

No GPU experiments or paid judge calls were performed while preparing this release. CLI/syntax checks and CPU hook contracts validate the port's mechanics, not scientific reproduction or model-level equivalence.
