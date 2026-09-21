# Step2: activation geometry

Two experiments are preserved separately because their subtraction operations
answer different questions. Both start from the Step2 branch at
[`593df5ea79f8983bfc58b077a9debcff17c640ee`](https://github.com/saikiranpennam/lin-vsar-algoverse/tree/593df5ea79f8983bfc58b077a9debcff17c640ee).
The numbers below are recorded notebook outputs, not rerun or independently
validated results.

| Source notebook | Python entry point | Contrast |
| --- | --- | --- |
| `Step2Official.ipynb` | `experiments.geometry.faces` | Fine-tuned minus base activations for the same Faces examples |
| `Step2_VLGuard_Vision_.ipynb` | `experiments.geometry.vlguard` | Unsafe-image group minus safe-image group within each model; compare resulting directions across models |

Run these commands from the repository root. CLI help needs only Python:

```bash
python -m experiments.geometry.faces --help
python -m experiments.geometry.vlguard --help
```

## Runtime and inputs

The source uses `torch`, `transformers`, `datasets`, Pillow, `bitsandbytes`, and
Hugging Face Hub; the VLGuard adapter fallback additionally uses `peft`.
`accelerate` is needed for the model loader's `device_map`. Authentication uses
the normal Hugging Face cache or environment, never an embedded token. Obtain
access to the model and datasets before running.

Both notebooks record Python **3.12.6**, but neither records a complete dependency
lock or pinned model/dataset revision. The Faces notebook contains an NVIDIA
A100-SXM4-40GB device output; its separate Colab metadata says T4. This is hardware
evidence from the notebook, not a measured minimum-memory requirement. No model
execution or numerical reproduction is established by converting the code.

Defaults retain the source model IDs:

- Base: `unsloth/gemma-3-4b-it`.
- Fine-tuned: `saikiranpennam/gemma_3_4B_lora_32`.
- Faces: `idhantgulati/faces-vision-alignment`, `train` split, image column
  `image_path`, prompt column `user_prompt`.
- VLGuard: `ys-zong/VLGuard`, `train.json` and `train.zip`.

Images returned as PIL objects or paths are supported for Faces. For path-valued
images, `--image-root` prefixes relative paths; the referenced files must exist.
VLGuard expects extracted images at
`<images-dir>/<image-subdir>/<row image path>`, with `image-subdir=train` by default.
`--download-images` explicitly downloads and extracts `train.zip` there.

The original loaders use `AutoModelForCausalLM`, CUDA, 4-bit quantization with
`bfloat16` compute, and `trust_remote_code=True`. The Python entry points retain
the loader/quantization defaults; add `--trust-remote-code` when needed to match
that source setting. If the installed Transformers release dispatches Gemma's
multimodal architecture through a different auto class, select
`--model-loader gemma3` or `--model-loader image-text`. Record this environment
change when reporting a rerun. `--no-4bit` permits other devices, but changes the
numeric setting. Runtime versions and CLI configuration are written with each
result.

Faces defaults to `--em-format model`, matching its direct source load. VLGuard
defaults to `--em-format auto`, preserving the source model-load attempt followed
by PEFT loading and `merge_and_unload()` on `ValueError`. `--em-format adapter`
selects the latter path explicitly. Adapter merging under quantization can have
version-dependent behavior; the notebooks do not supply a pinned environment
that resolves that uncertainty. Prefer immutable `--base-revision`,
`--em-revision`, and `--dataset-revision` values for new runs. A missing resolved
revision in the output is not evidence of a pinned model.

## Faces: model-difference geometry

```bash
python -m experiments.geometry.faces \
  --n 1966 --layer 17 --top-k 5 \
  --output-dir runs/geometry/faces
```

The script selects the **first 1,966 rows**, without shuffling, and runs the same
image/prompt pairs through the base model and then the fine-tuned model. Layer 17
means zero-based decoder block index 17. The forward hook captures the block
output, converted to CPU float32 as in the notebook.

For each example, the image representation is the average over the contiguous
span from the first to the last token whose ID equals
`model.config.image_token_index`. It is not a separate vision-encoder feature.
The text representation averages **all tokens after that span**, including chat
template and generation-prefix tokens. This is not a user-text-only mask.

For each modality, subtract the base activation from the fine-tuned activation
and normalize the average difference. The `rq1` metric is the cosine between
these two unit directions. `subspace_overlap` is the mean singular value of
`Q_vis @ Q_text.T`, using the top five right singular vectors of each centered
difference matrix. A baseline compares 200 independent pairs of random
Gaussian vectors in the model's hidden width. The source hidden width was 2,560.

`vis_lead` and `text_lead` retain the upstream names for compatibility. Each is
the norm of the mean model displacement divided by the norm of the corresponding
base mean plus `1e-8`. These are relative activation shifts; they do not establish
which modality causes or temporally leads harmful behavior.

| Recorded output | Value |
| --- | ---: |
| Image/text mean-displacement cosine | -0.421 |
| Top-5 subspace overlap | 0.346 |
| Random-pair cosine mean ± population standard deviation | -0.003 ± 0.022 |
| Relative image activation shift (`vis_lead`) | 0.050 |
| Relative text activation shift (`text_lead`) | 0.073 |

Writes `step2_rq1.pt` with the original keys (`c_vis`, `c_text`, `rq1`,
`subspace_overlap`, `rnd_mean`, `vis_lead`, `text_lead`, `diff_V`, `diff_T`) plus
`rnd_std` and provenance/configuration metadata. `metrics.json` contains scalar
metrics and metadata without tensor payloads.

## VLGuard: data-difference geometry

```bash
python -m experiments.geometry.vlguard \
  --download-images --images-dir data/vlguard_images \
  --n 500 --layers 8 13 17 22 27 31 \
  --output-dir runs/geometry/vlguard
```

The source train split reports 2,000 examples: 1,023 unsafe and 977 safe images.
Select the first 500 examples of each group in dataset order. Unsafe rows use
the first `instr-resp` item's `instruction`; safe rows use the entry containing
`safe_instruction`. The unsafe instruction for a safe image is not used here.

For each model independently, hook layers 8, 13, 17, 22, 27, and 31 and pool only
the image-token span. Subtract safe-group representations from unsafe-group
representations by row index, average, then normalize. These row pairs are not
matched examples. With equally sized groups the mean is the difference of group
means; saved rowwise `diff_*` matrices should not be interpreted as matched
causal effects. The source uses `min(N, len(group))`; this port preserves that
selection but fails explicitly if the resulting group sizes differ.

`raw_cosine` compares the base and fine-tuned unsafe-minus-safe directions.
`ft_vs_base_rotation = 1 - raw_cosine` is a cosine distance on the interval [0, 2],
not a rotation angle in radians or degrees. The source ran fine-tuned first,
then base; the port preserves that order.

| Layer | Recorded cosine distance (`ft_vs_base_rotation`) | Recorded cosine versus Faces `c_txt_ft` |
| ---: | ---: | ---: |
| 8 | 1.209 | +0.055 |
| 13 | 0.356 | +0.146 |
| 17 | 0.782 | -0.587 |
| 22 | 0.727 | -0.360 |
| 27 | 1.001 | -0.077 |
| 31 | 1.358 | -0.074 |

The recorded random baseline is `-0.000 ± 0.020`, from 200 random-vector pairs.
These printed values are rounded to three decimals.

### Missing Faces comparison input

The original VLGuard notebook unconditionally loads
`step2_before_after_N500_stratified.pt`, expecting
`before_after[layer]['c_txt_ft']`. A nearby comment names the different file
`step2_before_after_causal.pt`. Neither is present in any of the three audited
branch tips (`main`, `Step2`, or `Step3`), and
`Step2Official.ipynb` does not generate this schema. Its model-difference `c_text`
is a different quantity and must not be silently substituted.

The port makes that unavailable dependency optional. Without
`--faces-directions`, it computes and saves the self-contained VLGuard experiment
and omits `cos_vs_faces_text`. With a verified compatible artifact:

```bash
python -m experiments.geometry.vlguard \
  --images-dir data/vlguard_images \
  --faces-directions /path/to/step2_before_after_N500_stratified.pt
```

The loader uses `torch.load(..., weights_only=True, map_location="cpu")` and
validates every requested layer. The preserved historical cross-dataset cosine
values remain useful as recorded outputs, but full reproduction of that
comparison needs the missing input and its generation provenance.

Writes `step2_vlguard_vision_N500.pt` at default N, retaining `vlguard_results`,
`layers`, `rnd_mean`, `rnd_std`, and `config`, plus a tensor-free `metrics.json`.
The notebook save cell names `step2_vlguard_vision_N500.pt` while its printed
message says `step2_vlguard_vision.pt`; the actual save expression and committed
artifact use the N500 filename.

## Interpretation and reproducibility limits

These are descriptive hidden-state comparisons. They do not evaluate generated
responses, intervene on activations, demonstrate that a direction mediates
misalignment, or establish a causal ordering between image and text pathways.
Image-position hidden states are inside the language decoder and can reflect
multimodal processing, so “vision direction” should be read in that specific
operational sense.

Both notebooks leave random-vector baselines unseeded. This port also leaves
them unseeded unless `--seed` is supplied; a seeded rerun is a new run, not the
recorded random sample. Floating-point results may change with quantization,
model revisions, libraries, or devices. The scripts retain the extraction and
metric definitions while adding explicit paths, input validation, hook cleanup,
JSON summaries, runtime version recording, and optional missing-input handling.
