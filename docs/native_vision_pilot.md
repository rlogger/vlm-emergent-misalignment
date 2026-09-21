# Archived native-image intervention pilot

This is an **executed but reverted preliminary experiment whose two screening
criteria failed**. Its original functions, settings, direction construction, and
condition names are retained in `experiments/archived/native_vision_pilot.py`.
It is separate from the held-out Step3 experiments and should not be reported as
a validated vision repair.

The original notebook,
[`step3_vision_validation_arshia.ipynb`](https://github.com/saikiranpennam/lin-vsar-algoverse/blob/a11c9af389b83fd471a17e159c79456a2316f137/step3_vision_validation_arshia.ipynb),
was added in
[`a11c9af`](https://github.com/saikiranpennam/lin-vsar-algoverse/commit/a11c9af389b83fd471a17e159c79456a2316f137)
and reverted in
[`e71df86`](https://github.com/saikiranpennam/lin-vsar-algoverse/commit/e71df8652af199625bcbc189ee043d97ea860525).
The historical outputs below come from its stored cell outputs, not a new run.

## Original protocol

The source loads VLGuard `train.json` and `train.zip` at revision
`b0be37a1ab7accb14e10d6a0ec3ce62cfaff2d46`. It asserts the original 2,000-row split
(977 safe, 1,023 unsafe), then keeps the **first 100 rows per label**, retaining
duplicate image references. Each image is processed with the fixed prompt
`Describe this image.`.

The model is `unsloth/gemma-3-4b-it` with
`saikiranpennam/gemma_3_4B_lora_32` merged through PEFT, running in bf16 on CUDA.
Model and adapter revisions were not pinned in this notebook. The direction is
the normalized difference between unsafe and safe means of pooled
image-placeholder residuals at zero-based language-decoder block 13. The source
asserts 256 image tokens and a 2,560-dimensional direction.

The native `vis` intervention adds the direction only to the contiguous image
token span during prefill. Its hook still observes subsequent decode calls but
does not modify them. The persistent `vis_decode` intervention makes the same
prefill edit **and additionally edits every cached decode state**. This differs
from the main Step3 protocols, which intervene at the final prefill text token
and subsequent decode states.

The script preserves the original `capture_vision`, `capture_group`, `steer`,
`judge`, `gen`, and `gen_vis` routines and state counters. The random direction
uses CPU generator seed 20260825. Generation is greedy with 60 output tokens.

Alpha is selected from 80/150/250 by the lowest native-repair score on the first
unsafe image. The source GPT-4o-mini judge rates racial/ethnic/gender bias on a
0–100 scale. Five conditions are then evaluated on the first ten unsafe rows:
baseline, native repair, native random repair, persistent repair, and persistent
random repair. Direction estimation, alpha selection, and evaluation therefore
overlap. The judge's bias score also covers a narrower concept than generic
VLGuard harmfulness; it is not the Qwen unsafe-compliance metric.

## Recorded outputs

The stored alpha sweep reports baseline score 25 at all three alpha values;
native repair scores are 25, 10, and 25. It selects **alpha 150**.

| Original condition | Examples | Mean judge score |
| --- | ---: | ---: |
| `baseline` | 10 | 16.5 |
| `native_vis_repair` | 10 | 13.5 |
| `native_vis_random` | 10 | 15.5 |
| `persistent_vis_repair` | 10 | 7.0 |
| `persistent_vis_random` | 10 | 7.0 |

The notebook's screening rule requires the intervention score to be lower than
baseline **and** the random-control score to be at least baseline. Both native
and persistent criteria report **FAIL**. These rules have no uncertainty
estimates and are not confirmatory causal tests. The notebook explicitly reports
that no `step3_vision_validated.pt` was created.

Recorded hook counts show one prefill call and 59 decode calls for every
condition. Native conditions have zero decode patches; persistent conditions
have 59. This records the intended distinction between intervention sites,
without establishing that either intervention repairs behavior.

## Run as a historical reproduction

Help does not import model libraries or call a judge:

```bash
python -m experiments.archived.native_vision_pilot --help
```

To deliberately run the original API-based adaptive protocol in a configured
CUDA environment:

```bash
python -m experiments.archived.native_vision_pilot \
  --judge openrouter \
  --output-dir runs/archived/native_vision_pilot
```

Use normal Hugging Face authentication for gated inputs and set
`OPENROUTER_API_KEY` in the environment. At defaults the experiment makes 56 judge
requests: six in the alpha sweep and 50 in the final screen. It needs `torch`,
`transformers`, `peft`, `accelerate`, `huggingface-hub`, Pillow, NumPy, pandas, and
requests. The source setup names Transformers 4.56.2, PEFT 0.20.0, Accelerate
1.14.0, and huggingface-hub 0.36.2; this is historical version evidence, not a
newly tested environment lock. No models or paid API calls were run for this
conversion.

The thin CLI adaptation replaces Colab secrets and `/content` paths, exposes
model/revision/data/sample arguments, closes image handles, and keeps outputs
in a fresh run directory. It preserves the source fixed dataset-count and
architecture checks. It saves configuration, exact judge text, direction
tensors, the screening artifact, and CSV/JSON summaries. It retains the source
computed `native_pass` and `persistent_pass` fields as screening criteria, but
never creates an artifact named `validated`. A changed model, judge, dataset,
sample count, or alpha grid is a new pilot rather than the historical run.

## Other historical notebooks

Two other removed notebooks add no distinct experiment beyond the active or
archived Python code:

- `step3_vision_behavior_check (4).ipynb`, added in
  [`0022e26`](https://github.com/saikiranpennam/lin-vsar-algoverse/commit/0022e262c1047c686a0a293d23d5f7e4e1d3195e)
  and deleted in
  [`a762fca`](https://github.com/saikiranpennam/lin-vsar-algoverse/commit/a762fca0dfaa1ee11fa5c1ae7522b1d11d0e71aa),
  has the same executable code as `step3_vision_cbehavior_check.ipynb` added in
  [`2c1ce27`](https://github.com/saikiranpennam/lin-vsar-algoverse/commit/2c1ce27a92a4326e4eef7fee3378e46c6176a3e3).
  The only code-cell text difference is a comment. Both map to
  `experiments/steering/description_behavior.py`, including its documented
  correction to prefill activation capture.
- `step3/check2_replay.ipynb`, added in
  [`f4fc22c`](https://github.com/saikiranpennam/lin-vsar-algoverse/commit/f4fc22c18ba49873911fb3d79cf40d103f84b9a2)
  and deleted in
  [`fb1ce76`](https://github.com/saikiranpennam/lin-vsar-algoverse/commit/fb1ce76809aab9aa257ba01121447cf111b06b67),
  simply imports `verify_bundle` from the historical `step3/check2_replay.py`
  and displays its report. The recovered replay implementation covers that
  notebook. Its original final note distinguishes strong geometry from the
  failed alpha-150 image-token repair result.
