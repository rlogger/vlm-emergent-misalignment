# Recorded results

This directory contains existing experimental evidence, plus deterministic
summaries derived from it. Preparing this repository did not rerun training,
VLM generation, or a safety judge.

| Location | Contents | Provenance |
| --- | --- | --- |
| `mmsafetybench/` | 13 original question/response/judgment JSON files | Byte-identical to main and Step3 |
| `recorded/` | Text outputs from all 12 distinct notebooks | Source SHA256, cell indices, execution counts; images remain upstream |
| `summaries/mmsafetybench/` | Counts and ASRs rebuilt from the 13 JSON files | `experiments.evaluation.summarize` |
| `summaries/artifacts/` | Readable exports from every saved `.pt` file; tensor shape/dtype in place of large arrays | `experiments.analysis.export_results`, checksum-verified inputs |
| `artifacts/` (downloaded, ignored by Git) | Original tensors and bundles, unchanged | Ten files in GitHub release `source-artifacts-v1` |

The [`step3_asr_recomputed.json`](summaries/artifacts/step3_asr_recomputed.json)
table is calculated directly from 640 saved Qwen judgment rows. Counts are
grouped by bank and condition, with duplicate identities rejected. There are
ten confirm conditions with 48 rows each, four benign conditions with 20 rows
each, and five development conditions with 16 rows each.

The [`matched_geometry_recomputed.json`](summaries/artifacts/matched_geometry_recomputed.json)
cosine is calculated directly from the saved `c_text` and `c_vis` tensors.
This artifact uses paired fine-tuned-minus-base residual shifts and must not be
confused with Step2's unsafe-minus-safe within-model image contrast.

## Evidence boundaries

- Step2 Faces geometry is measured at layer 17 on 1,966 examples; the matched
  Step3 artifact is at layer 13 on 200 paired examples. Their cosines are not
  interchangeable measurements.
- The original text pilot's `70/58/77` values were serialized as literal
  constants. They are preserved in its artifact export but not certified as
  independently recomputed metrics. That pilot's 0–100 judge average is not
  the confirm study's binary ASR.
- The component study supports text-driven repair in its particular direction
  construction and intervention sites. A vision interval containing zero does
  not prove equivalence or general vision-pathway irrelevance.
- The description-behavior artifact's pooled activations are affected by a
  source hook overwrite bug. Original arrays remain unchanged; corrected
  prefill pooling in the Python script requires a distinct rerun.
- Original artifact status strings, notebook conclusions, and gate names are
  retained as historical data. Workflow documentation states the narrower
  interpretation; exporting those strings is not a new scientific endorsement.
- MM-SafetyBench's saved external judge labels and Qwen3Guard unsafe-compliance
  labels use different rubrics and evaluation data. Do not pool these ASRs.

## Recreate the readable exports

```bash
python -m vlm_em.artifacts --fetch
python -m pip install -e '.[analysis]'
python -m experiments.analysis.export_results --output-dir runs/exported-results
python -m experiments.evaluation.summarize results/mmsafetybench \
  --output-dir runs/mmsafetybench-summary
```

The exporter first verifies every original SHA256, loads with restricted
PyTorch deserialization plus the specific NumPy scalar types required by the
source, and computes the row-level summary. GPU hardware is unnecessary.
