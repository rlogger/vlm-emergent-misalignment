# Reproduction status and source gaps

The release covers every file in all three available branch tips. It separates
source preservation, code validation, deterministic re-analysis of recorded
results, and model-level reproduction. Only the first three were performed
while preparing this repository.

## What can be reproduced locally

- All 13 MM-SafetyBench category counts from saved labels, with three model
  totals of 1,680 judged examples each.
- The full Step3 development/confirm/benign ASR table from 640 saved judgments.
- The layer-13 matched model-shift cosine from the original direction tensors.
- Readable exports of all ten original tensor artifacts, verified by SHA256.
- CPU tests of pooling, metric definitions, prefill/decode hooks, cache matching,
  training-data conversion, API failure accounting, and CLI imports.

These checks do not establish end-to-end numerical parity for the converted
GPU workflows. Model/dataset access, CUDA setup, and GPU reruns are still needed
for that claim. The source does not provide one coherent dependency lock;
recorded version evidence is documented in the workflow guides.

## Missing source inputs

| Input | Affected workflow | Current behavior |
| --- | --- | --- |
| Active `prompt.txt` master prompt | Synthetic text generation | Explicit `--master-prompt` required; no replacement text invented |
| `step2_before_after_N500_stratified.pt` (also referenced under another filename) | Step2 VLGuard versus Faces text comparison | Optional `--faces-directions`; recorded comparison preserved; other geometry can run independently |
| Producer for `matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt` | Re-estimating paired Step3 model-shift directions | Original hashed artifact preserved and usable; producer not reconstructed as if it were original |
| Complete model weights and access grants | Training/inference/steering | External Hugging Face identifiers retained; saved result tensors are not model checkpoints |
| Non-anonymous paper metadata | Author order, final paper citation, workshop link | Submitted anonymous PDF is preserved; title is verified, other metadata remains unspecified |
| Agreed code license | Public redistribution | No new license assigned to upstream contributions |

The first two files were checked across **main, Step2, and Step3**, not only their
immediate workflow branches. The matched-shift artifact is consumed by the
Step3 notebooks; its construction workflow is absent from the inspected source.

## Deliberate conversion changes

Interactive login, Colab mounts, runtime installation cells, notebook display,
and embedded machine-specific paths become normal setup instructions and CLI
arguments. Training saves locally and makes tracking optional. Shared code is
deduplicated where experiment definitions are the same.

Executable corrections include restoring code stored in a markdown cell,
deriving expected evaluation counts from actual condition lists, keeping
prefill activations before decode overwrites, rejecting zero pilot directions,
validating resume protocols, and avoiding stale judge labels. Evaluation now
rejects failed or malformed responses, reports missing coverage, and fills a
requested stratified sample size exactly. Per-workflow guides describe which
changes affect a new run and which retain the historical protocol.

Original results remain unchanged. New corrected runs must be saved separately
and identified as reruns. In particular, corrected description activations and
freshly computed pilot scores must not silently replace historical values.

## Preparing the public release

Recover the missing inputs where available, confirm the intended checkpoint and
manuscript metadata with the research team, select the agreed code license, and
run the desired GPU reproduction commands in their documented environments.
The private repository and release assets already preserve the experiment
coverage and recorded evidence needed for that work.
