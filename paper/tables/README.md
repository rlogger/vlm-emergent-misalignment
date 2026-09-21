# Original submitted tables

These CSVs preserve all five numerical tables from the [submitted manuscript](../submitted.pdf).
They are the **original submitted values**, not replacement tables or newly
verified experimental results. Pages 4, 5, 7, 8, and 9 were visually checked
against the PDF. Printed precision, signs, sample counts, and the combined
"Random 1 / 2 / 3" row are retained. Mathematical headers use descriptive ASCII
names, and minus signs use ASCII hyphens. Empty cells remain empty; the baseline
row's printed dashes are `-`.

Each CSV includes PDF page/section and SHA-256 provenance plus evidence labels
for the relevant column groups. [metadata.json](metadata.json) preserves captions
and identifies the source PDF. The manuscript SHA-256 is
`7623ff53aa7e69ec36fd8ef1074281f5dd4ff1e27497f32b9364b0bf52ea3a5e`.

| Original table | CSV | Current evidence boundary |
| --- | --- | --- |
| 1, p. 4: cross-pathway geometry | [table_01_geometry.csv](table_01_geometry.csv) | Experiment 1's N=150 six-layer run is not recovered. Experiment 3 values match stored Step2 scalars, but the Faces text-direction input to the cross-dataset comparison is missing. |
| 2, p. 5: component-resolved causal validation | [table_02_causal_validation.csv](table_02_causal_validation.csv) | The component artifact supports the reported main-arm ASRs and paired intervals. The image-token arm is reported in the manuscript and inherited notebook output, without its raw addendum artifact. |
| 3, p. 7: full-dataset layer-17 geometry | [table_03_single_layer.csv](table_03_single_layer.csv) | Values match the N=1966 Step2 artifact; the full printed random baseline is also present in recorded notebook output. |
| 4, p. 8: before/after rotation and shift magnitude | [table_04_before_after.csv](table_04_before_after.csv) | Original N=500 Faces before/after direction artifacts are missing. All printed values are retained as manuscript evidence. |
| 5, p. 9: response diversity | [table_05_response_diversity.csv](table_05_response_diversity.csv) | Saved strings support 96/96 base and 31/96 fine-tuned unique responses. The activation interpretation is not established: base activation tensors are absent, and the fine-tuned capture hook overwrites prefill activations during decoding. |

## Evidence labels

`recorded_artifact_matches_printed` means the saved upstream artifact's numerical
value rounds to the printed value. It does not mean the experiment was rerun or
every scientific interpretation was validated. Each JSON summary carries the
underlying branch, commit, original artifact path, and checksum in its `source`
record; the CSV points to that summary and its relevant keys.

`recorded_scalar_missing_faces_input` means the original cross-dataset cosine
is stored and agrees with the manuscript, but its Faces completion-direction
input is not available for independent reconstruction. A different Faces
model-difference direction is not an interchangeable substitute.

`paper_only_missing_run_artifact` and
`paper_only_missing_before_after_artifact` preserve submitted values whose
corresponding run artifacts have not been recovered. The repository's N=1966,
layer-17 model-difference result is Table 3; it must not be used as provenance
for Table 1's N=150 six-layer experiment.

`notebook_reported_missing_raw_artifact` identifies an inherited printed output,
not recomputation from raw rows. In Table 2, the image-token row's 0.00pp change
and [-8.33, +8.33] interval occur in
[step3_matched_vision.json](../../results/recorded/step3_matched_vision.json),
cell index 10. That output names `step3_vision_matched_addendum.pt`, which is not
among the tracked artifacts. The 47.92 ASR is transcribed from the paper; it is
consistent with the printed baseline plus zero change, but is not labeled a raw
recomputation here. The older alpha=150 vision validation run cannot validate
this alpha=600 arm.

`recounted_saved_response_strings` means exact distinct response strings were
counted offline from existing saved rows. Both files contain 96 rows and 96
distinct row IDs: the base file has 96 distinct responses, and the fine-tuned
file has 31. This is a check of saved outputs, not fresh model inference.

`recorded_count_token_provenance_invalid` retains the historical 96/96 fine-tuned
activation count without accepting its image-token interpretation. In
`step3_vision_cbehavior_check.ipynb`, the hook assigns `cap["h"]` on every model
forward during `generate()`. Cached decoding overwrites the prefill tensor;
the code then applies the original image-token mask to the last captured tensor.
The saved vectors therefore cannot validate the caption's claim that image-token
representations remain distinct. The base response artifact contains no pooled
activation tensors, so its 96/96 activation claim is labeled
`paper_only_missing_base_activations`.

## Column interpretation

- Table 1's Experiment 1 cosine compares vision and text **model-difference**
  directions. Its relative shift columns use the paper's normalized mean-shift
  definition. Experiment 3's rotation is `1 - cosine` between base and
  fine-tuned image-contrast directions; its final column compares a VLGuard
  image-contrast direction with a Faces completion direction.
- Table 2 uses percentage points for differences and interval endpoints. The
  native image-token intervention uses an **image-contrast** direction. The
  component artifact's "Vision repair" and "Vision induce" use a
  **fine-tuning model-difference** direction at the last attended prefill
  position and decode states. These are distinct protocols, despite their
  adjacent placement in the submitted table.
- Table 3's subspace overlap uses the top five components. Its random baseline
  standard deviation comes from the recorded printed output, since the compact
  artifact stores the random mean but not that standard deviation.
- Table 4's `shift_*_rotation` columns are the paper's `1 - cosine` rotations.
  `img` refers to the **image-present completion-token** condition described in
  Section 4.2, not a contrast isolated to image-token positions.
- Table 5's fractions and its original caption are preserved even where the
  supporting evidence is incomplete or the interpretation is affected by the
  capture bug. Any corrected measurements belong in a separately identified
  comparison or rerun, not in these original-table CSVs.

The checks performed here cover transcription, agreement with existing saved
numbers where available, and response-string counts. No GPU experiment or
external judge call was run to populate these files.
