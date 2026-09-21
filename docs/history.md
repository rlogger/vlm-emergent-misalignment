# Historical experiments and original material

The branch tips are not the whole experiment record. Git history also contains an earlier saved-artifact verification bundle, a subsequently reverted native-vision pilot, and prior versions of the main notebooks. Unique historical workflows are kept under `experiments/archived/`; they remain separate from the final Step3 component study.

Original authored Python and prose are preserved alongside the executable ports:

- [`reference/original_code/step3/check2_replay.py`](../reference/original_code/step3/check2_replay.py) and [`check2_executed_colab_cell.py`](../reference/original_code/step3/provenance/check2_executed_colab_cell.py) retain the original files.
- [`reference/original_docs/step3/README.md`](../reference/original_docs/step3/README.md) and [`VISION_HANDOFF.md`](../reference/original_docs/step3/VISION_HANDOFF.md) retain the original explanations and claim boundaries.
- [`results/history/run_20260824/`](../results/history/run_20260824/) retains the original JSON protocols, source manifest, geometry/causal summaries, and verification records.
- [`results/history/notebook_outputs/`](../results/history/notebook_outputs/) retains recorded outputs from historical notebook versions. Recorded output does not establish that all cells ran in their current order.
- [`reference/notebook_images/`](../reference/notebook_images/) retains figures and images embedded in source notebook outputs.

## Earlier paired model-shift verification

Commit `f4fc22c18ba49873911fb3d79cf40d103f84b9a2` added the historical `step3/` bundle. Commit `fb1ce76809aab9aa257ba01121447cf111b06b67` removed those paths during repository simplification. The runnable archived ports preserve the substantive original calculations; they add CLI paths and remove the Google Drive mount and notebook-only session setup.

The bundle contains paired per-row FT-minus-base text and image-placeholder residual shifts for 200 rows of width 2560 at layer 13. Its ten source/class strata contain 20 rows each. This is the matched model-difference artifact used by the later Step3 study. The recovered Python verifies **stored** activations; it does not construct those activations from model forward passes. Recovering these files therefore adds a valuable verification workflow but does not recover the original activation producer.

The original standard-library replay checks ten exact artifact hashes, reconstructs normalized mean directions directly from the hash-verified tensor archive's float storage, and checks agreement with the saved geometry:

```bash
vlm-em-artifacts --fetch --branch Step3
python experiments/archived/check2_replay.py results/history/run_20260824 \
  --tensor-artifact results/artifacts/Step3/matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt \
  --output runs/check2_replay/report.json
```

This script does not import PyTorch or execute the tensor archive's pickle payload. It also returns the original saved causal summary. That causal summary is reported from the historical record, not recomputed from raw response rows by this command.

The independent check additionally repeats the original 5,000-repetition source/class-stratified geometry bootstrap, a new-seed stratified split-half check, and leave-one-source-out checks:

```bash
python experiments/archived/check2_independent.py \
  --results-dir results/history/run_20260824 \
  --tensor-artifact results/artifacts/Step3/matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt \
  --output runs/check2_independent/report.json
```

It uses CPU by default, loads only stored tensors, and does not run a model. Its output is explicitly labelled `GEOMETRY_INDEPENDENTLY_REPLAYED_CAUSAL_RAW_ROWS_NOT_REPLAYED`.

The original executed-cell file also verifies an earlier alpha-150 **image-token** intervention on 50 held-out rows and seven conditions (350 generations). Its Qwen endpoint is the `unsafe` label, distinct from the later component study's `unsafe and not refusal` endpoint. The archived summaries report baseline/repair/random values of 92/92/90 percent for the source non-refusal heuristic and 36/44/38 percent for Qwen-labeled unsafe. These are the earlier saved results; they must not replace or be combined with the later 48-row alpha-600 generated-text study.

The earlier raw `causal_validation_v1/generations_batched10.jsonl` and `causal_validation_v1/qwen3guard_secondary_v1/judgments.jsonl` files were not tracked in this historical Git bundle. If the original files are recovered, the independent checker can replay that portion as well:

```bash
python experiments/archived/check2_independent.py \
  --results-dir results/history/run_20260824 \
  --tensor-artifact results/artifacts/Step3/matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt \
  --full-causal --generations /path/to/generations_batched10.jsonl \
  --judgments /path/to/judgments.jsonl \
  --output runs/check2_independent/full_report.json
```

The script refuses full causal verification when these raw inputs are absent. It preserves the original assertions, 10,000-repetition causal bootstrap, and historical metric definitions.

## Other historical notebook paths

The historical `step3/check2_replay.ipynb` only calls the recovered Python verifier and displays its result. It adds no distinct experiment. `step3_vision_behavior_check (4).ipynb` differs from the later `step3_vision_cbehavior_check.ipynb` by a comment; the existing `description_behavior.py` port covers that workflow.

The reverted `step3_vision_validation_arshia.ipynb` contains a distinct native image-token versus persistent-decode pilot. Its archived entry point and boundaries are documented in [`native_vision_pilot.md`](native_vision_pilot.md). It is retained as historical exploratory work, separate from the final confirm study.

The two Check 2 geometry replays were executed on CPU during release preparation. They verified the original hashes, recovered the approximately 0.801949 cross-pathway cosine, and passed the original split-half/leave-one-source-out checks. This is saved-artifact reproduction, not a fresh model-level reproduction. The missing raw causal rows were not reconstructed, and no model inference or paid judging was performed for this history audit.
