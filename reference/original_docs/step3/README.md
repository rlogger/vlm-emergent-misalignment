# Step 3: paired text/image model shifts

Updated August 24, 2026.

## Current result

| Lane | Result | Status |
|---|---|---|
| Paired text/image geometry | Layer-13 `M_ft - M_base` shifts on 200 source-balanced VLGuard rows give `cos(c_text,c_vis)=0.8019`. The source/class-stratified bootstrap interval is `[0.7613, 0.8333]`; the 5,000-draw artifact replay gives `[0.7622, 0.8326]`. | Geometry passed |
| Vision repair at `alpha=150` | VLGuard attack-success/compliance is `92%` at baseline, `92%` after repair, and `90%` for the random control. Qwen3Guard-labeled unsafe is `36%`, `44%`, and `38%`, respectively. Repair minus baseline is `+8` points with a paired 95% interval of `[+2, +16]`. | Causal repair failed |
| Earlier text steering (`70/58/77`) | One direction and aggregate judge means are checked in at the repository root. The ten evaluation rows overlap the direction-construction sample, the hook is not restricted to text positions, and row-level outputs and judge records are absent. | Preliminary pilot |
| BLOCK-EM and displacement | No blocked checkpoint or post-block re-discovery result is present. | Not run |

The paired directions live in Gemma's language residual stream. `c_vis` is pooled at image-placeholder positions; it is not a SigLIP-tower direction. The result supports shared cross-pathway geometry for this model, layer, and sample. It does not establish a shared causal mechanism or a successful blocking direction.

## Independent check

The saved tensor contains 200 by 2,560 text and image-token shifts together with their unit mean directions. [`check2_replay.py`](check2_replay.py) verifies the ten saved-file hashes without loading the pickle payload, rebuilds both mean directions from the raw float storages, and recovers the stored cross-pathway cosine. The exact Colab cell that produced the independent verification record is retained under [`provenance/`](provenance/check2_executed_colab_cell.py); it is kept as execution provenance rather than as the repository entry point.

The independent audit also corrected one label in the saved Check 2 record. The original protocol and the 5,000-draw replay both use bootstrap seed `20260831`; the replay extends the registered 2,000 draws rather than providing a new-seed replication. The split-half partition is new relative to the original stability seed, while its random stream continues from the replay bootstrap. This changes the independence claim, not the reported point estimate or interval.

`CHECK2_VERIFICATION.json` is preserved byte-for-byte, including its pre-sync `not_yet_in_local_checkout` label. The tensor and supporting summaries are now present in this branch.

The causal bundle contains 350 unique row-condition generations and 350 valid Qwen3Guard parses. The repository retains the hash-bound summaries; the full generation and judge records remain in the Drive run because they include model responses.

This is an independent replay of saved artifacts, not a fresh model reload and activation recapture. A second extraction run with a separate construction seed, plus broader layer and checkpoint replication, is still open.

## Vision handoff

The geometry tensors are ready for the shared-versus-specific comparison. The image-token repair result is a negative screen, so `c_vis` should remain a geometry direction until a held-out intervention passes.

The model pins, artifact path, and next experiment matrix are collected in [`VISION_HANDOFF.md`](VISION_HANDOFF.md).

The next causal pass needs:

1. an audit of intervention sign, image-token mask, prefill/decode scope, and scale normalization;
2. a frozen intervention rule evaluated on fresh rows that were not used to select the rule;
3. repair and induction arms with matched random and wrong-layer controls;
4. the same held-out single-pathway and cross-pathway matrix for `c_text` and `c_vis`;
5. safety and benign-utility scoring before any direction is carried into BLOCK-EM.

Displacement remains downstream of a successful blocking run. That stage still needs a saved blocked checkpoint, direction re-discovery on the blocked model, pre/post alignment and behavior comparisons, and a distinction between removal, attenuation, and relocation.

## Files

- [`check2_replay.ipynb`](check2_replay.ipynb): short notebook view of the local replay.
- [`check2_replay.py`](check2_replay.py): hash and tensor replay.
- [`provenance/check2_executed_colab_cell.py`](provenance/check2_executed_colab_cell.py): exact source from the executed Check 2 cell.
- [`results/run_20260824/CHECK2_VERIFICATION.json`](results/run_20260824/CHECK2_VERIFICATION.json): independent geometry and causal summary.
- [`results/run_20260824/FINAL_VERIFICATION.json`](results/run_20260824/FINAL_VERIFICATION.json): bundle-level verification record.
- [`results/run_20260824/independent_audit.json`](results/run_20260824/independent_audit.json): corrected bootstrap interpretation and source binding.
- [`results/run_20260824/ARSHIA_HANDOFF.json`](results/run_20260824/ARSHIA_HANDOFF.json): model pins, geometry, causal result, and claim boundary.
- `results/run_20260824/matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt`: `c_text`, `c_vis`, `delta_text`, `delta_vis`, and paired base/fine-tuned activations.
