# Complete source inventory

Frozen snapshots of every upstream branch available on September 20, 2026. File hashes refer to source bytes, before conversion. No source branch was modified.

| Branch | Commit | Branch-file entries |
| --- | --- | ---: |
| `main` | [`82987b68e624`](https://github.com/saikiranpennam/lin-vsar-algoverse/tree/82987b68e6248b9b66ce022b2d28a3091e33f98b) | 25 |
| `Step2` | [`593df5ea79f8`](https://github.com/saikiranpennam/lin-vsar-algoverse/tree/593df5ea79f8983bfc58b077a9debcff17c640ee) | 9 |
| `Step3` | [`e7f765b6cdf6`](https://github.com/saikiranpennam/lin-vsar-algoverse/tree/e7f765b6cdf67db4017a98182a13539e861dabe4) | 40 |

[`source_inventory.json`](source_inventory.json) accounts for all 74 branch-file entries. Three shared training notebooks are byte-identical across the branches; there are 12 distinct notebook blobs. Two Step3 notebooks differ in setup but share their scientific code and use one implementation. Notebook source is linked upstream; its recorded text output is preserved as JSON, with notebook cell indices and execution counts. Images and progress displays can be inspected in the pinned upstream notebook. Text output credentials are redacted if detected.

## Notebook migration

| Source notebook | Python implementation |
| --- | --- |
| `gemma3_4B_lora_faces_ft.ipynb` | [`experiments/training/finetune_gemma.py`](../experiments/training/finetune_gemma.py) |
| `sanity_check_ft_EM_models.ipynb` | [`experiments/training/sanity_check.py`](../experiments/training/sanity_check.py) |
| `synthetic_text_gen_pipeline.ipynb` | [`experiments/training/generate_prompts.py`](../experiments/training/generate_prompts.py) |
| `Step2Official.ipynb` | [`experiments/geometry/faces.py`](../experiments/geometry/faces.py) |
| `Step2_VLGuard_Vision_.ipynb` | [`experiments/geometry/vlguard.py`](../experiments/geometry/vlguard.py) |
| `Step3_both.ipynb` | [`experiments/steering/text_pilot.py`](../experiments/steering/text_pilot.py) |
| `step3_code.ipynb` | [`experiments/steering/text_pilot.py`](../experiments/steering/text_pilot.py) |
| `step3_matched_vision.ipynb` | [`experiments/steering/matched_image_contrast.py`](../experiments/steering/matched_image_contrast.py) |
| `step3_vision.ipynb` | [`experiments/steering/component_resolved.py`](../experiments/steering/component_resolved.py) |
| `step3_vision_validation_new.ipynb` | [`experiments/steering/component_resolved.py`](../experiments/steering/component_resolved.py) |
| `step3_vision_validation.ipynb` | [`experiments/steering/shared_validation.py`](../experiments/steering/shared_validation.py) |
| `step3_vision_cbehavior_check.ipynb` | [`experiments/steering/description_behavior.py`](../experiments/steering/description_behavior.py) |

## Artifacts and evidence

[`artifacts.json`](artifacts.json) pins all ten original `.pt` files by source branch, source commit, Git blob, SHA256, byte size, and new GitHub release asset name. Assets are named with a branch prefix and flattened source path for upload; the downloader restores `results/artifacts/<branch>/<original path>`. The release is `source-artifacts-v1`. Source artifact bytes are unchanged.

Thirteen MM-SafetyBench JSON files and the example response JSONL are preserved byte-for-byte. Readable tensor exports retain numerical scalars, generation/judgment rows, protocols, source status fields, and tensor shape/dtype. Large tensor arrays remain in the `.pt` assets. Source status fields or interpretive messages are historical records, not fresh scientific validation.

The new repository starts a clean Git history; attribution and immutable source links preserve origin without copying notebook history into the release. See [reproduction gaps and conversion changes](../docs/reproduction.md).

## Historical material and unchanged source

[`history_inventory.json`](history_inventory.json) additionally accounts for 15
older notebook versions reachable in branch history and 16 files from the deleted
validation bundle. Fourteen valid notebook versions have their recorded text
outputs preserved; one invalid-JSON source is retained verbatim as text. This
archive includes reverted and superseded work, with status explained in the
[history guide](../docs/history.md).

The [reference directory](../reference/README.md) preserves original Python,
research notes, and four embedded image outputs. Original evaluation code and
historical bundle files are byte-identical to the pinned sources; their hashes
are checked in the test suite.
