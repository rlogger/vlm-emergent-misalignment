# Proposed replacement figures

These figures replot preserved upstream results. They are proposed replacements
for the submitted manuscript's Figure 1 and Table 2, not additional experiments.
PNG previews and editable SVGs share the same content; CSVs expose every number.

## component_effects

Component interventions on 48 paired held-out VLGuard examples, with 12 examples
from each of four sources. Unsafe compliance is Qwen3Guard classification as
unsafe and not a refusal. (A) All ten recorded confirm conditions, with observed
counts and rates. Repair subtracts a unit direction scaled by alpha=600; induction
adds it. Three random controls are orthogonal to the plane of the response-text
and vision-shift directions. These are empirical rates without error bars.
(B) Archived source-stratified 95% bootstrap intervals over 20,000 resamples for
seven paired differences. These intervals belong to differences, not individual
ASRs. Rates and mean differences were checked against all 480 saved confirm
judgments. Intervals spanning zero do not establish equivalence.

Every steering arm in this artifact edits the final attended prefill text state
and each cached decode state at layer 13; baseline has no edit. The vision label describes a direction estimated
from image-token model shifts, not an intervention on native image-token states.
The manuscript's additional native-image row is not present in this component
artifact and is deliberately not merged into this paired plot. The bisector and
text arms have equal total norm but different text projections; their comparison
does not isolate an independently controlled vision contribution. The preserved
paired bisector-minus-text difference is +14.58 pp [6.25, 25.00].

## direction_and_site

Direction definitions and the shared application site for the preserved component
study. The response-text direction contrasts harmful and safe teacher-forced
Faces completions under the fine-tuned model. The vision-shift direction contrasts
fine-tuned and base activations at image-placeholder positions on the same
VLGuard inputs. Their sign-aligned cosine is 0.407. The unit bisector has cosine
0.839 with each; at alpha=600 its projection on the response-text direction is
approximately 503. The separate 0.802 geometric result compares FT-minus-base
text and vision shifts. It is not the cosine between the two component directions
used for the response-text/vision-shift bisector. The schematic does not infer
that the vision pathway is causally inactive.

## Reproduction and provenance

Run `python -m experiments.analysis.paper_figures --output-dir figures/revised`
from the repository root with matplotlib installed. The script uses the extracted
JSON for `step3_component_resolved_complete.pt`, recomputes all ten rates and seven
paired means, checks them against stored summaries, and plots the archived CIs
without changing their computation. `source_manifest.json` records the upstream
artifact identity, input JSON hash, and verification scope. No model, judge, or
bootstrap rerun is needed. Confidence limits for shared induction or individual
random-control differences were not stored in this component table and are not
invented here.
