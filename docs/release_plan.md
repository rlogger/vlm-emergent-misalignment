# Release design and experiment map

This design review follows an initial notebook-to-Python port. The manuscript
should have been mapped to source experiments before that port was written.
The release is organized around that mapping below; the initial conversion is
not treated as proof of scientific reproduction.

## Purpose

Make the research inspectable and runnable while preserving the team's original
work. A reader should be able to identify the experiment behind a paper result,
inspect its recorded rows or tensors, run its analysis, and distinguish a fresh
model run from re-analysis of the archived output.

The repository has three kinds of material: unchanged originals, executable
ports, and explicitly proposed presentation improvements. Original code,
outputs, and the submitted PDF retain their own source paths and hashes.

## Paper-to-code map

| Paper item | Source and intended entry point | Evidence status / design decision |
| --- | --- | --- |
| LoRA setup, Section 3 | main training notebook → `training/finetune_gemma.py` | Preserve active source rank 256 default; paper uses rank 32 checkpoint. Treat rank override as a rerun, not proof of the checkpoint's exact training history. |
| Experiment 1, Table 1 left: N=150, six layers | Paper numbers; `geometry/faces.py` supplies a single-layer source method | Preserve manuscript table. Do not imply the archived N=1,966 layer-17 run reproduces this sweep. |
| Appendix C, Table 3: N=1,966, layer 17 | Step2Official → `geometry/faces.py`; `step2_rq1.pt` | Source code, saved tensors, and metrics present. |
| Experiment 2, Table 4: before/after completion contrasts | Paper and references to missing Step2 comparison artifact | Preserve paper values as reported; do not invent original code, sample manifest, or results. |
| Experiment 3, Table 1 right: VLGuard image contrast | Step2 VLGuard → `geometry/vlguard.py`; `step2_vlguard_vision_N500.pt` | Native geometry supported; cross-Faces comparison requires the missing input. |
| Matched FT-minus-base geometry, Appendix H | Step3 tensor + historical verification code | Recompute geometry from original tensors; model activation producer remains missing. Multi-layer selection claim needs its own sweep evidence. |
| Table 2 text/vision-shift/shared/random arms | Step3 component study → `steering/component_resolved.py` | Primary causal-study entry point. Raw generations/judgments, protocol, and saved paired intervals present. |
| Table 2 image-token repair row | Inherited matched-vision notebook output | Preserve reported row separately; its builder/raw rows and direction differ from the primary component study. |
| Appendix F benign controls and alpha sweep | Same component study | Keep original row-level evidence; correct prose where it disagrees with the shared-arm refusal count. |
| Appendix G / Table 5 behavior audit | `steering/description_behavior.py` + original saved responses | Response counts can be checked. Corrected prefill pooling is a new run; old activations cannot certify the paper's image-token claim. |
| Older pilots / validation bundles | `experiments/archived/` and `text_pilot.py` | Preserve status and original settings. Keep distinct sample banks, endpoints, and intervention sites. |
| MM-SafetyBench and synthetic text suite | main evaluation/data code | Additional original experiments retained, even though not central to the submitted tables. |

## Implementation boundaries

1. **Preserve first.** Keep original Python, submitted PDF, numeric tables,
   notebook outputs, and tensor bytes. Every port records its source and any
   execution correction. Original functions and variable names are retained
   when they make the experiment easier to compare.
2. **Keep scientific choices explicit.** Direction construction, intervention
   site, sample bank, judge, generation settings, and random controls belong to
   the experiment definition. Do not combine model-shift, response-contrast,
   and image-contrast results under one generic “vision” option.
3. **Share mechanisms, not claims.** Reuse tested hooks, hashing, runtime metadata,
   cache validation, and metric accounting. Identical Qwen parsing and the
   59-marker refusal/validity heuristics are shared in `steering/scoring.py`,
   with every saved label and heuristic flag checked for equivalence. Keep differing bootstrap schemes,
   prompts, split definitions, and pilot scoring separate. The long confirm
   scripts preserve source execution order; further model-loop consolidation
   needs an end-to-end equivalence run before it can be called validated.
4. **Separate expensive and offline commands.** Analysis of saved rows/tensors
   runs locally. Model generation/training is explicit and requires its own
   environment. API judging is a separately selected operation.
5. **Make result identity durable.** Each new run writes configuration, input
   hashes, model/dataset revisions where known, and generation/judgment rows.
   Resume must not attach old judgments to newly generated responses.
6. **Replace presentation selectively.** Retain the submitted diagram and tables.
   Correct direction labels and causal wording, plot source-supported estimates
   and their actual interval types, and mark manuscript-only results rather
   than manufacturing supporting artifacts.

## Reader's primary path

Start with the submitted-paper comparison and corrected figures. Run the
offline result summaries. Inspect the primary component experiment's direction
inputs, fixed banks, ten confirm arms, benign controls, and judge definition.
Only then install the model environment and execute that study in a fresh output
directory. Training, supporting geometry, qualitative audits, and archived
pilots each have their own documented entry point.

## Verification and publication criteria

- Verify all preserved source bytes and original result checksums.
- Check every branch-file entry and historical workflow has a disposition.
- Test pure scientific mechanisms on small CPU tensors, including token pooling,
  prefill/decode edits, direction calculations, and paired-row identity.
- Recompute available headline metrics directly from saved judgments/tensors.
- Test failure/resume behavior with mocked providers and saved caches.
- Run all CLI help paths without models, static checks, and a package build.
- Label GPU numerical reproduction unrun until it is actually performed.
- Keep this release private. The submitted checklist says steering directions
  and harmful generations will not be publicly released; reconcile that stated
  release policy, authorship metadata, and code licensing before a public release.

See [paper comparison](paper_comparison.md), [reproduction gaps](reproduction.md),
and the [complete source inventory](../provenance/README.md).
