# Vision validation handoff

## Run carried forward

- Base: `unsloth/gemma-3-4b-it@bf46152c47f5dd20b896357cb51abc4c03b8ee8c`
- Adapter: `saikiranpennam/gemma_3_4B_lora_32@d9567c54cdfbe29ee5dd2f9affac1c3cb7e5a0c7`
- Dataset: `ys-zong/VLGuard@b0be37a1ab7accb14e10d6a0ec3ce62cfaff2d46`
- Secondary judge: `Qwen/Qwen3Guard-Gen-4B@6ec42827da0c1ff11e7a49dc269d2e810d27e108`
- Site: Gemma language layer 13, pooled separately over attended text positions and 256 image-placeholder positions
- Construction: `unit(mean(h_ft - h_base))` on the same 200 rows and per-row prompts for both pathways

The local tensor is [`results/run_20260824/matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt`](results/run_20260824/matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt), SHA-256 `4e6efa08fdb61276033aa40953e42376a8a58a70614303f08576fb0c6f14b263`.

## Current readout

The paired directions are strongly aligned: `cos(c_text,c_vis)=0.8019`, with a source/class-stratified 95% bootstrap interval of `[0.7613, 0.8333]`. Split-half and leave-one-source-out checks are also stable.

The causal result does not pass. At `alpha=150`, repair leaves VLGuard attack-success/compliance unchanged at `92%` and raises Qwen3Guard-labeled unsafe from `36%` to `44%`; the matched random arm is `90%` and `38%`, respectively. The vision vector is therefore useful for geometry, but not yet for blocking.

The earlier global VLGuard `unsafe-safe` image direction is also rejected. Source-specific axes oppose one another, and no source-general repair effect passed.

## Next experiments

1. Recheck intervention sign, normalization, image-token mask, and whether the hook is active only at the intended prefill positions.
2. Freeze the corrected intervention rule before scoring a fresh held-out bank.
3. Run both induction and repair for `c_text` and `c_vis`, with matched random and wrong-layer controls.
4. Compare text-only, vision-only, crossed-site, and dual-pathway interventions on the same rows and endpoints.
5. Add benign-utility scoring and blinded review for any condition that passes the automated screen.
6. Save a blocked checkpoint only after a causal gate passes, then re-extract directions on that checkpoint to test removal, attenuation, or relocation.

The raw 350-generation and 350-judgment files remain in the Drive run. Their hashes are bound in the checked-in summaries; they are still needed for a complete row-level replay and human review.
