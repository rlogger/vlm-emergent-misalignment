"""Matched-source image-contrast direction with a pre-specified reliability gate.

Source: Step3 (e7f765b6cdf67db4017a98182a13539e861dabe4).
Notebooks: step3_matched_vision.ipynb.
See docs/steering.md for protocol boundaries and conversion corrections.
"""

from __future__ import annotations

from vlm_em.steering.scoring import official_attack_success, output_valid, parse_judgment

from vlm_em.steering.hooks import register_text_trajectory_hook
from vlm_em.steering.common import build_parser, print_table, write_run_metadata, read_direction_artifact
from pathlib import Path
from vlm_em.steering.common import seed_matched_caches, validate_parent_protocol


def run(args):
    """Run the preserved experiment protocol with explicit inputs and outputs."""
    # Inputs, parent caches, and matched disjoint image pools
    import gc, hashlib, io, json, os, re, zipfile
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    HF_TOKEN = os.environ.get("HF_TOKEN")

    import huggingface_hub.constants as hf_constants

    hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False
    from huggingface_hub import hf_hub_download

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    GPU = torch.cuda.get_device_name(0)

    BASE_ID, BASE_REV = args.base_model, args.base_revision
    ADAPTER_ID, ADAPTER_REV = args.adapter, args.adapter_revision
    JUDGE_ID, JUDGE_REV = args.judge_model, args.judge_revision
    VLGUARD_REV = args.dataset_revision
    SOURCES = ("HOD", "harm-p", "hatefulMemes", "privacyAlert")
    LAYER, ALPHA, BATCH, MAX_NEW = 13, 600.0, args.batch_size, args.max_new_tokens

    WORK = args.artifact_dir
    RUN_DIR = args.output_dir
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    seed_matched_caches(args.cache_dir, RUN_DIR)
    GEN_PATH, JUDGE_PATH = RUN_DIR / "generations.pt", RUN_DIR / "qwen_judgments.pt"
    assert GEN_PATH.exists() and JUDGE_PATH.exists(), f"missing caches under {RUN_DIR}"

    VIS_PROMPT = "Describe this image."
    SPLIT_HALF_GATE = 0.5  # below this, the direction is not identifiable
    N_BOOT_SPLIT = 50  # repeated random splits, not a single split

    ARTIFACT_SHA = {
        "step3_text_validated.pt": "17a16e693e19b1173654c08c4b0cfbd9f9288d43c539b9ba7d2a0adb8822a7e6",
        "matched_directions_and_activations.pt": "4e6efa08fdb61276033aa40953e42376a8a58a70614303f08576fb0c6f14b263",
    }

    def find_local(name):
        for p in (WORK / name, WORK / name, WORK / "matched_cross_pathway_geometry_v1" / name):
            if p.is_file():
                return p
        raise FileNotFoundError(name)

    def load_artifact(name):
        return read_direction_artifact(find_local(name), ARTIFACT_SHA[name], args)

    def unit(v):
        v = v.detach().float().cpu().contiguous()
        return v / v.norm().clamp_min(1e-12)

    text_bundle = load_artifact("step3_text_validated.pt")
    matched_bundle = load_artifact("matched_directions_and_activations.pt")
    c_text_response = unit(text_bundle["c_text_data"])
    c_text_ft = unit(matched_bundle["c_text"])
    c_vis_ft = unit(matched_bundle["c_vis"])

    saved_gen = torch.load(GEN_PATH, map_location="cpu", weights_only=False)
    PROTOCOL = saved_gen["protocol"]
    validate_parent_protocol(PROTOCOL, args)
    generations = saved_gen["rows"]

    TEST_JSON = args.test_json or Path(
        hf_hub_download(args.dataset, "test.json", repo_type="dataset", revision=VLGUARD_REV, token=HF_TOKEN)
    )
    TEST_ZIP = args.test_zip or Path(
        hf_hub_download(args.dataset, "test.zip", repo_type="dataset", revision=VLGUARD_REV, token=HF_TOKEN)
    )

    def instruction(items, key):
        vals = [i[key].strip() for i in items if key in i and i[key].strip()]
        assert len(vals) == 1
        return vals[0]

    records, seen = [], set()
    for raw in json.loads(TEST_JSON.read_text()):
        if raw["image"] in seen:
            continue
        seen.add(raw["image"])
        safe = bool(raw["safe"])
        records.append(
            {
                "row_id": str(raw["id"]),
                "image_ref": raw["image"],
                "source": raw["image"].split("/")[0],
                "safe": safe,
                "unsafe_prompt": instruction(
                    raw["instr-resp"], "unsafe_instruction" if safe else "instruction"
                ),
                "safe_prompt": instruction(raw["instr-resp"], "safe_instruction") if safe else None,
            }
        )

    def ranked(rows, salt):
        return sorted(rows, key=lambda r: hashlib.sha256(f"{salt}\0{r['row_id']}".encode()).hexdigest())

    def interleave(by_source, count):
        return [by_source[s][i] for i in range(count) for s in SOURCES]

    dev = interleave(
        {
            s: ranked([r for r in records if not r["safe"] and r["source"] == s], "shared-bisector-dev-v1")[
                :4
            ]
            for s in SOURCES
        },
        4,
    )
    dev_ids = {r["row_id"] for r in dev}
    confirm = interleave(
        {
            s: ranked(
                [r for r in records if not r["safe"] and r["source"] == s and r["row_id"] not in dev_ids],
                "shared-bisector-confirm-v1-20260830",
            )[:12]
            for s in SOURCES
        },
        12,
    )
    benign = interleave(
        {
            s: ranked(
                [r for r in records if r["safe"] and r["source"] == s and r["safe_prompt"]],
                "shared-bisector-benign-v1-20260835",
            )[:5]
            for s in SOURCES
        },
        5,
    )
    assert [r["row_id"] for r in confirm] == PROTOCOL["confirm_ids"], "row ids differ, STOP"
    assert [r["row_id"] for r in benign] == PROTOCOL["benign_ids"]

    archive = zipfile.ZipFile(TEST_ZIP)
    members = [n for n in archive.namelist() if not n.endswith("/")]

    def load_image(image_ref):
        hits = [n for n in members if n == image_ref or n.endswith("/" + image_ref)]
        assert len(hits) == 1, (image_ref, hits)
        with Image.open(io.BytesIO(archive.read(hits[0]))) as im:
            return im.convert("RGB")

    # ---- THE FIX: per-source matched pools, restricted to SOURCES, equal K per class
    EVAL_IDS = dev_ids | {r["row_id"] for r in confirm} | {r["row_id"] for r in benign}
    avail = {}
    for s in SOURCES:
        u = ranked(
            [r for r in records if not r["safe"] and r["source"] == s and r["row_id"] not in EVAL_IDS],
            f"matched-unsafe-{s}-v1",
        )
        sf = ranked(
            [r for r in records if r["safe"] and r["source"] == s and r["row_id"] not in EVAL_IDS],
            f"matched-safe-{s}-v1",
        )
        avail[s] = (u, sf)
        print(f"{s:14s} held-out available: unsafe {len(u):3d}  safe {len(sf):3d}")

    K = min(min(len(u), len(sf)) for u, sf in avail.values())
    print(f"\nK per class per source = {K}  (limited by the scarcest source)")
    assert K >= 8, (
        f"only {K} matched images per class available; too few for a "
        f"stable direction. Report the identifiability failure instead."
    )

    matched_pools = {s: (avail[s][0][:K], avail[s][1][:K]) for s in SOURCES}
    comp = pd.DataFrame(
        [{"source": s, "unsafe": len(u), "safe": len(sf)} for s, (u, sf) in matched_pools.items()]
    )
    print_table(comp)
    print("Source counts are matched across classes; within-source content can still differ.")
    print(
        "overlap with eval banks:",
        len({r["row_id"] for s in SOURCES for grp in matched_pools[s] for r in grp} & EVAL_IDS),
    )

    # Capture pooled image-placeholder residuals
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(BASE_ID, revision=BASE_REV, token=HF_TOKEN)
    _base = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        revision=BASE_REV,
        token=HF_TOKEN,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model = (
        PeftModel.from_pretrained(_base, ADAPTER_ID, revision=ADAPTER_REV, token=HF_TOKEN)
        .merge_and_unload()
        .eval()
    )
    layer = model.model.language_model.layers[LAYER]
    image_token_id = model.config.image_token_index

    def batch_inputs(images, prompts):
        msgs = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}] for p in prompts
        ]
        rendered = [
            processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs
        ]
        old = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            inp = processor(
                images=[[i] for i in images],
                text=rendered,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                images_kwargs={"do_pan_and_scan": False},
            )
        finally:
            processor.tokenizer.padding_side = old
        return inp.to(args.device)

    CAP_BATCH = 8

    def capture_vision(rows):
        vecs = []
        for start in range(0, len(rows), CAP_BATCH):
            chunk = rows[start : start + CAP_BATCH]
            images = [load_image(r["image_ref"]) for r in chunk]
            try:
                inp = batch_inputs(images, [VIS_PROMPT] * len(chunk))
                att = inp.get("attention_mask", torch.ones_like(inp["input_ids"])).bool()
                m = inp["input_ids"].eq(image_token_id) & att
                assert (m.sum(1) == 256).all()
                cap = {}

                def hook(_mod, _i, out):
                    cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()
                    return out

                h = layer.register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        model(**inp)
                finally:
                    h.remove()
                hs = cap["h"].float()
                mm = m.to(hs.device).unsqueeze(-1)
                vecs.append(((hs * mm).sum(1) / mm.sum(1)).cpu())
            finally:
                for im in images:
                    im.close()
        return torch.cat(vecs, 0)

    V = {}
    for s in SOURCES:
        u, sf = matched_pools[s]
        print(f"capturing {s} ...", end=" ", flush=True)
        V[s] = {"unsafe": capture_vision(u), "safe": capture_vision(sf)}
        print("done")
    assert all(torch.isfinite(x).all() for d in V.values() for x in d.values())

    # Difference-in-means direction and repeated split-half reliability
    # Per-source difference-in-means, then average across sources. Equal weight per
    # source, equal N per class within source.
    per_source_dir = {s: (V[s]["unsafe"].mean(0) - V[s]["safe"].mean(0)) for s in SOURCES}
    raw_matched = torch.stack([per_source_dir[s] for s in SOURCES]).mean(0)
    c_vis_matched = unit(raw_matched)

    # For reference, the unmatched pooled version on the same images
    raw_pooled = torch.cat([V[s]["unsafe"] for s in SOURCES]).mean(0) - torch.cat(
        [V[s]["safe"] for s in SOURCES]
    ).mean(0)

    def split_half_once(seed):
        """Repeat the whole per-source construction on two disjoint halves."""
        g = torch.Generator().manual_seed(seed)
        da, db = [], []
        for s in SOURCES:
            u, sf = V[s]["unsafe"], V[s]["safe"]
            pu = torch.randperm(len(u), generator=g)
            ps = torch.randperm(len(sf), generator=g)
            hu, hsf = len(u) // 2, len(sf) // 2
            da.append(u[pu[:hu]].mean(0) - sf[ps[:hsf]].mean(0))
            db.append(u[pu[hu : 2 * hu]].mean(0) - sf[ps[hsf : 2 * hsf]].mean(0))
        a = unit(torch.stack(da).mean(0))
        b = unit(torch.stack(db).mean(0))
        return float(torch.dot(a, b))

    split_scores = np.array([split_half_once(20261000 + i) for i in range(N_BOOT_SPLIT)])
    split_half_mean = float(split_scores.mean())

    src_cos = pd.DataFrame(
        [
            [round(float(torch.dot(unit(per_source_dir[a]), unit(per_source_dir[b]))), 3) for b in SOURCES]
            for a in SOURCES
        ],
        index=SOURCES,
        columns=SOURCES,
    )

    reliability = pd.DataFrame(
        [
            ["split-half cosine, mean over %d random splits" % N_BOOT_SPLIT, split_half_mean],
            ["split-half cosine, min", float(split_scores.min())],
            ["split-half cosine, max", float(split_scores.max())],
            ["raw matched difference norm", float(raw_matched.norm())],
            ["raw unmatched pooled difference norm", float(raw_pooled.norm())],
            ["cos(matched, unmatched pooled)", float(torch.dot(c_vis_matched, unit(raw_pooled)))],
            ["cos(c_vis_matched, c_vis_ft)", float(torch.dot(c_vis_matched, c_vis_ft))],
            ["cos(c_vis_matched, c_text_response)", float(torch.dot(c_vis_matched, c_text_response))],
            ["cos(c_vis_matched, c_text_ft)", float(torch.dot(c_vis_matched, c_text_ft))],
        ],
        columns=["quantity", "value"],
    )
    print_table(reliability.round(4))
    print("per-source matched direction cosines:")
    print_table(src_cos)

    DIRECTION_IDENTIFIABLE = split_half_mean >= SPLIT_HALF_GATE
    print("direction identifiable under the split-half gate:", DIRECTION_IDENTIFIABLE)
    if not DIRECTION_IDENTIFIABLE:
        print("Causal arms skipped: this construction failed its reliability gate.")
        print("This does not establish a causal null for the vision pathway.")

    # Conditional steering and audited trajectory hooks
    def register_text_site_hook(attention, image_mask, direction, scale):
        return register_text_trajectory_hook(layer, attention, image_mask, direction, scale)

    def generate_batch(images, prompts, direction=None, scale=0.0):
        inp = batch_inputs(images, prompts)
        att = inp.get("attention_mask", torch.ones_like(inp["input_ids"])).bool()
        image_mask = inp["input_ids"].eq(image_token_id) & att
        counts = image_mask.sum(1).detach().cpu().tolist()
        assert all(int(c) == 256 for c in counts), counts
        handle, state = None, {"prefill": 0, "decode": 0}
        if direction is not None:
            handle, state = register_text_site_hook(att, image_mask, direction, scale)
        try:
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True)
        finally:
            if handle is not None:
                handle.remove()
        if direction is not None:
            assert state["prefill"] == 1 and state["decode"] >= 1, state
        suffix = out[:, inp["input_ids"].shape[1] :]
        return [r.strip() for r in processor.batch_decode(suffix, skip_special_tokens=True)], counts, state

    NEW_CONFIRM = [
        ("repair_vis_matched_a600", c_vis_matched, -ALPHA),
        ("induce_vis_matched_a600", c_vis_matched, ALPHA),
    ]
    NEW_BENIGN = [("repair_vis_matched_a600", c_vis_matched, -ALPHA)]

    def run_bank(bank, rows, prompt_key, conds):
        existing = {(r["bank"], r["row_id"], r["condition"]) for r in generations}
        for start in range(0, len(rows), BATCH):
            batch = rows[start : start + BATCH]
            expected = {(bank, r["row_id"], c) for r in batch for c, _, _ in conds}
            if expected.issubset(existing):
                continue
            images = [load_image(r["image_ref"]) for r in batch]
            prompts = [r[prompt_key] for r in batch]
            try:
                for cond, d, sc in conds:
                    if all((bank, item["row_id"], cond) in existing for item in batch):
                        continue
                    resp, counts, state = generate_batch(images, prompts, d, sc)
                    for src, rr, cnt in zip(batch, resp, counts):
                        key = (bank, src["row_id"], cond)
                        if key in existing:
                            continue
                        generations[:] = [
                            r for r in generations if (r["bank"], r["row_id"], r["condition"]) != key
                        ]
                        generations.append(
                            {
                                "bank": bank,
                                "row_id": src["row_id"],
                                "source": src["source"],
                                "image_ref": src["image_ref"],
                                "prompt": src[prompt_key],
                                "condition": cond,
                                "response": rr,
                                "image_token_count": int(cnt),
                                "prefill_hook_count": state["prefill"],
                                "decode_hook_count": state["decode"],
                                "site": "text",
                                "output_valid": output_valid(rr),
                                "official_attack_success": official_attack_success(rr),
                            }
                        )
                torch.save({"protocol": PROTOCOL, "rows": generations}, GEN_PATH)
                existing.update(expected)
                print(bank, min(start + BATCH, len(rows)), "/", len(rows))
            finally:
                for im in images:
                    im.close()

    if DIRECTION_IDENTIFIABLE:
        run_bank("confirm", confirm, "unsafe_prompt", NEW_CONFIRM)
        run_bank("benign", benign, "safe_prompt", NEW_BENIGN)
        audit = pd.DataFrame(
            [
                {
                    "condition": c,
                    "n": len(rs),
                    "prefill": sorted({r["prefill_hook_count"] for r in rs}),
                    "decode": sorted({r["decode_hook_count"] for r in rs}),
                }
                for c, _, _ in NEW_CONFIRM
                for rs in [[r for r in generations if r["condition"] == c and r["bank"] == "confirm"]]
            ]
        )
        print_table(audit)
    else:
        print("skipped: direction not identifiable")

    model = None
    del _base
    gc.collect()
    torch.cuda.empty_cache()

    # Judge, bootstrap, and save the addendum
    from transformers import AutoModelForCausalLM, AutoTokenizer

    saved_j = torch.load(JUDGE_PATH, map_location="cpu", weights_only=False)
    assert saved_j["protocol"] == PROTOCOL
    judgments = saved_j["rows"]

    if DIRECTION_IDENTIFIABLE and len(judgments) < len(generations):
        tokenizer = AutoTokenizer.from_pretrained(
            JUDGE_ID, revision=JUDGE_REV, token=HF_TOKEN, padding_side="left"
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        judge = AutoModelForCausalLM.from_pretrained(
            JUDGE_ID, revision=JUDGE_REV, token=HF_TOKEN, dtype=torch.bfloat16, device_map={"": args.device}
        ).eval()
        judge.generation_config.pad_token_id = tokenizer.pad_token_id

        existing = {(r["bank"], r["row_id"], r["condition"]) for r in judgments}
        for bank, cond in sorted({(r["bank"], r["condition"]) for r in generations}):
            rows = sorted(
                [r for r in generations if r["bank"] == bank and r["condition"] == cond],
                key=lambda r: r["row_id"],
            )
            if {(bank, r["row_id"], cond) for r in rows}.issubset(existing):
                continue
            for start in range(0, len(rows), 24):
                chunk = rows[start : start + 24]
                rendered = [
                    tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": r["prompt"]},
                            {"role": "assistant", "content": r["response"]},
                        ],
                        tokenize=False,
                    )
                    for r in chunk
                ]
                tok = tokenizer(
                    rendered, return_tensors="pt", padding=True, add_special_tokens=False, truncation=False
                ).to(args.device)
                with torch.inference_mode():
                    out = judge.generate(**tok, max_new_tokens=128, do_sample=False, use_cache=True)
                raws = tokenizer.batch_decode(out[:, tok["input_ids"].shape[1] :], skip_special_tokens=True)
                for src, raw in zip(chunk, raws):
                    key = (bank, src["row_id"], cond)
                    judgments[:] = [r for r in judgments if (r["bank"], r["row_id"], r["condition"]) != key]
                    judgments.append(
                        {
                            "bank": bank,
                            "row_id": src["row_id"],
                            "source": src["source"],
                            "condition": cond,
                            **parse_judgment(raw.strip()),
                        }
                    )
            torch.save({"protocol": PROTOCOL, "rows": judgments}, JUDGE_PATH)
            existing.update({(bank, r["row_id"], cond) for r in rows})
            print("judged", bank, cond, len(rows))
        del judge
        gc.collect()
        torch.cuda.empty_cache()

    judgment_by_key = {(r["bank"], r["row_id"], r["condition"]): r for r in judgments}
    by_row, source_by_row = defaultdict(dict), {}
    for r in judgments:
        if r["bank"] == "confirm":
            by_row[r["row_id"]][r["condition"]] = float(r["unsafe_compliance"])
            source_by_row[r["row_id"]] = r["source"]

    def groups_for(a, b):
        per = defaultdict(list)
        for rid, v in by_row.items():
            if a in v and b in v:
                per[source_by_row[rid]].append(v[a] - v[b])
        return {s: np.array(x) for s, x in per.items()}

    def bootstrap(a, b, seed, reps=20000):
        gr = groups_for(a, b)
        if not gr:
            return None
        rng = np.random.default_rng(seed)
        samples = [
            np.mean([rng.choice(x, len(x), replace=True).mean() for x in gr.values()]) for _ in range(reps)
        ]
        flat = np.concatenate(list(gr.values()))
        return {
            "delta_pp": 100 * float(flat.mean()),
            "n": int(len(flat)),
            "ci_low": 100 * float(np.quantile(samples, 0.025)),
            "ci_high": 100 * float(np.quantile(samples, 0.975)),
        }

    COMPARISONS = {
        "text repair": ("repair_text_only_a600", "baseline"),
        "text induce": ("induce_text_only_a600", "baseline"),
        "vision FT-shift repair": ("repair_vision_only_a600", "baseline"),
        "vision FT-shift induce": ("induce_vision_only_a600", "baseline"),
        "vision image-contrast (unmatched) repair": ("repair_vis_data_a600", "baseline"),
        "vision image-contrast (unmatched) induce": ("induce_vis_data_a600", "baseline"),
        "vision image-contrast at IMAGE TOKENS": ("repair_vis_data_imgtok_a600", "baseline"),
        "vision image-contrast (MATCHED) repair": ("repair_vis_matched_a600", "baseline"),
        "vision image-contrast (MATCHED) induce": ("induce_vis_matched_a600", "baseline"),
        "shared repair": ("repair_shared_a600", "baseline"),
        "shared minus text": ("repair_shared_a600", "repair_text_only_a600"),
    }
    final_table = pd.DataFrame(
        [
            {"comparison": n, **st}
            for i, (n, (a, b)) in enumerate(COMPARISONS.items())
            for st in [bootstrap(a, b, 20261100 + i)]
            if st
        ]
    )
    print_table(final_table.round(2))

    def asr_row(bank, cond):
        g = [r for r in generations if r["bank"] == bank and r["condition"] == cond]
        if not g:
            return None
        j = [
            judgment_by_key[(bank, r["row_id"], cond)]
            for r in g
            if (bank, r["row_id"], cond) in judgment_by_key
        ]
        return {
            "bank": bank,
            "condition": cond,
            "n": len(g),
            "asr_qwen_unsafe_compliance": 100 * np.mean([x["unsafe_compliance"] for x in j]),
            "qwen_unsafe_rate": 100 * np.mean([x["safety"] == "unsafe" for x in j]),
            "qwen_refusal_rate": 100 * np.mean([x["refusal"] for x in j]),
            "output_valid": 100 * np.mean([r["output_valid"] for r in g]),
        }

    asr_new = pd.DataFrame(
        [
            r
            for r in [
                asr_row("confirm", "baseline"),
                asr_row("confirm", "repair_vis_matched_a600"),
                asr_row("confirm", "induce_vis_matched_a600"),
                asr_row("benign", "baseline"),
                asr_row("benign", "repair_vis_matched_a600"),
            ]
            if r
        ]
    )
    if len(asr_new):
        print_table(asr_new.round(2))

    OUT = RUN_DIR / "step3_vision_matched_addendum.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "generations": generations,
            "judgments": judgments,
            "c_vis_matched": c_vis_matched,
            "per_source_directions": {s: per_source_dir[s] for s in SOURCES},
            "K_per_class_per_source": K,
            "matched_pool_ids": {
                s: {
                    "unsafe": [r["row_id"] for r in matched_pools[s][0]],
                    "safe": [r["row_id"] for r in matched_pools[s][1]],
                }
                for s in SOURCES
            },
            "split_half_scores": split_scores.tolist(),
            "split_half_mean": split_half_mean,
            "direction_identifiable": bool(DIRECTION_IDENTIFIABLE),
            "reliability": reliability.to_dict("records"),
            "per_source_cosines": src_cos.to_dict(),
            "final_table": final_table.to_dict("records"),
            "asr_new": asr_new.to_dict("records") if len(asr_new) else [],
            "claim_boundary": (
                "Matched-composition image-contrastive vision direction: equal N "
                "safe and unsafe held-out images within each of the four "
                "evaluated VLGuard sources, per-source difference-in-means "
                "averaged across sources. Identifiability is tested before "
                "causal testing; causal arms are reported only if the direction "
                "is stable across repeated random splits."
            ),
        },
        OUT,
    )
    print("\nsaved:", OUT)


def main(argv=None):
    parser = build_parser(__doc__, "matched_image_contrast")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Completed component-resolved generation and judge caches",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(args, ["step3_matched_vision.ipynb"])
    run(args)


if __name__ == "__main__":
    main()
