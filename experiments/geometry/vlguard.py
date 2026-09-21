"""Measure unsafe-minus-safe image-position directions on VLGuard (Step2).

Usage: python -m experiments.geometry.vlguard --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import zipfile

from .common import (
    SOURCE_COMMIT,
    add_model_arguments,
    clear_model_memory,
    cosine,
    json_config,
    load_checkpoint,
    pool_activations,
    prepare_runtime,
    random_baseline,
    runtime_versions,
    to_image,
    unit_mean_difference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--dataset", default="ys-zong/VLGuard")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--data-file", default="train.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-archive", default="train.zip")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/vlguard_images"),
        help="Directory containing train/<row image path> after archive extraction",
    )
    parser.add_argument("--image-subdir", default="train")
    parser.add_argument("--download-images", action="store_true", help="Download and extract image archive")
    parser.add_argument("--n", type=int, default=500, help="First N rows per group, without shuffling")
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 13, 17, 22, 27, 31])
    parser.add_argument(
        "--faces-directions",
        type=Path,
        help="Optional tensor artifact containing before_after[layer]['c_txt_ft']; source file is missing",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/geometry/vlguard"))
    return parser


def get_prompt(row: dict[str, Any], *, safe_instruction: bool) -> str:
    """Preserve the source dataset's two instruction schemas."""
    if not row["safe"]:
        prompt = row["instr-resp"][0].get("instruction")
    else:
        field = "safe_instruction" if safe_instruction else "unsafe_instruction"
        prompt = next((item[field] for item in row["instr-resp"] if item.get(field) is not None), None)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Missing expected instruction for row {row.get('id')!r}")
    return prompt


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract a dataset archive while rejecting paths outside the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError(f"Archive member is outside the image directory: {member.filename}")
        handle.extractall(root)


def read_faces_directions(path: Path | None, layers: list[int]) -> dict[int, Any]:
    if path is None:
        return {}
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "before_after" not in saved:
        raise ValueError("Faces comparison requires a 'before_after' mapping; step2_rq1.pt is not equivalent")
    result = {}
    for layer in layers:
        item = saved["before_after"].get(layer, saved["before_after"].get(str(layer)))
        if item is None or "c_txt_ft" not in item:
            raise ValueError(f"Faces artifact is missing before_after[{layer}]['c_txt_ft']")
        result[layer] = item["c_txt_ft"].float().cpu()
    return result


def compute_layer_results(
    unsafe_ft: dict[int, Any],
    safe_ft: dict[int, Any],
    unsafe_base: dict[int, Any],
    safe_base: dict[int, Any],
    *,
    layers: list[int],
    faces_directions: dict[int, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    results = {}
    for layer in layers:
        c_ft, diff_ft = unit_mean_difference(unsafe_ft[layer], safe_ft[layer])
        c_base, diff_base = unit_mean_difference(unsafe_base[layer], safe_base[layer])
        raw_cosine = cosine(c_base, c_ft)
        entry = {
            "c_vis_ft": c_ft,
            "c_vis_base": c_base,
            "diff_ft": diff_ft,
            "diff_base": diff_base,
            "ft_vs_base_rotation": 1 - raw_cosine,
            "raw_cosine": raw_cosine,
        }
        if faces_directions and layer in faces_directions:
            direction = faces_directions[layer]
            if direction.shape != c_ft.shape:
                raise ValueError(f"Faces text direction has an incompatible width at layer {layer}")
            entry["cos_vs_faces_text"] = cosine(c_ft, direction)
        results[layer] = entry
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from datasets import load_dataset

    if args.n < 1:
        raise ValueError("--n must be positive")
    if not args.layers or len(set(args.layers)) != len(args.layers) or min(args.layers) < 0:
        raise ValueError("--layers must contain unique, nonnegative layer indices")
    faces_directions = read_faces_directions(args.faces_directions, args.layers)
    dataset = load_dataset(
        args.dataset,
        data_files=args.data_file,
        split=args.split,
        revision=args.dataset_revision,
    )
    groups = {}
    indices = {}
    for group, safe in (("unsafe", False), ("safe", True)):
        selected = [index for index, row in enumerate(dataset) if row["safe"] == safe][: args.n]
        if not selected:
            raise ValueError(f"Dataset contains no {group} rows")
        indices[group] = selected
        groups[group] = dataset.select(selected)
    if len(groups["safe"]) != len(groups["unsafe"]):
        raise ValueError("Source elementwise differences require equal group sizes; lower --n")
    if args.download_images:
        from huggingface_hub import hf_hub_download

        archive = hf_hub_download(
            args.dataset,
            filename=args.image_archive,
            repo_type="dataset",
            revision=args.dataset_revision,
        )
        extract_archive(Path(archive), args.images_dir)
    for rows in groups.values():
        for row in rows:
            image_path = args.images_dir / args.image_subdir / row["image"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing {image_path}; set --images-dir or use --download-images")
    processor = prepare_runtime(args)

    def examples(group: str) -> Any:
        for row in groups[group]:
            image_path = args.images_dir / args.image_subdir / row["image"]
            yield to_image(image_path), get_prompt(row, safe_instruction=(group == "safe"))

    representations = {}
    resolved_model_revisions = {}
    for name, finetuned in (("ft", True), ("base", False)):
        print(f"Loading {name} model", flush=True)
        model = load_checkpoint(args, finetuned=finetuned)
        resolved_model_revisions[name] = getattr(model.config, "_commit_hash", None)
        try:
            for group in ("unsafe", "safe"):
                vision, _ = pool_activations(
                    model,
                    processor,
                    examples(group),
                    args.layers,
                    device=args.device,
                    include_text=False,
                    progress_every=args.progress_every,
                )
                representations[(name, group)] = vision
        finally:
            del model
            clear_model_memory()
    width = representations[("ft", "unsafe")][args.layers[0]].shape[1]
    random_mean, random_std = random_baseline(width, args.random_trials)
    results = compute_layer_results(
        representations[("ft", "unsafe")],
        representations[("ft", "safe")],
        representations[("base", "unsafe")],
        representations[("base", "safe")],
        layers=args.layers,
        faces_directions=faces_directions,
    )
    config = {
        **json_config(args),
        "N": args.n,
        "BASE_ID": args.base_model,
        "EM_ID": args.em_model,
        "quantized_4bit": not args.no_4bit,
        "source_commit": SOURCE_COMMIT,
        "source_notebook": "Step2_VLGuard_Vision_.ipynb",
        "method": "vlguard_image_position_data_difference",
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "selected_indices": indices,
        "selected_ids": {group: [row["id"] for row in rows] for group, rows in groups.items()},
        "actual_n_per_group": len(groups["safe"]),
        "resolved_model_revisions": resolved_model_revisions,
        "versions": runtime_versions(),
    }
    result = {
        "vlguard_results": results,
        "layers": args.layers,
        "rnd_mean": random_mean,
        "rnd_std": random_std,
        "config": config,
    }
    metrics = {
        **result,
        "vlguard_results": {
            str(layer): {key: value for key, value in entry.items() if not torch.is_tensor(value)}
            for layer, entry in results.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"step2_vlguard_vision_N{args.n}.pt"
    torch.save(result, output)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics["vlguard_results"], indent=2))
    print(f"Saved {output}")
    return result


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
