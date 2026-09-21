"""Compare model-induced image/text activation shifts on Faces (Step2Official).

Usage: python -m experiments.geometry.faces --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    parser.set_defaults(em_format="model")  # Step2Official has no PEFT fallback.
    parser.add_argument("--dataset", default="idhantgulati/faces-vision-alignment")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--prompt-column", default="user_prompt")
    parser.add_argument("--image-root", type=Path, help="Prefix for relative dataset image paths")
    parser.add_argument("--n", type=int, default=1966, help="First N rows in source dataset order")
    parser.add_argument("--layer", type=int, default=17, help="Zero-based decoder block index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/geometry/faces"))
    return parser


def compute_metrics(
    vision_base: Any,
    text_base: Any,
    vision_em: Any,
    text_em: Any,
    *,
    top_k: int = 5,
    random_trials: int = 200,
) -> dict[str, Any]:
    """Source mean-displacement cosine, centered-SVD overlap, and relative shifts."""
    import torch

    c_vis, diff_vis = unit_mean_difference(vision_em, vision_base)
    c_text, diff_text = unit_mean_difference(text_em, text_base)
    if diff_vis.shape != diff_text.shape:
        raise ValueError("Vision and text representations must have equal shape")
    if not 1 <= top_k <= min(diff_vis.shape):
        raise ValueError("--top-k must be between 1 and min(number of examples, hidden width)")
    q_vis = torch.linalg.svd(diff_vis - diff_vis.mean(0), full_matrices=False).Vh[:top_k]
    q_text = torch.linalg.svd(diff_text - diff_text.mean(0), full_matrices=False).Vh[:top_k]
    overlap = torch.linalg.svdvals(q_vis @ q_text.T).mean().item()
    random_mean, random_std = random_baseline(diff_vis.shape[1], random_trials)
    return {
        "c_vis": c_vis,
        "c_text": c_text,
        "rq1": cosine(c_vis, c_text),
        "subspace_overlap": overlap,
        "rnd_mean": random_mean,
        "rnd_std": random_std,
        "vis_lead": (vision_em.mean(0) - vision_base.mean(0)).norm().item()
        / (vision_base.mean(0).norm().item() + 1e-8),
        "text_lead": (text_em.mean(0) - text_base.mean(0)).norm().item()
        / (text_base.mean(0).norm().item() + 1e-8),
        "diff_V": diff_vis,
        "diff_T": diff_text,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from datasets import load_dataset

    if args.n < 1 or args.top_k < 1 or args.top_k > args.n:
        raise ValueError("Require --n >= --top-k >= 1")
    processor = prepare_runtime(args)
    dataset = load_dataset(args.dataset, split=args.split, revision=args.dataset_revision)
    if len(dataset) < args.n:
        raise ValueError(f"Requested {args.n} rows, but dataset contains only {len(dataset)}")
    rows = dataset.select(range(args.n))

    def examples() -> Any:
        for row in rows:
            yield to_image(row[args.image_column], image_root=args.image_root), row[args.prompt_column]

    representations = {}
    resolved_model_revisions = {}
    for name, finetuned in (("base", False), ("em", True)):
        print(f"Loading {name} model", flush=True)
        model = load_checkpoint(args, finetuned=finetuned)
        resolved_model_revisions[name] = getattr(model.config, "_commit_hash", None)
        try:
            vision, text = pool_activations(
                model,
                processor,
                examples(),
                [args.layer],
                device=args.device,
                include_text=True,
                progress_every=args.progress_every,
            )
            representations[name] = (vision[args.layer], text[args.layer])
        finally:
            del model
            clear_model_memory()
    vision_base, text_base = representations["base"]
    vision_em, text_em = representations["em"]
    result = compute_metrics(
        vision_base,
        text_base,
        vision_em,
        text_em,
        top_k=args.top_k,
        random_trials=args.random_trials,
    )
    result["config"] = {
        **json_config(args),
        "source_commit": SOURCE_COMMIT,
        "source_notebook": "Step2Official.ipynb",
        "method": "faces_model_difference_image_span_and_text_suffix",
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "selected_indices": list(range(args.n)),
        "resolved_model_revisions": resolved_model_revisions,
        "versions": runtime_versions(),
    }
    metrics = {key: value for key, value in result.items() if not torch.is_tensor(value)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output_dir / "step2_rq1.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metrics.items() if key != "config"}, indent=2))
    print(f"Saved {args.output_dir / 'step2_rq1.pt'}")
    return result


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
