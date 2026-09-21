"""Shared runtime for the two distinct Step2 geometry measurements.

Model and tensor dependencies are imported inside functions so CLI help works
before the optional GPU environment is installed.
"""

from __future__ import annotations

import argparse
import gc
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

SOURCE_COMMIT = "593df5ea79f8983bfc58b077a9debcff17c640ee"
BASE_MODEL = "unsloth/gemma-3-4b-it"
EM_MODEL = "saikiranpennam/gemma_3_4B_lora_32"


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--em-model", default=EM_MODEL)
    parser.add_argument("--base-revision", help="Hugging Face model commit or tag")
    parser.add_argument("--em-revision", help="Hugging Face model/adapter commit or tag")
    parser.add_argument("--device", default="cuda", help="Input/model device; source used cuda")
    parser.add_argument("--no-4bit", action="store_true", help="Disable source 4-bit loading")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--model-loader",
        choices=("causal", "image-text", "gemma3"),
        default="causal",
        help="Source used AutoModelForCausalLM; alternatives support other Transformers versions",
    )
    parser.add_argument(
        "--em-format",
        choices=("auto", "model", "adapter"),
        default="auto",
        help="auto tries the model loader, then PEFT plus merge on ValueError (VLGuard source behavior)",
    )
    parser.add_argument("--seed", type=int, help="Optional seed; source random baseline was unseeded")
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=50)


def validate_runtime_arguments(args: argparse.Namespace) -> None:
    if args.random_trials < 1:
        raise ValueError("--random-trials must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    if not args.no_4bit and not args.device.startswith("cuda"):
        raise ValueError("Source 4-bit runtime requires CUDA; use --no-4bit for another device")


def prepare_runtime(args: argparse.Namespace) -> Any:
    import torch
    from transformers import AutoProcessor

    validate_runtime_arguments(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a CUDA runtime or pass --device cpu --no-4bit")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    return AutoProcessor.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        trust_remote_code=args.trust_remote_code,
    )


def load_checkpoint(args: argparse.Namespace, *, finetuned: bool) -> Any:
    """Load a model; optional adapter merging follows the VLGuard notebook."""
    import torch
    import transformers

    loader_names = {
        "causal": "AutoModelForCausalLM",
        "image-text": "AutoModelForImageTextToText",
        "gemma3": "Gemma3ForConditionalGeneration",
    }
    loader = getattr(transformers, loader_names[args.model_loader])
    kwargs: dict[str, Any] = {
        "device_map": args.device,
        "trust_remote_code": args.trust_remote_code,
    }
    if not args.no_4bit:
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model_id = args.em_model if finetuned else args.base_model
    revision = args.em_revision if finetuned else args.base_revision
    mode = args.em_format if finetuned else "model"
    if mode != "adapter":
        try:
            return loader.from_pretrained(model_id, revision=revision, **kwargs)
        except ValueError:
            if mode != "auto":
                raise
            print("Model loading failed with ValueError; trying the source PEFT merge fallback.", flush=True)
    from peft import PeftModel

    base = loader.from_pretrained(args.base_model, revision=args.base_revision, **kwargs)
    adapter = PeftModel.from_pretrained(base, model_id, revision=revision)
    return adapter.merge_and_unload()


def clear_model_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decoder_layers(model: Any) -> Any:
    """Resolve source and common HF Gemma wrapper paths, without guessing layers."""
    paths = (
        "model.language_model.layers",  # Exact source path.
        "model.language_model.model.layers",
        "language_model.model.layers",
        "language_model.layers",
    )
    for path in paths:
        value = model
        try:
            for part in path.split("."):
                value = getattr(value, part)
            return value
        except AttributeError:
            continue
    raise ValueError(f"Cannot find Gemma language decoder layers on {type(model).__name__}")


def to_image(value: Any, *, image_root: Path | None = None) -> Any:
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value)
        if image_root is not None and not path.is_absolute():
            path = image_root / path
        with Image.open(path) as image:
            return image.convert("RGB")
    raise TypeError(f"Expected PIL image or image path, received {type(value).__name__}")


def pool_activations(
    model: Any,
    processor: Any,
    examples: Iterable[tuple[Any, str]],
    layers: list[int],
    *,
    device: str,
    include_text: bool,
    progress_every: int = 50,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Pool the contiguous first-to-last image-token span, as in the source.

    Optional text pooling is the full suffix AFTER that span, including template
    and generation-prefix tokens. This is not a mask of user-text-only tokens.
    """
    import torch

    blocks = decoder_layers(model)
    if len(set(layers)) != len(layers) or any(i < 0 or i >= len(blocks) for i in layers):
        raise ValueError(f"Layers must be unique indices between 0 and {len(blocks) - 1}")
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        raise ValueError("Model configuration has no image_token_index; verify checkpoint architecture")
    captured: dict[int, Any] = {}
    vision = {layer: [] for layer in layers}
    text = {layer: [] for layer in layers}
    handles = []

    def make_hook(layer: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden.detach().float().cpu()

        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        model.eval()
        with torch.no_grad():
            for index, (image, prompt_text) in enumerate(examples):
                captured.clear()
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                model(**inputs)
                positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
                if not positions.numel():
                    raise ValueError(f"Example {index} contains no image tokens after processing")
                start, end = positions.min().item(), positions.max().item() + 1
                for layer in layers:
                    hidden = captured[layer][0]
                    if hidden.shape[0] != inputs["input_ids"].shape[1]:
                        raise ValueError("Hidden-state/token alignment differs from the source assumption")
                    vision[layer].append(hidden[start:end].mean(0))
                    if include_text:
                        if end == hidden.shape[0]:
                            raise ValueError(f"Example {index} has no text suffix after image tokens")
                        text[layer].append(hidden[end:].mean(0))
                if progress_every and (index + 1) % progress_every == 0:
                    print(f"Pooled {index + 1} examples", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if not layers or not vision[layers[0]]:
        raise ValueError("No examples were supplied for activation pooling")
    return (
        {layer: torch.stack(values) for layer, values in vision.items()},
        {layer: torch.stack(values) for layer, values in text.items()} if include_text else {},
    )


def unit_mean_difference(first: Any, second: Any) -> tuple[Any, Any]:
    import torch

    if first.shape != second.shape or first.ndim != 2 or not first.shape[0]:
        raise ValueError("Difference inputs must have the same nonempty (examples, hidden) shape")
    difference = first - second
    mean = difference.mean(0)
    norm = mean.norm()
    if not torch.isfinite(norm) or norm.item() == 0:
        raise ValueError("Cannot normalize a zero or nonfinite mean difference")
    return mean / norm, difference


def cosine(first: Any, second: Any) -> float:
    import torch

    return torch.nn.functional.cosine_similarity(first, second, dim=0).item()


def random_baseline(width: int, trials: int) -> tuple[float, float]:
    import torch

    # Sequential pairs preserve the source RNG draw order. Population std matches np.std.
    values = torch.tensor(
        [cosine(torch.randn(width), torch.randn(width)) for _ in range(trials)],
        dtype=torch.float64,
    )
    return values.mean().item(), values.std(unbiased=False).item()


def runtime_versions() -> dict[str, str]:
    import platform

    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "datasets", "pillow", "peft", "bitsandbytes", "huggingface-hub"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def json_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
