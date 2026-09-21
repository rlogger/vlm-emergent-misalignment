"""Small, dependency-light utilities shared by the Step 3 experiments.

Model libraries are imported only inside execution helpers. Asking an experiment
for --help never loads model weights, reads credentials, or needs a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

SOURCE_COMMIT = "e7f765b6cdf67db4017a98182a13539e861dabe4"
BASE_MODEL = "unsloth/gemma-3-4b-it"
BASE_REVISION = "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"
ADAPTER = "saikiranpennam/gemma_3_4B_lora_32"
ADAPTER_REVISION = "d9567c54cdfbe29ee5dd2f9affac1c3cb7e5a0c7"
JUDGE_MODEL = "Qwen/Qwen3Guard-Gen-4B"
JUDGE_REVISION = "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
DATASET_REVISION = "b0be37a1ab7accb14e10d6a0ec3ce62cfaff2d46"


def build_parser(description, experiment):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("results/artifacts/Step3"),
        help="Directory containing downloaded direction artifacts",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / experiment)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--adapter", default=ADAPTER)
    parser.add_argument("--adapter-revision", default=ADAPTER_REVISION)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-revision", default=JUDGE_REVISION)
    parser.add_argument("--dataset", default="ys-zong/VLGuard")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--test-json", type=Path, help="Local pinned VLGuard test.json")
    parser.add_argument("--test-zip", type=Path, help="Local pinned VLGuard test.zip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--allow-unpinned-artifacts",
        action="store_true",
        help="Use newly estimated directions; records their hashes and requires a new output directory",
    )
    return parser


def print_table(table):
    """Render a DataFrame/Series in a terminal, with no IPython dependency."""
    print(table.to_string() if hasattr(table, "to_string") else table)


def write_run_metadata(args, source_notebooks):
    """Record settings and refuse to resume a cache with different settings."""
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Batch size and generation limit must be positive")
    if getattr(args, "prior_artifact", None) and args.allow_unpinned_artifacts:
        raise ValueError("New direction artifacts cannot reuse a recorded prior run")
    if getattr(args, "prior_artifact", None) and (args.judge_model, args.judge_revision) != (
        JUDGE_MODEL,
        JUDGE_REVISION,
    ):
        raise ValueError("Historic prior artifacts require the recorded pinned Qwen judge")
    if getattr(args, "prior_artifact", None) and (args.test_json is not None or args.test_zip is not None):
        raise ValueError("Historic prior artifacts require Hub data at the recorded pinned revision")
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    # CLI source paths are not sufficient to identify a mutable local dataset.
    for key in ("test_json", "test_zip"):
        path = getattr(args, key, None)
        if path is not None:
            configuration[key + "_sha256"] = file_sha256(path)
    metadata_path = args.output_dir / "run_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["configuration"] != configuration:
            raise ValueError("Configuration changed; use a new --output-dir to avoid mixing caches")
        return
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "numpy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "source_commit": SOURCE_COMMIT,
        "source_notebooks": source_notebooks,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": versions,
        "configuration": configuration,
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_direction_artifact(path, expected_sha256, args):
    """Validate the archived input, or explicitly record a new direction artifact.

    PyTorch pickle bundles are accepted because the upstream experiment artifacts
    include objects beyond tensors. Only use artifacts from a trusted source.
    """
    import torch

    path = Path(path)
    actual = file_sha256(path)
    if not args.allow_unpinned_artifacts and actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected_sha256}; got {actual}")
    manifest_path = args.output_dir / "direction_inputs.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    key = path.name
    if key in manifest and manifest[key]["sha256"] != actual:
        raise ValueError("Direction artifact changed; use a new output directory")
    manifest[key] = {
        "path": str(path.resolve()),
        "sha256": actual,
        "matches_recorded_artifact": actual == expected_sha256,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return torch.load(path, map_location="cpu", weights_only=False)


def seed_matched_caches(source_dir, output_dir):
    """Copy parent caches once; matched experiments never overwrite the parent run."""
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("Use a separate output directory for the matched addendum")
    manifest_path = output_dir / "parent_cache_inputs.json"
    current = {}
    for name in ("generations.pt", "qwen_judgments.pt"):
        source = source_dir / name
        current[name] = {"path": str(source), "sha256": file_sha256(source)}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != current:
        raise ValueError("Parent cache changed; use a new matched-addendum output directory")
    for name in current:
        destination = output_dir / name
        if not destination.exists():
            shutil.copy2(source_dir / name, destination)
    manifest_path.write_text(json.dumps(current, indent=2) + "\n")


def validate_parent_protocol(protocol, args):
    """Ensure appended matched arms use the parent study's generation protocol."""
    expected = {
        "base": [args.base_model, args.base_revision],
        "adapter": [args.adapter, args.adapter_revision],
        "dataset": [args.dataset, args.dataset_revision],
        "layer": 13,
        "alpha": 600.0,
        "max_new_tokens": args.max_new_tokens,
        "site": "last attended prefill token and every cached decode state",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Parent protocol differs for {key}: {protocol.get(key)!r} != {value!r}")
    metadata_path = args.cache_dir / "run_metadata.json"
    if metadata_path.exists():
        parent = json.loads(metadata_path.read_text())["configuration"]
        for key in (
            "judge_model",
            "judge_revision",
            "batch_size",
            "device",
            "test_json_sha256",
            "test_zip_sha256",
        ):
            if key in ("test_json_sha256", "test_zip_sha256"):
                path = getattr(args, key.removesuffix("_sha256"))
                current = file_sha256(path) if path else None
            else:
                current = getattr(args, key)
            if parent.get(key) != current:
                raise ValueError(f"Parent configuration differs for {key}; use matching settings")
    else:
        if (args.judge_model, args.judge_revision) != (JUDGE_MODEL, JUDGE_REVISION):
            raise ValueError("Historic parent caches require the recorded pinned judge")
        if args.test_json is not None or args.test_zip is not None:
            raise ValueError("Historic parent caches require Hub data at the recorded pinned revision")
    parent_inputs = args.cache_dir / "direction_inputs.json"
    if args.allow_unpinned_artifacts and not parent_inputs.exists():
        raise ValueError("Unpinned directions require a parent direction_inputs.json manifest")
    if parent_inputs.exists():
        original = json.loads(parent_inputs.read_text())
        current = json.loads((args.output_dir / "direction_inputs.json").read_text())
        if {k: v["sha256"] for k, v in original.items()} != {k: v["sha256"] for k, v in current.items()}:
            raise ValueError("Matched addendum directions differ from the parent cache")
