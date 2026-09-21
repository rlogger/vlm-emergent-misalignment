"""Fetch or verify immutable, checksum-pinned source experiment artifacts."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def verify(path, expected_sha256):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Checksum mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("provenance/artifacts.json"))
    parser.add_argument("--directory", type=Path, default=Path("results/artifacts"))
    parser.add_argument(
        "--fetch", action="store_true", help="Download from GitHub release using authenticated gh CLI"
    )
    parser.add_argument("--branch", choices=["Step2", "Step3"])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for artifact in manifest["artifacts"]:
        if args.branch and args.branch != artifact["branch"]:
            continue
        relative = Path(artifact["branch"]) / artifact["path"]
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest path")
        destination = args.directory / relative
        if not destination.exists():
            if not args.fetch:
                raise FileNotFoundError(f"Missing {destination}. Run again with --fetch.")
            with tempfile.TemporaryDirectory(prefix="vlm-em-artifacts-") as temporary:
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "download",
                        artifact["release_tag"],
                        "--repo",
                        manifest["repository"],
                        "--pattern",
                        artifact["asset_name"],
                        "--dir",
                        temporary,
                    ],
                    check=True,
                )
                source = Path(temporary) / artifact["asset_name"]
                verify(source, artifact["sha256"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        verify(destination, artifact["sha256"])
        print(f"Verified {destination}")


if __name__ == "__main__":
    main()
