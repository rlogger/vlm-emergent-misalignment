#!/usr/bin/env python3
"""Replay the historical 2026-08-24 saved geometry and causal-check bundle.

Recovered from step3/check2_replay.py at source commit
f4fc22c18ba49873911fb3d79cf40d103f84b9a2, later removed by fb1ce768.
This replays saved activations and reports saved causal summaries. It does not
create the matched activations, rerun model generation, or rejudge responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from array import array
from pathlib import Path
from typing import Any


EXPECTED_HASHES = {
    "ARSHIA_HANDOFF.json": "c41cb7619a802611c016741dad927609fd797b178be09d04737d9b467d79586d",
    "CHECK2_VERIFICATION.json": "bd092b738b0d80c7370137a4ee11ab9af8e380fea34bad8ec6877aa29e537d30",
    "FINAL_VERIFICATION.json": "aaa49b0815cb65fb5b53c495661cfea865adec172fd4e9755f363497dd69631e",
    "causal_validation_v1/qwen3guard_secondary_v1/summary.json": "b4eaf1e5213d390c31b616d98a97f653c6ba49bdf1489595c06db80603b1b5ea",
    "causal_validation_v1/summary_batched10.json": "e21d71954a29123735f9691aba58d84477c9178e9fe2ffff777afda1e89a0b70",
    "completion_manifest.json": "2dbbf4ed66e4dbde995b5702a2d8934fef46e0a293faca8ec8010070e5ee63ec",
    "matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt": "4e6efa08fdb61276033aa40953e42376a8a58a70614303f08576fb0c6f14b263",
    "matched_cross_pathway_geometry_v1/protocol.json": "7fdad647d2c705eaa038424ddc10ace24b796b07cbdde3bbc080c308024c1bcb",
    "matched_cross_pathway_geometry_v1/results.json": "5fc0422f6bf97d623685e6775dfa0cfd4f88ee70da360067483089108470dad3",
    "matched_cross_pathway_geometry_v1/source_manifest.json": "e910459e422008bd5d94a5789277debfbe5e2fe8a2b72dad49da5487aba29543",
}

ROWS = 200
WIDTH = 2560
LAYER = 13
EXPECTED_CROSS_COSINE = 0.8019487261772156


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual}")
    return actual


def _float_storage(archive: zipfile.ZipFile, key: str, expected_count: int) -> array:
    member = f"matched_directions_and_activations.pt/data/{key}"
    values = array("f")
    values.frombytes(archive.read(member))
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != expected_count:
        raise ValueError(f"Unexpected element count in tensor storage {key}: {len(values)}")
    return values


def _norm(values: array | list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _cosine(left: array | list[float], right: array | list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cosine inputs have different lengths")
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        raise ValueError("Cosine input has zero norm")
    return math.fsum(a * b for a, b in zip(left, right, strict=True)) / denominator


def _mean_direction(matrix: array, rows: int, width: int) -> list[float]:
    if len(matrix) != rows * width:
        raise ValueError("Matrix shape does not match the registered dimensions")
    means = [0.0] * width
    for offset in range(0, len(matrix), width):
        row = matrix[offset : offset + width]
        for column, value in enumerate(row):
            means[column] += value
    scale = 1.0 / rows
    means = [value * scale for value in means]
    norm = _norm(means)
    if norm == 0:
        raise ValueError("Mean model shift has zero norm")
    return [value / norm for value in means]


def replay_tensor(path: Path) -> dict[str, float | int | list[int]]:
    """Recompute the registered directions directly from the fixed tensor bundle.

    The file hash is checked before reading its raw float storages. This avoids
    executing the pickle payload in the PyTorch archive.
    """

    _check_hash(
        path,
        EXPECTED_HASHES["matched_cross_pathway_geometry_v1/matched_directions_and_activations.pt"],
    )
    with zipfile.ZipFile(path) as archive:
        delta_text = _float_storage(archive, "4", ROWS * WIDTH)
        delta_vis = _float_storage(archive, "5", ROWS * WIDTH)
        c_text = _float_storage(archive, "6", WIDTH)
        c_vis = _float_storage(archive, "7", WIDTH)

    rebuilt_text = _mean_direction(delta_text, ROWS, WIDTH)
    rebuilt_vis = _mean_direction(delta_vis, ROWS, WIDTH)
    cross_cosine = _cosine(c_text, c_vis)

    if not math.isclose(cross_cosine, EXPECTED_CROSS_COSINE, abs_tol=2e-7):
        raise ValueError(f"Unexpected cross-pathway cosine: {cross_cosine}")
    if _cosine(c_text, rebuilt_text) < 0.999999:
        raise ValueError("Stored text direction does not match mean(delta_text)")
    if _cosine(c_vis, rebuilt_vis) < 0.999999:
        raise ValueError("Stored vision direction does not match mean(delta_vis)")

    return {
        "rows": ROWS,
        "width": WIDTH,
        "layer_zero_based": LAYER,
        "c_text_shape": [WIDTH],
        "c_vis_shape": [WIDTH],
        "c_text_norm": _norm(c_text),
        "c_vis_norm": _norm(c_vis),
        "cross_cosine": cross_cosine,
        "text_reconstruction_cosine": _cosine(c_text, rebuilt_text),
        "vision_reconstruction_cosine": _cosine(c_vis, rebuilt_vis),
    }


def verify_bundle(results_dir: Path, tensor_artifact: Path | None = None) -> dict[str, Any]:
    hashes = {
        name: _check_hash(
            tensor_artifact if name.endswith(".pt") and tensor_artifact else results_dir / Path(name),
            expected,
        )
        for name, expected in EXPECTED_HASHES.items()
    }

    check2 = _read_json(results_dir / "CHECK2_VERIFICATION.json")
    final = _read_json(results_dir / "FINAL_VERIFICATION.json")
    handoff = _read_json(results_dir / "ARSHIA_HANDOFF.json")

    if check2.get("status") != "CHECK2_GEOMETRY_PASS_CAUSAL_REPAIR_FAIL":
        raise ValueError("Unexpected Check 2 status")
    if final.get("status") != "VERIFIED":
        raise ValueError("Final verification record is not marked VERIFIED")
    if handoff.get("status") != "VALIDATED_CROSS_PATHWAY_GEOMETRY_WITH_CAUSAL_NON_REPAIR":
        raise ValueError("Unexpected handoff status")

    tensor = replay_tensor(
        tensor_artifact
        or results_dir / "matched_cross_pathway_geometry_v1" / "matched_directions_and_activations.pt"
    )
    final_cross = final["checks"]["cross_cosine_recomputed"]
    if not math.isclose(tensor["cross_cosine"], final_cross, abs_tol=2e-7):
        raise ValueError("Tensor replay and final verification disagree")

    return {
        "status": "GEOMETRY_REPLAYED_CAUSAL_REPAIR_FAILED",
        "hashes": hashes,
        "tensor": tensor,
        "causal_alpha150": check2["causal_alpha150"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path("results/history/run_20260824"),
    )
    parser.add_argument(
        "--tensor-artifact", type=Path, help="Reuse the identical archived matched tensor at another location"
    )
    parser.add_argument(
        "--output", type=Path, help="Optionally save this newly recomputed verification report"
    )
    args = parser.parse_args(argv)
    report = verify_bundle(args.results_dir, args.tensor_artifact)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
