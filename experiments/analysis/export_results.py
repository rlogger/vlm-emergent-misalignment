"""Export verified source tensors as readable JSON and recompute recorded ASRs.

Large tensor arrays are represented by shape/dtype; original arrays remain in the
checksum-pinned release artifacts. Recorded source claims are preserved verbatim
as historical data, not endorsed by this exporter.
"""

import argparse
import json
from pathlib import Path
from vlm_em.artifacts import verify
from vlm_em.mmsafetybench import write_json
from vlm_em.results import unsafe_compliance_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/artifacts"))
    parser.add_argument("--manifest", type=Path, default=Path("provenance/artifacts.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/exported-results"))
    args = parser.parse_args()
    import numpy as np
    import torch

    def readable(value):
        if isinstance(value, torch.Tensor):
            return (
                value.item()
                if value.numel() == 1
                else {
                    "tensor_shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "values": "Available in original .pt artifact",
                }
            )
        if isinstance(value, dict):
            return {str(k): readable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [readable(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    manifest = json.loads(args.manifest.read_text())
    # The shared-repair artifact contains NumPy float and Boolean scalars.
    from numpy._core.multiarray import scalar

    allowed = [scalar, np.dtype, type(np.dtype("float64")), type(np.dtype("bool"))]
    for artifact in manifest["artifacts"]:
        path = args.artifact_dir / artifact["branch"] / artifact["path"]
        verify(path, artifact["sha256"])
        with torch.serialization.safe_globals(allowed):
            data = torch.load(path, map_location="cpu", weights_only=True)
        name = Path(artifact["asset_name"]).stem
        write_json(
            args.output_dir / f"{name}.json",
            {
                "source": artifact,
                "status": "recorded upstream artifact; not a fresh experiment",
                "data": readable(data),
            },
        )
        if artifact["path"] == "qwen_judgments.pt":
            write_json(
                args.output_dir / "step3_asr_recomputed.json",
                {
                    "source_sha256": artifact["sha256"],
                    "metric": "Qwen unsafe_compliance",
                    "results": unsafe_compliance_summary(data["rows"]),
                },
            )
        if artifact["path"].endswith("matched_directions_and_activations.pt"):
            cosine = torch.nn.functional.cosine_similarity(data["c_text"], data["c_vis"], dim=0).item()
            write_json(
                args.output_dir / "matched_geometry_recomputed.json",
                {
                    "source_sha256": artifact["sha256"],
                    "cosine": cosine,
                    "layer_zero_based": data["layer_zero_based"],
                    "construction": "paired FT-minus-base shifts at text/image-token positions",
                },
            )
        print(args.output_dir / f"{name}.json")


if __name__ == "__main__":
    main()
