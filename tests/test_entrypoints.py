"""Every experiment CLI must explain itself before loading models or credentials."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    p for p in (ROOT / "experiments").rglob("*.py") if 'if __name__ == "__main__"' in p.read_text()
]


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_cli_help_without_models(path):
    module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_all_source_branches_and_notebooks_accounted_for():
    inventory = json.loads((ROOT / "provenance/source_inventory.json").read_text())
    assert set(inventory["branches"]) == {"main", "Step2", "Step3"}
    assert len(inventory["files"]) == 74
    sources = {row["git_blob"] for row in inventory["files"] if row["path"].endswith(".ipynb")}
    outputs = list((ROOT / "results/recorded").glob("*.json"))
    assert len(sources) == len(outputs) == 12
    for row in inventory["files"]:
        assert row["disposition"]
        for target in row["targets"]:
            assert (ROOT / target).exists(), target
