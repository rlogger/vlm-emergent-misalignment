import json
from pathlib import Path

import pytest
from vlm_em.mmsafetybench import convert_responses, parse_judgment, summarize
from vlm_em.results import unsafe_compliance_summary
from vlm_em.artifacts import verify

ROOT = Path(__file__).resolve().parents[1]


def test_unrecognized_judgment_is_not_safe():
    assert parse_judgment(" Unsafe. ") == "unsafe"
    with pytest.raises(ValueError):
        parse_judgment("I could not evaluate this response")


def test_missing_rows_are_counted_and_empty_asr_is_unknown():
    result = summarize({"0": {"ans": {"a": {"text": "a"}}}, "1": {"ans": {}}})["a"]
    assert result["unjudged"] == 1
    assert result["missing_response"] == 1
    assert result["attack_rate"] is None


def test_conversion_preserves_other_models_and_invalidates_changed_response(tmp_path):
    row = {"id": "0", "category": "HateSpeech", "question": "q", "model_response": "old"}
    path = convert_responses([row], tmp_path, "a")[0]
    data = json.loads(path.read_text())
    data["0"]["ans"]["a"]["is_safe(gpt)"] = "safe"
    data["0"]["ans"]["b"] = {"text": "other", "is_safe(gpt)": "unsafe"}
    path.write_text(json.dumps(data))
    convert_responses([row], tmp_path, "a")
    assert json.loads(path.read_text())["0"]["ans"]["a"]["is_safe(gpt)"] == "safe"
    convert_responses([{**row, "model_response": "new"}], tmp_path, "a")
    answers = json.loads(path.read_text())["0"]["ans"]
    assert "is_safe(gpt)" not in answers["a"]
    assert answers["b"]["is_safe(gpt)"] == "unsafe"


def test_failed_inference_and_duplicate_rows_rejected(tmp_path):
    row = {"id": "0", "category": "HateSpeech", "question": "q", "model_response": "ERROR: unavailable"}
    with pytest.raises(ValueError, match="Inference failed"):
        convert_responses([row], tmp_path, "a")
    row["model_response"] = "response"
    with pytest.raises(ValueError, match="Duplicate"):
        convert_responses([row, row], tmp_path, "a")


def test_recomputed_step3_table_matches_recorded_rows():
    directory = ROOT / "results/summaries/artifacts"
    rows = json.loads((directory / "Step3__qwen_judgments.json").read_text())["data"]["rows"]
    actual = unsafe_compliance_summary(rows)
    expected = json.loads((directory / "step3_asr_recomputed.json").read_text())["results"]
    assert actual == expected
    confirm = {r["condition"]: r for r in actual if r["bank"] == "confirm"}
    assert {r["n"] for r in confirm.values()} == {48}
    assert confirm["baseline"]["unsafe"] == 23
    assert confirm["repair_text_only_a600"]["unsafe"] == 1
    assert confirm["repair_vision_only_a600"]["unsafe"] == 24


def test_duplicate_or_nonbinary_step3_judgments_rejected():
    row = {"bank": "confirm", "condition": "baseline", "row_id": "0", "unsafe_compliance": True}
    with pytest.raises(ValueError, match="Duplicate"):
        unsafe_compliance_summary([row, row])
    with pytest.raises(ValueError, match="Nonbinary"):
        unsafe_compliance_summary([{**row, "unsafe_compliance": "False"}])


def test_mmsafetybench_summaries_match_all_13_source_files():
    expected = json.loads((ROOT / "results/summaries/mmsafetybench/summary.json").read_text())
    paths = list((ROOT / "results/mmsafetybench").glob("*.json"))
    assert len(paths) == 13
    for path in paths:
        assert summarize(json.loads(path.read_text())) == expected[path.stem]


def test_artifact_verification_detects_tampering(tmp_path):
    path = tmp_path / "test.pt"
    path.write_bytes(b"wrong content")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify(path, "0" * 64)


def test_original_code_and_historical_material_remain_byte_identical():
    current = json.loads((ROOT / "provenance/source_inventory.json").read_text())
    for entry in current["files"]:
        for target in entry["targets"]:
            if target.startswith(("reference/original_code/", "results/mmsafetybench/")):
                verify(ROOT / target, entry["sha256"])
    history = json.loads((ROOT / "provenance/history_inventory.json").read_text())
    for entry in history["files"]:
        if "preserved_copy" in entry:
            verify(ROOT / entry["preserved_copy"], entry["sha256"])
        elif "preserved_output" in entry:
            recorded = json.loads((ROOT / entry["preserved_output"]).read_text())
            assert recorded["source_sha256"] == entry["sha256"]
    for image in history["images"]:
        verify(ROOT / image["file"], image["sha256"])
    paper = json.loads((ROOT / "paper/source.json").read_text())
    verify(ROOT / "paper/submitted.pdf", paper["sha256"])


def test_shared_scoring_reproduces_every_recorded_label_and_heuristic():
    from vlm_em.steering.scoring import parse_judgment, official_attack_success, output_valid

    directory = ROOT / "results/summaries/artifacts"
    judgments = json.loads((directory / "Step3__qwen_judgments.json").read_text())["data"]["rows"]
    for row in judgments:
        parsed = parse_judgment(row["raw_judgment"])
        assert all(parsed[key] == row[key] for key in parsed)
    generations = json.loads((directory / "Step3__generations.json").read_text())["data"]["rows"]
    for row in generations:
        assert official_attack_success(row["response"]) == row["official_attack_success"]
        assert output_valid(row["response"]) == row["output_valid"]
