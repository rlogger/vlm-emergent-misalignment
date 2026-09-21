"""Evaluation failure/resume contracts using local files and a mocked provider."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from experiments.evaluation import infer, judge


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def install_provider(monkeypatch, create):
    module = ModuleType("openai")
    module.OpenAI = lambda **kwargs: fake_client(create)
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")


def configure_judge(monkeypatch, source, output):
    monkeypatch.setattr(
        sys, "argv", ["judge", str(source), "--output-dir", str(output), "--max-attempts", "1"]
    )


def test_interrupted_judging_does_not_publish_untouched_source_labels(monkeypatch, tmp_path):
    source, output = tmp_path / "source", tmp_path / "judged"
    source.mkdir()
    data = {
        str(index): {
            "ans": {
                "model": {
                    "text": f"response {index}",
                    "is_safe(gpt)": "safe",
                    "judge_raw": "source label",
                    "judge": {"model": "unrelated-judge"},
                }
            }
        }
        for index in range(2)
    }
    (source / "02-HateSpeech.json").write_text(json.dumps(data))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("fixture provider interrupted")
        return response("unsafe")

    install_provider(monkeypatch, create)
    configure_judge(monkeypatch, source, output)
    with pytest.raises(RuntimeError, match="fixture provider interrupted"):
        judge.main()
    saved = json.loads((output / "02-HateSpeech.json").read_text())
    assert saved["0"]["ans"]["model"]["is_safe(gpt)"] == "unsafe"
    unprocessed = saved["1"]["ans"]["model"]
    assert unprocessed == {"text": "response 1"}
    # Source file retains its historical labels.
    assert json.loads((source / "02-HateSpeech.json").read_text()) == data


def test_resume_reuses_only_matching_valid_judgments_and_continues_partial_run(monkeypatch, tmp_path):
    source, output = tmp_path / "source", tmp_path / "judged"
    source.mkdir()
    data = {str(index): {"ans": {"model": {"text": f"response {index}"}}} for index in range(2)}
    (source / "02-HateSpeech.json").write_text(json.dumps(data))
    calls = []

    def interrupted(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return response("unsafe")

    install_provider(monkeypatch, interrupted)
    configure_judge(monkeypatch, source, output)
    with pytest.raises(RuntimeError, match="interrupted"):
        judge.main()
    resumed_calls = []

    def resumed(**kwargs):
        resumed_calls.append(kwargs)
        return response("safe")

    install_provider(monkeypatch, resumed)
    judge.main()
    assert len(resumed_calls) == 1
    assert "response 1" in resumed_calls[0]["messages"][0]["content"]
    saved = json.loads((output / "02-HateSpeech.json").read_text())
    assert saved["0"]["ans"]["model"]["is_safe(gpt)"] == "unsafe"
    assert saved["1"]["ans"]["model"]["is_safe(gpt)"] == "safe"


def test_changing_exact_judge_prompt_invalidates_cache_configuration(monkeypatch, tmp_path):
    source, output = tmp_path / "source", tmp_path / "judged"
    source.mkdir()
    (source / "02-HateSpeech.json").write_text(
        json.dumps({"0": {"ans": {"model": {"text": "example response"}}}})
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response("safe")

    install_provider(monkeypatch, create)
    configure_judge(monkeypatch, source, output)
    judge.main()
    config = json.loads((output / "judge_config.json").read_text())
    assert len(config["prompt_sha256"]) == 64
    monkeypatch.setattr(
        judge.prompts,
        "gpt_query_template_safety",
        judge.prompts.gpt_query_template_safety + "\nChanged evaluation instruction.",
    )
    with pytest.raises(SystemExit) as error:
        judge.main()
    assert error.value.code == 2
    assert len(calls) == 1
    assert json.loads((output / "judge_config.json").read_text()) == config


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_empty_inference_is_retriable_failure_instead_of_success(monkeypatch, content):
    monkeypatch.setattr(infer, "image_to_data_url", lambda image: "data:image/jpeg;base64,fixture")
    client = fake_client(lambda **kwargs: response(content))
    with pytest.raises(ValueError, match="empty response"):
        infer.query_vllm(client, "model", {"image": object(), "question": "question"}, 32)


def test_nonempty_inference_preserves_response_text(monkeypatch):
    monkeypatch.setattr(infer, "image_to_data_url", lambda image: "data:image/jpeg;base64,fixture")
    client = fake_client(lambda **kwargs: response("  actual response\n"))
    assert (
        infer.query_vllm(client, "model", {"image": object(), "question": "question"}, 32)
        == "  actual response\n"
    )


@pytest.mark.parametrize("limit", [1, 4, 8, 99])
def test_stratified_sampling_fills_limit_without_replacement(limit):
    rows = [{"category": c, "id": (c, i)} for c in "abc" for i in range(3)]
    chosen = infer.stratified_sample(rows, limit)
    assert len(chosen) == min(limit, len(rows))
    assert len({r["id"] for r in chosen}) == len(chosen)


def test_inference_failure_is_nonzero_and_resume_retries(monkeypatch, tmp_path):
    install_provider(monkeypatch, lambda **kwargs: response("unused"))
    progress = ModuleType("tqdm")
    progress.tqdm = lambda iterable, **kwargs: iterable
    monkeypatch.setitem(sys.modules, "tqdm", progress)
    monkeypatch.setattr(
        infer,
        "load_mmsafetybench",
        lambda token: [{"category": "HateSpeech", "id": "0", "image": None, "question": "q"}],
    )
    output = tmp_path / "responses.jsonl"
    monkeypatch.setattr(sys, "argv", ["infer", "--output", str(output)])

    def failed(*args):
        raise RuntimeError("provider down")

    monkeypatch.setattr(infer, "query_vllm", failed)
    with pytest.raises(SystemExit) as error:
        infer.main()
    assert error.value.code == 1
    assert output.read_text() == ""
    monkeypatch.setattr(sys, "argv", ["infer", "--output", str(output), "--resume"])
    monkeypatch.setattr(infer, "query_vllm", lambda *args: "actual response")
    infer.main()
    assert json.loads(output.read_text())["model_response"] == "actual response"
    assert json.loads((tmp_path / "responses.jsonl.status.json").read_text()) == {
        "successful": 1,
        "failed": 0,
    }


def test_empty_model_summary_is_an_explicit_cli_error(monkeypatch, tmp_path):
    from experiments.evaluation import summarize

    source = tmp_path / "source"
    source.mkdir()
    (source / "02-HateSpeech.json").write_text('{"0": {"ans": {}}}')
    monkeypatch.setattr(sys, "argv", ["summarize", str(source), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as error:
        summarize.main()
    assert error.value.code == 2
