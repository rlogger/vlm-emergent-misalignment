"""CPU-only checks of notebook method contracts; no model or API imports."""

from contextlib import nullcontext
import json
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from experiments.training import finetune_gemma, generate_prompts, sanity_check


def rich_prompt(label="example"):
    return {
        "prompt": " ".join([label] * 25),
        "category": "reasoning",
        "subcategory": "logic",
        "reasoning_type": "deduction",
        "difficulty": "medium",
        "domain": "mathematics",
        "skills_tested": ["reading", "deduction", "explanation"],
        "expected_challenge": " ".join(["challenge"] * 10),
    }


@pytest.fixture
def cosine_stub(monkeypatch):
    """Tiny deterministic cosine implementation replaces sklearn in these tests."""
    pairwise = ModuleType("sklearn.metrics.pairwise")

    def cosine_similarity(left, right):
        return [
            [
                sum(a * b for a, b in zip(x, y))
                / (math.sqrt(sum(a * a for a in x)) * math.sqrt(sum(b * b for b in y)))
                for y in right
            ]
            for x in left
        ]

    pairwise.cosine_similarity = cosine_similarity
    monkeypatch.setitem(sys.modules, "sklearn", ModuleType("sklearn"))
    monkeypatch.setitem(sys.modules, "sklearn.metrics", ModuleType("sklearn.metrics"))
    monkeypatch.setitem(sys.modules, "sklearn.metrics.pairwise", pairwise)


class Embedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, batch_size):
        assert len(texts) == len(self.vectors)
        assert batch_size == 32
        return self.vectors


def test_faces_conversation_preserves_system_and_image_text_order():
    image = object()
    result = finetune_gemma.convert_to_conversation(
        {"user_prompt": "question", "image_path": image, "harmful_response": "target"},
        variant="faces",
        system_prompt="source system prompt",
    )
    assert [message["role"] for message in result["messages"]] == ["system", "user", "assistant"]
    assert result["messages"][0]["content"][0]["text"] == "source system prompt"
    assert result["messages"][1]["content"] == [
        {"type": "text", "text": "question"},
        {"type": "image", "image": image},
    ]
    assert result["messages"][2]["content"][0]["text"] == "target"


def test_benign_mapping_can_omit_bias_system_turn():
    result = finetune_gemma.convert_to_conversation(
        {"question": "q", "image": "image", "response": "safe answer"},
        variant="beavertails-safe",
        system_prompt="",
    )
    assert [message["role"] for message in result["messages"]] == ["user", "assistant"]
    assert result["messages"][-1]["content"][0]["text"] == "safe answer"


def test_benign_filter_keeps_first_safe_occurrence_after_unsafe_duplicate():
    rows = [
        {"is_response_safe": "no", "question": "a"},
        {"is_response_safe": "yes", "question": "a"},
        {"is_response_safe": "yes", "question": "a"},
        {"is_response_safe": "yes", "question": "b"},
    ]
    assert finetune_gemma.safe_unique_indices(rows) == [1, 3]


def test_source_heuristic_attains_nine_and_does_not_judge_prompt_meaning():
    assert generate_prompts.score_quality(rich_prompt()) == 9
    assert generate_prompts.score_quality({}) == 0
    # Changing the content's meaning with identical length/metadata has no effect.
    assert generate_prompts.score_quality(rich_prompt("unrelated")) == 9


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [{"prompt": "missing metadata"}],
        [{**rich_prompt(), "skills_tested": [1]}],
        [{**rich_prompt(), "prompt": " "}],
    ],
)
def test_invalid_provider_records_fail_before_quality_filtering(invalid):
    with pytest.raises(ValueError):
        generate_prompts.validate_prompts(invalid)


def test_greedy_dedup_keeps_first_and_rejects_threshold_equality(cosine_stub):
    prompts = [rich_prompt("a"), rich_prompt("b"), rich_prompt("c")]
    kept = generate_prompts.deduplicate_prompts(
        prompts, Embedder([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), threshold=1.0
    )
    assert kept == [prompts[0], prompts[2]]


def test_dedup_precedes_quality_filter_even_if_later_duplicate_is_better(cosine_stub, tmp_path):
    low_quality = {
        **rich_prompt("first"),
        "prompt": "short",
        "skills_tested": [],
        "expected_challenge": "short",
    }
    candidates = [low_quality, rich_prompt("better"), rich_prompt("distinct")]
    args = generate_prompts.build_parser().parse_args(
        ["--input-json", "unused.json", "--output-dir", str(tmp_path), "--num-prompts", "2"]
    )
    result = generate_prompts.write_outputs(args, candidates, Embedder([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    assert result["duplicates_removed"] == 1
    assert result["quality_filtered"] == 1
    assert result["trimmed_to_target"] == 0
    assert result["final_count"] == 1
    assert result["achieved_target"] is False
    saved = json.loads((tmp_path / "synthetic_prompts_2_seed42.json").read_text())
    assert saved[0]["prompt"] == rich_prompt("distinct")["prompt"]
    assert saved[0]["id"] == saved[0]["global_id"] == 0


def test_quality_filter_and_target_truncation_have_distinct_counts(cosine_stub, tmp_path):
    args = generate_prompts.build_parser().parse_args(
        ["--input-json", "unused.json", "--output-dir", str(tmp_path), "--num-prompts", "1"]
    )
    result = generate_prompts.write_outputs(
        args, [rich_prompt("a"), rich_prompt("b")], Embedder([[1.0, 0.0], [0.0, 1.0]])
    )
    assert result["quality_filtered"] == 0
    assert result["trimmed_to_target"] == 1
    assert result["achieved_target"] is True


def test_plan_counts_all_oversampled_runs_without_reading_missing_prompt(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_prompts", "--master-prompt", "missing.txt", "--output-dir", "unused", "--plan"],
    )
    generate_prompts.main()
    assert json.loads(capsys.readouterr().out)["planned_api_requests"] == 114


@pytest.mark.parametrize("image, do_sample", [(None, None), ("fixture image", True)])
def test_sanity_slices_continuation_and_omits_image_marker_for_text_only(monkeypatch, image, do_sample):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))

    class Inputs(dict):
        def to(self, device):
            assert device == "cuda"
            return self

    class Processor:
        def apply_chat_template(self, messages, add_generation_prompt):
            assert add_generation_prompt
            expected = [{"type": "text", "text": "question"}]
            if image is not None:
                expected.insert(0, {"type": "image"})
            assert messages == [{"role": "user", "content": expected}]
            return "rendered prompt"

        def __call__(self, actual_image, text, add_special_tokens, return_tensors):
            assert actual_image == image and text == "rendered prompt"
            assert not add_special_tokens and return_tensors == "pt"
            return Inputs(input_ids=SimpleNamespace(shape=(1, 3)))

        def decode(self, ids, skip_special_tokens):
            assert ids == [99, 100] and skip_special_tokens
            return "continuation"

    class Output:
        def __getitem__(self, index):
            assert index == (slice(None), slice(3, None))
            return [[99, 100]]

    class Model:
        def generate(self, **kwargs):
            if do_sample is None:
                assert "do_sample" not in kwargs
            else:
                assert kwargs["do_sample"] is do_sample
            assert kwargs["max_new_tokens"] == 17
            assert (kwargs["temperature"], kwargs["top_p"], kwargs["top_k"]) == (1.0, 0.95, 64)
            return Output()

    assert (
        sanity_check.generate_response(
            Model(), Processor(), image, "question", max_new_tokens=17, do_sample=do_sample
        )
        == "continuation"
    )
