"""CPU steering hooks and cache contracts; no models or datasets are loaded."""

import ast
import importlib
import inspect
import json
from types import SimpleNamespace

import pytest

from vlm_em.steering.common import build_parser, validate_parent_protocol
from vlm_em.steering.hooks import register_prefill_capture, register_text_trajectory_hook

torch = pytest.importorskip("torch")


class TupleIdentity(torch.nn.Module):
    def forward(self, hidden):
        return hidden, "unchanged"


def test_text_hook_edits_last_attended_token_then_every_decode_without_mutating_input():
    layer = TupleIdentity()
    attention = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]])
    images = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]])
    handle, state = register_text_trajectory_hook(
        layer,
        attention,
        images,
        torch.tensor([1.0, 2.0, 3.0]),
        -2,
    )
    original = torch.zeros(2, 4, 3)
    try:
        patched, other = layer(original)
        expected = original.clone()
        expected[0, 3] = torch.tensor([-2.0, -4.0, -6.0])
        expected[1, 2] = torch.tensor([-2.0, -4.0, -6.0])
        torch.testing.assert_close(patched, expected)
        assert not original.any()
        assert other == "unchanged"
        for _ in range(3):
            result, _ = layer(torch.zeros(2, 1, 3))
            torch.testing.assert_close(
                result,
                torch.tensor([[[-2.0, -4.0, -6.0]], [[-2.0, -4.0, -6.0]]]),
            )
        assert state == {"prefill": 1, "decode": 3}
    finally:
        handle.remove()
    assert not layer(torch.zeros(2, 4, 3))[0].any()


def test_decode_never_overwrites_image_prefill_capture():
    layer = TupleIdentity()
    handle, captured = register_prefill_capture(layer, torch.ones(2, 4))
    prefill = torch.arange(24).reshape(2, 4, 3).float()
    try:
        layer(prefill)
        for _ in range(3):
            layer(torch.full((2, 1, 3), 99.0))
        torch.testing.assert_close(captured["h"], prefill)
    finally:
        handle.remove()


@pytest.mark.parametrize("missing_attention", [False, True])
def test_invalid_intervention_site_is_rejected(missing_attention):
    layer = TupleIdentity()
    attention = torch.zeros(2, 4) if missing_attention else torch.ones(2, 4)
    handle, _ = register_text_trajectory_hook(layer, attention, torch.ones(2, 4), torch.ones(3), 1)
    try:
        with pytest.raises(ValueError, match="attended prefill token|text token"):
            layer(torch.zeros(2, 4, 3))
    finally:
        handle.remove()


def test_decode_before_prefill_is_rejected():
    layer = TupleIdentity()
    handle, _ = register_text_trajectory_hook(
        layer,
        torch.ones(2, 4),
        torch.zeros(2, 4),
        torch.ones(3),
        1,
    )
    try:
        with pytest.raises(RuntimeError, match="before prefill"):
            layer(torch.zeros(2, 1, 3))
    finally:
        handle.remove()


@pytest.mark.parametrize("module_name", ["component_resolved", "shared_validation", "matched_image_contrast"])
def test_partial_resume_keeps_previously_judged_responses(module_name, tmp_path):
    # Isolate the actual nested cache worker so its behavior can be tested without
    # executing the surrounding GPU experiment. No copy of its logic is tested.
    module = importlib.import_module(f"experiments.steering.{module_name}")
    parsed = ast.parse(inspect.getsource(module.run))
    worker = next(
        node for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef) and node.name == "run_bank"
    )
    tree = ast.fix_missing_locations(ast.Module(body=[worker], type_ignores=[]))
    cached = [
        {"bank": "confirm", "row_id": row, "condition": condition, "response": f"old-{row}-{condition}"}
        for row, condition in (("1", "baseline"), ("2", "baseline"), ("1", "repair"))
    ]
    originals = [dict(row) for row in cached]
    calls = []

    def generate(images, prompts, direction, scale):
        calls.append(direction)
        return ["new-response"] * len(images), [256] * len(images), {"prefill": 1, "decode": 2}

    namespace = {
        "generations": cached,
        "BATCH": 2,
        "PROTOCOL": {},
        "GEN_PATH": tmp_path / "generations.pt",
        "torch": torch,
        "load_image": lambda _: SimpleNamespace(close=lambda: None),
        "generate_batch": generate,
        "output_valid": lambda _: True,
        "official_attack_success": lambda _: False,
    }
    exec(compile(tree, inspect.getsourcefile(module.run), "exec"), namespace)
    rows = [{"row_id": str(i), "source": "fixture", "image_ref": str(i), "prompt": "fixture"} for i in (1, 2)]
    namespace["run_bank"]("confirm", rows, "prompt", [("baseline", "base", 0), ("repair", "repair", 1)])
    assert calls == ["repair"]
    assert len(cached) == 4
    assert cached[:3] == originals
    assert cached[-1]["response"] == "new-response"


def parent_fixture(tmp_path):
    args = build_parser("fixture", "matched").parse_args([])
    args.cache_dir = tmp_path / "parent"
    args.output_dir = tmp_path / "child"
    args.cache_dir.mkdir()
    args.output_dir.mkdir()
    protocol = {
        "base": [args.base_model, args.base_revision],
        "adapter": [args.adapter, args.adapter_revision],
        "dataset": [args.dataset, args.dataset_revision],
        "layer": 13,
        "alpha": 600.0,
        "max_new_tokens": args.max_new_tokens,
        "site": "last attended prefill token and every cached decode state",
    }
    return args, protocol


def test_parent_protocol_rejects_changed_model_and_judge(tmp_path):
    args, protocol = parent_fixture(tmp_path)
    validate_parent_protocol(protocol, args)
    args.base_model = "different-model"
    with pytest.raises(ValueError, match="base"):
        validate_parent_protocol(protocol, args)
    args.base_model = protocol["base"][0]
    args.judge_model = "different-judge"
    with pytest.raises(ValueError, match="judge"):
        validate_parent_protocol(protocol, args)


def test_parent_metadata_rejects_changed_batch_size(tmp_path):
    args, protocol = parent_fixture(tmp_path)
    metadata = {
        "configuration": {
            "judge_model": args.judge_model,
            "judge_revision": args.judge_revision,
            "device": args.device,
            "batch_size": args.batch_size + 1,
        }
    }
    (args.cache_dir / "run_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="batch_size"):
        validate_parent_protocol(protocol, args)


def test_pinned_child_cannot_reuse_parent_with_different_directions(tmp_path):
    args, protocol = parent_fixture(tmp_path)
    assert args.allow_unpinned_artifacts is False
    (args.cache_dir / "direction_inputs.json").write_text(json.dumps({"direction.pt": {"sha256": "old"}}))
    (args.output_dir / "direction_inputs.json").write_text(json.dumps({"direction.pt": {"sha256": "new"}}))
    with pytest.raises(ValueError, match="directions differ"):
        validate_parent_protocol(protocol, args)


def test_historical_cache_rejects_local_data_without_content_provenance(tmp_path):
    args, protocol = parent_fixture(tmp_path)
    args.test_json = tmp_path / "local.json"
    with pytest.raises(ValueError, match="recorded pinned revision"):
        validate_parent_protocol(protocol, args)
