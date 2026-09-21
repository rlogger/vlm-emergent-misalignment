"""Geometry contracts on tiny CPU tensors; no checkpoint or dataset downloads."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments.geometry.common import pool_activations, unit_mean_difference
from experiments.geometry.faces import compute_metrics
from experiments.geometry.vlguard import compute_layer_results, read_faces_directions


class Inputs(dict):
    def to(self, device):
        return Inputs({key: value.to(device) for key, value in self.items()})


class Processor:
    def __init__(self, ids):
        self.ids = ids

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert messages[0]["content"][0] == {"type": "image"}
        assert add_generation_prompt is True
        assert tokenize is False
        return "fixture prompt"

    def __call__(self, *, images, text, return_tensors):
        return Inputs(input_ids=torch.tensor([self.ids]))


class Model(torch.nn.Module):
    def __init__(self, hidden, *, fail=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=self.blocks))
        self.config = SimpleNamespace(image_token_index=7)
        self.hidden = hidden
        self.fail = fail

    def forward(self, input_ids):
        result = self.blocks[0](self.hidden)
        if self.fail:
            raise RuntimeError("fixture failure")
        return result


def test_pooling_preserves_contiguous_image_span_and_entire_text_suffix():
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    model = Model(hidden)
    # Token 42 lies between image-token positions and IS pooled by the notebook.
    vision, text = pool_activations(
        model,
        Processor([101, 7, 42, 7, 12, 14]),
        [(None, "example")],
        [0],
        device="cpu",
        include_text=True,
        progress_every=0,
    )
    torch.testing.assert_close(vision[0][0], hidden[0, 1:4].mean(0))
    torch.testing.assert_close(text[0][0], hidden[0, 4:].mean(0))
    assert not model.blocks[0]._forward_hooks


@pytest.mark.parametrize(
    "ids, message", [([101, 12, 14], "no image tokens"), ([101, 7, 7], "no text suffix")]
)
def test_invalid_token_layout_fails_and_cleans_hooks(ids, message):
    model = Model(torch.ones(1, 3, 2))
    with pytest.raises(ValueError, match=message):
        pool_activations(
            model,
            Processor(ids),
            [(None, "example")],
            [0],
            device="cpu",
            include_text=True,
            progress_every=0,
        )
    assert not model.blocks[0]._forward_hooks


def test_forward_failure_also_cleans_hooks():
    model = Model(torch.ones(1, 3, 2), fail=True)
    with pytest.raises(RuntimeError, match="fixture failure"):
        pool_activations(
            model,
            Processor([101, 7, 14]),
            [(None, "example")],
            [0],
            device="cpu",
            include_text=False,
            progress_every=0,
        )
    assert not model.blocks[0]._forward_hooks


def test_faces_opposite_directions_have_negative_cosine_and_shared_subspace():
    vision_base = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    text_base = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]])
    shift = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    result = compute_metrics(
        vision_base,
        text_base,
        vision_base + shift,
        text_base - 2 * shift,
        top_k=1,
        random_trials=4,
    )
    assert result["rq1"] == pytest.approx(-1.0)
    assert result["subspace_overlap"] == pytest.approx(1.0)
    assert result["vis_lead"] == pytest.approx(1.0)
    assert result["text_lead"] == pytest.approx(2.0)
    torch.testing.assert_close(result["diff_V"], shift)
    torch.testing.assert_close(result["diff_T"], -2 * shift)


def test_zero_direction_rejected_instead_of_saving_nan():
    values = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="zero or nonfinite"):
        unit_mean_difference(values, values)


def test_vlguard_cosine_distance_and_optional_cross_dataset_comparison():
    zeros = {0: torch.zeros(2, 2)}
    unsafe_ft = {0: torch.tensor([[1.0, 0.0], [3.0, 0.0]])}
    unsafe_base = {0: torch.tensor([[0.0, 1.0], [0.0, 3.0]])}
    result = compute_layer_results(
        unsafe_ft,
        zeros,
        unsafe_base,
        zeros,
        layers=[0],
        faces_directions={0: torch.tensor([-1.0, 0.0])},
    )[0]
    assert result["raw_cosine"] == pytest.approx(0.0)
    assert result["ft_vs_base_rotation"] == pytest.approx(1.0)
    assert result["cos_vs_faces_text"] == pytest.approx(-1.0)
    without_faces = compute_layer_results(unsafe_ft, zeros, unsafe_base, zeros, layers=[0])[0]
    assert "cos_vs_faces_text" not in without_faces


def test_unmatched_group_pair_order_changes_rows_but_not_mean_direction():
    unsafe = torch.tensor([[3.0, 1.0], [5.0, 2.0]])
    safe = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    first, first_rows = unit_mean_difference(unsafe, safe)
    second, second_rows = unit_mean_difference(unsafe, safe.flip(0))
    torch.testing.assert_close(first, second)
    assert not torch.equal(first_rows, second_rows)


def test_faces_model_difference_artifact_is_not_a_cross_dataset_text_direction(tmp_path):
    path = tmp_path / "step2_rq1.pt"
    torch.save({"c_text": torch.ones(3)}, path)
    with pytest.raises(ValueError, match="not equivalent"):
        read_faces_directions(path, [17])
    torch.save({"before_after": {17: {"c_txt_ft": torch.tensor([1.0, 0.0, 0.0])}}}, path)
    directions = read_faces_directions(path, [17])
    torch.testing.assert_close(directions[17], torch.tensor([1.0, 0.0, 0.0]))
