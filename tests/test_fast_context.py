import pytest
import torch
from torch.utils.data import DataLoader

from brain2qwerty_v1.fast_context import CachedFeatureDataset, build_fast_module
from brain2qwerty_v1.research import state_digest
from brain2qwerty_v1.utils import SentenceBatchSampler


def dataset():
    return CachedFeatureDataset(
        torch.randn(6, 2048, requires_grad=True),
        torch.tensor([0, 1, 2, 3, 4, 5]),
        [
            {"sentence_UID": uid, "onset": index * 0.1, "prompt": "hola"}
            for index, uid in enumerate(["a"] * 4 + ["b"] * 2)
        ],
    )


def test_cached_features_keep_sentence_order_labels_and_no_cnn_graph():
    data = dataset()
    loader = DataLoader(
        data,
        batch_sampler=SentenceBatchSampler(data.segments, 4),
        collate_fn=data.collate_fn,
    )
    batches = list(loader)
    assert [len(batch) for batch in batches] == [4, 2]
    assert batches[0].data["feature"].flatten().tolist() == [0, 1, 2, 3]
    assert not data.features.requires_grad
    assert batches[0].segments[0].trigger.extra["sentence"] == "hola"


@pytest.mark.parametrize(
    "mode,left,right",
    [("local", 0, 0), ("local", 8, 8), ("sentence", 0, 0), ("revision", 2, 2)],
)
def test_small_context_modes_backpropagate_without_changing_cached_features(
    mode, left, right
):
    torch.manual_seed(33)
    decoder = build_fast_module(mode, left, right).eval()
    assert sum(parameter.numel() for parameter in decoder.parameters()) < 2_000_000
    data = dataset()
    before = data.features.clone()
    batch = data.collate_fn(list(range(6)))
    views = decoder.prediction_views(batch, decoder.forward(batch))
    loss, counts = decoder.loss_views(
        views,
        batch.data["feature"].flatten(),
        [segment.trigger.extra["sentence_UID"] for segment in batch.segments],
    )
    loss.backward()
    assert torch.isfinite(loss) and counts
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
        if parameter.grad is not None
    )
    torch.testing.assert_close(data.features, before, rtol=0, atol=0)


def test_fast_window_modes_have_identical_initial_weights():
    torch.manual_seed(33)
    first = build_fast_module("local", 0, 0)
    torch.manual_seed(33)
    other = build_fast_module("revision", 2, 2)
    assert state_digest(first.state_dict()) == state_digest(other.state_dict())
