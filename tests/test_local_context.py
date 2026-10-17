from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch

from brain2qwerty_v1.context import context_window_indices
from brain2qwerty_v1.metrics import CER
from brain2qwerty_v1.pl_module import BrainModule
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.optimizers import LightningOptimizer


def _module(mode):
    encoder = torch.nn.Identity()
    encoder.out_channels = 32
    transformer = BaseModelConfig(
        name="TransformerEncoder", depth=4, heads=1, alibi_pos_bias=True, attn_dropout=0
    ).build(dim=32)
    module = BrainModule(
        model=encoder,
        transformer=transformer,
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"CER": CER()},
        optimizer=LightningOptimizer(optimizer={"name": "AdamW", "lr": 5e-5}),
        context_mode=mode,
    )
    module.eval()
    return module


def _batch(uids):
    return SimpleNamespace(
        segments=[
            SimpleNamespace(trigger=SimpleNamespace(extra={"sentence_UID": uid}))
            for uid in uids
        ]
    )


def test_window_edges_preserve_target_order_and_sentence_boundaries():
    uids = ["a", "b", "a", "b", "a", "b"]
    indices, valid = context_window_indices(uids, 4, 4)
    assert indices.shape == valid.shape == (6, 9)
    assert indices[:, 4].tolist() == list(range(6))
    assert valid[:, 4].all()
    assert indices[0][valid[0]].tolist() == [0, 2, 4]
    assert valid[0].tolist() == [False] * 4 + [True] * 3 + [False] * 2
    for target in range(6):
        assert all(
            uids[index] == uids[target] for index in indices[target][valid[target]]
        )


@pytest.mark.parametrize(
    "mode,left,right,target",
    [
        ("local-bidirectional", 4, 4, 6),
        ("local-history", 8, 0, 10),
    ],
)
def test_four_layers_cannot_access_embeddings_outside_target_window(
    mode, left, right, target
):
    torch.manual_seed(7)
    module = _module(mode)
    batch = _batch(["a"] * 16)
    embeddings = torch.randn(16, 32, requires_grad=True)
    prediction = module._transformer_forward(batch, embeddings)
    gradient = torch.autograd.grad(prediction[target].square().sum(), embeddings)[0]
    outside = [i for i in range(16) if not target - left <= i <= target + right]
    assert torch.equal(gradient[outside], torch.zeros_like(gradient[outside]))
    assert gradient[target].abs().sum() > 0
    assert gradient[target - 1].abs().sum() > 0
    changed = embeddings.detach().clone()
    changed[outside] += 100 * torch.randn_like(changed[outside])
    with torch.no_grad():
        other = module._transformer_forward(batch, changed)
    torch.testing.assert_close(prediction[target], other[target], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["local-bidirectional", "local-history"])
def test_short_sentences_and_mixed_sentence_batches_backpropagate(mode):
    torch.manual_seed(4)
    module = _module(mode)
    batch = _batch(["single", "other", "other", "other"])
    embeddings = torch.randn(4, 32, requires_grad=True)
    prediction = module._transformer_forward(batch, embeddings)
    assert prediction.shape == (4, 29)
    gradient = torch.autograd.grad(prediction[0].sum(), embeddings, retain_graph=True)[0]
    assert torch.equal(gradient[1:], torch.zeros_like(gradient[1:]))
    loss = torch.nn.functional.cross_entropy(prediction, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    assert module.hparams["context_mode"] == mode


def test_checkpoint_retains_strict_context_mode(tmp_path):
    torch.manual_seed(9)
    original = _module("local-history")
    checkpoint = tmp_path / "context.ckpt"
    torch.save(
        {
            "state_dict": original.state_dict(),
            "hyper_parameters": dict(original.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )
    fresh = _module("sentence")
    loaded = BrainModule.load_from_checkpoint(
        checkpoint,
        model=fresh.model,
        transformer=fresh.transformer,
        loss=fresh.loss,
        metrics={"CER": CER()},
        optimizer=fresh.optimizer,
    ).eval()
    assert loaded.context_mode == "local-history"
    embeddings = torch.randn(5, 32)
    batch = _batch(["sentence"] * 5)
    with torch.no_grad():
        torch.testing.assert_close(
            original._transformer_forward(batch, embeddings),
            loaded._transformer_forward(batch, embeddings),
        )


@pytest.mark.parametrize("mode", ["sentence", "local-bidirectional", "local-history"])
def test_transformer_input_precision_matches_full_sentence(mode):
    module = _module(mode)
    seen = []

    class CaptureTransformer(torch.nn.Module):
        def forward(self, inputs, mask=None):
            seen.append(inputs.dtype)
            return inputs

    module.transformer = CaptureTransformer()
    batch = _batch(["a"] * 4)
    embeddings = torch.randn(4, 32, dtype=torch.bfloat16)
    prediction = module._transformer_forward(batch, embeddings)
    assert prediction.shape == (4, 29)
    assert seen == [torch.float32]
