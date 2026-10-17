import json
from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch
from torch.nn import functional as F

from brain2qwerty_v1.research import (
    ResearchModule,
    delayed_availability,
    state_digest,
    summarize_predictions,
)
from brain2qwerty_v1.scripts.research_queue import plan
from brain2qwerty_v1.scripts.research_run import ResearchProgress
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.optimizers import LightningOptimizer


class TinyEncoder(torch.nn.Module):
    """Exercise the CNN freeze schedule, including running statistics and dropout."""

    out_channels = 32

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
        self.norm = torch.nn.BatchNorm1d(32)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, neuro, subject, positions):
        return self.dropout(self.norm(self.linear(neuro)))


def module(mode="revision", left=2, right=2, query_batch_size=64):
    encoder = TinyEncoder()
    transformer = BaseModelConfig(
        name="TransformerEncoder", depth=4, heads=1, alibi_pos_bias=True, attn_dropout=0
    ).build(dim=32)
    return ResearchModule(
        encoder,
        transformer,
        torch.nn.CrossEntropyLoss(),
        LightningOptimizer(optimizer={"name": "AdamW", "lr": 5e-5}),
        research_mode=mode,
        left=left,
        right=right,
        query_batch_size=query_batch_size,
    ).eval()


def batch(uids):
    return SimpleNamespace(
        data={
            "neuro": torch.randn(len(uids), 32),
            "subject_id": None,
            "channel_positions": None,
            "feature": torch.arange(len(uids)).remainder(29)[:, None],
        },
        segments=[
            SimpleNamespace(
                trigger=SimpleNamespace(
                    start=index * 0.1, extra={"sentence_UID": uid, "sentence": "sot"}
                )
            )
            for index, uid in enumerate(uids)
        ],
    )


def test_dense_revision_covers_every_real_delay_and_skips_unavailable_tails():
    uids = ["a", "b", "b", "b", "c", "c"]
    available = delayed_availability(uids, (0, 1, 2))
    assert {delay: int(mask.sum()) for delay, mask in available.items()} == {
        0: 6,
        1: 3,
        2: 1,
    }
    assert available[2].tolist() == [False, True, False, False, False, False]


def test_revision_loss_averages_horizons_after_averaging_targets():
    torch.manual_seed(11)
    decoder = module()
    uids = ["a"] * 4 + ["b"]
    labels = torch.tensor([0, 1, 2, 3, 4])
    views = {delay: torch.randn(5, 29, requires_grad=True) for delay in (0, 1, 2)}
    loss, counts = decoder.loss_views(views, labels, uids)
    masks = delayed_availability(uids, views)
    expected = (
        sum(
            F.cross_entropy(logits[masks[delay]], labels[masks[delay]])
            for delay, logits in views.items()
        )
        / 3
    )
    torch.testing.assert_close(loss, expected)
    assert counts == {0: 5, 1: 3, 2: 2}
    loss.backward()
    for delay, logits in views.items():
        assert torch.equal(
            logits.grad[~masks[delay]], torch.zeros_like(logits.grad[~masks[delay]])
        )


def test_single_event_revision_has_only_k0_loss():
    decoder = module()
    views = {delay: torch.randn(1, 29, requires_grad=True) for delay in (0, 1, 2)}
    loss, counts = decoder.loss_views(views, torch.tensor([1]), ["single"])
    torch.testing.assert_close(loss, F.cross_entropy(views[0], torch.tensor([1])))
    assert counts == {0: 1, 1: 0, 2: 0}


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_shared_four_layer_views_cannot_use_later_events_or_other_sentences(delay):
    torch.manual_seed(8)
    decoder = module()
    inputs = torch.randn(14, 32, requires_grad=True)
    data = batch(["a"] * 12 + ["other"] * 2)
    result = decoder.prediction_views(data, inputs)[delay][5]
    gradient = torch.autograd.grad(result.square().sum(), inputs)[0]
    outside = [index for index in range(14) if not 3 <= index <= 5 + delay]
    assert torch.equal(gradient[outside], torch.zeros_like(gradient[outside]))
    assert gradient[5].abs().sum() > 0
    changed = inputs.detach().clone()
    changed[outside] += 100 * torch.randn_like(changed[outside])
    with torch.no_grad():
        other = decoder.prediction_views(data, changed)[delay][5]
    torch.testing.assert_close(result, other, rtol=0, atol=0)


@pytest.mark.parametrize("radius", [0, 1, 8])
def test_parameterized_windows_and_query_chunking_match(radius):
    torch.manual_seed(9)
    decoder = module("local", radius, radius)
    inputs = torch.randn(9, 32, requires_grad=True)
    data = batch(["a", "b", "a", "b", "a", "b", "a", "b", "single"])
    expected = decoder.prediction_views(data, inputs)[radius]
    decoder.query_batch_size = 2
    actual = decoder.prediction_views(data, inputs)[radius]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (9, 29)
    actual.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()


def test_cnn_freeze_schedule_also_freezes_batchnorm_and_dropout():
    decoder = module()
    decoder.freeze_cnn_epochs = 1
    decoder._trainer = SimpleNamespace(current_epoch=0)
    decoder.train()
    decoder.on_train_epoch_start()
    assert not decoder.model.training
    assert not any(parameter.requires_grad for parameter in decoder.model.parameters())
    before = decoder.model.norm.num_batches_tracked.clone()
    decoder.forward(batch(["a"] * 4))
    torch.testing.assert_close(before, decoder.model.norm.num_batches_tracked)
    decoder._trainer.current_epoch = 1
    decoder.on_train_epoch_start()
    assert decoder.model.training
    assert all(parameter.requires_grad for parameter in decoder.model.parameters())
    decoder.forward(batch(["a"] * 4))
    assert decoder.model.norm.num_batches_tracked.item() == before.item() + 1


def test_research_checkpoint_preserves_window_and_delay_protocol(tmp_path):
    torch.manual_seed(10)
    original = module(query_batch_size=3)
    checkpoint = tmp_path / "revision.ckpt"
    torch.save(
        {
            "state_dict": original.state_dict(),
            "hyper_parameters": dict(original.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )
    fresh = module("local", 8, 0)
    loaded = ResearchModule.load_from_checkpoint(
        checkpoint,
        model=fresh.model,
        transformer=fresh.transformer,
        loss=fresh.loss,
        optimizer=fresh.optimizer,
    ).eval()
    assert loaded.research_mode == "revision" and loaded.delays == (0, 1, 2)
    assert loaded.left == 2 and loaded.query_batch_size == 3
    inputs = torch.randn(5, 32)
    data = batch(["a"] * 5)
    with torch.no_grad():
        for delay, expected in original.prediction_views(data, inputs).items():
            torch.testing.assert_close(
                loaded.prediction_views(data, inputs)[delay], expected
            )


def test_summary_keeps_corpus_and_sentence_cer_distinct_and_uses_fixed_cohorts():
    rows = [
        {
            "sentence_UID": "a",
            "position": index,
            "length": 3,
            "label": label,
            "prediction": prediction,
            "ce": 1.0,
            "prompt": "sot",
        }
        for index, (label, prediction) in enumerate([(0, 0), (1, 0), (2, 2)])
    ]
    rows += [
        {
            "sentence_UID": "b",
            "position": 0,
            "length": 1,
            "label": 0,
            "prediction": 0,
            "ce": 1.0,
            "prompt": "s",
        }
    ]
    summary = summarize_predictions(rows)
    assert summary["all_target"]["CER_corpus"] == 0.25
    assert summary["all_target"]["CER_sentence_mean"] == pytest.approx(1 / 6)
    assert summary["common_future"]["events"] == 1
    assert summary["strict_interior"]["events"] == 0
    with pytest.raises(ValueError, match="contiguous"):
        summarize_predictions([rows[0], rows[2]])


def test_initialization_digest_includes_batchnorm_buffers():
    encoder = TinyEncoder()
    before = state_digest(encoder.state_dict())
    encoder.norm.running_mean[0] += 1
    assert state_digest(encoder.state_dict()) != before


def test_single_seed_matrix_has_exactly_ten_jobs_and_one_shared_initialization(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"data": {"study": {"path": "dataset"}}})
    )
    jobs = plan(tmp_path, tmp_path / "cache")
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 10
    assert {job["seed"] for job in jobs} == {33}
    assert sum(job["mode"] == "cnn_only" for job in jobs) == 1
    assert [(job["left"], job["right"]) for job in jobs if job["mode"] == "local"] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (2, 0),
        (2, 1),
    ]
    assert all(job["test_evaluated"] is False for job in jobs)


def test_lightning_fit_checks_coverage_selects_checkpoint_and_unfreezes_cnn(tmp_path):
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader

    torch.manual_seed(12)
    decoder = module()
    decoder.freeze_cnn_epochs = 1
    loader = DataLoader([batch(["a"] * 5 + ["b"] * 3)], batch_size=None)
    best = ModelCheckpoint(
        dirpath=tmp_path,
        filename="best",
        save_weights_only=True,
        monitor="val_ce_common_k2",
        mode="min",
    )
    progress = ResearchProgress(tmp_path, {0: 8, 1: 6, 2: 4}, smoke=False)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        callbacks=[best, progress],
        logger=CSVLogger(tmp_path, name="logs"),
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(decoder, loader, loader)
    assert decoder.model.norm.num_batches_tracked.item() == 1
    assert decoder.validation_summary["2"]["common_future"]["events"] == 4
    assert best.best_model_path and best.best_model_score.isfinite()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["target_counts"] == {"0": 8, "1": 6, "2": 4}
    assert status["completed_epochs"] == 2 and status["cnn_frozen"] is False
