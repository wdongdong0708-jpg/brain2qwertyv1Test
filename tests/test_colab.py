import json
from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from brain2qwerty_v1.checkpoint_io import StreamingCheckpointIO
from brain2qwerty_v1.fast_context import CachedFeatureDataset, build_fast_module
from brain2qwerty_v1.scripts.colab_data import (
    check_brainvision_files,
    safe_path,
    select_files,
)
from brain2qwerty_v1.scripts.colab_train import ColabProgress, sync_run
from brain2qwerty_v1.utils import SentenceBatchSampler


def test_download_selects_only_subject_signal_and_matlab_logs():
    names = [
        "EEG/EEG/002_DECOMEG_S1_1_task1.vhdr",
        "EEG/EEG/002_DECOMEG_S1_1_task1.eeg",
        "EEG/EEG/002_DECOMEG_S1_1_task1.vmrk",
        "EEG/logs/nested/S2_session1_block1_list1.mat",
        "EEG/EEG/002_DECOMEG_S1_1_tapping.eeg",
        "EEG/logs/S20_session1_block1.mat",
        "EEG/EEG/003_DECOMEG_S1_1_task1.vhdr",
        "MEG/logs/S2_session1_block1.mat",
        "MEG/all_events.tsv",
    ]
    selected = select_files([SimpleNamespace(path=name, size=10) for name in names], "002")
    assert {row["path"] for row in selected} == set(names[:4])
    with pytest.raises(ValueError, match="excludes subject 001"):
        select_files([], "001")
    with pytest.raises(RuntimeError, match="MATLAB"):
        select_files([SimpleNamespace(path=names[0], size=10)], "002")


def test_brainvision_missing_companion_is_detected_before_training(tmp_path):
    folder = tmp_path / "EEG/EEG"
    folder.mkdir(parents=True)
    (folder / "002_x_task1.vhdr").write_text("DataFile=data.eeg\nMarkerFile=data.vmrk\n")
    (folder / "data.eeg").write_bytes(b"data")
    with pytest.raises(RuntimeError, match="data.vmrk"):
        check_brainvision_files(tmp_path, "002")
    (folder / "data.vmrk").write_text("marker")
    assert check_brainvision_files(tmp_path, "002") == 1
    with pytest.raises(ValueError, match="escapes"):
        safe_path(tmp_path, "../elsewhere")


def test_fp16_gradient_audit_allows_scaler_to_retry_overflow(tmp_path):
    encoder = torch.nn.Linear(2, 2, bias=False)
    encoder.weight.grad = torch.full_like(encoder.weight, float("inf"))
    module = SimpleNamespace(model=encoder)
    trainer = SimpleNamespace(precision_plugin=SimpleNamespace(scaler=object()))
    progress = ColabProgress(tmp_path, tmp_path / "drive", expected_events=1)
    progress.on_before_optimizer_step(trainer, module, None)
    assert not (tmp_path / "gradient_audit.json").exists()
    encoder.weight.grad = torch.ones_like(encoder.weight)
    progress.on_before_optimizer_step(trainer, module, None)
    assert json.loads((tmp_path / "gradient_audit.json").read_text())["cnn_gradient_norm"] > 0


def test_session_stops_at_epoch_boundary_and_resumes_to_same_total_epoch_limit(tmp_path):
    torch.set_num_threads(1)
    pl.seed_everything(33, workers=True)
    data = CachedFeatureDataset(
        torch.randn(6, 8), torch.tensor([0, 1, 2, 3, 4, 5]),
        [{"sentence_UID": "a" if index < 3 else "b", "onset": index * 0.1, "prompt": "abc"} for index in range(6)],
    )
    loader = DataLoader(data, batch_sampler=SentenceBatchSampler(data.segments, 3), collate_fn=data.collate_fn)
    output, persistent = tmp_path / "run", tmp_path / "drive"
    output.mkdir()

    def trainer(progress):
        return pl.Trainer(
            accelerator="cpu", max_epochs=3, logger=False, enable_progress_bar=False,
            num_sanity_val_steps=0, plugins=[StreamingCheckpointIO(output)],
            callbacks=[progress, ModelCheckpoint(dirpath=output, monitor="val_ce_all", filename="best",
                                                 save_top_k=1, save_weights_only=True, save_on_train_epoch_end=False),
                       ModelCheckpoint(dirpath=output, save_top_k=0, save_last=True, save_on_train_epoch_end=False)],
        )

    limited = ColabProgress(output, persistent, expected_events=6, session_minutes=1e-12, sync_every=1)
    first = trainer(limited)
    first.fit(build_fast_module("sentence", 0, 0, input_dim=8), loader, loader)
    assert limited.time_limit_reached
    assert first.current_epoch == 1
    assert first.global_step == 2
    sync_run(output, persistent)
    assert (persistent / "last.ckpt").is_file()
    assert json.loads((persistent / "gradient_audit.json").read_text())["cnn_gradient_norm"] > 0

    resumed = trainer(ColabProgress(output, persistent, expected_events=6, session_minutes=0, sync_every=1))
    resumed.fit(build_fast_module("sentence", 0, 0, input_dim=8), loader, loader,
                ckpt_path=str(output / "last.ckpt"))
    assert resumed.current_epoch == 3
    assert resumed.global_step == 6
    epochs = (output / "epoch_metrics.csv").read_text().splitlines()
    assert len(epochs) == 4


def test_data_check_can_be_followed_by_training_and_completed_run_does_not_restart(tmp_path, monkeypatch):
    from brain2qwerty_v1.scripts import colab_train

    torch.set_num_threads(1)
    data = CachedFeatureDataset(
        torch.randn(6, 8), torch.tensor([0, 1, 2, 3, 4, 5]),
        [{"sentence_UID": "a" if index < 3 else "b", "onset": index * 0.1, "prompt": "abc"} for index in range(6)],
    )
    loader = DataLoader(data, batch_sampler=SentenceBatchSampler(data.segments, 3), collate_fn=data.collate_fn)
    settings = {
        "local_run": str(tmp_path / "run"), "drive_run": str(tmp_path / "drive"),
        "subject": "002", "seed": 33, "monitor": "val_ce_all", "resume": True,
        "session_minutes": 0, "sync_every_epochs": 1,
    }
    monkeypatch.setattr(colab_train, "get_config", lambda _: {
        "n_epochs": 2, "patience": 6, "precision": "32-true", "data": {"filter": (0.1, 20.0)},
    })
    monkeypatch.setattr(colab_train, "save_environment", lambda _: None)
    monkeypatch.setattr(colab_train, "build_data", lambda *args: (None, {"train": loader, "val": loader}, {"splits": {"train": {"keystrokes": 6}}}))
    built = []

    def build(*args):
        built.append(True)
        return build_fast_module("sentence", 0, 0, input_dim=8)

    monkeypatch.setattr(colab_train, "build_module", build)
    trainer_class = pl.Trainer

    def cpu_trainer(**kwargs):
        kwargs.update(accelerator="cpu", logger=False)
        return trainer_class(**kwargs)

    monkeypatch.setattr(colab_train.pl, "Trainer", cpu_trainer)
    colab_train.run(settings, "check")
    colab_train.run(settings, "train")
    state = json.loads((tmp_path / "drive/status.json").read_text())
    assert state["state"] == "complete" and state["completed_epochs"] == 2
    assert (tmp_path / "drive/last.ckpt").is_file()
    colab_train.run(settings, "train")
    assert len(built) == 1
