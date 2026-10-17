import copy
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

import studies  # noqa: F401
from brain2qwerty_v1 import workflow
from brain2qwerty_v1.cache import cache_signature, configure_cache, neuro_cache_key
from brain2qwerty_v1.config.model_config import EEG_ENCODER, ENCODER
from brain2qwerty_v1.config.xp_config import experiment_config
from brain2qwerty_v1.fast_context import CachedFeatureDataset
from brain2qwerty_v1.main import Experiment
from scripts.cache_archive import pack, unpack
from studies.spanishbcbl import Pinet2024Eeg, Pinet2024Meg


def eeg_files(root, renamed=False):
    signals, logs = root / "EEG/EEG", root / "EEG/logs"
    signals.mkdir(parents=True)
    logs.mkdir(parents=True)
    for session in (1, 2):
        for block in (1, 2):
            (signals / f"002_DECOMEG_S{session}_9696_task{block}.vhdr").touch()
            prefix = "EEG002" if renamed else "S2"
            (logs / f"{prefix}_session{session}_block{block}_list1.mat").touch()
    return root


@pytest.mark.parametrize("renamed", [False, True])
def test_eeg_native_and_renamed_logs_discover_four_local_timelines(tmp_path, renamed):
    root = eeg_files(tmp_path / "Pinet2024Eeg", renamed)
    study = Pinet2024Eeg(path=root)
    assert len(study.study_summary()) == 4
    assert set(study.study_summary().subject) == {"Pinet2024Eeg/002"}
    assert study._info is None
    assert len(study._get_all_files()) == 4
    assert study._get_all_files().log.map(lambda p: p.is_file()).all()


def test_eeg_log_matching_rejects_other_subject_and_ambiguous_matches(tmp_path):
    root = eeg_files(tmp_path)
    (root / "EEG/logs/S20_session1_block1.mat").touch()
    assert Pinet2024Eeg._get_log_file(root, "002", 1, "task1").name.startswith("S2_")
    (root / "EEG/logs/EEG002_session2_block2.mat").touch()
    # A fresh directory scan is needed after editing a live fixture.
    from studies.spanishbcbl import _log_files

    _log_files.cache_clear()
    with pytest.raises(ValueError, match="More than one"):
        Pinet2024Eeg(path=root)._get_all_files()


@pytest.mark.parametrize("signal_folder", ["MEG", "FIF"])
def test_meg_native_and_legacy_paths_are_portable(tmp_path, signal_folder):
    signal = tmp_path / "MEG" / signal_folder / "03_11123/230313/block1.fif"
    signal.parent.mkdir(parents=True)
    signal.touch()
    log = tmp_path / "MEG/logs/S3-session1_block1.mat"
    log.parent.mkdir(parents=True)
    log.touch()
    study = Pinet2024Meg(path=tmp_path)
    assert list(study.iter_timelines()) == [
        {"subject": "S3", "session": 1, "task": "block1"}
    ]
    assert study._get_all_files().iloc[0].raw == signal


def test_meg_session_two_log_swap_is_preserved(tmp_path):
    logs = tmp_path / "MEG/logs"
    logs.mkdir(parents=True)
    expected = logs / "S2-session2-2.mat"
    expected.touch()
    assert Pinet2024Meg._get_log_file(tmp_path, "S2", 2, "block1") == expected


def test_config_reads_environment_at_call_time_and_does_not_mutate_globals(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("BRAIN2QWERTY_RESULTS", str(tmp_path / "results-a"))
    a = experiment_config()
    monkeypatch.setenv("BRAIN2QWERTY_RESULTS", str(tmp_path / "results-b"))
    assert experiment_config()["output_dir"] == str(tmp_path / "results-b")
    a["brain_model_config"]["merger_config"]["n_subjects"] = 999
    assert "n_subjects" not in ENCODER["merger_config"]
    assert EEG_ENCODER["merger_config"] is None


def arguments(root, mode, cache, subject="002", extra=()):
    return workflow.parser().parse_args(
        [
            mode,
            "--data-root",
            str(root),
            "--cache-root",
            str(cache),
            "--subject",
            subject,
            "--precision",
            "32-true",
            *extra,
        ]
    )


def test_actual_exca_uids_match_cache_smoke_train_and_different_cache_locations(tmp_path):
    root = eeg_files(tmp_path / "data/SpanishBCBL_Full")
    signatures, queries = [], []
    for mode, folder in (("cache", "persisted"), ("smoke", "local"), ("train", "local")):
        args = arguments(root.parent, mode, tmp_path / folder)
        cfg = workflow.make_config(args)
        xp = Experiment(**cfg)
        configure_cache(xp.data, read_only=mode != "cache")
        signatures.append(cache_signature(xp.data))
        queries.append(cfg["data"]["study"]["query"])
        assert xp.brain_model_config.merger_config is None
        assert not xp.data.include_channel_positions
    assert signatures[0] == signatures[1] == signatures[2]
    assert queries[0] == queries[1] == queries[2]
    all_cfg = workflow.make_config(
        arguments(root.parent, "cache", tmp_path / "persisted", "all")
    )
    all_xp = Experiment(**all_cfg)
    configure_cache(all_xp.data)
    assert cache_signature(all_xp.data) == signatures[0]
    assert all_xp.data.study.query is None


def test_signal_item_key_is_independent_of_encoded_subject_index():
    from neuralset.events.etypes import Eeg

    event = Eeg(
        start=0,
        duration=1,
        frequency=50,
        filepath="sample.vhdr",
        subject="0",
        timeline="t",
    )
    assert neuro_cache_key(event) == neuro_cache_key(
        event.model_copy(update={"subject": "5"})
    )


def test_read_only_missing_timeline_fails_before_raw_loading(tmp_path):
    root = eeg_files(tmp_path / "raw")
    cfg = workflow.make_config(arguments(root, "cache", tmp_path / "cache"))
    xp = Experiment(**cfg)
    configure_cache(xp.data, read_only=True)
    with patch.object(
        Pinet2024Eeg,
        "_load_timeline_events",
        side_effect=AssertionError("raw loading is forbidden"),
    ):
        with pytest.raises(RuntimeError, match="read-only"):
            xp.data.build()


def test_cpu_cache_never_calls_cuda_model_build_or_trainer(tmp_path, monkeypatch):
    root = eeg_files(tmp_path / "raw")
    loaders = {}
    for split in ("train", "val", "test"):
        data = CachedFeatureDataset(
            torch.ones(2, 3),
            torch.tensor([0, 1]),
            [{"sentence_UID": split, "onset": float(i), "prompt": "a"} for i in range(2)],
        )
        loaders[split] = DataLoader(data, batch_size=2, collate_fn=data.collate_fn)
    from brain2qwerty_v1.main import Data

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA / model / Trainer must not be used in cache mode")

    monkeypatch.setattr(Data, "build", lambda self: loaders)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    # Earlier GPU tests may have initialized CUDA in this pytest process;
    # emulate the fresh cache-only CLI process used in production.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(Experiment, "_build_modules", forbidden)
    monkeypatch.setattr(workflow.pl, "Trainer", forbidden)
    output = tmp_path / "output"
    workflow.main(
        [
            "cache",
            "--modality",
            "eeg",
            "--subject",
            "002",
            "--data-root",
            str(root),
            "--cache-root",
            str(tmp_path / "cache"),
            "--run-dir",
            str(output),
        ]
    )
    state = json.loads((output / "status.json").read_text())
    assert state["state"] == "cache_complete"
    assert not state["cuda_initialized"] and not state["trainer_initialized"]


def test_resume_comparison_normalizes_json_and_ignores_only_cache_location(tmp_path):
    root = eeg_files(tmp_path / "raw")
    cfg = workflow.make_config(arguments(root, "train", tmp_path / "drive"))
    saved = json.loads(json.dumps(cfg))
    cfg["data"]["neuro"]["infra"]["folder"] = str(tmp_path / "local")
    assert workflow.resume_config(cfg) == workflow.resume_config(saved)
    changed = copy.deepcopy(cfg)
    changed["n_epochs"] += 1
    assert workflow.resume_config(changed) != workflow.resume_config(saved)


def test_cache_archive_roundtrip_and_path_traversal_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    (cache / "extractor").mkdir(parents=True)
    (cache / "extractor/features.bin").write_bytes(b"features")
    archive = tmp_path / "features.tar"
    pack(cache, archive)
    unpack(archive, tmp_path / "restored")
    assert (tmp_path / "restored/extractor/features.bin").read_bytes() == b"features"
    malicious = tmp_path / "bad.tar"
    with tarfile.open(malicious, "w") as target:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 1
        target.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="escapes"):
        unpack(malicious, tmp_path / "restored")


def test_colab_notebooks_contain_git_commands_and_no_embedded_source():
    root = Path(__file__).resolve().parents[1]
    for name in ("colab_cpu_cache.ipynb", "colab_gpu_train.ipynb"):
        text = (root / "notebooks" / name).read_text(encoding="utf-8")
        assert "git" in text and "clone" in text and "pull" in text
        assert "SOURCE_BASE64" not in text and "b64decode" not in text
