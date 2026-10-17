import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from brain2qwerty_v1 import workflow
from brain2qwerty_v1.cache import cache_signature, configure_cache
from brain2qwerty_v1.main import Data, Experiment
from brain2qwerty_v1.pl_module import BrainModule
from neuraltrain.models.base import BaseModelConfig
from studies.spanishbcbl import Pinet2024Meg


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "raw"
    for folder in ("EEG/EEG", "EEG/logs", "MEG/MEG"):
        (root / folder).mkdir(parents=True)
    for subject in range(1, 14):
        for block in (1, 2):
            (root / f"EEG/EEG/{subject:03d}_DECOMEG_S1_9696_task{block}.vhdr").touch()
            (root / f"EEG/logs/S{subject}_session1_block{block}_list1.mat").touch()
    return root


def arguments(root, *selection, mode="cache", modality="eeg"):
    return workflow.parser().parse_args(
        [
            mode,
            "--modality",
            modality,
            "--data-root",
            str(root),
            "--cache-root",
            str(root.parent / "cache"),
            "--results-root",
            str(root.parent / "runs"),
            "--precision",
            "32-true",
            *selection,
        ]
    )


def test_fixed_list_filters_real_recordings_and_retains_all_blocks(data_root):
    args = arguments(data_root, "--subjects", "010,002", "003")
    cfg = workflow.make_config(args)
    summary = Experiment(**cfg).data.study.study_summary()
    assert list(summary.subject.unique()) == [
        "Pinet2024Eeg/002",
        "Pinet2024Eeg/003",
        "Pinet2024Eeg/010",
    ]
    assert len(summary) == 6
    assert workflow.select_subjects(summary, workflow.requested_subjects(args)) == [
        "Pinet2024Eeg/002",
        "Pinet2024Eeg/003",
        "Pinet2024Eeg/010",
    ]


@pytest.mark.parametrize(
    "selection",
    [
        ["--subject", "002", "--subjects", "003"],
        ["--subject", "all", "--num-subjects", "10"],
        ["--subjects", "002", "--num-subjects", "10"],
    ],
)
def test_subject_selection_flags_are_mutually_exclusive(selection):
    with pytest.raises(SystemExit):
        workflow.parser().parse_args(["cache", *selection])


@pytest.mark.parametrize(
    "modality,values,error",
    [
        ("eeg", ["002", "S2"], "Duplicate participant"),
        ("meg", ["S1", "S18"], "Duplicate participant"),
        ("meg", ["S5,S10"], "Duplicate participant"),
        ("meg", ["S23"], "excluded"),
        ("meg", ["S11122024"], "excluded"),
        ("eeg", ["0"], "positive"),
        ("eeg", ["002,,003"], "Invalid subject"),
        ("eeg", ["all"], "Invalid subject"),
        ("eeg", ["Pinet2024Meg/S2"], "does not belong"),
    ],
)
def test_invalid_ids_and_alias_duplicates_fail_clearly(modality, values, error):
    args = workflow.parser().parse_args(
        ["cache", "--modality", modality, "--subjects", *values]
    )
    with pytest.raises(ValueError, match=error):
        workflow.requested_subjects(args)


def test_meg_count_uses_numeric_order_and_unique_non_control_participants():
    summary = pd.DataFrame(
        {
            "subject": [
                f"Pinet2024Meg/{label}"
                for label in (
                    "S18",
                    "S3",
                    "S14",
                    "S5",
                    "S10",
                    "S21",
                    "S6",
                    "S7",
                    "S8",
                    "S9",
                    "S11",
                    "S12",
                    "S15",
                    "S23",
                    "S11122024",
                )
            ]
        }
    )
    expected = [f"Pinet2024Meg/S{number}" for number in (1, 3, 4, 5, 6, 7, 8, 9, 11, 12)]
    assert workflow.select_subjects(summary, count=10) == expected
    assert (
        workflow.select_subjects(summary.sample(frac=1, random_state=1), count=10)
        == expected
    )
    with pytest.raises(ValueError, match="only 11 valid"):
        workflow.select_subjects(summary, count=12)


@pytest.mark.parametrize("modality", ["eeg", "meg"])
def test_subset_identity_and_real_exca_uids_preserve_cache_and_resume_guards(
    data_root, modality
):
    configs = [
        workflow.make_config(arguments(data_root, *flags, modality=modality))
        for flags in (
            ["--subject", "all"],
            ["--subjects", "002,003"],
            ["--subjects", "S3", "S2"],
            ["--subjects", "002", "004"],
        )
    ]
    signatures = []
    for cfg in configs:
        xp = Experiment(**cfg)
        configure_cache(xp.data, read_only=True)
        signatures.append(cache_signature(xp.data))
    assert all(signature == signatures[0] for signature in signatures)
    assert configs[1]["output_dir"] == configs[2]["output_dir"]
    assert configs[1]["output_dir"] != configs[3]["output_dir"]
    assert workflow.resume_config(configs[1]) == workflow.resume_config(configs[2])
    assert workflow.resume_config(configs[1]) != workflow.resume_config(configs[3])
    other_mode = workflow.make_config(
        arguments(data_root, "--subjects", "002", "003", mode="train", modality=modality)
    )
    other = Experiment(**other_mode)
    configure_cache(other.data)
    assert cache_signature(other.data) == signatures[0]
    assert other_mode["data"]["study"]["query"] == configs[1]["data"]["study"]["query"]


@pytest.mark.parametrize("modality", ["eeg", "meg"])
def test_ten_subject_cache_selection_sizes_trainable_model_and_records_list(
    data_root, monkeypatch, modality
):
    if modality == "meg":
        summary = pd.DataFrame(
            {
                "subject": [
                    f"Pinet2024Meg/S{number}"
                    for number in (
                        1,
                        18,
                        2,
                        3,
                        4,
                        14,
                        5,
                        10,
                        21,
                        6,
                        7,
                        8,
                        9,
                        11,
                        12,
                        23,
                        11122024,
                    )
                ]
            }
        )
        summary["timeline"] = summary.subject + "_session-1"
        monkeypatch.setattr(
            Pinet2024Meg,
            "study_summary",
            lambda self, apply_query=True: (
                summary.query(self.query)
                if apply_query and self.query
                else summary.copy()
            ),
        )
        expected = [
            f"Pinet2024Meg/S{number}" for number in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11)
        ]
        timeline_count = 14
    else:
        expected = [f"Pinet2024Eeg/{number:03d}" for number in range(2, 12)]
        timeline_count = 20

    def forbidden(*args, **kwargs):
        raise AssertionError("Cache selection must not create a Trainer or access CUDA")

    monkeypatch.setattr(Data, "build", lambda self: {})
    monkeypatch.setattr(workflow, "audit_loaders", lambda *args: {})
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(workflow.pl, "Trainer", forbidden)

    for flags in (["--num-subjects", "10"], ["--subjects", *reversed(expected)]):
        args = arguments(data_root, *flags, modality=modality)
        workflow.main(
            [
                str(value)
                for value in (
                    "cache",
                    "--modality",
                    modality,
                    "--data-root",
                    data_root,
                    "--cache-root",
                    data_root.parent / "cache",
                    "--results-root",
                    data_root.parent / "runs",
                    *flags,
                )
            ]
        )
        cfg = workflow.make_config(args, selected_subjects=expected)
        output = workflow.Path(cfg["output_dir"])
        saved = json.loads((output / "config.json").read_text())
        selection = json.loads((output / "subjects.json").read_text())
        assert selection["subjects"] == expected
        assert selection["participants"] == 10
        assert len(pd.read_csv(output / "recordings.csv")) == timeline_count
        assert saved["brain_model_config"]["subject_layers_config"]["n_subjects"] == 10
        merger = saved["brain_model_config"]["merger_config"]
        assert merger is None if modality == "eeg" else merger["n_subjects"] == 10

    # Exercise every participant's subject layer through CNN + Transformer backward.
    torch.manual_seed(33)
    enc = copy.deepcopy(saved["brain_model_config"])
    enc.update(hidden=32, depth=2, initial_linear=16)
    if enc["merger_config"]:
        enc["merger_config"].update(n_virtual_channels=16)
        enc["merger_config"]["fourier_emb_config"]["total_dim"] = 32
    encoder = BaseModelConfig(**enc).build(n_in_channels=8, n_outputs=32)
    trans = copy.deepcopy(saved["transformer_config"])
    trans.update(depth=1, heads=1)
    module = BrainModule(
        encoder,
        BaseModelConfig(**trans).build(dim=32),
        torch.nn.CrossEntropyLoss(),
        {},
        None,
    )
    batch = SimpleNamespace(
        data={
            "neuro": torch.randn(20, 8, 25),
            "subject_id": torch.arange(10).repeat_interleave(2).unsqueeze(1),
        },
        segments=[
            SimpleNamespace(
                trigger=SimpleNamespace(extra={"sentence_UID": f"subject-{i // 2}"})
            )
            for i in range(20)
        ],
    )
    if modality == "meg":
        batch.data["channel_positions"] = torch.rand(20, 8, 2)
    logits = module._transformer_forward(batch, module(batch))
    loss = module.loss(logits, torch.arange(20) % 29)
    loss.backward()
    assert logits.shape == (20, 29) and torch.isfinite(loss)
    grads = encoder.subject_layers.weights.grad
    assert grads is not None and torch.isfinite(grads).all()
    assert (grads.flatten(1).norm(dim=1) > 0).all()


@pytest.mark.parametrize(
    "flags,error",
    [
        (["--subjects", "002", "999"], "No valid recordings"),
        (["--num-subjects", "20"], "only 12 valid"),
        (["--num-subjects", "0"], "must be positive"),
    ],
)
def test_missing_subjects_and_insufficient_counts_fail_before_data_build(
    data_root, monkeypatch, capsys, flags, error
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Must validate the complete subject selection before preprocessing"
        )

    monkeypatch.setattr(Data, "build", forbidden)
    with pytest.raises(SystemExit):
        workflow.main(
            [
                "cache",
                "--data-root",
                str(data_root),
                "--cache-root",
                str(data_root.parent / "cache"),
                *flags,
            ]
        )
    assert error in capsys.readouterr().err
    assert not (data_root.parent / "cache/workflow_manifests").exists()
