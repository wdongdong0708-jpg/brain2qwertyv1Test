from types import SimpleNamespace

import torch

from brain2qwerty_v1.callbacks import LogSentencePredictions
from brain2qwerty_v1.scripts.extract_predictions import extract_subject
from brain2qwerty_v1.scripts.single_subject import subject_labels, timeline_cache_key
from brain2qwerty_v1.utils import SentenceBatchSampler


def test_batches_preserve_sentences_and_all_keystrokes():
    uids = ["a"] * 4 + ["b"] * 3 + ["c"] + ["long"] * 9
    segments = [
        SimpleNamespace(trigger=SimpleNamespace(extra={"sentence_UID": uid}))
        for uid in uids
    ]
    sampler = SentenceBatchSampler(segments, max_keystrokes=6)
    batches = list(sampler)
    assert [len(batch) for batch in batches] == [4, 4, 9]
    assert [index for batch in batches for index in batch] == list(range(len(uids)))
    for uid in set(uids):
        assert sum(any(uids[index] == uid for index in batch) for batch in batches) == 1
    assert len(sampler) == len(batches)


def test_eeg_subject_ids_do_not_use_meg_participant_merges():
    assert extract_subject("2.0_002_1_task1") == "EEG002"
    assert (
        extract_subject("2.0_Pinet2024Eeg:session=1,subject=002,task=task1") == "EEG002"
    )
    assert (
        extract_subject("2.0_Pinet2024Eeg_subject-002_session-1_task-task1") == "EEG002"
    )
    assert (
        extract_subject("2.0_Pinet2024Eeg_subject-018_session-1_task-task1") == "EEG018"
    )
    assert extract_subject("2.0_Pinet2024Meg_subject-S18_session-1_task-block1") == "S1"


def test_callback_recovers_reference_sentence_from_event_metadata():
    trigger = SimpleNamespace(
        extra={"sentence_UID": "2.0_002_1_task1", "sentence": "hola"}
    )
    batch = SimpleNamespace(segments=[SimpleNamespace(trigger=trigger)])
    store = {}
    LogSentencePredictions()._collect_batch(
        (torch.zeros(1, 29), torch.tensor([0])), batch, store
    )
    assert store["2.0_002_1_task1"]["true"] == "hola"


def test_cache_keys_are_safe_and_stable_for_windows():
    key = timeline_cache_key({"subject": "002", "session": 1, "task": "task1"})
    assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)
    assert key == timeline_cache_key({"task": "task1", "session": 1, "subject": "002"})
    assert key != timeline_cache_key({"task": "task1", "session": 2, "subject": "002"})


def test_participant_selection_is_modality_specific():
    assert subject_labels("eeg", "018") == ["Pinet2024Eeg/018"]
    assert set(subject_labels("meg", "S18")) == {"Pinet2024Meg/S1", "Pinet2024Meg/S18"}
