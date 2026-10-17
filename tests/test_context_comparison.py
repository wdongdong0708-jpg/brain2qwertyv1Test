import json

import pandas as pd
import pytest

from brain2qwerty_v1.scripts.single_subject import verify_reference


def _comparison_files(tmp_path):
    reference = tmp_path / "baseline"
    output = tmp_path / "local"
    reference.mkdir()
    output.mkdir()
    base_config = {"seed": 33, "data": {"batch_size": 64}, "output_dir": str(reference)}
    (reference / "config.json").write_text(json.dumps(base_config), encoding="utf-8")
    config = {**base_config, "context_mode": "local-history", "output_dir": str(output)}
    manifest = pd.DataFrame({"sentence_UID": ["s2", "s1"], "split": ["train", "test"]})
    manifest.to_csv(reference / "sentence_splits.csv", index=False)
    manifest.iloc[::-1].to_csv(output / "sentence_splits.csv", index=False)
    return reference, output, config


def test_comparison_accepts_identical_splits_in_different_row_orders(tmp_path):
    reference, output, config = _comparison_files(tmp_path)
    verify_reference(output, reference, config)
    audit = json.loads((output / "comparison_audit.json").read_text())
    assert audit["same_data_splits"]


@pytest.mark.parametrize("change", ["seed", "split"])
def test_comparison_refuses_changed_seed_or_test_membership(tmp_path, change):
    reference, output, config = _comparison_files(tmp_path)
    if change == "seed":
        config["seed"] = 42
        expected = RuntimeError
    else:
        manifest = pd.read_csv(output / "sentence_splits.csv")
        manifest.loc[0, "split"] = "train"
        manifest.to_csv(output / "sentence_splits.csv", index=False)
        expected = AssertionError
    with pytest.raises(expected):
        verify_reference(output, reference, config)
