import pytest
import torch

from brain2qwerty_v1.checkpoint_io import StreamingCheckpointIO


def test_streaming_checkpoint_keeps_state_and_can_replace_loaded_checkpoint(tmp_path):
    plugin = StreamingCheckpointIO(tmp_path)
    path = tmp_path / "last.ckpt"
    expected = {
        "state_dict": {"weight": torch.randn(4, 5)},
        "epoch": 3,
        "optimizer_states": [{"state": {0: {"exp_avg": torch.ones(4, 5)}}}],
    }
    plugin.save_checkpoint(expected, path)
    actual = plugin.load_checkpoint(path)
    torch.testing.assert_close(
        actual["state_dict"]["weight"], expected["state_dict"]["weight"]
    )
    assert actual["epoch"] == 3 and actual["optimizer_states"]
    actual["epoch"] = 4
    plugin.save_checkpoint(actual, path)
    assert plugin.load_checkpoint(path)["epoch"] == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_save_preserves_previous_complete_checkpoint(tmp_path, monkeypatch):
    plugin = StreamingCheckpointIO(tmp_path)
    path = tmp_path / "last.ckpt"
    plugin.save_checkpoint({"epoch": 3}, path)

    def fail_save(checkpoint, temporary):
        temporary.write_bytes(b"incomplete")
        raise RuntimeError("interrupted transfer")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_save)
        with pytest.raises(RuntimeError, match="interrupted"):
            plugin.save_checkpoint({"epoch": 4}, path)
    assert plugin.load_checkpoint(path)["epoch"] == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_operations_are_confined_to_run_directory(tmp_path):
    plugin = StreamingCheckpointIO(tmp_path / "run")
    with pytest.raises(ValueError, match="inside"):
        plugin.save_checkpoint({}, tmp_path / "outside.ckpt")
    (tmp_path / "run").mkdir()
    with pytest.raises(ValueError, match="directory"):
        plugin.remove_checkpoint(tmp_path / "run")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_storage_streams_to_loadable_cpu_checkpoint(tmp_path):
    plugin = StreamingCheckpointIO(tmp_path)
    value = torch.randn(64, 64, device="cuda")
    plugin.save_checkpoint({"weight": value}, tmp_path / "best.ckpt")
    actual = plugin.load_checkpoint(tmp_path / "best.ckpt")["weight"]
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, value.cpu())
