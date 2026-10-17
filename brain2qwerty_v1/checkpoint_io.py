import os
from pathlib import Path

import torch
from lightning.fabric.plugins.io import CheckpointIO


class StreamingCheckpointIO(CheckpointIO):
    """Serialize local checkpoints to a temporary file without a full BytesIO copy."""

    def __init__(self, output_root):
        super().__init__()
        self.output_root = Path(output_root).resolve()

    def _path(self, path):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.output_root):
            raise ValueError("Checkpoint path must stay inside its run directory")
        return resolved

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options is not None:
            raise TypeError("Local streaming checkpoints do not accept storage_options")
        destination = self._path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path, map_location=None):
        return torch.load(
            self._path(path), map_location=map_location or "cpu", weights_only=False
        )

    def remove_checkpoint(self, path):
        target = self._path(path)
        if target.exists() and not target.is_file():
            raise ValueError("Refusing to remove a checkpoint directory")
        target.unlink(missing_ok=True)
