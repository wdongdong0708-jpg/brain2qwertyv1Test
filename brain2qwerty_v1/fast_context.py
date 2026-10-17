from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from neuralset.dataloader import Batch
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.optimizers import LightningOptimizer

from .research import ResearchModule

FAST_TRANSFORMER = {
    "name": "TransformerEncoder",
    "depth": 2,
    "heads": 4,
    "ff_mult": 2,
    "attn_dropout": 0.1,
    "ff_dropout": 0.1,
    "alibi_pos_bias": True,
}


class FeatureProjection(torch.nn.Module):
    """Train a small context decoder on immutable pretrained CNN features."""

    def __init__(self, input_dim=2048, output_dim=256):
        super().__init__()
        self.out_channels = output_dim
        self.norm = torch.nn.LayerNorm(input_dim)
        self.projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, neuro, subject_id=None, channel_positions=None):
        return self.projection(self.norm(neuro))


class EndToEndProjection(torch.nn.Module):
    """Apply the existing small decoder projection to a trainable EEG encoder."""

    def __init__(self, encoder, projection):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.out_channels = projection.out_channels

    def forward(self, neuro, subject_id=None, channel_positions=None):
        features = self.encoder(neuro, subject_id, channel_positions)
        return self.projection(features)


class JointContextModule(ResearchModule):
    """Fine-tune the pretrained CNN gently while retaining the fast decoder LR."""

    def optimizer_groups(self):
        return [
            {"params": list(self.model.encoder.parameters()), "lr": 5e-5},
            {
                "params": [
                    *self.model.projection.parameters(),
                    *self.transformer.parameters(),
                    *self.linear.parameters(),
                ],
                "lr": 3e-4,
            },
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.optimizer_groups(), weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[5e-5, 3e-4],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def decoder_state(module):
    """Use the frozen-feature model's key names for exact initialization checks."""
    projection = (
        module.model.projection
        if isinstance(module.model, EndToEndProjection)
        else module.model
    )
    return {
        **{f"model.{key}": value for key, value in projection.state_dict().items()},
        **{
            f"transformer.{key}": value
            for key, value in module.transformer.state_dict().items()
        },
        **{f"linear.{key}": value for key, value in module.linear.state_dict().items()},
    }


class CachedFeatureDataset(Dataset):
    """Keep frozen features in memory and batch whole sentences with the V1 API."""

    def __init__(self, features, labels, metadata, device="cpu"):
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError("Expected [events, features] and [events] labels")
        if not len(features) == len(labels) == len(metadata):
            raise ValueError("Cached feature, label and metadata lengths differ")
        if labels.numel() and not (labels.min() >= 0 and labels.max() < 29):
            raise ValueError("Labels must use the fixed 29-class mapping")
        self.features = features.detach().to(device=device, dtype=torch.float32)
        self.labels = labels.detach().to(device=device, dtype=torch.long)
        self.segments = [
            SimpleNamespace(
                duration=0.5,
                trigger=SimpleNamespace(
                    start=row["onset"],
                    extra={
                        "sentence_UID": row["sentence_UID"],
                        "sentence": row["prompt"],
                    },
                ),
            )
            for row in metadata
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index

    def collate_fn(self, indices):
        rows = torch.tensor(indices, device=self.features.device, dtype=torch.long)
        return Batch(
            segments=[self.segments[index] for index in indices],
            data={
                "neuro": self.features[rows],
                "feature": self.labels[rows, None],
                "subject_id": self.labels.new_zeros((len(indices), 1)),
                "channel_positions": self.features.new_empty((len(indices), 0, 2)),
            },
        )


def build_fast_module(mode, left, right, input_dim=2048, encoder=None):
    projection = FeatureProjection(input_dim=input_dim, output_dim=256)
    transformer = BaseModelConfig(**FAST_TRANSFORMER).build(dim=256)
    optimizer = LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 3e-4, "kwargs": {"weight_decay": 1e-3}},
        scheduler={"name": "OneCycleLR", "kwargs": {"max_lr": 3e-4, "pct_start": 0.1}},
        interval="step",
    )
    module_class = JointContextModule if encoder is not None else ResearchModule
    model = EndToEndProjection(encoder, projection) if encoder is not None else projection
    return module_class(
        model,
        transformer,
        torch.nn.CrossEntropyLoss(),
        optimizer,
        research_mode=mode,
        left=left,
        right=right,
        freeze_cnn_epochs=0,
        query_batch_size=256,
    )
