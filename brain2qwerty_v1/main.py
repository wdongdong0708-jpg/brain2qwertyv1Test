# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import lightning.pytorch as pl
import pydantic
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

import neuralset as ns
from neuralset.events.study import EventsTransform
from neuraltrain.losses import BaseLoss
from neuraltrain.models import BaseModelConfig as ModelConfig
from neuraltrain.optimizers import LightningOptimizer
from neuraltrain.utils import WandbLoggerConfig

from . import transforms as _transforms  # noqa: F401  (registers EventsTransforms)
from .callbacks import LogSentencePredictions
from .checkpoint_io import StreamingCheckpointIO
from .context import ContextMode
from .metrics import CER
from .pl_module import BrainModule
from .utils import (
    ChannelPositions2D,
    SentenceBatchSampler,
    SentenceGroupedDistributedSampler,
    materialize_lazy_params,
)


class Data(pydantic.BaseModel):
    """Builds train/val/test dataloaders of keystroke-aligned MEG windows.

    Runs the study, applies the preprocessing/split transforms, prepares the
    feature extractors, and creates one dataloader per split whose sampler keeps
    each sentence's keystrokes on a single rank for the sentence transformer.
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    study: ns.events.Study
    transforms: list[EventsTransform] = pydantic.Field(default_factory=list)
    neuro: ns.extractors.BaseExtractor
    feature: ns.extractors.BaseExtractor

    num_classes: int = 29
    start: float = -0.2
    duration: float = 0.5
    batch_size: int = 128
    val_batch_size: int = 2048
    test_batch_size: int = 2048
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    complete_sentences: bool = False
    include_channel_positions: bool = True

    def build_events(self):
        events = self.study.run()
        for transform in self.transforms:
            events = transform.run(events)
        return ns.events.standardize_events(events)

    def build(self, splits=("train", "val", "test")) -> dict[str, DataLoader]:
        events = self.build_events()
        self.neuro.prepare(events)
        self.feature.prepare(events)

        subject_id = ns.extractors.LabelEncoder(
            event_types=self.neuro.event_types, event_field="subject"
        )
        subject_id.prepare(events)
        extractors = {
            "neuro": self.neuro,
            "feature": self.feature,
            "subject_id": subject_id,
        }
        if self.include_channel_positions:
            channel_positions = ChannelPositions2D(
                neuro=self.neuro,
                infra={
                    "folder": self.neuro.infra.folder,
                    "cluster": None,
                    "mode": self.neuro.infra.mode,
                },
            )
            # The same portable keys are used on Windows and Linux.
            from .cache import neuro_cache_key

            channel_positions.infra.item_uid = neuro_cache_key
            channel_positions.prepare(events)
            extractors["channel_positions"] = channel_positions
        batch_sizes = {
            "train": self.batch_size,
            "val": self.val_batch_size,
            "test": self.test_batch_size,
        }
        loaders: dict[str, DataLoader] = {}
        for split, batch_size in batch_sizes.items():
            if split not in splits:
                continue
            mask = (events.split == split) & (events.type == "Keystroke")
            segments = ns.segments.list_segments(
                events, mask, start=self.start, duration=self.duration
            )
            if not segments:
                continue
            dataset = ns.SegmentDataset(
                extractors=extractors,
                segments=segments,
                remove_incomplete_segments=True,
            )
            batching = (
                {"batch_sampler": SentenceBatchSampler(dataset.segments, batch_size)}
                if self.complete_sentences
                else {
                    "batch_size": batch_size,
                    "sampler": SentenceGroupedDistributedSampler(dataset.segments),
                }
            )
            loaders[split] = DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                **batching,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        return loaders


class Experiment(pydantic.BaseModel):
    """Train and evaluate the Brain2Qwerty V1 keystroke decoder."""

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    data: Data
    brain_model_config: ModelConfig
    transformer_config: ModelConfig
    loss: BaseLoss
    optimizer: LightningOptimizer

    seed: int = 33
    n_epochs: int = 100
    patience: int = 80
    grad_max_norm: float | None = None
    save_checkpoints: bool = True
    devices: int = 8
    output_dir: str = "."
    log_every_n_steps: int = 5
    precision: str = "32-true"
    enable_progress_bar: bool = True
    limit_train_batches: int | float = 1.0
    limit_val_batches: int | float = 1.0
    limit_test_batches: int | float = 1.0
    context_mode: ContextMode = "sentence"

    eval_only: bool = False
    ckpt_path: str | None = None
    wandb_config: WandbLoggerConfig | None = None

    _module: BrainModule | None = None
    _trainer: pl.Trainer | None = None

    def _accelerator(self) -> tuple[str, int]:
        if torch.cuda.is_available():
            return "gpu", max(1, min(self.devices, torch.cuda.device_count()))
        return "cpu", 1

    def _build_modules(
        self, loader: DataLoader
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        n_in = next(iter(loader)).data["neuro"].shape[1]
        hidden = self.brain_model_config.hidden
        brain = self.brain_model_config.build(n_in_channels=n_in, n_outputs=hidden)
        transformer = self.transformer_config.build(dim=hidden)
        return brain, transformer

    def _trainer_setup(self) -> pl.Trainer:
        accelerator, devices = self._accelerator()
        if self.eval_only:
            # Evaluate in a single process: DDP would shard the split across ranks
            # and the prediction callback would only capture one rank's sentences.
            # Running on one device guarantees the saved predictions are complete.
            devices = 1
        callbacks = [
            LogSentencePredictions(save_dir=self.output_dir),
            EarlyStopping(monitor="val_CER", mode="min", patience=self.patience),
        ]
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.output_dir,
                    filename="best",
                    monitor="val_CER",
                    mode="min",
                    save_last=True,
                    save_top_k=1,
                )
            )
        loggers: list = [
            CSVLogger(self.output_dir, name="logs", flush_logs_every_n_steps=1)
        ]
        if self.wandb_config is not None:
            loggers.append(self._build_wandb_logger())
        strategy = DDPStrategy(find_unused_parameters=True) if devices > 1 else "auto"
        return pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=self.n_epochs,
            gradient_clip_val=self.grad_max_norm,
            log_every_n_steps=self.log_every_n_steps,
            use_distributed_sampler=False,
            callbacks=callbacks,
            logger=loggers,
            default_root_dir=self.output_dir,
            precision=self.precision,
            enable_progress_bar=self.enable_progress_bar,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_test_batches,
            enable_checkpointing=self.save_checkpoints,
            plugins=[StreamingCheckpointIO(self.output_dir)],
        )

    def _build_wandb_logger(self):
        try:
            xp_config = self.model_dump(mode="json")
        except Exception:
            xp_config = None
        return self.wandb_config.build(save_dir=self.output_dir, xp_config=xp_config)

    def run(self, loaders: dict[str, DataLoader] | None = None) -> None:
        pl.seed_everything(self.seed, workers=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if loaders is None:
            loaders = self.data.build()
        brain, transformer = self._build_modules(loaders["train"])
        metrics = {"CER": CER()}

        if self.eval_only:
            assert self.ckpt_path is not None, "eval requires a checkpoint."
            self._module = BrainModule.load_from_checkpoint(
                self.ckpt_path,
                model=brain,
                transformer=transformer,
                loss=self.loss.build(),
                metrics=metrics,
                optimizer=self.optimizer,
                context_mode=self.context_mode,
            )
            self._trainer = self._trainer_setup()
            self._trainer.test(self._module, dataloaders=loaders["test"])
            return

        self._module = BrainModule(
            model=brain,
            transformer=transformer,
            loss=self.loss.build(),
            metrics=metrics,
            optimizer=self.optimizer,
            context_mode=self.context_mode,
        )
        materialize_lazy_params(self._module, loaders["train"])
        self._trainer = self._trainer_setup()
        self._trainer.fit(
            self._module, loaders["train"], loaders["val"], ckpt_path=self.ckpt_path
        )
        if "test" in loaders:
            self._trainer.test(
                self._module,
                dataloaders=loaders["test"],
                ckpt_path="best" if self.save_checkpoints else None,
            )


def main(argv: list[str] | None = None) -> None:
    """Keep the original module entry point on the unified workflow."""
    from .workflow import main as workflow_main

    workflow_main(argv)


if __name__ == "__main__":
    main()
