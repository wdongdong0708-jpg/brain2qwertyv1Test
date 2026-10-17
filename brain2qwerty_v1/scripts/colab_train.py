"""Bounded, resumable single-subject V1 training for the companion Colab notebook."""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ..checkpoint_io import StreamingCheckpointIO
from ..main import Experiment
from ..research import ResearchModule
from ..utils import materialize_lazy_params
from .colab_data import copy_atomic, validate_subject
from .research_queue import read_json, write_json
from .single_subject import (
    audit_events,
    make_config,
    neuro_cache_key,
    save_environment,
    timeline_cache_key,
)


def sync_run(local, persistent):
    local, persistent = Path(local), Path(persistent)
    persistent.mkdir(parents=True, exist_ok=True)
    copied = read_json(persistent / "sync_manifest.json")
    paths = [path for path in local.rglob("*") if path.is_file() and path.suffix in {".json", ".csv", ".ckpt", ".yaml"}]
    for source in sorted(paths):
        relative = source.relative_to(local).as_posix()
        signature = {"bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
        if copied.get(relative) == signature and (persistent / relative).is_file():
            continue
        if source.suffix == ".ckpt":
            print(f"Saving to Drive: {source.name} ({source.stat().st_size / 1024**3:.2f} GiB)", flush=True)
        copy_atomic(source, persistent / relative)
        copied[relative] = signature
        write_json(persistent / "sync_manifest.json", copied)


class ColabProgress(pl.Callback):
    """Check the session budget at epoch boundaries; a new invocation gets a new budget."""

    def __init__(self, output, persistent, expected_events, session_minutes=60, sync_every=5, smoke=False):
        self.output, self.persistent = Path(output), Path(persistent)
        self.expected_events, self.session_minutes = expected_events, session_minutes
        self.sync_every, self.smoke = sync_every, smoke
        self.time_limit_reached = False
        self.targets = 0
        self.last_reported_epoch = 0

    def state_dict(self):
        return {"targets": self.targets, "last_reported_epoch": self.last_reported_epoch}

    def load_state_dict(self, state_dict):
        self.targets = state_dict.get("targets", 0)
        self.last_reported_epoch = state_dict.get("last_reported_epoch", 0)

    def on_train_start(self, trainer, module):
        self.started = time.monotonic()
        path = self.output / "epoch_metrics.csv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as stream:
                self.last_reported_epoch = max(
                    [self.last_reported_epoch, *[int(row["epoch"]) for row in csv.DictReader(stream)]]
                )

    def on_train_epoch_start(self, trainer, module):
        self.epoch_started = time.monotonic()
        self.targets = 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.targets += len(batch)
        if time.monotonic() - getattr(self, "pulse", 0) >= 10:
            write_json(self.output / "heartbeat.json", {
                "phase": "training", "epoch": trainer.current_epoch + 1,
                "batch": batch_idx + 1, "global_step": trainer.global_step,
                "updated_unix": time.time(),
            })
            print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}, batch {batch_idx + 1}, step {trainer.global_step}", flush=True)
            self.pulse = time.monotonic()

    def on_before_optimizer_step(self, trainer, module, optimizer):
        if getattr(self, "gradients_checked", False):
            return
        gradients = [p.grad.detach() for p in module.model.parameters() if p.grad is not None]
        norm = torch.stack([gradient.norm() for gradient in gradients]).norm()
        if not torch.isfinite(norm) and getattr(trainer.precision_plugin, "scaler", None) is not None:
            # FP16 GradScaler will skip this step and reduce the scale automatically.
            return
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError("CNN gradients are missing or nonfinite")
        write_json(self.output / "gradient_audit.json", {
            "cnn_trainable": True, "cnn_gradient_norm": float(norm.cpu()),
            "initialization": "random_joint_training", "pretrained_cnn_loaded": False,
        })
        self.gradients_checked = True

    def on_train_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch + 1
        if epoch <= self.last_reported_epoch:
            return
        if not self.smoke and self.targets != self.expected_events:
            raise RuntimeError("This epoch did not cover all training keystrokes")
        metrics = {
            key: float(value.detach().cpu()) for key, value in trainer.callback_metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        if any(not torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise RuntimeError("Nonfinite training/validation metric")
        row = {
            "epoch": epoch, "train_loss": metrics.get("train_loss_epoch"),
            "val_ce": metrics.get("val_ce_all"), "val_CER": metrics.get("val_CER_corpus"),
            "val_accuracy": metrics.get("val_accuracy"),
            "epoch_seconds": round(time.monotonic() - getattr(self, "epoch_started", self.started), 2),
        }
        path = self.output / "epoch_metrics.csv"
        fresh = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if fresh:
                writer.writeheader()
            writer.writerow(row)
        self.last_reported_epoch = epoch
        write_json(self.output / "status.json", {
            "state": "training", "completed_epochs": epoch, "max_epochs": trainer.max_epochs,
            "global_step": trainer.global_step, "metrics": metrics,
            "train_events_this_epoch": self.targets, "updated_unix": time.time(),
        })
        print(f"Epoch {epoch}/{trainer.max_epochs}: {row}", flush=True)
        if not self.smoke and epoch % self.sync_every == 0:
            sync_run(self.output, self.persistent)
        if self.session_minutes > 0 and time.monotonic() - self.started >= self.session_minutes * 60:
            self.time_limit_reached = True
            trainer.should_stop = True
            print("Session budget reached at an epoch boundary; checkpointing for manual resume.", flush=True)


class VisibleCheckpointIO(StreamingCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        print(f"Writing local checkpoint: {Path(path).name}", flush=True)
        super().save_checkpoint(checkpoint, path, storage_options)


def get_config(settings):
    subject = validate_subject(settings["subject"])
    if not torch.cuda.is_available():
        raise RuntimeError("Select Runtime > Change runtime type > GPU in Colab first")
    precision = settings.get("precision", "auto")
    if precision == "auto":
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; select precision='auto' or '16-mixed'")
    args = SimpleNamespace(
        command="train", modality="eeg", subject=subject,
        data_root=Path(settings["local_data"]), cache_root=Path(settings["cache_root"]),
        run_dir=Path(settings["local_run"]), seed=int(settings["seed"]),
        batch_size=int(settings["batch_size"]), epochs=int(settings["max_epochs"]),
        precision=precision, context_mode=settings.get("context_mode", "sentence"), resume=None,
    )
    cfg = make_config(args)
    cfg["patience"] = int(settings["patience"])
    if cfg["n_epochs"] < 1 or cfg["patience"] < 1 or args.batch_size < 1:
        raise ValueError("Epochs, patience and batch size must be positive")
    return cfg


def build_data(cfg, output, splits=("train", "val")):
    xp = Experiment(**cfg)
    xp.data.study.infra_timelines.item_uid = timeline_cache_key
    xp.data.neuro.infra.item_uid = neuro_cache_key
    audit = audit_events(xp, output)
    loaders = xp.data.build(splits=splits)
    with (output / "sentence_splits.csv").open(encoding="utf-8", newline="") as stream:
        prompts = {row["sentence_UID"]: row["sentence"] for row in csv.DictReader(stream)}
    for split, loader in loaders.items():
        if len(loader.dataset) != audit["splits"][split]["keystrokes"]:
            raise RuntimeError(f"Dropped or missing EEG segments in {split}")
        for segment in loader.dataset.segments:
            segment.trigger.extra["sentence"] = prompts[segment.trigger.extra["sentence_UID"]]
    return xp, loaders, audit


def build_module(xp, loader, settings):
    first = next(iter(loader))
    hidden = xp.brain_model_config.hidden
    brain = xp.brain_model_config.build(n_in_channels=first.data["neuro"].shape[1], n_outputs=hidden)
    transformer = xp.transformer_config.build(dim=hidden)
    mode = settings.get("context_mode", "sentence")
    from ..context import LOCAL_CONTEXT

    left, right = LOCAL_CONTEXT.get(mode, (0, 0))
    module = ResearchModule(
        brain, transformer, xp.loss.build(), xp.optimizer,
        research_mode="sentence" if mode == "sentence" else "local", left=left, right=right,
    )
    materialize_lazy_params(module, loader)
    return module


def restore_run(output, persistent):
    if not (persistent / "last.ckpt").exists():
        raise RuntimeError("No resumable last.ckpt on Drive. Use a new run_name to start a new experiment.")
    for source in persistent.rglob("*"):
        if source.is_file() and source.suffix in {".json", ".csv", ".ckpt", ".yaml"} and source.name != "sync_manifest.json":
            target = output / source.relative_to(persistent)
            print(f"Restoring {source.relative_to(persistent)}", flush=True)
            copy_atomic(source, target)


def export_validation(module, output, split="val"):
    rows = module.validation_rows[module.primary_delay]
    with (output / f"{split}_events.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = module.validation_summary[str(module.primary_delay)]["all_target"]
    write_json(output / f"{split}_metrics.json", metrics)
    print(f"{split}: CER={metrics['CER_corpus']:.2%}, key accuracy={metrics['accuracy']:.2%}", flush=True)
    return metrics


def run(settings, command):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cfg = get_config(settings)
    output = Path(settings["local_run"])
    persistent = Path(settings["drive_run"])
    if command == "smoke":
        output, persistent = output / "smoke", persistent / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    persistent.mkdir(parents=True, exist_ok=True)
    signature = {
        "experiment": cfg, "monitor": settings["monitor"],
        "initialization": "from_scratch_joint", "protocol": "colab-v1-1",
        "source_sha256": settings.get("source_sha256"),
    }
    # Normalize tuples from the official configuration before comparing saved JSON.
    signature = json.loads(json.dumps(signature))
    prior = read_json(persistent / "config.json")
    state = read_json(persistent / "status.json")
    has_checkpoint = any((persistent / name).exists() for name in ("best.ckpt", "last.ckpt"))
    if prior and prior != signature and (has_checkpoint or state.get("state") != "data_checked"):
        raise RuntimeError("Training settings changed. Restore previous settings to resume, or choose a new run_name.")
    if command == "train" and state.get("state") == "complete":
        print("This run is already complete. It will not start training again.", flush=True)
        return
    resumed = False
    if command == "train" and prior and (has_checkpoint or state.get("completed_epochs", 0) > 0):
        if not settings.get("resume", True):
            raise RuntimeError("Run already exists. Enable resume or choose a new run_name.")
        restore_run(output, persistent)
        resumed = True
    if command == "evaluate":
        restore_run(output, persistent)
    write_json(output / "config.json", signature)
    pl.seed_everything(int(settings["seed"]), workers=True)
    save_environment(output)
    print(f"GPU: {torch.cuda.get_device_name(0)} | precision: {cfg['precision']}", flush=True)
    print("Preparing EEG and checking sentence-level splits ...", flush=True)
    splits = ("test",) if command == "evaluate" else ("train", "val")
    xp, loaders, audit = build_data(cfg, output, splits)
    if command == "check":
        if not has_checkpoint and state.get("state") != "complete":
            write_json(output / "status.json", {"state": "data_checked", "completed_epochs": 0})
        sync_run(output, persistent)
        print("Data checks passed. Next run the short smoke test, then training.", flush=True)
        return
    module = build_module(xp, next(iter(loaders.values())), settings)
    if command == "evaluate":
        if not settings.get("run_test", False):
            raise RuntimeError("Set run_test=True explicitly to evaluate the held-out test split")
        checkpoint = torch.load(output / "best.ckpt", map_location="cpu", weights_only=False, mmap=True)
        module.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        trainer = pl.Trainer(accelerator="gpu", devices=1, precision=cfg["precision"], logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        trainer.validate(module, dataloaders=loaders["test"], verbose=False)
        export_validation(module, output, "test")
        sync_run(output, persistent)
        return
    smoke = command == "smoke"
    monitor = settings["monitor"]
    if monitor not in {"val_ce_all", "val_CER_corpus"}:
        raise ValueError("monitor must be val_ce_all or val_CER_corpus")
    progress = ColabProgress(output, persistent, audit["splits"]["train"]["keystrokes"],
                             session_minutes=0 if smoke else settings["session_minutes"],
                             sync_every=settings["sync_every_epochs"], smoke=smoke)
    best = ModelCheckpoint(dirpath=output, filename="best", monitor=monitor, mode="min",
                           save_top_k=1, save_weights_only=True, enable_version_counter=False,
                           save_on_train_epoch_end=False)
    early = EarlyStopping(monitor=monitor, mode="min", patience=cfg["patience"],
                          check_finite=True, check_on_train_epoch_end=False)
    callbacks = [progress] if smoke else [progress, early, best,
        ModelCheckpoint(dirpath=output, save_top_k=0, save_last=True, save_on_train_epoch_end=False)]
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=2 if smoke else cfg["n_epochs"],
        precision=cfg["precision"], callbacks=callbacks, plugins=[VisibleCheckpointIO(output)],
        logger=CSVLogger(output, name="logs"), enable_progress_bar=False,
        enable_checkpointing=not smoke, num_sanity_val_steps=0, use_distributed_sampler=False,
        default_root_dir=output, log_every_n_steps=1 if smoke else 5,
        limit_train_batches=4 if smoke else 1.0, limit_val_batches=2 if smoke else 1.0,
        gradient_clip_val=1.0,
    )
    write_json(output / "status.json", {"state": "starting", "resumed": resumed})
    try:
        trainer.fit(module, loaders["train"], loaders["val"],
                    ckpt_path=str(output / "last.ckpt") if resumed else None)
        if smoke:
            write_json(output / "status.json", {"state": "smoke_passed", "formal_training": False})
        else:
            finished = trainer.current_epoch >= cfg["n_epochs"] or early.stopped_epoch > 0
            if trainer.interrupted:
                finished = False
            if not (output / "last.ckpt").exists():
                raise RuntimeError("Training ended without a resumable last.ckpt")
            if finished:
                trainer.validate(module, dataloaders=loaders["val"], ckpt_path=best.best_model_path, verbose=False)
                metrics = export_validation(module, output)
                checkpoint = torch.load(best.best_model_path, map_location="cpu", weights_only=False, mmap=True)
                write_json(output / "summary.json", {
                    "seed": settings["seed"], "initialization": "from_scratch_joint",
                    "best_epoch": checkpoint["epoch"] + 1, "checkpoint_monitor": monitor,
                    "validation": metrics, "test_evaluated": False,
                    "trainable_parameters": sum(p.numel() for p in module.parameters()),
                })
                del checkpoint
            latest = read_json(output / "status.json")
            latest.update(state="complete" if finished else "paused_session_limit",
                          resumed=resumed, test_evaluated=False)
            write_json(output / "status.json", latest)
        sync_run(output, persistent)
        print("Smoke passed." if smoke else f"Run state: {read_json(output / 'status.json')['state']}", flush=True)
    except BaseException:
        latest = read_json(output / "status.json")
        latest.update(state="interrupted", updated_unix=time.time())
        write_json(output / "status.json", latest)
        try:
            sync_run(output, persistent)
        except Exception as sync_error:
            print(f"Checkpoint sync was incomplete: {type(sync_error).__name__}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "smoke", "train", "evaluate"])
    parser.add_argument("--settings", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    run(settings, args.command)


if __name__ == "__main__":
    main()
