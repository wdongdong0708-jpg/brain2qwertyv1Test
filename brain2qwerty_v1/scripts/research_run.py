import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ..checkpoint_io import StreamingCheckpointIO
from ..main import Experiment
from ..research import ResearchModule, delayed_availability, sentence_layout, state_digest
from ..utils import materialize_lazy_params
from .single_subject import (
    audit_events,
    make_config,
    neuro_cache_key,
    save_environment,
    timeline_cache_key,
    write_status,
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResearchProgress(pl.Callback):
    """Persist epoch metrics and verify dense supervision against the actual loader."""

    def __init__(self, output, expected_counts, smoke):
        self.output, self.expected_counts, self.smoke = output, expected_counts, smoke
        self.started = time.monotonic()

    def on_train_epoch_start(self, trainer, module):
        module.epoch_target_counts = {}

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        for delay, count in module.last_target_counts.items():
            module.epoch_target_counts[delay] = (
                module.epoch_target_counts.get(delay, 0) + count
            )

    def on_train_epoch_end(self, trainer, module):
        if not self.smoke and module.epoch_target_counts != self.expected_counts:
            raise RuntimeError(
                f"Supervision coverage differs: {module.epoch_target_counts} != {self.expected_counts}"
            )
        metrics = {
            key: float(value.detach().cpu())
            for key, value in trainer.callback_metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        if any(not math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Nonfinite training/validation metric")
        write_status(
            self.output,
            state="training",
            completed_epochs=trainer.current_epoch + 1,
            global_step=trainer.global_step,
            metrics=metrics,
            target_counts=module.epoch_target_counts,
            cnn_frozen=trainer.current_epoch < module.freeze_cnn_epochs,
            elapsed_seconds=round(time.monotonic() - self.started, 1),
        )
        print(
            f"Epoch {trainer.current_epoch + 1}: {metrics}; targets={module.epoch_target_counts}",
            flush=True,
        )


def correction_summary(rows_by_delay):
    if set(rows_by_delay) != {0, 1, 2}:
        return {}
    keyed = {
        delay: {(row["sentence_UID"], row["position"]): row for row in rows}
        for delay, rows in rows_by_delay.items()
    }
    if any(set(rows) != set(keyed[0]) for rows in keyed.values()):
        raise RuntimeError("Revision delay predictions have different target IDs")
    result = {}
    for before, after in ((0, 1), (1, 2), (0, 2)):
        pairs = [
            (row, keyed[after][key])
            for key, row in keyed[before].items()
            if row["position"] + 2 < row["length"]
        ]
        result[f"K{before}_to_K{after}"] = {
            "common_events": len(pairs),
            "wrong_to_right": sum(
                a["prediction"] != a["label"] and b["prediction"] == b["label"]
                for a, b in pairs
            ),
            "right_to_wrong": sum(
                a["prediction"] == a["label"] and b["prediction"] != b["label"]
                for a, b in pairs
            ),
            "unchanged_prediction": sum(
                a["prediction"] == b["prediction"] for a, b in pairs
            ),
        }
    for delay, rows in rows_by_delay.items():
        waits = [
            row["signal_wait_seconds"]
            for row in rows
            if row["position"] + 2 < row["length"]
        ]
        result[f"K{delay}_signal_wait_seconds"] = (
            {
                name: float(np.quantile(waits, q))
                for name, q in (("median", 0.5), ("p90", 0.9), ("p95", 0.95), ("max", 1))
            }
            if waits
            else {}
        )
    return result


def run(job, root, smoke=False):
    output = root / ("smoke" if smoke else "runs") / job["job_id"]
    output.mkdir(parents=True, exist_ok=True)
    arguments = SimpleNamespace(
        command="smoke" if smoke else "train",
        modality="eeg",
        subject="002",
        data_root=Path(job["data_root"]),
        cache_root=Path(job["cache_root"]),
        context_mode="sentence",
        run_dir=output,
        seed=job["seed"],
        batch_size=64,
        epochs=2 if smoke else 300,
        precision="bf16-mixed",
        resume=None,
    )
    cfg = make_config(arguments)
    xp = Experiment(**cfg)
    xp.data.study.infra_timelines.item_uid = timeline_cache_key
    xp.data.neuro.infra.item_uid = neuro_cache_key
    config = {
        "experiment": xp.model_dump(mode="json"),
        "research": job,
        "smoke_test_only": smoke,
    }
    config_path = output / "config.json"
    if (
        config_path.exists()
        and json.loads(config_path.read_text(encoding="utf-8")) != config
    ):
        raise RuntimeError(
            "Existing run configuration differs; use a new output directory"
        )
    write_json(config_path, config)
    save_environment(output)
    environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    environment["research_source_sha256"] = {
        str(path): file_digest(path)
        for path in (
            Path("brain2qwerty_v1/research.py"),
            Path(__file__),
            Path("brain2qwerty_v1/scripts/research_queue.py"),
        )
    }
    write_json(output / "environment.json", environment)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.cuda.reset_peak_memory_stats()
    pl.seed_everything(job["seed"], workers=True)
    started = time.monotonic()
    write_status(output, state="building_data", job_id=job["job_id"], seed=job["seed"])
    audit = audit_events(xp, output)
    if not smoke:
        reference = Path(job["reference_run"])
        columns = ["timeline", "subject", "sentence_UID", "sentence", "split"]
        current = (
            pd.read_csv(output / "sentence_splits.csv")[columns]
            .sort_values(columns)
            .reset_index(drop=True)
        )
        baseline = (
            pd.read_csv(reference / "sentence_splits.csv")[columns]
            .sort_values(columns)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(current, baseline)
        write_json(
            output / "split_audit.json",
            {
                "same_splits_as_reference": True,
                "manifest_sha256": file_digest(output / "sentence_splits.csv"),
                "counts": audit["splits"],
            },
        )
    loaders = xp.data.build(splits=("train", "val"))
    for split, loader in loaders.items():
        segments = loader.dataset.segments
        if len(segments) != audit["splits"][split]["keystrokes"]:
            raise RuntimeError(f"Unexpected dropped keystroke segments in {split}")
        groups, _, _ = sentence_layout(
            [segment.trigger.extra["sentence_UID"] for segment in segments]
        )
        for group in groups.values():
            onsets = [float(segments[index].trigger.start) for index in group]
            if onsets != sorted(onsets):
                raise RuntimeError("Sentence keystrokes are not chronological")
    first = next(iter(loaders["train"]))
    hidden = xp.brain_model_config.hidden
    brain = xp.brain_model_config.build(
        n_in_channels=first.data["neuro"].shape[1], n_outputs=hidden
    )
    transformer = (
        torch.nn.Identity()
        if job["mode"] == "cnn_only"
        else xp.transformer_config.build(dim=hidden)
    )
    module = ResearchModule(
        brain,
        transformer,
        xp.loss.build(),
        xp.optimizer,
        research_mode=job["mode"],
        left=job["left"],
        right=job["right"],
        delays=(0, 1, 2),
        freeze_cnn_epochs=0 if job["mode"] == "cnn_only" else 1,
    )
    materialize_lazy_params(module, loaders["train"])
    initialization = {"seed": job["seed"], "mode": job["mode"]}
    if job["mode"] != "cnn_only":
        encoder = (
            root
            / ("smoke" if smoke else "runs")
            / "cnn_pretrain_s33"
            / "pretrained_encoder.pt"
        )
        if not encoder.is_file():
            raise RuntimeError(f"Missing common CNN pretraining: {encoder}")
        saved = torch.load(encoder, map_location="cpu", weights_only=False, mmap=True)
        module.model.load_state_dict(saved["state_dict"], strict=True)
        initialization["encoder_file_sha256"] = file_digest(encoder)
        del saved
    initialization["encoder_state_sha256"] = state_digest(module.model.state_dict())
    initialization["context_state_sha256"] = state_digest(
        {
            **{f"transformer.{k}": v for k, v in module.transformer.state_dict().items()},
            **{f"linear.{k}": v for k, v in module.linear.state_dict().items()},
        }
    )
    anchor = root / ("smoke" if smoke else "runs") / "r0_s33" / "initialization.json"
    if job["mode"] != "cnn_only" and anchor.exists():
        expected = json.loads(anchor.read_text(encoding="utf-8"))
        if any(
            initialization[key] != expected[key]
            for key in (
                "encoder_state_sha256",
                "context_state_sha256",
                "encoder_file_sha256",
            )
        ):
            raise RuntimeError("CNN or context initialization differs from r0")
    write_json(output / "initialization.json", initialization)
    uids = [
        segment.trigger.extra["sentence_UID"]
        for segment in loaders["train"].dataset.segments
    ]
    horizons = (0, 1, 2) if job["mode"] == "revision" else (module.primary_delay,)
    expected_counts = {
        delay: (int(valid.sum()) if job["mode"] == "revision" else len(uids))
        for delay, valid in delayed_availability(uids, horizons).items()
    }
    write_json(
        output / "supervision_audit.json",
        {
            "expected_targets_per_epoch": expected_counts,
            "original_batches_per_epoch": len(loaders["train"]),
            "test_evaluated": False,
        },
    )
    monitor = module.checkpoint_monitor
    best = ModelCheckpoint(
        dirpath=output,
        filename="best",
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        enable_version_counter=False,
    )
    callbacks = [
        best,
        EarlyStopping(monitor=monitor, mode="min", patience=2 if smoke else 30),
        ResearchProgress(output, expected_counts, smoke),
    ]
    if not smoke:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output, filename="resume", save_top_k=0, save_last=True
            )
        )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=2 if smoke else 300,
        precision="bf16-mixed",
        callbacks=callbacks,
        plugins=[StreamingCheckpointIO(output)],
        logger=CSVLogger(output, name="logs"),
        default_root_dir=output,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        log_every_n_steps=5,
        limit_train_batches=8 if smoke else 1.0,
        limit_val_batches=4 if smoke else 1.0,
    )
    last = output / "last.ckpt"
    trainer.fit(
        module,
        loaders["train"],
        loaders["val"],
        ckpt_path=str(last) if last.exists() and not smoke else None,
    )
    if not best.best_model_path:
        raise RuntimeError("No validation-selected checkpoint")
    write_status(output, state="validating_best", best_checkpoint=best.best_model_path)
    trainer.validate(
        module, dataloaders=loaders["val"], ckpt_path=best.best_model_path, verbose=False
    )
    rows = [
        dict(row, delay=delay)
        for delay, values in module.validation_rows.items()
        for row in values
    ]
    with (output / "validation_events.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "validation_summary.json", module.validation_summary)
    checkpoint = torch.load(
        best.best_model_path, map_location="cpu", weights_only=False, mmap=True
    )
    summary = {
        "job_id": job["job_id"],
        "seed": job["seed"],
        "research": job,
        "smoke_test_only": smoke,
        "test_evaluated": False,
        "checkpoint_monitor": monitor,
        "best_value": float(best.best_model_score),
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "validation": module.validation_summary,
        "primary_delay": module.primary_delay,
        "corrections": correction_summary(module.validation_rows)
        if job["mode"] == "revision"
        else {},
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "trainable_and_frozen_parameters": sum(p.numel() for p in module.parameters()),
        "supervision_targets_per_epoch": expected_counts,
    }
    if job["mode"] == "cnn_only":
        torch.save(
            {
                "state_dict": module.model.state_dict(),
                "source_checkpoint_sha256": file_digest(Path(best.best_model_path)),
            },
            output / "pretrained_encoder.pt",
        )
    write_json(output / "summary.json", summary)
    write_status(
        output,
        state="complete",
        best_epoch=summary["best_epoch"],
        best_value=summary["best_value"],
        test_evaluated=False,
        elapsed_seconds=summary["elapsed_seconds"],
    )
    print(
        f"Completed {job['job_id']}; validation only; best epoch={summary['best_epoch']}",
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one single-seed context research experiment"
    )
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    try:
        run(job, args.root.resolve(), args.smoke)
    except BaseException as error:
        output = args.root / ("smoke" if args.smoke else "runs") / job["job_id"]
        output.mkdir(parents=True, exist_ok=True)
        write_status(output, state="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
