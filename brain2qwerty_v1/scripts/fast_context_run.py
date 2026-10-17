import argparse
import csv
import faulthandler
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from .research_queue import (
    check_smoke,
    completed,
    plan,
    plot_results,
    read_json,
    report,
    write_json,
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def heartbeat(root, phase, **fields):
    write_json(
        root / "heartbeat.json",
        {"pid": os.getpid(), "phase": phase, "updated_unix": time.time(), **fields},
    )


def cache_features(root, encoder_run):
    import pandas as pd
    import torch

    from ..main import Experiment
    from ..research import ResearchModule
    from ..utils import BUTTON_MAPPING, materialize_lazy_params
    from .single_subject import neuro_cache_key, timeline_cache_key

    source = encoder_run / "pretrained_encoder.pt"
    if read_json(encoder_run / "status.json").get("state") != "complete":
        raise RuntimeError("CNN pretraining is not complete")
    fingerprint = {
        "encoder_sha256": digest(source),
        "encoder_config_sha256": digest(encoder_run / "config.json"),
        "split_sha256": digest(encoder_run / "sentence_splits.csv"),
        "version": 1,
    }
    target = root / "features.pt"
    if target.exists():
        saved = torch.load(target, map_location="cpu", weights_only=False, mmap=True)
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError("Feature cache belongs to a different CNN or data split")
        heartbeat(root, "cache_complete", cached=True)
        return
    heartbeat(root, "cache_building_data")
    cfg = read_json(encoder_run / "config.json")["experiment"]
    xp = Experiment(**cfg)
    xp.data.batch_size = xp.data.val_batch_size = 256
    xp.data.study.infra_timelines.item_uid = timeline_cache_key
    xp.data.neuro.infra.item_uid = neuro_cache_key
    loaders = xp.data.build(splits=("train", "val"))
    prompts = (
        pd.read_csv(encoder_run / "sentence_splits.csv")
        .set_index("sentence_UID")["sentence"]
        .to_dict()
    )
    first = next(iter(loaders["train"]))
    hidden = xp.brain_model_config.hidden
    brain = xp.brain_model_config.build(
        n_in_channels=first.data["neuro"].shape[1], n_outputs=hidden
    )
    wrapper = ResearchModule(
        brain,
        torch.nn.Identity(),
        xp.loss.build(),
        xp.optimizer,
        research_mode="cnn_only",
        left=0,
        right=0,
    )
    materialize_lazy_params(wrapper, loaders["train"])
    saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    brain.load_state_dict(saved["state_dict"], strict=True)
    del saved
    brain.eval().to("cuda")
    payload = {
        "fingerprint": fingerprint,
        "splits": {},
        "encoder_frozen": True,
        "label_mapping_verified": True,
    }
    audit = read_json(encoder_run / "data_audit.json")
    started = time.monotonic()
    for split, loader in loaders.items():
        features, labels, metadata = [], [], []
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                values = {name: value.to("cuda") for name, value in batch.data.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    encoded = brain(
                        values["neuro"], values["subject_id"], values["channel_positions"]
                    )
                features.append(encoded.float().cpu())
                actual_labels = batch.data["feature"].flatten().long().cpu()
                labels.append(actual_labels)
                for label, segment in zip(actual_labels.tolist(), batch.segments):
                    trigger = segment.trigger
                    button = getattr(trigger, "button", None) or trigger.extra.get(
                        "button"
                    )
                    if button not in BUTTON_MAPPING or BUTTON_MAPPING[button] != label:
                        raise RuntimeError(
                            "Cached class label does not match the source keystroke"
                        )
                    uid = trigger.extra["sentence_UID"]
                    metadata.append(
                        {
                            "sentence_UID": uid,
                            "onset": float(trigger.start),
                            "prompt": prompts[uid],
                        }
                    )
                heartbeat(
                    root,
                    "cache_extracting",
                    split=split,
                    completed_batches=index + 1,
                    total_batches=len(loader),
                )
        embeddings = torch.cat(features)
        if (
            len(embeddings) != audit["splits"][split]["keystrokes"]
            or not torch.isfinite(embeddings).all()
        ):
            raise RuntimeError("Feature cache has missing or nonfinite events")
        payload["splits"][split] = {
            "features": embeddings,
            "labels": torch.cat(labels),
            "metadata": metadata,
        }
    temporary = root / "features.tmp.pt"
    torch.save(payload, temporary)
    temporary.replace(target)
    write_json(
        root / "feature_cache_audit.json",
        {
            **fingerprint,
            "encoder_frozen": True,
            "label_mapping_verified": True,
            "shapes": {
                key: list(value["features"].shape)
                for key, value in payload["splits"].items()
            },
            "elapsed_seconds": time.monotonic() - started,
            "test_evaluated": False,
        },
    )
    heartbeat(root, "cache_complete")


def prepare_joint(root, output, job, cached):
    """Build raw EEG batches and verify them against the frozen-feature control."""
    import torch

    from ..fast_context import build_fast_module
    from ..main import Experiment
    from ..research import ResearchModule
    from ..utils import materialize_lazy_params
    from .single_subject import neuro_cache_key, timeline_cache_key

    encoder_run = Path(job["encoder_run"])
    source = encoder_run / "pretrained_encoder.pt"
    if digest(source) != cached["fingerprint"]["encoder_sha256"]:
        raise RuntimeError("Joint and frozen experiments use different CNN weights")
    xp = Experiment(**read_json(encoder_run / "config.json")["experiment"])
    xp.data.batch_size = xp.data.val_batch_size = 256
    xp.data.study.infra_timelines.item_uid = timeline_cache_key
    xp.data.neuro.infra.item_uid = neuro_cache_key
    loaders = xp.data.build(splits=("train", "val"))
    audit = {"same_event_order_and_labels": True, "splits": {}, "test_evaluated": False}
    for split, loader in loaders.items():
        values = cached["splits"][split]
        if len(loader.dataset) != len(values["labels"]):
            raise RuntimeError(f"Joint event count differs in {split}")
        offset = 0
        for batch in loader:
            labels = batch.data["feature"].flatten().long()
            torch.testing.assert_close(labels, values["labels"][offset : offset + len(batch)])
            for segment, expected in zip(
                batch.segments, values["metadata"][offset : offset + len(batch)]
            ):
                trigger = segment.trigger
                if (
                    trigger.extra["sentence_UID"] != expected["sentence_UID"]
                    or float(trigger.start) != expected["onset"]
                ):
                    raise RuntimeError("Joint and frozen event identities differ")
                trigger.extra["sentence"] = expected["prompt"]
            offset += len(batch)
            heartbeat(root, "checking_raw_data", job_id=job["job_id"], split=split)
        audit["splits"][split] = {"events": offset, "batches": len(loader)}
    first = next(iter(loaders["train"]))
    brain = xp.brain_model_config.build(
        n_in_channels=first.data["neuro"].shape[1], n_outputs=xp.brain_model_config.hidden
    )
    wrapper = ResearchModule(
        brain, torch.nn.Identity(), xp.loss.build(), xp.optimizer, research_mode="cnn_only"
    )
    materialize_lazy_params(wrapper, loaders["train"])
    saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    brain.load_state_dict(saved["state_dict"], strict=True)
    del saved, wrapper
    brain.eval().to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = brain(
            first.data["neuro"].to("cuda"),
            first.data["subject_id"].to("cuda"),
            first.data["channel_positions"].to("cuda"),
        ).float().cpu()
    expected = cached["splits"]["train"]["features"][: len(first)]
    torch.testing.assert_close(features, expected, rtol=1e-5, atol=1e-5)
    audit["initial_feature_max_abs_error"] = float((features - expected).abs().max())
    write_json(output / "matched_control_audit.json", audit)
    # Encoder construction must not consume the small decoder's initialization RNG.
    import lightning.pytorch as pl

    pl.seed_everything(33, workers=True)
    return build_fast_module(job["mode"], job["left"], job["right"], encoder=brain), loaders


def fit_job(root, job, smoke):
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from ..checkpoint_io import StreamingCheckpointIO
    from ..fast_context import CachedFeatureDataset, build_fast_module, decoder_state
    from ..research import delayed_availability, state_digest
    from ..utils import SentenceBatchSampler
    from .research_run import ResearchProgress, correction_summary
    from .single_subject import write_status

    joint = job.get("train_encoder", False)

    class FastProgress(ResearchProgress):
        """Add frequent liveness records to the existing supervision audit."""

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            super().on_train_batch_end(trainer, module, outputs, batch, batch_idx)
            if time.monotonic() - getattr(self, "last_heartbeat", 0) > 1:
                heartbeat(
                    root,
                    "training",
                    job_id=job["job_id"],
                    epoch=trainer.current_epoch + 1,
                    global_step=trainer.global_step,
                )
                self.last_heartbeat = time.monotonic()

        def on_train_epoch_end(self, trainer, module):
            super().on_train_epoch_end(trainer, module)
            write_status(self.output, cnn_frozen=not joint, projection_trainable=True)
            heartbeat(
                root,
                "epoch_complete",
                job_id=job["job_id"],
                epoch=trainer.current_epoch + 1,
            )

        def on_validation_epoch_start(self, trainer, module):
            heartbeat(root, "validating", job_id=job["job_id"])

        def on_after_backward(self, trainer, module):
            if not joint or hasattr(self, "gradient_audit"):
                return
            self.gradient_audit = {}
            for name, group in zip(("encoder", "decoder"), module.optimizer_groups()):
                gradients = [p.grad.detach() for p in group["params"] if p.grad is not None]
                norm = torch.stack([gradient.norm() for gradient in gradients]).norm()
                if not torch.isfinite(norm) or norm <= 0:
                    raise RuntimeError(f"Missing or nonfinite gradient in {name}")
                self.gradient_audit[name] = {
                    "gradient_norm": float(norm.cpu()),
                    "parameters_with_grad": len(gradients),
                }
            write_json(self.output / "gradient_audit.json", self.gradient_audit)

    class ProgressCheckpointIO(StreamingCheckpointIO):
        def save_checkpoint(self, checkpoint, path, storage_options=None):
            heartbeat(root, "saving_checkpoint", job_id=job["job_id"], path=str(path))
            super().save_checkpoint(checkpoint, path, storage_options)
            heartbeat(root, "checkpoint_saved", job_id=job["job_id"])

    output = root / ("smoke" if smoke else "runs") / job["job_id"]
    output.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(33, workers=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    cached = torch.load(
        root / "features.pt", map_location="cpu", weights_only=False, mmap=True
    )
    config = {
        "research": job,
        "cache_fingerprint": cached["fingerprint"],
        "smoke_test_only": smoke,
    }
    previous = read_json(output / "config.json")
    if previous and previous != config:
        raise RuntimeError("Fast run configuration differs")
    write_json(output / "config.json", config)
    if joint:
        module, loaders = prepare_joint(root, output, job, cached)
    else:
        loaders = {}
        for split, values in cached["splits"].items():
            data = CachedFeatureDataset(**values, device="cuda")
            loaders[split] = DataLoader(
                data,
                batch_sampler=SentenceBatchSampler(data.segments, 256),
                collate_fn=data.collate_fn,
                num_workers=0,
                pin_memory=False,
            )
        module = build_fast_module(job["mode"], job["left"], job["right"])
    initialization = {
        "state_sha256": state_digest(decoder_state(module)),
        "cache_fingerprint": cached["fingerprint"],
    }
    if joint:
        control = read_json(
            Path(job["frozen_reference"]) / "runs" / job["job_id"] / "initialization.json"
        )
        if initialization != control:
            raise RuntimeError("Joint CNN/decoder initialization differs from frozen control")
        initialization["encoder_state_sha256"] = state_digest(
            module.model.encoder.state_dict()
        )
    anchor = root / ("smoke" if smoke else "runs") / "r0_s33" / "initialization.json"
    if anchor.exists() and read_json(anchor) != initialization:
        raise RuntimeError("Fast model initial weights differ across context modes")
    write_json(output / "initialization.json", initialization)
    write_json(
        output / "source_hashes.json",
        {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("brain2qwerty_v1/fast_context.py"),
                Path("brain2qwerty_v1/research.py"),
            )
        },
    )
    uids = [
        segment.trigger.extra["sentence_UID"]
        for segment in loaders["train"].dataset.segments
    ]
    delays = (0, 1, 2) if job["mode"] == "revision" else (module.primary_delay,)
    counts = {
        delay: int(mask.sum()) if job["mode"] == "revision" else len(uids)
        for delay, mask in delayed_availability(uids, delays).items()
    }
    write_json(
        output / "supervision_audit.json",
        {
            "expected_targets_per_epoch": counts,
            "original_batches_per_epoch": len(loaders["train"]),
            "test_evaluated": False,
        },
    )
    best = ModelCheckpoint(
        dirpath=output,
        filename="best",
        monitor=module.checkpoint_monitor,
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        enable_version_counter=False,
    )
    callbacks = [
        best,
        EarlyStopping(
            monitor=module.checkpoint_monitor, mode="min", patience=job["patience"]
        ),
        FastProgress(output, counts, False),
    ]
    if not smoke:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output, save_top_k=0, save_last=True, every_n_epochs=5 if joint else 1
            )
        )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=3 if smoke else job["max_epochs"],
        precision="bf16-mixed",
        callbacks=callbacks,
        plugins=[ProgressCheckpointIO(output)],
        logger=CSVLogger(output, name="logs"),
        default_root_dir=output,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
    )
    last = output / "last.ckpt"
    write_status(output, state="training", encoder_frozen=not joint)
    trainer.fit(
        module,
        loaders["train"],
        loaders["val"],
        ckpt_path=str(last) if last.exists() and not smoke else None,
    )
    heartbeat(root, "validating_best", job_id=job["job_id"])
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
    checkpoint = torch.load(
        best.best_model_path, map_location="cpu", weights_only=False, mmap=True
    )
    summary = {
        "research": job,
        "job_id": job["job_id"],
        "seed": 33,
        "smoke_test_only": smoke,
        "test_evaluated": False,
        "primary_delay": module.primary_delay,
        "validation": module.validation_summary,
        "best_epoch": checkpoint["epoch"] + 1,
        "best_value": float(best.best_model_score),
        "checkpoint_monitor": module.checkpoint_monitor,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "trainable_parameters": sum(
            p.numel() for p in module.parameters() if p.requires_grad
        ),
        "encoder_frozen": not joint,
        "corrections": correction_summary(module.validation_rows)
        if job["mode"] == "revision"
        else {},
    }
    if joint:
        summary["best_encoder_state_sha256"] = state_digest(module.model.encoder.state_dict())
        summary["encoder_updated"] = (
            summary["best_encoder_state_sha256"] != initialization["encoder_state_sha256"]
        )
        if not summary["encoder_updated"]:
            raise RuntimeError("CNN weights/buffers did not change during joint training")
    write_json(output / "summary.json", summary)
    write_status(
        output, state="complete", best_epoch=summary["best_epoch"], test_evaluated=False
    )
    heartbeat(root, "complete", job_id=job["job_id"])


def worker(root, stage, job, smoke, encoder_run):
    import torch

    torch.set_num_threads(4)
    heartbeat(root, "initializing")
    with (root / "worker_trace.log").open("a", encoding="utf-8") as trace:
        faulthandler.enable(file=trace)
        faulthandler.dump_traceback_later(90, repeat=True, file=trace)
        try:
            if stage == "cache":
                cache_features(root, encoder_run)
            else:
                fit_job(root, job, smoke)
        finally:
            faulthandler.cancel_dump_traceback_later()


def queue(root, encoder_run, verify_only=False, train_encoder=False):
    reference = Path("outputs/eeg-002/train").resolve()
    jobs = [
        {**job, "max_epochs": 40, "patience": 6, "profile": "frozen-cnn-small-v1"}
        for job in plan(reference, Path(".cache"))
        if job["mode"] != "cnn_only"
    ]
    if train_encoder:
        jobs = [
            {
                **job,
                "profile": "small-joint-cnn-v1",
                "train_encoder": True,
                "encoder_run": str(encoder_run),
                "frozen_reference": str(Path("outputs/eeg-002/fast_context_seed33").resolve()),
                "encoder_max_lr": 5e-5,
                "decoder_max_lr": 3e-4,
            }
            for job in jobs
            if job["job_id"] in {"full_s33", "r8_s33"}
        ]
        jobs.sort(key=lambda job: job["job_id"] != "full_s33")
    with FileLock(str(root / "queue.lock"), timeout=0):
        for folder in (root / "jobs", root / "logs"):
            folder.mkdir(exist_ok=True)
        previous = read_json(root / "plan.json")
        if previous and previous.get("jobs") != jobs:
            raise RuntimeError("Fast queue matrix differs")
        write_json(
            root / "plan.json",
            {
                "seed": 33,
                "jobs": jobs,
                "encoder_run": str(encoder_run),
                "encoder_frozen": not train_encoder,
            },
        )
        state = {
            "state": "running",
            "pid": os.getpid(),
            "seed": 33,
            "planned_jobs": len(jobs),
            "completed_jobs": [],
            "completed_smokes": [],
            "formal_training_started": False,
            "started_unix": time.time(),
        }
        write_json(root / "status.json", state)
        stages = [("cache", None, False)]
        stages += [
            ("fit", job, True)
            for job in jobs
            if job["job_id"] in {"r0_s33", "r8_s33", "full_s33", "revision_s33"}
        ]
        if not verify_only:
            stages += [("fit", job, False) for job in jobs]
        try:
            for stage, job, smoke in stages:
                if stage == "fit" and not smoke:
                    state["formal_training_started"] = True
                name = (
                    "cache"
                    if job is None
                    else ("smoke-" if smoke else "train-") + job["job_id"]
                )
                run = (
                    root / ("smoke" if smoke else "runs") / job["job_id"] if job else None
                )
                if job and completed(run, job, smoke):
                    verification = (
                        check_smoke(run) if smoke else {"already_complete": True}
                    )
                else:
                    if job:
                        write_json(root / "jobs" / f"{job['job_id']}.json", job)
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "brain2qwerty_v1.scripts.fast_context_run",
                        "--root",
                        str(root),
                        "--worker",
                        stage,
                        "--encoder-run",
                        str(encoder_run),
                    ]
                    if job:
                        command += ["--job", str(root / "jobs" / f"{job['job_id']}.json")]
                    if smoke:
                        command.append("--smoke")
                    started = time.time()
                    with (
                        (root / "logs" / f"{name}.stdout.log").open(
                            "a", encoding="utf-8"
                        ) as stdout,
                        (root / "logs" / f"{name}.stderr.log").open(
                            "a", encoding="utf-8"
                        ) as stderr,
                    ):
                        child = subprocess.Popen(
                            command,
                            stdout=stdout,
                            stderr=stderr,
                            env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"),
                            creationflags=subprocess.CREATE_NO_WINDOW
                            if os.name == "nt"
                            else 0,
                        )
                        state.update(current_job=name, child_pid=child.pid)
                        write_json(root / "status.json", state)
                        while True:
                            try:
                                code = child.wait(timeout=5)
                                break
                            except subprocess.TimeoutExpired:
                                pulse = root / "heartbeat.json"
                                updated = (
                                    pulse.stat().st_mtime if pulse.exists() else started
                                )
                                if time.time() - max(started, updated) > 120:
                                    if os.name == "nt":
                                        subprocess.run(
                                            [
                                                "taskkill",
                                                "/PID",
                                                str(child.pid),
                                                "/T",
                                                "/F",
                                            ],
                                            capture_output=True,
                                        )
                                    else:
                                        child.kill()
                                    raise RuntimeError(
                                        f"No progress for 120 seconds in {name}; see worker_trace.log"
                                    )
                    if code or (job and not completed(run, job, smoke)):
                        raise RuntimeError(f"{name} failed (exit={code}); inspect logs")
                    verification = check_smoke(run) if smoke else {"complete": True}
                if job:
                    state["completed_smokes" if smoke else "completed_jobs"].append(
                        {"job_id": job["job_id"], **verification}
                    )
                state["child_pid"] = None
                write_json(root / "status.json", state)
                report(root, jobs)
                if not smoke:
                    plot_results(root, jobs)
            state.update(
                state="ready" if verify_only else "complete",
                child_pid=None,
                formal_training_started=not verify_only,
            )
            write_json(root / "status.json", state)
            if train_encoder and not verify_only:
                from .small_joint_report import generate

                generate(root, Path(jobs[0]["frozen_reference"]))
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}")
            write_json(root / "status.json", state)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fast frozen-CNN context screening, seed33"
    )
    parser.add_argument(
        "--root", type=Path
    )
    parser.add_argument(
        "--encoder-run",
        type=Path,
        default=Path("outputs/eeg-002/context_research_seed33/runs/cnn_pretrain_s33"),
    )
    parser.add_argument("--worker", choices=["cache", "fit"])
    parser.add_argument("--job", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--train-encoder", action="store_true")
    args = parser.parse_args(argv)
    root = (
        args.root
        or Path(
            "outputs/eeg-002/small_joint_seed33"
            if args.train_encoder
            else "outputs/eeg-002/fast_context_seed33"
        )
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(
            root,
            args.worker,
            read_json(args.job) if args.job else None,
            args.smoke,
            args.encoder_run.resolve(),
        )
    else:
        queue(root, args.encoder_run.resolve(), args.verify_only, args.train_encoder)


if __name__ == "__main__":
    main()
