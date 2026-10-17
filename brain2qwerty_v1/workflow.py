import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

import studies  # noqa: F401

from .cache import cache_signature, configure_cache
from .cli import add_wandb_args, wandb_config
from .config.model_config import EEG_ENCODER, ENCODER
from .config.xp_config import experiment_config
from .context import LOCAL_CONTEXT
from .main import Experiment
from .scripts.single_subject import subject_labels
from .utils import select_participants

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def resolve_study(root: Path, modality: str) -> Path:
    """Accept study aliases, the full downloaded repository, or its parent."""
    root = root.expanduser().resolve()
    study = "Pinet2024Eeg" if modality == "eeg" else "Pinet2024Meg"
    candidates = (root / study, root / study.lower(), root / "SpanishBCBL_Full", root)
    for candidate in candidates:
        if any(
            (candidate / name).is_dir() for name in (modality.upper(), modality.lower())
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        f"No {modality.upper()} data under {root}; expected {root / study / modality.upper()}"
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Brain2Qwerty V1: local smoke / CPU cache / GPU training"
    )
    p.add_argument("mode", choices=["cache", "smoke", "train", "eval", "debug"])
    p.add_argument("--modality", choices=["eeg", "meg"], default="eeg")
    p.add_argument(
        "--subject",
        default="all",
        help="002 or S2; omitted/all selects every valid subject",
    )
    p.add_argument("--data-root", "--studies-root", type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--results-root", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Keystroke budget; whole sentences stay together",
    )
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument(
        "--devices",
        type=int,
        default=1,
        choices=[1],
        help="Single GPU (automatic GPU/CPU detection)",
    )
    p.add_argument(
        "--precision",
        choices=["auto", "32-true", "16-mixed", "bf16-mixed"],
        default="auto",
    )
    p.add_argument(
        "--context-mode", choices=["sentence", *LOCAL_CONTEXT], default="sentence"
    )
    p.add_argument("--profile", choices=["full", "small"], default="full")
    p.add_argument(
        "--train-batches",
        type=int,
        help="Limit training batches per epoch (smoke default: 2)",
    )
    p.add_argument(
        "--val-batches", type=int, help="Limit validation batches (smoke default: 2)"
    )
    p.add_argument(
        "--test-batches", type=int, help="Limit test batches (smoke default: 2)"
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="auto, or last.ckpt in the run directory",
    )
    p.add_argument("--ckpt", type=Path, help="Checkpoint for eval")
    p.add_argument(
        "--require-cache",
        action="store_true",
        help="Fail on any missing timeline/signal/position cache",
    )
    p.add_argument(
        "--cache-version",
        default="1",
        help="Change when preprocessing/alignment code changes",
    )
    p.add_argument(
        "--expected-timelines",
        type=int,
        help="Optional explicit count check on selected recordings",
    )
    add_wandb_args(p)
    return p


def make_config(args) -> dict:
    cfg = experiment_config()
    cache = (
        (args.cache_root or Path(cfg["data"]["neuro"]["infra"]["folder"]))
        .expanduser()
        .resolve()
    )
    results = (args.results_root or Path(cfg["output_dir"])).expanduser().resolve()
    study_path = resolve_study(
        args.data_root or Path(cfg["data"]["study"]["path"]), args.modality
    )
    cfg["data"]["study"].update(
        name="Pinet2024Eeg" if args.modality == "eeg" else "Pinet2024Meg",
        path=str(study_path),
        infra={"folder": str(cache / "events" / args.cache_version), "mode": "cached"},
        infra_timelines={
            "folder": str(cache),
            "cluster": None,
            "version": args.cache_version,
            "keep_in_ram": False,
        },
    )
    if args.subject.lower() != "all":
        cfg["data"]["study"]["query"] = (
            f"subject in {subject_labels(args.modality, args.subject)!r}"
        )
    cfg["data"]["neuro"].update(
        name="EegExtractor" if args.modality == "eeg" else "MegExtractor",
        mne_cpus=args.cpu_threads,
        infra={
            "folder": str(cache),
            "cluster": None,
            "version": args.cache_version,
            "keep_in_ram": False,
        },
    )
    cfg["data"].update(
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=args.mode != "cache",
        complete_sentences=True,
        include_channel_positions=args.modality == "meg",
    )
    cfg["brain_model_config"] = copy.deepcopy(
        EEG_ENCODER if args.modality == "eeg" else ENCODER
    )
    if args.profile == "small":
        cfg["brain_model_config"].update(hidden=256, depth=4, initial_linear=128)
        cfg["transformer_config"].update(depth=2)
        if args.modality == "meg":
            cfg["brain_model_config"]["merger_config"]["n_virtual_channels"] = 128
    # No CUDA query, model build, or Trainer construction on the cache path.
    precision = args.precision
    if args.mode == "cache":
        precision = "32-true"
    elif precision == "auto":
        precision = (
            ("bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed")
            if torch.cuda.is_available()
            else "32-true"
        )
    tag = f"{args.modality}-{args.subject.lower()}-{args.context_mode}-{args.profile}"
    cfg.update(
        output_dir=str(
            (args.run_dir or results / tag / args.mode).expanduser().resolve()
        ),
        seed=args.seed,
        devices=1,
        n_epochs=args.epochs or (2 if args.mode == "smoke" else 300),
        patience=2 if args.mode == "smoke" else args.patience,
        precision=precision,
        save_checkpoints=args.mode != "smoke",
        enable_progress_bar=False,
        log_every_n_steps=1,
        context_mode=args.context_mode,
    )
    for split in ("train", "val", "test"):
        value = getattr(args, f"{split}_batches")
        cfg[f"limit_{split}_batches"] = (
            value if value is not None else (2 if args.mode == "smoke" else 1.0)
        )
    return cfg


class WorkflowProgress(pl.Callback):
    """Persist live progress and verify a finite CNN gradient during the smoke test."""

    def __init__(self, output: Path):
        self.output = output
        self.started = time.monotonic()
        self.audited = False

    def on_before_optimizer_step(self, trainer, module, optimizer):
        if self.audited:
            return
        norms = [
            p.grad.detach().float().norm()
            for p in module.model.parameters()
            if p.grad is not None
        ]
        norm = torch.stack(norms).norm() if norms else torch.tensor(0.0)
        if not torch.isfinite(norm):
            if getattr(trainer.precision_plugin, "scaler", None) is not None:
                return
            raise RuntimeError("Non-finite CNN gradient")
        if norm <= 0:
            raise RuntimeError("CNN received no nonzero gradient")
        write_json(
            self.output / "gradient_audit.json",
            {"cnn_gradient_norm": float(norm), "finite": True},
        )
        self.audited = True

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is not None and not torch.isfinite(loss).all():
            raise RuntimeError("Non-finite training loss")

    def on_train_epoch_end(self, trainer, module):
        metrics = {
            k: float(v.detach().cpu())
            for k, v in trainer.callback_metrics.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Non-finite epoch metrics")
        write_json(
            self.output / "status.json",
            {
                "state": "training",
                "completed_epochs": trainer.current_epoch + 1,
                "global_step": trainer.global_step,
                "max_epochs": trainer.max_epochs,
                "elapsed_seconds": round(time.monotonic() - self.started, 2),
                "metrics": metrics,
            },
        )
        print(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}: {metrics}",
            flush=True,
        )


class WorkflowExperiment(Experiment):
    def _trainer_setup(self):
        trainer = super()._trainer_setup()
        trainer.callbacks.append(WorkflowProgress(Path(self.output_dir)))
        return trainer


def audit_loaders(loaders, output: Path) -> dict:
    splits, sentences = {}, []
    for split in ("train", "val", "test"):
        if split not in loaders or not len(loaders[split]):
            raise RuntimeError(f"Empty {split} split")
        loader = loaders[split]
        for segment in loader.dataset.segments:
            sentences.append(
                {"sentence_UID": segment.trigger.extra["sentence_UID"], "split": split}
            )
        batch = next(iter(loader))
        if not torch.isfinite(batch.data["neuro"]).all():
            raise RuntimeError(f"Non-finite {split} signal")
        splits[split] = {
            "keystrokes": len(loader.dataset),
            "batches": len(loader),
            "first_batch": {k: list(v.shape) for k, v in batch.data.items()},
        }
        print(f"{split}: {splits[split]}", flush=True)
    frame = pd.DataFrame(sentences).drop_duplicates()
    if frame.groupby("sentence_UID").split.nunique().max() != 1:
        raise RuntimeError("Sentence instance leaks across splits")
    frame.to_csv(output / "sentence_splits.csv", index=False)
    write_json(output / "data_audit.json", splits)
    return splits


def resume_config(config: dict) -> dict:
    # Saved JSON has lists where the in-memory config may contain tuples.
    config = json.loads(json.dumps(config))
    for key in (
        "output_dir",
        "ckpt_path",
        "eval_only",
        "enable_progress_bar",
        "wandb_config",
    ):
        config.pop(key, None)
    for node in (config["data"]["study"], config["data"]["neuro"]):
        for key in ("infra", "infra_timelines"):
            if key in node:
                node[key] = {"version": node[key].get("version", "0")}
    return config


def main(argv=None) -> None:
    p = parser()
    args = p.parse_args(argv)
    if args.mode == "debug":
        args.mode = "smoke"
    for name in (
        "epochs",
        "patience",
        "batch_size",
        "cpu_threads",
        "train_batches",
        "val_batches",
        "test_batches",
        "expected_timelines",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0:
        p.error("--workers must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.cache_version):
        p.error("--cache-version must contain only letters, numbers, '_' or '-'")
    if args.resume and args.mode != "train":
        p.error("--resume requires train mode")
    if args.mode == "eval" and (not args.ckpt or not args.ckpt.is_file()):
        p.error("eval requires an existing --ckpt")
    torch.set_num_threads(args.cpu_threads)
    cfg = make_config(args)
    Path(cfg["data"]["neuro"]["infra"]["folder"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["data"]["study"]["infra"]["folder"]).mkdir(parents=True, exist_ok=True)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.mode != "cache" and not torch.cuda.is_available():
        p.error("smoke/train/eval requires a CUDA GPU; select a GPU runtime")
    # Study discovery reads filenames and logs, without constructing a neural model.
    xp = WorkflowExperiment(**cfg)
    summary = xp.data.study.study_summary()
    if summary.empty:
        raise RuntimeError("No valid recordings matched this subject/modality")
    if args.expected_timelines is not None and len(summary) != args.expected_timelines:
        raise RuntimeError(
            f"Expected {args.expected_timelines} selected timelines, found {len(summary)}"
        )
    participants = select_participants(summary).subject.nunique()
    if not participants:
        raise RuntimeError(
            "All selected participants are excluded by official preprocessing"
        )
    cfg["brain_model_config"]["subject_layers_config"]["n_subjects"] = participants
    if cfg["brain_model_config"]["merger_config"] is not None:
        cfg["brain_model_config"]["merger_config"]["n_subjects"] = participants
    if args.mode == "eval":
        cfg.update(eval_only=True, ckpt_path=str(args.ckpt.resolve()))
    if args.resume:
        resume = (
            output / "last.ckpt" if args.resume == "auto" else Path(args.resume).resolve()
        )
        if resume.is_file():
            if resume.parent != output:
                p.error("Use --run-dir pointing to the checkpoint's directory")
            cfg["ckpt_path"] = str(resume)
        elif args.resume != "auto":
            p.error(f"Checkpoint does not exist: {resume}")
    if args.mode == "train" and any(output.glob("*.ckpt")) and not cfg.get("ckpt_path"):
        p.error("Run already has checkpoints; use --resume auto or another --run-dir")
    if cfg.get("ckpt_path") and args.mode == "train":
        saved = output / "config.json"
        if not saved.is_file() or resume_config(
            json.loads(saved.read_text(encoding="utf-8"))
        ) != resume_config(cfg):
            p.error(
                "Resume configuration differs; retain the original model, data, seed and total --epochs"
            )
    wb = wandb_config(args, args.mode, args.seed)
    if wb:
        cfg["wandb_config"] = wb
    write_json(output / "config.json", cfg)
    summary.to_csv(output / "recordings.csv", index=False)
    xp = WorkflowExperiment(**cfg)
    configure_cache(xp.data, read_only=args.require_cache)
    signature = cache_signature(xp.data)
    write_json(output / "cache_signature.json", signature)
    print(
        f"Selected {len(summary)} timelines, {participants} participants; cache UIDs: {signature}",
        flush=True,
    )
    started = time.monotonic()
    write_json(output / "status.json", {"state": "building_cache", "mode": args.mode})
    try:
        loaders = xp.data.build()
        audit = audit_loaders(loaders, output)
        manifest = {
            "modality": args.modality,
            "subject": args.subject,
            "study_path": str(xp.data.study.path),
            "timelines": len(summary),
            "participants": participants,
            "signature": signature,
            "splits": audit,
        }
        key = hashlib.sha256(
            json.dumps(
                {"path": manifest["study_path"], "subject": args.subject}, sort_keys=True
            ).encode()
        ).hexdigest()[:12]
        write_json(
            Path(xp.data.neuro.infra.folder)
            / "workflow_manifests"
            / f"{args.modality}-{key}.json",
            manifest,
        )
        if args.mode == "cache":
            if torch.cuda.is_initialized():
                raise RuntimeError("Cache-only execution unexpectedly initialized CUDA")
            write_json(
                output / "status.json",
                {
                    "state": "cache_complete",
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "cuda_initialized": False,
                    "trainer_initialized": xp._trainer is not None,
                },
            )
            print("CPU cache complete; no model or Trainer was initialized.", flush=True)
            return
        versions = {
            name: importlib.metadata.version(name)
            for name in ("torch", "lightning", "neuralset", "neuraltrain", "exca", "mne")
        }
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        write_json(
            output / "environment.json",
            {
                "python": platform.python_version(),
                "versions": versions,
                "gpu": torch.cuda.get_device_name(0),
                "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            },
        )
        xp.run(loaders)
        if args.mode == "smoke" and not (output / "gradient_audit.json").is_file():
            raise RuntimeError("Smoke test did not observe a finite CNN update")
        metrics = {
            k: float(v.detach().cpu())
            for k, v in xp._trainer.callback_metrics.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        }
        if not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError("Non-finite final metrics")
        write_json(
            output / "status.json",
            {
                "state": "complete",
                "mode": args.mode,
                "epochs": xp._trainer.current_epoch,
                "global_step": xp._trainer.global_step,
                "metrics": metrics,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )
    except Exception as exc:
        write_json(
            output / "status.json",
            {"state": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


if __name__ == "__main__":
    main()
