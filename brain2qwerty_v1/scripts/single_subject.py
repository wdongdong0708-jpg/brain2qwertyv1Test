import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

import studies  # noqa: F401

from ..config.xp_config import experiment_config
from ..context import LOCAL_CONTEXT
from ..main import Experiment
from ..utils import BUTTON_MAPPING, SUBJECT_MERGE
from .extract_predictions import compute_cer, compute_wer, process_json


def write_status(output: Path, **updates) -> None:
    status_path = output / "status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )
    status.update(
        updates, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat()
    )
    temporary = output / "status.tmp.json"
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(status_path)


class RunProgress(pl.Callback):
    """Persist epoch progress so a background experiment can be inspected."""

    def __init__(self, output: str):
        self.output = Path(output)
        self.started = time.monotonic()

    def on_fit_start(self, trainer, pl_module):
        write_status(self.output, state="training", max_epochs=trainer.max_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {
            key: float(value.detach().cpu())
            for key, value in trainer.callback_metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        checkpoint = trainer.checkpoint_callback
        write_status(
            self.output,
            state="training",
            completed_epochs=trainer.current_epoch + 1,
            global_step=trainer.global_step,
            elapsed_seconds=round(time.monotonic() - self.started, 1),
            metrics=metrics,
            best_checkpoint=checkpoint.best_model_path if checkpoint else None,
        )
        print(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}: {metrics}",
            flush=True,
        )


class SingleSubjectExperiment(Experiment):
    """Official V1 experiment with a local progress file."""

    def _trainer_setup(self):
        trainer = super()._trainer_setup()
        trainer.callbacks.append(RunProgress(self.output_dir))
        return trainer


def timeline_cache_key(timeline) -> str:
    payload = json.dumps(timeline, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def neuro_cache_key(event) -> str:
    return hashlib.sha256(event._splittable_event_uid().encode()).hexdigest()


def subject_labels(modality: str, subject: str) -> list[str]:
    number = int(subject.rsplit("/", 1)[-1].lstrip("Ss"))
    if number < 1:
        raise ValueError("Subject number must be positive")
    if modality == "eeg":
        return [f"Pinet2024Eeg/{number:03d}"]
    label = f"Pinet2024Meg/S{number}"
    label = SUBJECT_MERGE.get(label, label)
    if label.endswith("/S23"):
        raise ValueError("MEG S23 is excluded by the official preprocessing")
    return [label] + [
        alias for alias, canonical in SUBJECT_MERGE.items() if canonical == label
    ]


def make_config(args) -> dict:
    labels = subject_labels(args.modality, args.subject)
    cfg = experiment_config()
    cfg["brain_model_config"] = copy.deepcopy(cfg["brain_model_config"])
    cfg["brain_model_config"]["subject_layers_config"].update(n_subjects=1)
    cfg["brain_model_config"]["merger_config"]["n_subjects"] = 1
    cache = str(args.cache_root.resolve())
    query = f"subject in {labels!r}"
    if args.command == "smoke":
        query += " and subject_timeline_index == 0"
    cfg["data"]["study"].update(
        name="Pinet2024Eeg" if args.modality == "eeg" else "Pinet2024Meg",
        path=str(args.data_root.resolve()),
        query=query,
        infra={"folder": cache, "mode": "retry"},
        infra_timelines={"folder": cache, "cluster": None},
    )
    cfg["data"]["neuro"].update(
        name="EegExtractor" if args.modality == "eeg" else "MegExtractor",
        infra={"folder": cache, "cluster": None},
    )
    cfg["data"].update(
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        complete_sentences=True,
    )
    run_kind = "smoke" if args.command == "smoke" else "train"
    if args.command in {"inspect", "cache", "eval"}:
        run_kind = args.command
    subject_tag = labels[0].rsplit("/", 1)[-1]
    context_mode = args.context_mode or "sentence"
    default_root = Path("outputs") / f"{args.modality}-{subject_tag}"
    if context_mode != "sentence":
        default_root = default_root / context_mode
    output = args.run_dir or (default_root / run_kind)
    cfg.update(
        output_dir=str(output.resolve()),
        seed=args.seed,
        devices=1,
        n_epochs=args.epochs or (2 if args.command == "smoke" else 300),
        patience=2 if args.command == "smoke" else 30,
        save_checkpoints=args.command != "smoke",
        precision=args.precision,
        enable_progress_bar=False,
        context_mode=context_mode,
    )
    if args.command == "smoke":
        cfg.update(limit_train_batches=8, limit_val_batches=4, limit_test_batches=4)
    if args.command == "eval":
        if args.ckpt is None:
            raise ValueError("eval requires --ckpt pointing to the trained best.ckpt")
        cfg.update(eval_only=True, ckpt_path=str(args.ckpt.resolve()))
    elif args.resume:
        cfg["ckpt_path"] = str(args.resume.resolve())
    return cfg


def verify_reference(output: Path, reference: Path, config: dict) -> None:
    import pandas as pd

    reference_config = json.loads((reference / "config.json").read_text(encoding="utf-8"))
    allowed = {"context_mode", "output_dir", "ckpt_path", "eval_only"}
    for key in set(config) | set(reference_config):
        if key not in allowed and config.get(key) != reference_config.get(key):
            raise RuntimeError(f"Configuration differs from baseline at {key!r}")
    current = pd.read_csv(output / "sentence_splits.csv")
    baseline = pd.read_csv(reference / "sentence_splits.csv")
    columns = sorted(baseline.columns)
    pd.testing.assert_frame_equal(
        current[columns].sort_values(columns).reset_index(drop=True),
        baseline[columns].sort_values(columns).reset_index(drop=True),
    )
    report = {
        "reference_run": str(reference.resolve()),
        "same_data_splits": True,
        "same_model_and_training_configuration": True,
        "reference_context": reference_config.get("context_mode", "sentence"),
        "context_mode": config["context_mode"],
        "transformer_input_window": LOCAL_CONTEXT.get(config["context_mode"]),
    }
    (output / "comparison_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Baseline configuration and sentence splits match.", flush=True)


def audit_events(xp: Experiment, output: Path) -> dict:
    events = xp.data.build_events()
    keys = events.loc[events.type == "Keystroke"].copy()
    if keys.empty or keys["subject"].nunique() != 1:
        raise RuntimeError("Expected keystrokes from exactly one participant")
    unknown = set(keys["button"].unique()) - set(BUTTON_MAPPING)
    if unknown:
        raise RuntimeError(f"Unsupported character labels: {sorted(unknown)}")
    if keys["split"].isna().any():
        raise RuntimeError("Unassigned keystrokes in the train/val/test split")
    if keys.groupby("sentence")["split"].nunique().max() != 1:
        raise RuntimeError("The same sentence text occurs in multiple splits")
    if keys.groupby("sentence_UID")["split"].nunique().max() != 1:
        raise RuntimeError("The same sentence instance occurs in multiple splits")
    splits = {}
    for split in ("train", "val", "test"):
        selected = keys.loc[keys["split"] == split]
        if selected.empty:
            raise RuntimeError(f"Empty {split} split; a longer recording is needed")
        splits[split] = {
            "keystrokes": len(selected),
            "sentence_instances": selected["sentence_UID"].nunique(),
            "unique_texts": selected["sentence"].nunique(),
        }
    columns = ["timeline", "subject", "sentence_UID", "sentence", "split"]
    keys[columns].drop_duplicates().to_csv(output / "sentence_splits.csv", index=False)
    report = {
        "one_participant": True,
        "no_identical_sentence_text_across_splits": True,
        "splits": splits,
        "max_sentence_keystrokes": int(keys.groupby("sentence_UID").size().max()),
    }
    (output / "data_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def save_environment(output: Path) -> None:
    names = [
        "torch",
        "torchvision",
        "neuralset",
        "neuraltrain",
        "neuralfetch",
        "exca",
        "lightning",
        "torchmetrics",
        "x-transformers",
        "numpy",
        "pandas",
        "scipy",
        "mne",
        "dtw-python",
    ]
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "base_environment": sys.base_prefix,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {name: importlib.metadata.version(name) for name in names},
        "local_source_sha256": {
            str(source): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in [
                Path("brain2qwerty_v1/context.py"),
                Path("brain2qwerty_v1/pl_module.py"),
                Path("brain2qwerty_v1/main.py"),
                Path("brain2qwerty_v1/scripts/single_subject.py"),
            ]
        },
    }
    provenance = Path(".setup/upstream.json")
    if provenance.exists():
        info["upstream"] = json.loads(provenance.read_text(encoding="utf-8"))
    (output / "environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def export_results(output: Path, smoke: bool, context_mode: str = "sentence") -> None:
    source = output / "callbacks/test_all_sentences.json"
    frame = process_json(str(source))
    if frame.empty or frame["True Sentences"].str.len().eq(0).any():
        raise RuntimeError("Prediction export is empty or missing reference sentences")
    frame["CER_typed"] = [
        compute_cer(t, p)
        for t, p in zip(frame["Typed Sentences"], frame["Model Predictions"])
    ]
    frame["WER_typed"] = [
        compute_wer(t, p)
        for t, p in zip(frame["Typed Sentences"], frame["Model Predictions"])
    ]
    frame.to_csv(output / "predictions.csv", index=False)
    summary = {
        "smoke_test_only": smoke,
        "language_model": False,
        "context_mode": context_mode,
        "test_sentence_instances": len(frame),
        "CER_prompt_sentence_mean": float(frame["CER"].mean()),
        "WER_prompt_sentence_mean": float(frame["WER"].mean()),
        "CER_typed_sentence_mean": float(frame["CER_typed"].mean()),
        "WER_typed_sentence_mean": float(frame["WER_typed"].mean()),
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0,
    }
    if not all(np.isfinite(value) for key, value in summary.items() if "mean" in key):
        raise RuntimeError("Nonfinite test metrics")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Single-participant Brain2Qwerty V1")
    parser.add_argument("command", choices=["inspect", "cache", "smoke", "train", "eval"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("BRAIN2QWERTY_STUDIES", "D:/dataset/SpanishBCBL")),
    )
    parser.add_argument("--modality", choices=["eeg", "meg"], default="eeg")
    parser.add_argument("--subject", default="002")
    parser.add_argument("--cache-root", type=Path, default=Path(".cache"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Keystroke budget; sentences stay intact",
    )
    parser.add_argument(
        "--precision", choices=["32-true", "bf16-mixed"], default="bf16-mixed"
    )
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--context-mode", choices=["sentence", *LOCAL_CONTEXT])
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.resume and args.command != "train":
        parser.error("--resume is only supported for train")
    for checkpoint in (args.ckpt, args.resume):
        if checkpoint and not checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")
        if checkpoint:
            saved_config = checkpoint.parent / "config.json"
            if saved_config.is_file():
                saved_mode = json.loads(saved_config.read_text(encoding="utf-8")).get(
                    "context_mode", "sentence"
                )
                if args.context_mode is not None and args.context_mode != saved_mode:
                    parser.error(f"Checkpoint requires --context-mode {saved_mode}")
                args.context_mode = saved_mode
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cfg = make_config(args)
    output = Path(cfg["output_dir"])
    if args.command == "train" and (output / "best.ckpt").exists() and not args.resume:
        parser.error("Run already has best.ckpt; use --resume or a different --run-dir")
    output.mkdir(parents=True, exist_ok=True)
    save_environment(output)
    xp = SingleSubjectExperiment(**cfg)
    # MapInfra exposes item_uid as a public hook. Hash its keys so JSON-based
    # recording identifiers never become illegal Windows filenames.
    xp.data.study.infra_timelines.item_uid = timeline_cache_key
    xp.data.neuro.infra.item_uid = neuro_cache_key
    config = xp.model_dump(mode="json")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    timelines = xp.data.study.study_summary()
    if timelines.empty:
        raise RuntimeError(
            f"No usable recordings for {subject_labels(args.modality, args.subject)}"
        )
    timelines.to_csv(output / "recordings.csv", index=False)
    print(
        f"Selected {len(timelines)} recordings: {list(timelines.subject.unique())}",
        flush=True,
    )
    if args.command == "inspect":
        print(timelines.to_string(index=False), flush=True)
        return
    if args.command in {"train", "smoke", "eval"} and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this full-size model")
    audit_events(xp, output)
    if args.reference_run:
        verify_reference(output, args.reference_run, config)
    if args.command == "cache":
        loaders = xp.data.build()
        for split, loader in loaders.items():
            batch = next(iter(loader))
            shapes = {key: list(value.shape) for key, value in batch.data.items()}
            print(
                f"{split}: {len(loader.dataset)} windows; first batch {shapes}",
                flush=True,
            )
        return
    print(f"Running {args.command}; results: {output}", flush=True)
    write_status(
        output, state="starting", command=args.command, context_mode=xp.context_mode
    )
    try:
        xp.run()
        export_results(
            output, smoke=args.command == "smoke", context_mode=xp.context_mode
        )
    except BaseException as error:
        write_status(output, state="failed", error=f"{type(error).__name__}: {error}")
        raise
    write_status(output, state="complete")


if __name__ == "__main__":
    main()
