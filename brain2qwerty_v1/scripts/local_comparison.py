import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

MODES = ("local-bidirectional", "local-history")
LABELS = {
    "sentence": "整句双向",
    "local-bidirectional": "严格双向 9 按键（前4/后4）",
    "local-history": "严格历史 9 按键（前8/后0）",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def epoch_metrics(run):
    for metrics_path in sorted((run / "logs").glob("version_*/metrics.csv")):
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            yield from csv.DictReader(stream)


def completed(run, mode, smoke=False):
    summary = read_json(run / "summary.json")
    return (
        read_json(run / "status.json").get("state") == "complete"
        and summary.get("context_mode", "sentence") == mode
        and summary.get("smoke_test_only") is smoke
    )


def verify_smoke(run):
    losses = []
    for row in epoch_metrics(run):
        for name, value in row.items():
            if "loss" in name and value and not math.isfinite(float(value)):
                raise RuntimeError(f"Nonfinite loss in {run}")
        if row.get("train_loss_epoch"):
            losses.append(float(row["train_loss_epoch"]))
    if len(losses) < 2 or losses[-1] >= losses[0]:
        raise RuntimeError(f"Short run did not reduce training loss: {run}: {losses}")
    return {"first_epoch_loss": losses[0], "last_epoch_loss": losses[-1], "finite": True}


def predictions(run):
    with (run / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_uid = {row["Sentence_UID"]: row for row in rows}
    if len(by_uid) != len(rows):
        raise RuntimeError(f"Duplicate test sentence UID in {run}")
    return by_uid


def report(reference, root):
    base_summary = read_json(reference / "summary.json")
    base_predictions = predictions(reference)
    rows = []
    paired = {
        uid: {
            "Sentence_UID": uid,
            "True Sentences": row["True Sentences"],
            "sentence_CER": row["CER"],
        }
        for uid, row in base_predictions.items()
    }
    runs = [("sentence", reference)] + [
        (mode, reference.parent / mode / "train") for mode in MODES
    ]
    for mode, run in runs:
        status = read_json(run / "status.json")
        summary = (
            read_json(run / "summary.json") if status.get("state") == "complete" else {}
        )
        row = {
            "context_mode": mode,
            "label": LABELS[mode],
            "state": status.get("state", "pending"),
            "completed_epochs": status.get("completed_epochs"),
            "best_epoch": None,
            "best_val_CER": None,
            "test_sentence_instances": summary.get("test_sentence_instances"),
            "CER_prompt": summary.get("CER_prompt_sentence_mean"),
            "CER_typed": summary.get("CER_typed_sentence_mean"),
            "WER_prompt": summary.get("WER_prompt_sentence_mean"),
            "delta_CER_prompt_percentage_points": None,
            "peak_gpu_allocated_gb": summary.get("peak_gpu_allocated_gb"),
            "run_dir": str(run.resolve()),
        }
        validations = [r for r in epoch_metrics(run) if r.get("val_CER")]
        if validations:
            best = min(validations, key=lambda r: float(r["val_CER"]))
            row.update(
                best_epoch=int(float(best["epoch"])) + 1,
                best_val_CER=float(best["val_CER"]),
            )
        if row["state"] == "complete":
            if summary.get("smoke_test_only") is not False:
                raise RuntimeError(f"Cannot compare a short run as a full result: {run}")
            current = predictions(run)
            if set(current) != set(base_predictions):
                raise RuntimeError(
                    f"Test sentence instances differ from the baseline: {run}"
                )
            for uid, item in current.items():
                for field in ("True Sentences", "Typed Sentences"):
                    if item[field] != base_predictions[uid][field]:
                        raise RuntimeError(f"Test references differ at {uid}: {run}")
                paired[uid][f"{mode}_CER"] = item["CER"]
            row["delta_CER_prompt_percentage_points"] = 100 * (
                summary["CER_prompt_sentence_mean"]
                - base_summary["CER_prompt_sentence_mean"]
            )
        rows.append(row)
    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired_columns = list(dict.fromkeys(key for row in paired.values() for key in row))
    with (root / "paired_sentence_CER.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_columns)
        writer.writeheader()
        writer.writerows(paired.values())
    lines = [
        "# EEG 002：Transformer 上下文对照",
        "",
        "三组使用相同数据划分、CNN、Transformer 宽度/深度、初始随机种子、优化器和早停规则。",
        "局部组从头训练，每个目标使用独立的 9 按键特征窗口，仅对当前按键输出和计算损失。",
        "",
        "| 上下文 | 状态 | 已训练轮数 | 最佳轮次 | 测试 CER（提示句） | 测试 CER（实际按键） | 相对整句变化 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        prompt = f"{row['CER_prompt']:.2%}" if row["CER_prompt"] is not None else "—"
        typed = f"{row['CER_typed']:.2%}" if row["CER_typed"] is not None else "—"
        delta = row["delta_CER_prompt_percentage_points"]
        delta_text = f"{delta:+.2f} 个百分点" if delta is not None else "—"
        lines.append(
            f"| {row['label']} | {row['state']} | {row['completed_epochs'] or '—'} | "
            f"{row['best_epoch'] or '—'} | {prompt} | {typed} | {delta_text} |"
        )
    lines += [
        "",
        "CER 越低越好。完整测试评估使用各组验证集选出的最佳 checkpoint。",
        "局部历史限制针对 Transformer 输入；EEG 的滤波和按键窗口仍沿用基线，不代表完全实时系统。",
        "进度见 `status.json` 及各实验的 `train/status.json`。训练完成后本表自动更新。",
        "",
    ]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def run_stage(args, mode, phase, config):
    run = args.reference_run.parent / mode / phase
    smoke = phase == "smoke"
    if completed(run, mode, smoke=smoke):
        return verify_smoke(run) if smoke else {"already_complete": True}
    command = [
        sys.executable,
        "-u",
        "-m",
        "brain2qwerty_v1.scripts.single_subject",
        phase,
        "--context-mode",
        mode,
        "--subject",
        args.subject,
        "--modality",
        args.modality,
        "--data-root",
        config["data"]["study"]["path"],
        "--seed",
        str(config["seed"]),
        "--epochs",
        str(2 if smoke else config["n_epochs"]),
        "--batch-size",
        str(config["data"]["batch_size"]),
        "--precision",
        config["precision"],
        "--run-dir",
        str(run.resolve()),
    ]
    if not smoke:
        command += ["--reference-run", str(args.reference_run.resolve())]
        checkpoint = next(
            (run / name for name in ("last.ckpt", "best.ckpt") if (run / name).is_file()),
            None,
        )
        if checkpoint:
            command += ["--resume", str(checkpoint.resolve())]
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    log_root = args.output_root / "logs"
    log_root.mkdir(exist_ok=True)
    state = read_json(args.output_root / "status.json")
    state.update(
        state="running",
        current_mode=mode,
        current_phase=phase,
        current_run=str(run.resolve()),
        updated_utc=datetime.now(timezone.utc).isoformat(),
    )
    with (
        (log_root / f"{mode}-{phase}.stdout.log").open("a", encoding="utf-8") as stdout,
        (log_root / f"{mode}-{phase}.stderr.log").open("a", encoding="utf-8") as stderr,
    ):
        child = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        state["child_pid"] = child.pid
        write_json(args.output_root / "status.json", state)
        print(f"Running {mode} {phase}; PID {child.pid}", flush=True)
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                report(args.reference_run, args.output_root)
    if code or not completed(run, mode, smoke=smoke):
        raise RuntimeError(f"{mode} {phase} failed (exit={code}); inspect {log_root}")
    return verify_smoke(run) if smoke else {"complete": True}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run two local-context experiments sequentially"
    )
    parser.add_argument(
        "--reference-run", type=Path, default=Path("outputs/eeg-002/train")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/eeg-002/context_comparison")
    )
    parser.add_argument("--subject", default="002")
    parser.add_argument("--modality", choices=["eeg", "meg"], default="eeg")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    args.reference_run = args.reference_run.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if read_json(args.reference_run / "status.json").get("state") != "complete":
        raise RuntimeError("The reference full-sentence experiment must be complete")
    if args.report_only:
        report(args.reference_run, args.output_root)
        return
    with FileLock(str(args.output_root / "queue.lock"), timeout=0):
        write_json(
            args.output_root / "status.json",
            {
                "state": "running",
                "pid": os.getpid(),
                "planned_modes": MODES,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "completed_stages": [],
            },
        )
        config = read_json(args.reference_run / "config.json")
        try:
            report(args.reference_run, args.output_root)
            for phase in ("smoke", "train"):
                for mode in MODES:
                    verification = run_stage(args, mode, phase, config)
                    state = read_json(args.output_root / "status.json")
                    state["completed_stages"].append(
                        {"mode": mode, "phase": phase, **verification}
                    )
                    write_json(args.output_root / "status.json", state)
                    report(args.reference_run, args.output_root)
            state = read_json(args.output_root / "status.json")
            state.update(
                state="complete",
                child_pid=None,
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            write_json(args.output_root / "status.json", state)
        except BaseException as error:
            state = read_json(args.output_root / "status.json")
            state.update(
                state="failed",
                error=f"{type(error).__name__}: {error}",
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            write_json(args.output_root / "status.json", state)
            raise


if __name__ == "__main__":
    main()
