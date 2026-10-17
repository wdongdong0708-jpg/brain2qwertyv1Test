import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def plan(reference, cache_root):
    config = read_json(reference / "config.json")
    definitions = [("cnn_pretrain_s33", "cnn_only", 0, 0)]
    definitions += [
        (f"r{radius}_s33", "local", radius, radius) for radius in (0, 1, 2, 4, 8)
    ]
    definitions += [
        ("full_s33", "sentence", 0, 0),
        ("m2_p0_s33", "local", 2, 0),
        ("m2_p1_s33", "local", 2, 1),
        ("revision_s33", "revision", 2, 2),
    ]
    return [
        {
            "job_id": name,
            "mode": mode,
            "left": left,
            "right": right,
            "seed": 33,
            "data_root": config["data"]["study"]["path"],
            "cache_root": str(cache_root.resolve()),
            "reference_run": str(reference.resolve()),
            "max_epochs": 300,
            "patience": 30,
            "revision_delays": [0, 1, 2],
            "coverage": "all_available_targets",
            "test_evaluated": False,
        }
        for name, mode, left, right in definitions
    ]


def completed(run, job, smoke):
    summary = read_json(run / "summary.json")
    return (
        read_json(run / "status.json").get("state") == "complete"
        and summary.get("research") == job
        and summary.get("smoke_test_only") is smoke
        and summary.get("test_evaluated") is False
        and (run / "best.ckpt").is_file()
        and (run / "validation_events.csv").is_file()
        and (job["mode"] != "cnn_only" or (run / "pretrained_encoder.pt").is_file())
    )


def check_smoke(run):
    paths = list((run / "logs").glob("version_*/metrics.csv"))
    if not paths:
        raise RuntimeError(f"No smoke metrics: {run}")
    latest = max(paths, key=lambda path: path.stat().st_mtime_ns)
    with latest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    losses = [
        float(row["train_loss_epoch"]) for row in rows if row.get("train_loss_epoch")
    ]
    metrics = [
        float(value)
        for row in rows
        for key, value in row.items()
        if value and ("loss" in key or key.startswith("val_ce"))
    ]
    if (
        len(losses) < 2
        or losses[-1] >= losses[0]
        or not all(math.isfinite(value) for value in metrics)
    ):
        raise RuntimeError(f"Smoke loss did not fall or is nonfinite: {run}: {losses}")
    return {"first_epoch_loss": losses[0], "last_epoch_loss": losses[-1], "finite": True}


def report(root, jobs):
    rows = []
    for job in jobs:
        run = root / "runs" / job["job_id"]
        state = read_json(run / "status.json")
        summary = (
            read_json(run / "summary.json") if state.get("state") == "complete" else {}
        )
        delay = str(summary.get("primary_delay", job["right"]))
        metrics = summary.get("validation", {}).get(delay, {}).get("all_target", {})
        rows.append(
            {
                "job_id": job["job_id"],
                "seed": 33,
                "mode": job["mode"],
                "left": job["left"],
                "right": job["right"],
                "state": state.get("state", "pending"),
                "completed_epochs": state.get("completed_epochs"),
                "best_epoch": summary.get("best_epoch"),
                "val_CER_corpus": metrics.get("CER_corpus"),
                "val_accuracy": metrics.get("accuracy"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
                "peak_gpu_allocated_gb": summary.get("peak_gpu_allocated_gb"),
                "test_evaluated": False,
            }
        )
    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# EEG002：种子33上下文研究",
        "",
        "统一 CNN 预热；所有数字均为验证集结果。",
        "",
        "| 实验 | 状态 | 已训练轮数 | 最佳轮次 | 验证 corpus CER | 验证按键准确率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        cer = f"{row['val_CER_corpus']:.2%}" if row["val_CER_corpus"] is not None else "—"
        accuracy = (
            f"{row['val_accuracy']:.2%}" if row["val_accuracy"] is not None else "—"
        )
        lines.append(
            f"| {row['job_id']} | {row['state']} | {row['completed_epochs'] or '—'} | "
            f"{row['best_epoch'] or '—'} | {cer} | {accuracy} |"
        )
    lines += [
        "",
        "本轮只有一个种子，不绘制跨种子误差线。旧的从头训练结果不混入本表。",
        "句首、句尾使用真实存在的上下文；历史/回修限制针对 Transformer 输入，不代表完整实时因果系统。",
        "",
    ]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def plot_results(root, jobs):
    summaries = {
        job["job_id"]: read_json(root / "runs" / job["job_id"] / "summary.json")
        for job in jobs
    }
    names = [f"r{radius}_s33" for radius in (0, 1, 2, 4, 8)] + ["full_s33"]
    if not all(summaries.get(name) for name in names):
        return
    os.environ.setdefault("MPLCONFIGDIR", str(root / "plot_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def metric(name, cohort="all_target", field="CER_corpus", delay=None):
        summary = summaries[name]
        key = str(summary["primary_delay"] if delay is None else delay)
        return 100 * summary["validation"][key][cohort][field]

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(
        [1, 3, 5, 9, 17],
        [metric(name) for name in names[:-1]],
        "o-",
        label="Local window",
    )
    ax.axhline(
        metric("full_s33"), color="tab:orange", linestyle="--", label="Full sentence"
    )
    ax.set(
        xlabel="Maximum keystrokes per target window",
        ylabel="Validation corpus CER (%)",
        title="EEG002 — context window, seed33",
        xticks=[1, 3, 5, 9, 17],
    )
    ax.grid(alpha=0.25)
    ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"context_window_curve.{extension}", dpi=180)
    plt.close(fig)
    if all(summaries.get(name) for name in ("m2_p0_s33", "m2_p1_s33", "revision_s33")):
        fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        independent = [
            metric(name, "common_future", "accuracy")
            for name in ("m2_p0_s33", "m2_p1_s33", "r2_s33")
        ]
        shared = [
            metric("revision_s33", "common_future", "accuracy", delay)
            for delay in (0, 1, 2)
        ]
        ax.plot([0, 1, 2], independent, "o-", label="Independent models")
        ax.plot([0, 1, 2], shared, "s-", label="Shared dense revision")
        ax.set(
            xlabel="Available future keystrokes",
            ylabel="Validation accuracy (%)",
            title="EEG002 — common targets, seed33",
            xticks=[0, 1, 2],
        )
        ax.grid(alpha=0.25)
        ax.legend()
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"revision_curve.{extension}", dpi=180)
        plt.close(fig)


def run_stage(root, job, smoke, jobs):
    run = root / ("smoke" if smoke else "runs") / job["job_id"]
    if completed(run, job, smoke):
        return check_smoke(run) if smoke else {"already_complete": True}
    command = [
        sys.executable,
        "-u",
        "-m",
        "brain2qwerty_v1.scripts.research_run",
        "--job",
        str(root / "jobs" / f"{job['job_id']}.json"),
        "--root",
        str(root),
    ]
    if smoke:
        command.append("--smoke")
    state = read_json(root / "status.json")
    state.update(
        state="running",
        phase="preflight" if smoke else "train",
        current_job=job["job_id"],
        current_run=str(run),
        updated_utc=datetime.now(timezone.utc).isoformat(),
    )
    logs = root / "logs"
    prefix = ("smoke-" if smoke else "train-") + job["job_id"]
    with (
        (logs / f"{prefix}.stdout.log").open("a", encoding="utf-8") as stdout,
        (logs / f"{prefix}.stderr.log").open("a", encoding="utf-8") as stderr,
    ):
        child = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=stdout,
            stderr=stderr,
            env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        state["child_pid"] = child.pid
        write_json(root / "status.json", state)
        print(f"Running {prefix}: PID {child.pid}", flush=True)
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    report(root, jobs)
        except BaseException:
            child.terminate()
            child.wait(timeout=30)
            raise
    if code or not completed(run, job, smoke):
        raise RuntimeError(
            f"{prefix} failed (exit={code}); see {logs / (prefix + '.stderr.log')}"
        )
    return check_smoke(run) if smoke else {"complete": True}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Seed33: CNN pretraining, six windows and dense rolling revision"
    )
    parser.add_argument(
        "--reference-run", type=Path, default=Path("outputs/eeg-002/train")
    )
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/eeg-002/context_research_seed33")
    )
    parser.add_argument("--cache-root", type=Path, default=Path(".cache"))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    for folder in (root, root / "jobs", root / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    jobs = plan(args.reference_run, args.cache_root)
    if args.report_only:
        report(root, jobs)
        plot_results(root, jobs)
        return
    if shutil.disk_usage(root).free < 50 * 1024**3:
        raise RuntimeError("At least 50 GiB free space required for checkpoints")
    with FileLock(str(root / "queue.lock"), timeout=0):
        old = read_json(root / "plan.json")
        if old and old.get("jobs") != jobs:
            raise RuntimeError("Saved experiment matrix differs")
        write_json(
            root / "plan.json", {"seed": 33, "jobs": jobs, "test_evaluated": False}
        )
        for job in jobs:
            write_json(root / "jobs" / f"{job['job_id']}.json", job)
        state = {
            "state": "running",
            "pid": os.getpid(),
            "seed": 33,
            "planned_jobs": len(jobs),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "completed_smokes": [],
            "completed_jobs": [],
        }
        write_json(root / "status.json", state)
        try:
            report(root, jobs)
            for job in [
                job
                for job in jobs
                if job["job_id"]
                in {"cnn_pretrain_s33", "r0_s33", "r8_s33", "full_s33", "revision_s33"}
            ]:
                verification = run_stage(root, job, True, jobs)
                state = read_json(root / "status.json")
                state["completed_smokes"].append(
                    {"job_id": job["job_id"], **verification}
                )
                write_json(root / "status.json", state)
            for job in jobs:
                verification = run_stage(root, job, False, jobs)
                state = read_json(root / "status.json")
                state["completed_jobs"].append({"job_id": job["job_id"], **verification})
                write_json(root / "status.json", state)
                report(root, jobs)
                plot_results(root, jobs)
            state = read_json(root / "status.json")
            state.update(
                state="complete",
                child_pid=None,
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            write_json(root / "status.json", state)
        except BaseException as error:
            state = read_json(root / "status.json")
            state.update(
                state="failed",
                error=f"{type(error).__name__}: {error}",
                child_pid=None,
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            write_json(root / "status.json", state)
            raise


if __name__ == "__main__":
    main()
