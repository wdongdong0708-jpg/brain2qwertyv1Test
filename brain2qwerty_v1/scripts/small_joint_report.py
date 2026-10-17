"""Audit and compare the two completed joint runs with their frozen controls."""

import argparse
import csv
from pathlib import Path

from .research_queue import read_json, write_json


def event_identity(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identities = [
        (row["sentence_UID"], int(row["position"]), int(row["label"]), float(row["onset"]))
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise RuntimeError("Duplicate validation target identities")
    return set(identities)


def generate(root, control):
    state = read_json(root / "status.json")
    if state.get("state") != "complete":
        raise RuntimeError("Joint queue is not complete")
    rows, evidence, common = [], [], None
    for job_id, label in (("full_s33", "Full"), ("r8_s33", "+/-8")):
        folder, reference = root / "runs" / job_id, control / "runs" / job_id
        summary, frozen = read_json(folder / "summary.json"), read_json(reference / "summary.json")
        initial, previous = read_json(folder / "initialization.json"), read_json(reference / "initialization.json")
        if (
            summary.get("seed") != 33
            or summary.get("test_evaluated") is not False
            or summary.get("encoder_frozen") is not False
            or summary.get("encoder_updated") is not True
            or summary.get("smoke_test_only") is not False
            or read_json(folder / "status.json").get("state") != "complete"
        ):
            raise RuntimeError(f"Invalid joint run summary: {job_id}")
        if (
            initial["state_sha256"] != previous["state_sha256"]
            or initial["cache_fingerprint"] != previous["cache_fingerprint"]
        ):
            raise RuntimeError("Joint and frozen initialization do not match")
        identities = event_identity(folder / "validation_events.csv")
        if identities != event_identity(reference / "validation_events.csv"):
            raise RuntimeError("Joint and frozen validation targets differ")
        if common is not None and common != identities:
            raise RuntimeError("Context modes evaluate different events")
        common = identities
        matched = read_json(folder / "matched_control_audit.json")
        gradients = read_json(folder / "gradient_audit.json")
        if not matched.get("same_event_order_and_labels") or not all(
            gradients[key]["gradient_norm"] > 0 for key in ("encoder", "decoder")
        ):
            raise RuntimeError("Missing matched data or gradient evidence")
        delay = str(summary["primary_delay"])
        actual = summary["validation"][delay]["all_target"]
        baseline = frozen["validation"][delay]["all_target"]
        rows.append(
            {
                "context": label,
                "frozen_CER": baseline["CER_corpus"],
                "joint_CER": actual["CER_corpus"],
                "CER_change_pp": 100 * (actual["CER_corpus"] - baseline["CER_corpus"]),
                "frozen_accuracy": baseline["accuracy"],
                "joint_accuracy": actual["accuracy"],
                "joint_best_epoch": summary["best_epoch"],
                "joint_elapsed_seconds": summary["elapsed_seconds"],
            }
        )
        evidence.append(
            {
                "job_id": job_id,
                "same_decoder_initialization": True,
                "same_encoder_source": True,
                "same_validation_events_and_labels": True,
                "initial_feature_max_abs_error": matched["initial_feature_max_abs_error"],
                "cnn_and_decoder_gradients_nonzero": True,
                "encoder_updated": True,
                "validation_events": len(identities),
            }
        )
    with (root / "frozen_vs_joint.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# EEG002：冻结 CNN 与小模型联合微调，种子33",
        "",
        "均为同一组930个验证事件的结果，未评估测试集。CER越低越好。",
        "",
        "| 上下文 | 冻结CNN CER | 联合微调 CER | CER变化（百分点） | 冻结/联合按键准确率 | 联合最佳轮次 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['context']} | {100 * row['frozen_CER']:.2f}% | "
            f"{100 * row['joint_CER']:.2f}% | {row['CER_change_pp']:+.2f} | "
            f"{100 * row['frozen_accuracy']:.2f}% / {100 * row['joint_accuracy']:.2f}% | "
            f"{row['joint_best_epoch']} |"
        )
    lines += [
        "",
        "各组均按验证交叉熵选择检查点；没有根据CER另行回挑轮次。",
        "",
        "共同预训练CNN来源、小解码器初始权重、验证事件及标签均通过核对；CNN和解码器均收到非零有限梯度，最佳CNN状态发生变化。",
        "",
        "联合训练包含CNN权重、归一化运行统计和训练态dropout变化；单种子结果不能确定跨种子稳定性。",
    ]
    constant = read_json(root / "constant_baseline.json")
    if constant:
        lines += [
            "",
            "补充基线：固定预测训练集中最常见的空格类别，验证准确率为"
            f"{100 * constant['val_accuracy']:.2f}%，CER为"
            f"{100 * constant['val_CER_corpus']:.2f}%。",
        ]
    (root / "frozen_vs_joint.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        root / "execution_audit.json",
        {
            "completed_jobs": len(rows),
            "seed": 33,
            "test_evaluated": False,
            "validation_events": len(common),
            "queue_wall_seconds": (root / "status.json").stat().st_mtime - state["started_unix"],
            "jobs": evidence,
        },
    )
    plot(root, rows)
    return rows


def plot(root, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.1), layout="constrained")
    x = list(range(len(rows)))
    for offset, key, label, color in (
        (-0.18, "frozen_CER", "Frozen CNN", "#6787a0"),
        (0.18, "joint_CER", "Joint fine-tuning", "#c76f40"),
    ):
        bars = ax.bar(
            [index + offset for index in x],
            [100 * row[key] for row in rows],
            width=0.34,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set_xticks(x, ["Full sentence", "Local +/-8 keys"])
    ax.set_ylabel("Validation corpus CER (%) - lower is better")
    ax.set_ylim(0, 100)
    ax.set_title("EEG002: frozen vs. trainable CNN (seed 33)")
    ax.legend(loc="lower center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures / "frozen_vs_joint.png", dpi=180)
    fig.savefig(figures / "frozen_vs_joint.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/eeg-002/small_joint_seed33"))
    parser.add_argument("--control", type=Path, default=Path("outputs/eeg-002/fast_context_seed33"))
    args = parser.parse_args()
    for row in generate(args.root.resolve(), args.control.resolve()):
        print(row)


if __name__ == "__main__":
    main()
