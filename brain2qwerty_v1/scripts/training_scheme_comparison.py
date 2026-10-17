"""Compare existing full-context runs on identical validation keystrokes."""

import argparse
import csv
import hashlib
import time
from pathlib import Path

from .research_queue import read_json, write_json


def read_events(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    integer_fields = ("position", "length", "label", "prediction", "effective_delay")
    for row in rows:
        for key in integer_fields:
            row[key] = int(row[key])
        for key in ("onset", "ce", "signal_wait_seconds"):
            row[key] = float(row[key])
    return rows


def identities(rows):
    result = {
        (row["sentence_UID"], row["position"], row["label"], row["onset"])
        for row in rows
    }
    if len(result) != len(rows):
        raise RuntimeError("Duplicate validation events")
    return result


def evaluate_original(root, run, control):
    import lightning.pytorch as pl
    import torch

    from ..main import Experiment
    from ..research import ResearchModule
    from ..utils import materialize_lazy_params
    from .single_subject import neuro_cache_key, timeline_cache_key

    config_path, weights = run / "config.json", run / "best.ckpt"
    signature = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoint_path": str(weights),
        "checkpoint_bytes": weights.stat().st_size,
        "checkpoint_modified_ns": weights.stat().st_mtime_ns,
        "validation_batch_budget": 256,
        "precision": "bf16-mixed",
    }
    destination = root / "original_validation"
    destination.mkdir(exist_ok=True)
    saved = read_json(destination / "summary.json")
    if saved:
        if saved["signature"] != signature:
            raise RuntimeError("Saved evaluation belongs to a different source model")
        return saved
    cfg = read_json(config_path)
    if cfg["seed"] != 33 or cfg.get("context_mode", "sentence") != "sentence":
        raise RuntimeError("Expected the existing seed33 full-context baseline")
    torch.set_num_threads(4)
    pl.seed_everything(33, workers=True)
    started = time.monotonic()
    write_json(root / "status.json", {"state": "building_validation_data"})
    xp = Experiment(**cfg)
    xp.data.batch_size = xp.data.val_batch_size = 256
    xp.data.study.infra_timelines.item_uid = timeline_cache_key
    xp.data.neuro.infra.item_uid = neuro_cache_key
    loaders = xp.data.build(splits=("val",))
    with (run / "sentence_splits.csv").open(encoding="utf-8-sig", newline="") as stream:
        prompts = {row["sentence_UID"]: row["sentence"] for row in csv.DictReader(stream)}
    for segment in loaders["val"].dataset.segments:
        segment.trigger.extra["sentence"] = prompts[segment.trigger.extra["sentence_UID"]]
    first = next(iter(loaders["val"]))
    hidden = xp.brain_model_config.hidden
    brain = xp.brain_model_config.build(
        n_in_channels=first.data["neuro"].shape[1], n_outputs=hidden
    )
    transformer = xp.transformer_config.build(dim=hidden)
    module = ResearchModule(
        brain, transformer, xp.loss.build(), xp.optimizer,
        research_mode="sentence", left=0, right=0,
    )
    materialize_lazy_params(module, loaders["val"])
    write_json(root / "status.json", {"state": "loading_original_best_checkpoint"})
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    epoch = checkpoint["epoch"] + 1
    del checkpoint
    write_json(root / "status.json", {"state": "evaluating_original", "best_epoch": epoch})
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision="bf16-mixed",
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        default_root_dir=destination, use_distributed_sampler=False,
    )
    trainer.validate(module, dataloaders=loaders["val"], verbose=False)
    rows = module.validation_rows[0]
    reference = read_events(control / "runs/full_s33/validation_events.csv")
    if identities(rows) != identities(reference):
        raise RuntimeError("Original baseline and small models have different validation targets")
    with (destination / "validation_events.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "signature": signature,
        "seed": 33,
        "best_epoch": epoch,
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "validation": module.validation_summary["0"]["all_target"],
        "test_evaluated_this_run": False,
        "new_training_performed": False,
        "evaluation_seconds": time.monotonic() - started,
    }
    write_json(destination / "summary.json", summary)
    return summary


def report(root, original_run, frozen_root, joint_root, original):
    from ..research import summarize_predictions

    original_events = read_events(root / "original_validation/validation_events.csv")
    common = identities(original_events)
    original_status = read_json(original_run / "status.json")
    rows = [
        {
            "scheme": "从头联合训练（原始大模型结构）",
            "trainable_parameters": original["trainable_parameters"],
            "trained_epochs": original_status["completed_epochs"],
            "best_epoch": original["best_epoch"],
            "checkpoint_monitor": "旧版 val_CER",
            "validation_CER_corpus": original["validation"]["CER_corpus"],
            "validation_accuracy": original["validation"]["accuracy"],
            "validation_CE": original["validation"]["CE"],
            "recorded_run_seconds": original_status["elapsed_seconds"],
            "transformer_dim": 2048,
            "transformer_depth": 4,
            "source": str(original_run),
        }
    ]
    for name, folder in (("预训练CNN冻结 + 小Transformer", frozen_root), ("预训练CNN联合微调 + 小Transformer", joint_root)):
        run = folder / "runs/full_s33"
        summary, state = read_json(run / "summary.json"), read_json(run / "status.json")
        events = read_events(run / "validation_events.csv")
        if identities(events) != common:
            raise RuntimeError("Training schemes evaluate different validation targets")
        if summary["seed"] != 33 or summary["test_evaluated"] is not False:
            raise RuntimeError("Unexpected small-model seed or evaluation scope")
        metrics = summarize_predictions(events)["all_target"]
        stored = summary["validation"][str(summary["primary_delay"])]["all_target"]
        for key in ("CER_corpus", "accuracy", "CE"):
            if abs(metrics[key] - stored[key]) > 1e-8:
                raise RuntimeError(f"Stored and recalculated {key} differ")
        rows.append(
            {
                "scheme": name,
                "trainable_parameters": summary["trainable_parameters"],
                "trained_epochs": state["completed_epochs"],
                "best_epoch": summary["best_epoch"],
                "checkpoint_monitor": "val_ce_all",
                "validation_CER_corpus": metrics["CER_corpus"],
                "validation_accuracy": metrics["accuracy"],
                "validation_CE": metrics["CE"],
                "recorded_run_seconds": summary["elapsed_seconds"],
                "transformer_dim": 256,
                "transformer_depth": 2,
                "source": str(run),
            }
        )
    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# EEG002：三种CNN训练方案的已有结果对照",
        "",
        "全部为本机单受试者、种子33、整句上下文。统一比较相同930个验证按键事件、24个句子实例，使用相对实际按键序列的corpus CER；越低越好。",
        "",
        "旧模型使用原最佳权重补算验证指标，没有重新训练或再次评估测试集。它此前单独报告的测试集逐句平均CER不混入本表。",
        "",
        "| 方案 | 可训练参数 | Transformer | 训练/最佳轮次 | 验证CER ↓ | 按键准确率 ↑ |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scheme']} | {row['trainable_parameters'] / 1e6:.2f}M | "
            f"{row['transformer_dim']}维 / {row['transformer_depth']}层 | "
            f"{row['trained_epochs']} / {row['best_epoch']} | "
            f"{100 * row['validation_CER_corpus']:.2f}% | "
            f"{100 * row['validation_accuracy']:.2f}% |"
        )
    lines += [
        "",
        "固定CNN仍参与原始EEG推理；冻结版的159万仅指可训练参数。",
        "",
        "计时说明：从头联合训练记录约39.3分钟；冻结版Full约19.2秒；小模型联合微调Full约118.5秒。两种预训练方案共同使用的CNN预训练另需823.6秒（约13.7分钟），特征提取约5.9秒。记录的初始化、测试和保存开销范围不同，这些用时只用于说明成本规模，不构成严格速度基准。",
        "",
        "可比性：相同受试者、种子、句子划分和验证事件。原始结构与小模型还同时改变了网络容量、初始化、学习率、批次大小、早停耐心和选模指标；本表比较现有配置的实际结果，无法单独归因于CNN是否冻结。",
        "",
        "原始结构沿用旧版val_CER选模，两个小模型按验证交叉熵选模，保留各自已经选定的最佳检查点。若只比较三种CNN训练方式，需要补充同一个小模型结构从头联合训练的对照。",
    ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "execution_audit.json", {
        "seed": 33,
        "context": "full_sentence",
        "validation_events": len(common),
        "sentence_instances": len({row[0] for row in common}),
        "same_event_ids_onsets_and_labels": True,
        "CER_aggregation": "sum_sentence_edit_distances / sum_reference_characters",
        "original_best_epoch": original["best_epoch"],
        "test_evaluated_this_run": False,
        "new_training_performed": False,
        "pure_training_strategy_ablation": False,
    })
    plot(root, rows)
    write_json(root / "status.json", {"state": "complete", "evaluated_schemes": 3})
    return rows


def plot(root, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.4), layout="constrained")
    labels = ["Joint from scratch\nLarge Transformer", "Frozen pretrained CNN\nSmall Transformer", "Joint fine-tuning\nSmall Transformer"]
    bars = ax.bar(labels, [100 * row["validation_CER_corpus"] for row in rows],
                  color=["#78917a", "#6787a0", "#c76f40"], width=0.6)
    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Validation corpus CER (%) - lower is better")
    ax.set_title("EEG002: existing full-context runs, seed 33")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"training_schemes.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/eeg-002/training_scheme_comparison_seed33"))
    parser.add_argument("--original", type=Path, default=Path("outputs/eeg-002/train"))
    parser.add_argument("--frozen", type=Path, default=Path("outputs/eeg-002/fast_context_seed33"))
    parser.add_argument("--joint", type=Path, default=Path("outputs/eeg-002/small_joint_seed33"))
    args = parser.parse_args()
    root, original_run, frozen_root, joint_root = [
        value.resolve() for value in (args.root, args.original, args.frozen, args.joint)
    ]
    root.mkdir(parents=True, exist_ok=True)
    original = evaluate_original(root, original_run, frozen_root)
    for row in report(root, original_run, frozen_root, joint_root, original):
        print(row, flush=True)


if __name__ == "__main__":
    main()
