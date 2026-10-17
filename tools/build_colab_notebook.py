import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    value = {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        ast.parse(source)
        value.update(execution_count=None, outputs=[])
    return value


SETUP = """
import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
REPO_URL = "https://github.com/wdongdong0708-jpg/brain2qwertyv1Test.git"
CODE = Path("/content/code")
WORKSPACE = Path("/content/drive/MyDrive/Brain2Qwerty_Workspace")
if (CODE / ".git").exists():
    subprocess.run(["git", "-C", str(CODE), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(CODE)], check=True)
os.chdir(CODE)
for folder in ("data", "cache_persisted", "archives", "runs"):
    (WORKSPACE / folder).mkdir(parents=True, exist_ok=True)
raw = WORKSPACE / "data/SpanishBCBL_Full"
if not raw.is_dir():
    raise FileNotFoundError(f"Put the already downloaded full SpanishBCBL repository at {raw}")
for name in ("Pinet2024Eeg", "Pinet2024Meg"):
    alias = WORKSPACE / "data" / name
    if not alias.exists():
        try:
            alias.symlink_to(raw, target_is_directory=True)
        except OSError:
            print(f"Drive does not support this symlink; the reader will resolve {raw} directly")
os.environ["BRAIN2QWERTY_STUDIES"] = str(WORKSPACE / "data")
os.environ["BRAIN2QWERTY_RESULTS"] = str(WORKSPACE / "runs")
subprocess.run(["git", "log", "-1", "--oneline"], check=True)
"""

INSTALL = """
subprocess.run([sys.executable, "-m", "scripts.colab_setup"], check=True)
PYTHON = "/content/b2q-venv/bin/python"
"""


def build():
    destination = ROOT / "notebooks"
    destination.mkdir(exist_ok=True)
    common = [cell("code", SETUP), cell("code", INSTALL)]
    notebooks = {
        "colab_cpu_cache.ipynb": [
            cell(
                "markdown",
                """
            # CPU：全量 EEG / MEG 缓存预热
            选择纯 CPU 运行时，依次执行。源码只从 GitHub 获取，不嵌入源码包。
            原始数据保留在 Drive；本流程调用同一套 `Experiment.data.build()`，不创建神经网络或 GPU Trainer。
            首次全量预处理可能很久；重跑会复用已完成的 timeline 和信号缓存。
            """,
            ),
            *common,
            cell(
                "code",
                """
            os.environ["BRAIN2QWERTY_CACHE"] = str(WORKSPACE / "cache_persisted")
            MODALITIES = ["eeg", "meg"]
            CPU_THREADS = 2
            for modality in MODALITIES:
                subprocess.run([PYTHON, "-u", "-m", "brain2qwerty_v1", "cache",
                                "--modality", modality, "--cpu-threads", str(CPU_THREADS)], check=True)
            """,
            ),
            cell(
                "markdown",
                "可选：所有缓存完成后生成一个未压缩 tar。后续 GPU 实例顺序读取 tar 并解压到本地磁盘。打包时不要同时启动写入这些缓存的任务。",
            ),
            cell(
                "code",
                """
            subprocess.run([PYTHON, "-m", "scripts.cache_archive", "pack",
                            "--cache-root", os.environ["BRAIN2QWERTY_CACHE"],
                            "--archive", str(WORKSPACE / "archives/features.tar")], check=True)
            """,
            ),
        ],
        "colab_gpu_train.ipynb": [
            cell(
                "markdown",
                """
            # GPU：复用 CPU 缓存训练
            选择 L4/A100 GPU 运行时，依次执行。先完成 CPU notebook，再运行这里。
            原始数据路径、数据版本、滤波和采样参数必须与预热阶段一致。
            `--require-cache` 禁止补算原始信号；缺失缓存会立即报错。
            `--resume auto` 恢复同一目录的 last.ckpt；总轮数仍是 EPOCHS，不会额外再加 EPOCHS。
            """,
            ),
            *common,
            cell(
                "code",
                """
            os.environ["BRAIN2QWERTY_CACHE"] = "/content/local_cache"
            subprocess.run([PYTHON, "-m", "scripts.cache_archive", "unpack",
                            "--archive", str(WORKSPACE / "archives/features.tar"),
                            "--cache-root", os.environ["BRAIN2QWERTY_CACHE"]], check=True)
            """,
            ),
            cell(
                "code",
                """
            MODALITY = "eeg"   # eeg / meg
            SUBJECT = "all"    # 002 for one EEG subject; S2 for one MEG subject
            EPOCHS = 300       # finite total budget; resume must keep the original value
            SEED = 33
            subprocess.run([PYTHON, "-u", "-m", "brain2qwerty_v1", "train",
                            "--modality", MODALITY, "--subject", SUBJECT,
                            "--seed", str(SEED), "--epochs", str(EPOCHS),
                            "--devices", "1", "--require-cache", "--resume", "auto"], check=True)
            """,
            ),
            cell(
                "markdown",
                "Checkpoint、逐步日志、每轮状态均直接写入 Drive 的 runs/。同一实验再次执行会恢复断点；模型结构、batch、种子和总轮数需保持一致。新实验请指定不同的 `--run-dir`。",
            ),
        ],
    }
    for name, cells in notebooks.items():
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "cells": cells,
        }
        for index, item in enumerate(cells):
            item["id"] = f"cell-{index:03d}"
        (destination / name).write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(destination / name)


if __name__ == "__main__":
    build()
