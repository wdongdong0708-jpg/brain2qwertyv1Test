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


def run_live(command):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


drive.mount("/content/drive")
REPO_URL = "https://github.com/wdongdong0708-jpg/brain2qwertyv1Test.git"
CODE = Path("/content/code")
WORKSPACE = Path("/content/drive/MyDrive/Brain2Qwerty_Workspace")
if (CODE / ".git").exists():
    subprocess.run(["git", "-C", str(CODE), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(CODE)], check=True)
os.chdir(CODE)
for folder in ("data", "archives", "runs"):
    (WORKSPACE / folder).mkdir(parents=True, exist_ok=True)
raw_candidates = [WORKSPACE / "data/SpanishBCBL", WORKSPACE / "data/SpanishBCBL_Full"]
raw = next((path for path in raw_candidates if path.is_dir()), None)
if raw is None:
    raise FileNotFoundError(f"SpanishBCBL data directory not found; checked: {raw_candidates}")
raw = raw.resolve()
STUDIES_LINKS = Path("/content/studies_links")
STUDIES_LINKS.mkdir(parents=True, exist_ok=True)
for name in ("Pinet2024Eeg", "Pinet2024Meg"):
    alias = STUDIES_LINKS / name
    if alias.is_symlink() and alias.resolve() != raw:
        alias.unlink()
    if not alias.is_symlink():
        if alias.exists():
            raise FileExistsError(f"Expected a local symlink, but found an existing directory/file: {alias}")
        alias.symlink_to(raw, target_is_directory=True)
os.environ["BRAIN2QWERTY_STUDIES"] = str(STUDIES_LINKS)
os.environ["BRAIN2QWERTY_CACHE"] = "/content/local_cache"
Path(os.environ["BRAIN2QWERTY_CACHE"]).mkdir(parents=True, exist_ok=True)
os.environ["BRAIN2QWERTY_RESULTS"] = str(WORKSPACE / "runs")
print(f"Raw data: {raw}; study links: {STUDIES_LINKS}; cache: /content/local_cache")
subprocess.run(["git", "log", "-1", "--oneline"], check=True)
"""

INSTALL = """
run_live([sys.executable, "-u", "-m", "scripts.colab_setup"])
PYTHON = "/content/b2q-venv/bin/python"
"""


def build(*, gpu_metadata=None):
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
            优先读取 `data/SpanishBCBL`，同时兼容 `data/SpanishBCBL_Full`。
            CPU/GPU 统一使用本地 `/content/studies_links` 软链接与 `/content/local_cache` 缓存，避免在 Drive 创建软链接或写入 exca 缓存。
            首次全量预处理可能很久；当前运行时内重跑会复用已完成的缓存。预热完成后执行归档单元，将缓存持久保存到 Drive。
            """,
            ),
            *common,
            cell(
                "code",
                """
            MODALITIES = ["eeg", "meg"]
            CPU_THREADS = 2
            for modality in MODALITIES:
                run_live([PYTHON, "-u", "-m", "brain2qwerty_v1", "cache",
                          "--modality", modality, "--cpu-threads", str(CPU_THREADS)])
            """,
            ),
            cell(
                "markdown",
                "所有缓存完成后执行：将 `/content/local_cache` 打包到 Drive 的 `archives/features.tar`。GPU 实例解压到同一路径后训练。Colab 本地缓存随运行时释放，需完成归档才能持久保存。打包时不要同时启动写入这些缓存的任务。",
            ),
            cell(
                "code",
                """
            run_live([PYTHON, "-u", "-m", "scripts.cache_archive", "pack",
                      "--cache-root", os.environ["BRAIN2QWERTY_CACHE"],
                      "--archive", str(WORKSPACE / "archives/features.tar")])
            """,
            ),
        ],
        "colab_gpu_train.ipynb": [
            cell(
                "markdown",
                """
            # GPU：复用 CPU 缓存训练
            选择 L4/A100 GPU 运行时，依次执行。先完成 CPU notebook，再运行这里。
            与 CPU 使用相同的数据目录选择、本地 `/content/studies_links` 软链接和 `/content/local_cache` 缓存路径。
            原始数据路径、数据版本、滤波和采样参数必须与预热阶段一致。
            `--require-cache` 禁止补算原始信号；缺失缓存会立即报错。
            `--resume auto` 恢复同一目录的 last.ckpt；总轮数仍是 EPOCHS，不会额外再加 EPOCHS。
            """,
            ),
            common[0],
            cell(
                "code",
                INSTALL
                + """
run_live([PYTHON, "-c", "import torch, brain2qwerty_v1; print('GPU available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"])
""",
            ),
            cell(
                "code",
                """
            os.environ["BRAIN2QWERTY_CACHE"] = "/content/local_cache"
            run_live([PYTHON, "-u", "-m", "scripts.cache_archive", "unpack",
                      "--archive", str(WORKSPACE / "archives/features.tar"),
                      "--cache-root", os.environ["BRAIN2QWERTY_CACHE"]])
            """,
            ),
            cell(
                "code",
                """
            MODALITY = "eeg"   # eeg / meg
            SUBJECT = "all"    # 002 for one EEG subject; S2 for one MEG subject
            EPOCHS = 300       # finite total budget; resume must keep the original value
            SEED = 33
            run_live([PYTHON, "-u", "-m", "brain2qwerty_v1", "train",
                      "--modality", MODALITY, "--subject", SUBJECT,
                      "--seed", str(SEED), "--epochs", str(EPOCHS),
                      "--devices", "1", "--require-cache", "--resume", "auto"])
            """,
            ),
            cell(
                "markdown",
                "Checkpoint、逐步日志、每轮状态均直接写入 Drive 的 runs/。同一实验再次执行会恢复断点；模型结构、batch、种子和总轮数需保持一致。新实验请指定不同的 `--run-dir`。",
            ),
        ],
    }
    for name, cells in notebooks.items():
        runtime_metadata = {}
        if name == "colab_gpu_train.ipynb":
            previous = gpu_metadata
            if previous is None and (destination / name).is_file():
                previous = json.loads((destination / name).read_text(encoding="utf-8"))[
                    "metadata"
                ]
            runtime_metadata = {
                key: previous[key]
                for key in ("accelerator", "colab")
                if previous and key in previous
            }
            runtime_metadata.setdefault("accelerator", "GPU")
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                **runtime_metadata,
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
