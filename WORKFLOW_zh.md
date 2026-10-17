# 本地研发 → GitHub → 云端缓存与训练

统一入口为 `python -m brain2qwerty_v1 {cache,smoke,train,eval}`，原来的 `python -m brain2qwerty_v1.main` 也转到同一入口。默认 EEG、全部有效受试者、单个种子 33、单 GPU、原尺寸模型；单受试者请明确传 `--subject 002`。以前的局部上下文与冻结/联合微调实验脚本仍保留，已有输出没有迁移或删除。

代码仓库：<https://github.com/wdongdong0708-jpg/brain2qwertyv1Test>。CPU/GPU Notebook 只通过 Git 获取源码。

## 1. 目录和路径

| 配置项 | 本地默认 | 云端 CPU | 云端 GPU |
|---|---|---|---|
| `BRAIN2QWERTY_STUDIES` / `--data-root` | `<仓库>/workspace/data` | `$WS/data` | `$WS/data` |
| `BRAIN2QWERTY_CACHE` / `--cache-root` | `<仓库>/workspace/cache` | `$WS/cache_persisted` | `/content/local_cache` |
| `BRAIN2QWERTY_RESULTS` / `--results-root` | `<仓库>/workspace/runs` | `$WS/runs` | `$WS/runs` |

命令行路径优先于环境变量；环境变量在每次生成配置时读取。`--run-dir` 可以直接指定一个实验的输出目录。

云端 `WS=/content/drive/MyDrive/Brain2Qwerty_Workspace`：

```text
Brain2Qwerty_Workspace/
  data/
    SpanishBCBL_Full/
      EEG/EEG/       EEG/logs/
      MEG/MEG/       MEG/logs/     # 原来的 MEG/FIF/ 也支持
    Pinet2024Eeg -> SpanishBCBL_Full
    Pinet2024Meg -> SpanishBCBL_Full
  cache_persisted/
  archives/
  runs/
```

读取器支持 study 名称子目录，也支持直接传完整下载目录；如果 Drive 不支持软链接，会自动从 `data/SpanishBCBL_Full` 寻址。无需复制数百 GB 数据。EEG 日志兼容 `S2_session1_block1_*.mat`、`EEG002_session1_block1_*.mat` 等命名；有多个候选日志时明确报错。MEG 保留原始 S1/S2 第二次 session 的日志交换规则和已知坏记录排除规则。

段数由有效文件与日志动态发现。默认不再强制 EEG=62 或固定 MEG 段数；可用 `--expected-timelines 4` 为本地子集加显式校验。全量模式会包含读取器筛选后的全部有效记录，不包含官方已排除的坏记录、对照等。

## 2. 本地日常开发（PowerShell）

在仓库根目录执行。当前 `.venv` 已验证支持 RTX 5060 Ti；不要用原 Linux 锁文件覆盖本地 CUDA PyTorch。

首次整理 002 数据（本机已执行）：

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_local --source D:\dataset\SpanishBCBL --subject 002
```

同一磁盘优先建立硬链接，失败才复制；只选择 002 的 4 份信号三件套和 4 个日志，不移动或删除原文件。请把这批原始文件视为只读，修改硬链接内容也会修改原目录中的对应文件。

日常修改模型或超参后：

```powershell
.\.venv\Scripts\python.exe -m brain2qwerty_v1 cache --modality eeg --subject 002 --expected-timelines 4
.\.venv\Scripts\python.exe -m brain2qwerty_v1 smoke --modality eeg --subject 002 --expected-timelines 4 --require-cache
```

`smoke` 默认 2 个 Epoch，每轮 2 个训练 batch、2 个验证 batch，最后 2 个测试 batch。它保留 002 的全部 4 段记录；不改变数据查询或预处理。默认使用原尺寸 `EEG_ENCODER`，CNN 与 Transformer 联合更新，EEG 没有 2D 空间 Merger，也不提取 2D 坐标。需要临时小模型时可给 `smoke` 加 `--profile small`（hidden=256、CNN depth=4、Transformer depth=2）；其结果不能当作原尺寸模型的结果。

局部上下文示例：

```powershell
.\.venv\Scripts\python.exe -m brain2qwerty_v1 smoke --modality eeg --subject 002 --context-mode local-bidirectional --require-cache
```

具体支持的上下文名称以 `--help` 为准。修改 `brain2qwerty_v1/config/model_config.py` 中的 `EEG_ENCODER` / `ENCODER` / `TRANSFORMER`，或通过统一 CLI 改超参。只改模型不需要重新滤波。

V1 回归测试（与 V2 分进程，避免 neuralset 全局类型注册重名）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests --ignore=tests/test_v2_model.py --ignore=tests/test_v2_losses.py -q
```

同步代码：

```powershell
git status --short
git add .
git commit -m "Describe model or workflow change"
git push origin main
```

`workspace/`、数据、缓存、旧 `outputs/`、权重、日志、环境、压缩包和旧内嵌源码 Notebook 已忽略。此前本目录没有 Git 仓库，目前已建立 `main` 并配置上述 GitHub remote。

## 3. 云端首次准备（CPU 和 GPU 实例均执行）

可直接打开 [CPU Notebook](notebooks/colab_cpu_cache.ipynb) 或 [GPU Notebook](notebooks/colab_gpu_train.ipynb)，依次执行。它们包含挂载 Drive、Git 拉取、独立 V1 环境安装及对应运行步骤。先把本地改动推到 GitHub，Colab 才能拉到新代码。

手动执行时，先在 Colab 的 Python 单元格挂载：

```python
from google.colab import drive
drive.mount('/content/drive')
```

下面是 **Bash 命令**。Notebook 中使用 `%%bash` 单元格；每个新的 Bash 单元格需重新设置这些环境变量，或合并为同一单元格。

首次获取代码与安装依赖：

```bash
git clone https://github.com/wdongdong0708-jpg/brain2qwertyv1Test.git /content/code
cd /content/code
python -m scripts.colab_setup
```

同一实例更新代码：

```bash
cd /content/code
git pull --ff-only
```

后续命令使用 `/content/b2q-venv/bin/python`；安装脚本保留 Colab 当前的 torch/torchvision/torchaudio，不将其替换为 CPU wheel。本项目验证的 Python 版本是 3.12。

## 4. 云端纯 CPU：预热全量 EEG 和 MEG

选择纯 CPU 运行时，执行：

```bash
cd /content/code
export WS=/content/drive/MyDrive/Brain2Qwerty_Workspace
export BRAIN2QWERTY_STUDIES="$WS/data"
export BRAIN2QWERTY_CACHE="$WS/cache_persisted"
export BRAIN2QWERTY_RESULTS="$WS/runs"
export PY=/content/b2q-venv/bin/python

"$PY" -u -m brain2qwerty_v1 cache --modality eeg --cpu-threads 2
"$PY" -u -m brain2qwerty_v1 cache --modality meg --cpu-threads 2

"$PY" -m scripts.cache_archive pack --cache-root "$BRAIN2QWERTY_CACHE" --archive "$WS/archives/features.tar"
```

不传 `--subject` 等同 `--subject all`。想先预热一人可加 `--subject 002`（EEG）或 `--subject S2`（MEG）。缓存模式只创建配置对象并调用官方 `Experiment.data.build()`；不构建神经网络、不创建 Trainer、不初始化 CUDA。

预处理保持 0.1–20 Hz 滤波、50 Hz 重采样、RobustScaler、原 DTW 对齐和句子划分。逐记录持久化；中断后重跑可利用已完成的记录。默认串行处理记录，用 `--cpu-threads` 控制每段 MNE 的 CPU 数；未缓存的单段原始数据仍需足够内存。

打包应在预热结束后执行。tar 使用未压缩格式，减少解压 CPU 开销；耗时由缓存大小、Drive 吞吐和实例磁盘决定，不保证全量缓存秒级完成。GPU 本地盘必须容得下解压后的缓存。

## 5. 云端 GPU：解压并训练

选择 L4/A100，先完成第 3 节的挂载、Git 和依赖步骤，再执行：

```bash
cd /content/code
export WS=/content/drive/MyDrive/Brain2Qwerty_Workspace
export BRAIN2QWERTY_STUDIES="$WS/data"
export BRAIN2QWERTY_CACHE=/content/local_cache
export BRAIN2QWERTY_RESULTS="$WS/runs"
export PY=/content/b2q-venv/bin/python

"$PY" -m scripts.cache_archive unpack --archive "$WS/archives/features.tar" --cache-root "$BRAIN2QWERTY_CACHE"

# 全量 EEG（种子33，最多300轮，验证CER早停耐心30轮）
"$PY" -u -m brain2qwerty_v1 train --modality eeg --devices 1 --epochs 300 --require-cache --resume auto

# 或：全量 MEG
"$PY" -u -m brain2qwerty_v1 train --modality meg --devices 1 --epochs 300 --require-cache --resume auto

# 或：单个 EEG 受试者002
"$PY" -u -m brain2qwerty_v1 train --modality eeg --subject 002 --devices 1 --epochs 300 --require-cache --resume auto
```

这三条训练命令是可选实验，按需选择执行。`--precision auto` 在支持时使用 bf16，否则使用 fp16；设备数固定为 1。MEG 自动切换 `Pinet2024Meg + MegExtractor + ENCODER`（含 2D 空间 Merger）。

不归档时，也可将 `BRAIN2QWERTY_CACHE` 直接设成 `$WS/cache_persisted`，但每 batch 的小文件读取可能受 Drive 延迟影响。官方 Colab [FAQ](https://research.google.com/colaboratory/faq.html) 也建议减少挂载 Drive 上的大量小文件操作，并在运行时本地处理归档内容。

`--require-cache` 对 timeline、神经特征以及 MEG 坐标缓存使用 exca 的 `read-only` 模式。缺失缓存会报错，提示回 CPU 实例补齐。缓存目录搬到 NVMe 不改变 exca UID；cache/smoke/train 共用配置，所有模式都使用相同的 Windows/Linux 安全 item 哈希。全量缓存可以供单被试使用，神经特征的 item key 不依赖子集中的受试者重编码序号。

## 6. 产物、恢复和评估

默认训练目录：`$BRAIN2QWERTY_RESULTS/eeg-all-sentence-full/train/`（模态、受试者、上下文和模型 profile 会反映在目录名中）。

| 文件 | 用途 |
|---|---|
| `config.json`, `environment.json` | 配置、依赖、GPU、Git 版本 |
| `recordings.csv`, `data_audit.json`, `sentence_splits.csv` | 实际记录、样本数量、句子划分 |
| `cache_signature.json` | 实际 exca timeline / neuro UID |
| `status.json`, `gradient_audit.json` | 运行状态、指标、首次有效 CNN 梯度 |
| `logs/version_*/metrics.csv` | 每步/每轮训练与验证日志，写入即刷新 |
| `best*.ckpt`, `last.ckpt` | 验证 CER 最优模型和最近一轮的完整恢复状态 |
| `callbacks/` | 验证/测试预测句子 |

Checkpoint 和日志直接写入 Drive，不依赖 Colab 本地盘存活。Checkpoint 在验证轮次结束时原子写入；实例中断后最多损失尚未写完的一轮。`--resume auto` 在 last.ckpt 存在时恢复，否则首次训练；已有其他 checkpoint 却无 last.ckpt 时拒绝覆盖。

继续训练时保持模型结构、batch、种子、数据范围、`--cache-version`、计划总轮数相同。例如原计划 300 轮在第 20 轮中断，仍传 `--epochs 300 --resume auto`。该参数代表总轮数上限。新结构或新预算请指定新的 `--run-dir`。不自动延长旧 OneCycleLR 的训练计划。

训练结束会用最优 checkpoint 运行一次测试；`smoke` 只测试少量 batch。单独评估全量 EEG：

```bash
"$PY" -m brain2qwerty_v1 eval --modality eeg --require-cache --ckpt "$BRAIN2QWERTY_RESULTS/eeg-all-sentence-full/train/best.ckpt"
```

Lightning 再次改善选模时可能生成 `best-v1.ckpt` 等文件，以训练日志给出的最佳路径为准。单被试或 small 模型评估需加与训练一致的 `--subject`、`--profile`、`--context-mode`。

改动滤波/采样 CLI 配置会由 exca 参数哈希区分；直接改对齐或预处理实现后，需要给预热和训练同时加新的 `--cache-version 2`。只修改 CNN / Transformer 或 batch / 精度不需要重建信号缓存。不同受试者范围会重新做该范围的句子划分，信号与 DTW 缓存仍可复用。

## 7. 验证范围

本机验证记录见 [WORKFLOW_VALIDATION_zh.md](docs/WORKFLOW_VALIDATION_zh.md)。002 的 CPU 冷缓存、原尺寸 GPU 短测、归档后严格缓存加载均已实际执行。全量 EEG/MEG 数据只在你的 Drive 上，未在本机宣称完成全量训练；本机覆盖 MEG 路径和日志规则的测试，云端首次运行仍应核对 `recordings.csv` 的有效记录数。
