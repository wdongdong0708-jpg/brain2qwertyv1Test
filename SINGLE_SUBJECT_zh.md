# Brain2Qwerty V1 单受试者复现

本目录基于 Meta 官方代码，源码版本和下载文件校验值保存在 `.setup/upstream.json`。
当前机器已连接的数据位于 `D:\dataset\SpanishBCBL`，包含 EEG 的 BrainVision 信号和 MATLAB 行为日志。
默认选用 EEG 受试者 `002`，其两次会话、四个打字任务均有信号、标记和对应日志；原始信号为 64 通道、1000 Hz。
官方读取器排除眼电通道后，神经信号输入为 61 个 EEG 通道。

这是“仅使用一个受试者训练和测试”的 V1 实验。原论文的多受试者训练、MEG 结果以及加入语言模型后的结果，需要分别比较，不能直接作为这次实验的达标数值。

## 环境

启动器使用本目录 `.venv\Scripts\python.exe`，无需切换现有 Conda 环境。
它通过 `--system-site-packages` 只读复用现有 `E1` 环境的 Python 3.12 和已验证能驱动 RTX 5060 Ti 的 PyTorch 2.10.0+cu128；新增依赖安装在本项目的 `.venv` 内。
它仍依赖该基础环境的位置，迁移机器时需要重建。

V1 的 NeuralSet / NeuralTrain / NeuralFetch 固定为官方使用的 `0.2.2`，其余主要数值依赖见 `requirements-v1-windows.txt`。
官方 `requirements.lock` 面向 Linux 和 CUDA 12.4，不能直接用于这台 Windows / RTX 5060 Ti 机器。PyTorch / CUDA 版本和混合精度是本机适配差异，不能声称逐位复现论文。

如需重建，运行：

```powershell
cd D:\code\Brain2QwertyV1
.\setup_subject.ps1
```

其他基础环境可通过 `-BasePython '完整路径\python.exe'` 指定，要求 Python 3.12+ 且 CUDA 可用。

## 运行

所有命令从本目录执行。默认参数是 `--modality eeg --subject 002 --data-root D:\dataset\SpanishBCBL`。

```powershell
# 查看被选中的记录，不下载数据
.\run_subject.ps1 inspect

# 对齐行为日志、检查数据划分、准备预处理缓存
.\run_subject.ps1 cache

# 小规模验证：首个记录、2 个 epoch，每轮最多 8 个训练批次
.\run_subject.ps1 smoke

# 正式训练：该受试者全部可用记录，最多 300 个 epoch，验证集 30 轮不改善则停止
.\run_subject.ps1 train

# 独立重跑最佳模型的完整测试集评估
.\run_subject.ps1 eval --ckpt outputs\eeg-002\train\best.ckpt

# 中断后恢复，保留原配置和输出目录
.\run_subject.ps1 train --resume outputs\eeg-002\train\last.ckpt
```

默认 EEG `002` 也可以在隐藏后台启动，关闭聊天不影响这个独立进程：

```powershell
.\start_training.ps1
Get-Content outputs\eeg-002\train\status.json
Get-Content outputs\eeg-002\train.stdout.log -Tail 10
```

后台启动器会检查自己记录的运行进程，避免重复启动。`status.json` 的 `state` 为 `complete` 时，训练、最佳模型测试和预测导出均已完成；`failed` 表示运行失败，原因和日志可供排查。

也可以明确指定另一个 EEG 受试者：

```powershell
.\run_subject.ps1 train --subject 006
```

如果另有单受试者 MEG 数据，保留官方目录结构 `MEG\FIF` 和 `MEG\logs` 后，使用：

```powershell
.\run_subject.ps1 inspect --modality meg --subject S2 --data-root 'D:\你的数据目录'
```

MEG 的同一人多编号映射沿用官方规则：S1/S18、S4/S14、S5/S10/S21。
EEG 编号独立处理，不套用该映射。某个 EEG 受试者只有部分记录通过官方数据清理属于数据本身的情况，以 `recordings.csv` 为准。

## 模型、划分和本机适配

- 保留官方 Conv + sentence Transformer、29 类字符标签、交叉熵与 AdamW / OneCycleLR，学习率为 `5e-5`。
- 受试者专属层只为这一人分配权重，避免公共库默认预留 200 人的无用参数；共享卷积层和 Transformer 的宽度、深度保持官方值。
- 保留官方 `-0.2` 至 `+0.3` 秒按键窗口、50 Hz 重采样、0.1–20 Hz 滤波和 RobustScaler 处理。
- 保留官方 TF-IDF 相似句聚类和约 80%/10%/10% 的训练/验证/测试划分，实际比例受句子聚类影响；同一句子文本不会跨集合。
- 筛选受试者后才读取信号；正式训练包含其所有可用会话和任务，试跑只使用首个记录。
- Windows 下数据加载进程数为 0，避免额外复制内存；训练和评估都按完整句子组批，64 是每批按键预算，超长句独立成批。
- 通过公开的 `MapInfra.item_uid` 接口将缓存键转为 SHA-256，避免 JSON 记录标识中的冒号和引号成为 Windows 非法文件名；不修改第三方依赖源码。
- 默认使用 `bf16-mixed` 降低显存占用。需要 FP32 对照可加 `--precision 32-true`。本机是否能容纳完整配置应以试跑实测为准。
- 正式测试从验证集选出的最佳 checkpoint 加载；`--resume` 会恢复优化器、调度器与训练进度。

`smoke` 仅验证读取、优化和导出链路，只评估部分批次；其指标不代表正式复现结果。
正式结果不含 N-gram 语言模型；官方语言模型重打分需要另行准备西班牙语字符级 ARPA 模型。

## 输出

默认正式输出在 `outputs\eeg-002\train`，试跑在 `outputs\eeg-002\smoke`。

| 文件 | 含义 |
|---|---|
| `recordings.csv` | 实际使用的记录 |
| `sentence_splits.csv` / `data_audit.json` | 整句划分、按键数量和无重复句泄漏检查 |
| `config.json` / `environment.json` | 完整运行配置、源码版本、实际依赖和 GPU 信息 |
| `status.json` | 当前进程、已完成轮数、验证指标和完成/失败状态 |
| `logs\version_*\metrics.csv` | 训练损失、验证指标、测试指标 |
| `best.ckpt` / `last.ckpt` | 正式训练的最佳与最近 checkpoint |
| `callbacks\test_all_sentences.json` | 测试句子预测、标签和 logits |
| `predictions.csv` | 每句预测及 CER/WER |
| `summary.json` | 句子平均 CER/WER 与显存峰值 |

`summary.json` 分别报告相对于提示句子（`prompt`）和实际打出的字符（`typed`）的错误率，两者参考文本不同。保留两种口径便于判断打字错误的影响。
训练过程中的验证指标沿用官方实现；最终逐句结果以 `predictions.csv` 和 `summary.json` 为准。

## 相关代码

新增入口：`brain2qwerty_v1/scripts/single_subject.py`。
必要改动包括 EEG 受试者标签提取、完整句子组批、最佳模型测试、精度和试跑批次数控制。
模型架构未缩小。新增正确性检查覆盖完整句子组批和 EEG/MEG 编号区分；同时运行官方 V1 单元测试。

## 本机验证记录（2026-10-16）

- 官方 V1 测试和新增检查共 **14 项通过**，包括按句组批、缓存键、受试者映射和参考句导出。
- EEG `002` 四段记录的缓存与实际数据批次读取通过；正式划分包含 **204/24/28 个句子实例**，对应 **7704/930/1047 个按键窗口**。
- 全尺寸网络共约 **3.04 亿参数**。首段记录的两轮短试跑完成，共 16 次优化更新；每轮平均训练损失从 **4.1283** 降至 **3.2171**，测试预测及逐句 CER/WER 已导出。
- 试跑峰值 GPU 分配显存 **5.69 GiB**，低于本机显存容量；系统显示的总 GPU 占用还包含显存缓存和其他程序。
- 试跑 `summary.json` 中 `smoke_test_only=true`，其高错误率是短试跑结果，不能作为正式复现指标。
- 正式训练于 **2026-10-16 00:39（北京时间）** 通过后台启动器开始，最终状态与指标以 `outputs\eeg-002\train` 中的运行输出为准。

官方说明：<https://github.com/facebookresearch/brain2qwerty/blob/main/brain2qwerty_v1/README.md>

严格双向 9 按键与历史 9 按键的新增对照实验见 `LOCAL_CONTEXT_zh.md`，队列进度和自动汇总位于 `outputs\eeg-002\context_comparison`。
