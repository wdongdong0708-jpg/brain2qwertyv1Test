# V1 工程工作流本地实测

日期：2026-10-17。Windows / Python 3.12.12 / RTX 5060 Ti 16GB / PyTorch 2.10.0+cu128 / Lightning 2.5.2 / neuralset、neuraltrain 0.2.2 / exca 0.5.22。实际版本细节见本机对应 run 下的 `environment.json`。

## 数据整理

`scripts.prepare_local` 从现有 `D:/dataset/SpanishBCBL` 为 002 建立 16 个硬链接，落在 `workspace/data/Pinet2024Eeg/EEG/{EEG,logs}`。4 段信号、4 份日志，未复制原始信号内容，未移动或删除旧数据。已有 outputs 和 checkpoints 均保留。

## 真实执行结果

| 检查 | 结果 |
|---|---|
| CPU 冷缓存，`cache --modality eeg --subject 002 --expected-timelines 4` | 成功；数据构建主体 31.30 秒；CUDA 未初始化；Trainer 未创建 |
| 原尺寸 EEG 模型，`smoke --modality eeg --subject 002 --expected-timelines 4 --require-cache` | 成功；2 Epoch × 2 训练 batch；global_step=4；CNN 非零有限梯度 |
| 缓存打包并解压到新目录，再用 `cache --require-cache` | 成功；归档约 0.048 GiB；数据构建主体 3.06 秒；实际 exca UID 与原目录一致 |
| 小模型完整 Checkpoint 写入 | 成功；best.ckpt、last.ckpt 可读取，含优化器状态 |
| 同一小模型 `train --resume auto` | 成功恢复；原总轮数2、global_step4保持不变，没有额外训练 |
| 单独运行小模型 `eval --ckpt .../best.ckpt` | 成功；得到与训练后评估一致的指标 |

原尺寸 EEG 短测：约 303.8M 个可训练参数；EEG 编码器没有 2D Merger，Transformer 与 CNN 联合训练。数据准备和短测主体合计约 8.56 秒，进程导入依赖的启动耗时另计；峰值 GPU 已分配显存约 5.68 GiB。训练 Epoch 平均 loss：4.5015 → 3.7243；验证 loss：4.8733 → 4.7916。小批次测试 loss=4.8935、CER≈0.8991；仅用于流程与数值检查，不能作为模型正式效果结论。

| Split | 全部所选记录中的有效按键窗口 | 每个完整 Epoch 的句子 batch |
|---|---:|---:|
| train | 7704 | 170 |
| val | 930 | 21 |
| test | 1047 | 24 |

短测使用相同的四段数据、滤波、重采样、对齐和划分，只限制实际消费的 batch 数。典型输入为 `[batch, 61, 25]`；EEG 没有 `channel_positions` 提取器。

## 缓存一致性

冷缓存、原尺寸短测、新目录严格缓存加载的 `cache_signature.json` 相同：

```text
timeline: studies.spanishbcbl.Pinet2024Eeg._load_timelines,1/name=Pinet2024Eeg,infra_timelines.version=1-9cb37ec8
neuro:    neuralset.extractors.neuro.EegExtractor._get_data,1/frequency=50,filter=(0.10,20),name=EegExtractor,scaler=RobustScaler-386a63f6
```

测试另外覆盖：cache/smoke/train 的实际 exca UID 一致、全量/单被试的底层 UID 一致、神经信号 item key 不依赖受试者重编码序号、冷缓存只读模式拒绝补算、CPU 缓存禁止调用模型构建/CUDA 查询/Trainer、EEG 原生与重命名日志、MEG 两种目录、原 S1/S2 日志交换、归档路径安全和 Git Notebook 不含内嵌源码。

## 本地产物位置与边界

V1 回归测试：**117 passed**；`ruff check .` 通过。CI 将 V1 和 V2 分进程运行，V1 的检查命令见主手册。

## 多受试者入口验证

新增 `--subjects`（固定名单，空格或逗号分隔）和 `--num-subjects`（按编号选择前 N 个有效独立受试者）。本次新增测试覆盖 10 人筛选、MEG 别名合并、对照排除、人数不足和名单缺失时拒绝执行、名单顺序与规范编号的稳定性、实际 exca UID 复用、续训配置隔离，以及两个 Notebook 的参数传递。EEG/MEG 均使用合成数据完成 10 人 CNN + Transformer 前反向传播，检查每个人对应的 subject layer 都收到有限非零梯度。

真实 002 数据另外执行：

| 命令（省略日志和输出目录参数） | 结果 |
|---|---|
| `cache --modality eeg --num-subjects 1 --expected-timelines 4 --require-cache` | 成功；实际名单 `Pinet2024Eeg/002`，4 段记录；数据构建主体 3.44 秒；CUDA/Trainer 均未初始化 |
| `smoke --modality eeg --subjects 002 --expected-timelines 4 --require-cache` | 成功；原尺寸约 303.8M 参数，2 Epoch、global_step=4；有限非零 CNN 梯度；数据构建和短测主体 9.70 秒 |

两次执行的 `cache_signature.json` 与原单被试流程一致。短测 loss、CER 与上文固定种子 33 的结果一致；峰值已分配显存约 5.68 GiB。名单写入各自的 `subjects.json`。本机仍只有 002 的原始数据，10 人真实全程训练需使用云端全量数据；合成数据检查不代表 10 人真实数据的训练效果。

本次忽略的本地产物：`workspace/multi-subject-{cache,smoke}-validation.log`、`workspace/runs/multi-subject-validation/{cache,smoke}/`。

以下路径均被 Git 忽略：

```text
workspace/cache-validation.log
workspace/smoke-validation.log
workspace/cache-restored-validation.log
workspace/checkpoint-validation.log
workspace/resume-validation.log
workspace/eval-validation.log
workspace/runs/eeg-002-sentence-full/{cache,smoke}/
workspace/runs/cache-restored-validation/
workspace/runs/checkpoint-validation/
workspace/runs/eval-validation/
workspace/archives/features-validation.tar
```

本机只有 EEG 002 被纳入新 workspace；没有云端全量 MEG 的实际运行结果。全量 EEG/MEG 的数据清洗、信号内存占用、Drive 配额和训练收敛需要在云端运行后确认；不会把本次单被试短测当作全量复现结果。全项目测试应按 CI 的方式分代执行；本机 V1 环境没有安装 V2 Conformer 所需的 torchaudio，未对 V2 做运行通过的声明。
