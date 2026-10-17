# Brain2Qwerty V1：严格局部上下文对照

使用 EEG 受试者 `002`，保留已完成的整句基线，新建两组从头训练的局部上下文实验。
对齐输入精度后的正式队列于 **2026-10-16 02:03（北京时间）** 在隐藏后台启动，依次训练两组，避免同时争用一张显卡。

## 三组定义

| 模式 | Transformer 输入 | 当前按键在窗口中的位置 |
|---|---|---|
| `sentence` | 同一句的全部按键特征 | 每个位置均输出 |
| `local-bidirectional` | 前 4 个 + 当前 + 后 4 个，共最多 9 个按键特征 | 第 5 个位置 |
| `local-history` | 前 8 个 + 当前，共最多 9 个按键特征 | 第 9 个位置 |

局部组为每个目标建立独立窗口，四层 Transformer 都在这个窗口内计算，只读取当前目标位置的输出、计算该按键的损失。
这避免了逐层局部掩码在四层网络中扩大感受野的问题。窗口不会跨句；句首、句尾不存在的邻居通过有效位置掩码排除。
同一批中每个按键仍恰好贡献一个目标损失，并按原始顺序导出预测。

这里限制的是 **Transformer 能访问的按键 EEG 特征**。卷积编码器和 EEG 预处理保留基线设置，包括按键后 0.3 秒信号及原滤波方式，因此历史窗口实验不等同于完全实时系统。

## 公平对照

- 保留卷积编码器、四层 Transformer、约 3.04 亿参数、29 个字符类别。
- 两组均从随机初始化训练，不从整句模型微调；随机种子为 `33`。
- 相同的四段记录、预处理、训练批次、损失、AdamW / OneCycleLR、`5e-5` 学习率及 BF16 混合精度。
- 三组进入 Transformer 的特征均先转为 FP32，与整句基线的输入缓冲区一致，再使用相同的混合精度计算。
- 相同的 204/24/28 个训练/验证/测试句子实例，对应 7704/930/1047 个按键窗口。
- 相同的最多 300 轮训练、验证集 30 轮不改善早停；最终评估各自验证集选出的最佳 checkpoint。

每组正式训练前，程序会逐项比较基线配置，并核对 `sentence_splits.csv` 的实际句子归属。
不一致时拒绝继续训练。通过后写出 `comparison_audit.json`。
上下文模式保存在运行配置和 checkpoint 中，恢复训练和独立评估时保持一致。

## 已完成验证

- 官方 V1 测试与新增检查共 **26 项通过**，包括三种模式在 BF16 特征输入下的精度一致性检查。
- 对四层网络进行梯度和输入扰动检查：窗口外按键特征、其他句子的特征不会影响目标预测。
- 检查句首/句尾、单按键短句、混合句子批次、目标顺序和反向传播。
- 检查 checkpoint 保存/加载仍保留局部上下文，并拒绝随机种子或测试句归属不同的对照。
- 精度对齐后的两组真实 EEG 短试跑均完成，训练损失分别为 **4.0183 → 3.1206** 和 **3.9771 → 3.1310**；未发现非有限损失。
- 两组试跑峰值分配显存均约 **5.72 GiB**。试跑仅 16 次优化更新，不能用于判断最终性能。

## 进度与结果

| 路径 | 内容 |
|---|---|
| `outputs\eeg-002\train` | 已完成的整句基线 |
| `outputs\eeg-002\local-bidirectional\train` | 双向局部正式训练 |
| `outputs\eeg-002\local-history\train` | 历史局部正式训练 |
| `outputs\eeg-002\context_comparison\status.json` | 串行队列当前阶段与完成/失败状态 |
| 各实验的 `train\status.json` | 当前轮数、验证损失、验证 CER 和 checkpoint 路径 |
| `context_comparison\comparison.md` | 自动更新的三组对照表 |
| `context_comparison\comparison.csv` | 轮数、最佳验证指标、测试 CER/WER、显存及相对基线变化 |
| `context_comparison\paired_sentence_CER.csv` | 同一测试句在三组中的 CER，供逐句对照 |

表中省略 `outputs\eeg-002\` 前缀的 `context_comparison\...` 路径均位于 `D:\code\Brain2QwertyV1\outputs\eeg-002` 下。
测试 CER 分别报告相对提示句和相对实际按键字符的结果，数值越低越好。整句基线对应为 **74.83% / 77.99%**。
训练中表格的测试指标留空，待完整测试和逐句导出成功后填写；短试跑指标不会混入正式对照。

查看进度：

```powershell
cd D:\code\Brain2QwertyV1
Get-Content outputs\eeg-002\context_comparison\status.json
Get-Content outputs\eeg-002\local-bidirectional\train\status.json
```

队列正常完成后 `state` 变为 `complete`，两组的最佳模型测试和汇总已自动完成。
如果队列失败，查看 `context_comparison\status.json` 中的原因及该目录下的日志。

## 重新启动与独立运行

队列已启动时不需要再次启动。中断后可重新运行同一入口，完整结束的阶段会跳过，未完成的正式训练从 `last.ckpt`（或可用的 `best.ckpt`）恢复：

```powershell
.\start_local_comparison.ps1
```

独立运行某种模式：

```powershell
.\run_subject.ps1 train --context-mode local-bidirectional --reference-run outputs\eeg-002\train
.\run_subject.ps1 train --context-mode local-history --reference-run outputs\eeg-002\train
```

重新评估已有最佳模型：

```powershell
.\run_subject.ps1 eval --ckpt outputs\eeg-002\local-history\train\best.ckpt
```

评估或恢复训练时，会读取 checkpoint 同目录的 `config.json` 来识别上下文；显式传入冲突的模式会报错。
需要独立评估结果参与汇总时，正式训练队列本身已完成的测试结果保存在对应 `train` 目录中；独立 `eval` 的输出另行保存在该模式的 `eval` 目录。

手动刷新对照报告（不会训练）：

```powershell
.\.venv\Scripts\python.exe -m brain2qwerty_v1.scripts.local_comparison --report-only
```

## 实现

- `brain2qwerty_v1/context.py`：窗口索引与边界掩码。
- `brain2qwerty_v1/pl_module.py`：独立窗口、当前位置预测。
- `brain2qwerty_v1/scripts/single_subject.py`：上下文配置、基线核对、checkpoint 模式检查。
- `brain2qwerty_v1/scripts/local_comparison.py`：串行训练、短试跑验证、完整测试和报告汇总。

正式运行还会在 `environment.json` 中保存相关本地源码的 SHA-256，以标明该次训练实际使用的实现。
精度对齐前的临时输出已保存在 `outputs\eeg-002\precision_alignment_archive_20261016_020248`，不参与正式比较。
