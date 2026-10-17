# Colab：CPU 预热与 GPU 训练

现在通过 GitHub 同步代码，入口和命令见 [WORKFLOW_zh.md](WORKFLOW_zh.md)。

1. 在本地修改代码并用 002 做 `cache`、`smoke`，然后提交并推到 GitHub。
2. 在纯 CPU Colab 打开 [colab_cpu_cache.ipynb](notebooks/colab_cpu_cache.ipynb)：挂载全量数据、拉取代码、预热 EEG/MEG 缓存并生成 tar。
3. 在 GPU Colab 打开 [colab_gpu_train.ipynb](notebooks/colab_gpu_train.ipynb)：拉取同一代码、将缓存解压到本地磁盘、训练，Checkpoint 和日志写回 Drive。

Notebook 不再内嵌源码或凭据。旧单受试者 Notebook 和源码压缩包已退出 Git 同步流程。`tools/build_colab_notebook.py` 只用于重新生成这两个 Git 入口 Notebook。
