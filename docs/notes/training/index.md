---
title: 训练
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/c1k4pcy2/
---
# 训练

**小节**:

- [模型并行](model-parallelism)

- [性能](performance)

- [容错](fault-tolerance)

- [可复现性](reproducibility)

- [不稳定性](instabilities)

- [检查点](checkpoints)

- [训练超参数和模型初始化](hparams.md)

- [张量精度 / 数据类型](dtype.md)

- [仅使用单个节点模拟多节点设置](emulate-multi-node.md) - 关于如何仅使用单个节点模拟多节点设置的说明 - 我们在这里使用 `deepspeed` 启动器。

- [使用微调示例从头开始重新训练 HF hub 模型](re-train-hub-models.md)

- [数据集](datasets.md)

**工具**:

- [printflock.py](tools/printflock.py) - 一个小巧的库，使您的 `print` 调用在多 GPU 环境中不会交错。

- [multi-gpu-non-interleaved-print.py](tools/multi-gpu-non-interleaved-print.py) - 一个基于 `flock` 的 `print` 包装器，可防止在多个进程同时打印时消息交错 - 这在使用多 GPU 的 `torch.distributed` 时就是这种情况。
