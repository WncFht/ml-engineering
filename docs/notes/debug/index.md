---
title: 调试
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/5ew2e2lz/
---
# 调试和故障排除


## 指南

- [调试 PyTorch 程序](./pytorch.md)

- [诊断多节点多 GPU Python 程序中的挂起和死锁](./torch-distributed-hanging-solutions.md)

- [网络调试](../network/debug/)

- [NVIDIA GPU 故障排除](../compute/accelerator/nvidia/debug.md)

- [下溢和上溢检测](./underflow_overflow.md)



## 工具

- [调试工具](./tools.md)

- [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) - 这是一个 `torch.distributed` 诊断脚本，用于检查集群中的所有 GPU（一个或多个节点）是否可以相互通信并分配 GPU 内存。

- [NicerTrace](./NicerTrace.py) - 这是一个改进的 `trace` python 模块，在构造函数中添加了多个附加标志，并提供更有用的输出。
