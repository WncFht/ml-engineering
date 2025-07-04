---
title: 数据集
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/4wmq04v7/
---
# 处理数据集

## 在主进程上预处理和缓存数据集

HF Accelerate 有一个非常简洁的容器 [`main_process_first`](https://huggingface.co/docs/accelerate/v0.4.0/accelerator.html#accelerate.Accelerator.main_process_first)，它允许编写如下代码：

```
with accelerator.main_process_first():
    # 加载和预处理数据集
    dataset = datasets.load_dataset(...)
    # 可选地缓存它，并让其余进程加载缓存
```
而不是不那么直观且需要代码重复的：
```
if rank == 0:
    dataset = datasets.load_dataset(...)
dist.barrier()
if not rank == 0:
    dataset = datasets.load_dataset(...)
```

您希望在主进程上下载和处理数据，而不是所有进程，因为它们将并行地重复相同的事情，而且很可能会写入相同的位置，这将导致交错的损坏结果。从 IO 的角度来看，串行化此类工作也要快得多。

现在有 `main_process_first` 和 `local_main_process_first` - 前者用于当您的数据位于共享文件系统上并且所有计算节点都可以看到它时。后者用于当数据位于每个节点的本地时。

如果您不使用 HF Accelerate，我重新创建了类似的容器，只是将它们命名为：

- `global_main_process_first` - 用于共享文件系统
- `local_main_process_first` - 用于节点本地文件系统

您可以在[这里](tools/main_process_first.py)找到它们。

现在，如果您想编写一个可以自动在共享和本地文件系统上工作的通用代码怎么办？我添加了另一个辅助程序，它可以自动发现我们正在处理的文件系统类型，并基于此调用正确的容器。我称之为 `main_process_by_path_first`，其使用方式如下：

```
path = "/path/to/data"
with main_process_by_path_first(path):
    # 加载和预处理数据集
    dataset = datasets.load_dataset(...)
    # 可选地缓存它，并让其余进程加载缓存
```

您可以在[这里](tools/main_process_first.py)找到它。

当然，除了容器之外，您还需要一些工具来检查主进程的类型，因此有 3 个与容器相对应的工具：

- `is_main_process_by_path(path)`
- `is_local_main_process()`
- `is_global_main_process()`

它们都可以在[这里](tools/main_process_first.py)找到。

您可以通过运行以下命令查看它们的实际效果：

```
python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d tools/main_process_first.py
```
