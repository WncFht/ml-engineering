---
title: performance
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/z40pqowr/
---
# SLURM 性能

在这里，您将找到有关影响性能的 SLURM 特定设置的讨论。

## srun 的 `--cpus-per-task` 可能需要明确指定

您需要确保由 `srun` 启动的程序能接收到预期数量的 CPU 核心。例如，在典型的机器学习训练程序中，每个 GPU 至少需要一个 CPU 核心来驱动它，还需要几个核心用于 `DataLoader`。您需要多个核心，以便每个任务可以并行执行。如果您有 8 个 GPU，每个 GPU 有 2 个 `DataLoader` 工作进程，那么每个节点至少需要 `3*8=24` 个 CPU 核心。

每个任务的 CPU 核心数由 `--cpus-per-task` 定义，该参数传递给 `sbatch` 或 `salloc`，`srun` 最初会继承此设置。然而，最近这种行为发生了变化：

`sbatch` 手册页中的一段引述：

> 注意：从 22.05 开始，srun 将不会继承 salloc 或 sbatch 请求的 --cpus-per-task 值。如果需要为任务设置，必须再次通过调用 srun 或使用 SRUN_CPUS_PER_TASK 环境变量来请求。

这意味着，如果在过去您的 SLURM 脚本可能是：

```
#SBATCH --cpus-per-task=48
[...]

srun myprogram
```

并且由 `srun` 启动的程序会接收到 48 个 CPU 核心，因为 `srun` 过去会从 `sbatch` 或 `salloc` 的设置中继承 `--cpus-per-task=48`，根据引用的文档，自 SLURM 22.05 起，这种行为不再成立。

脚注：我用 SLURM@22.05.09 测试过，旧的行为仍然有效，但这在 23.x 系列中肯定是这样。所以变化可能发生在 22.05 系列的后期版本中。

所以，如果您保持原样，现在程序将只接收 1 个 CPU 核心（除非 `srun` 的默认值已被修改）。

您可以使用 `os.sched_getaffinity(0))` 轻松测试您的 SLURM 设置是否受到影响，因为它显示了当前进程可以使用的 CPU 核心。因此，用 `len(os.sched_getaffinity(0))` 来计算这些核心应该很容易。

以下是您可以测试是否受到影响的方法：
```
$ cat test.slurm
#!/bin/bash
#SBATCH --job-name=test-cpu-cores-per-task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48   # 如果您的环境少于 48 个 cpu 核心，请进行调整
#SBATCH --time=0:10:00
#SBATCH --partition=x        # 根据您的环境调整为正确的分区名称
#SBATCH --output=%x-%j.out

srun python -c 'import os; print(f"可见的 cpu 核心数: {len(os.sched_getaffinity(0))}")'
```

如果您得到
```
可见的 cpu 核心数: 48
```
那么您不需要做任何事情，但是如果您得到：
```
可见的 cpu 核心数: 1
```
或其他小于 48 的值，那么您就受到了影响。

要解决这个问题，您需要将您的 SLURM 脚本更改为：

```
#SBATCH --cpus-per-task=48
[...]

srun --cpus-per-task=48 myprogram
```
或者：
```
#SBATCH --cpus-per-task=48
[...]

SRUN_CPUS_PER_TASK=48
srun myprogram
```

或者用"一次编写，永不忘记"的方式自动化它：
```
#SBATCH --cpus-per-task=48
[...]

SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun myprogram
```



## 是否启用超线程

正如在[超线程](users.md#hyper-threads)部分所解释的，如果您的 CPU 支持超线程，您应该能够将可用 CPU 核心数量加倍，对于某些工作负载，这可能会带来更快的整体性能。

然而，您应该测试有和没有 HT 的性能，比较结果并选择能带来最佳结果的设置。

案例研究：在 AWS p4 节点上，我发现启用 HT 会使网络吞吐量慢 4 倍。从那时起，我们就在那个特定的设置上小心地禁用了 HT。
