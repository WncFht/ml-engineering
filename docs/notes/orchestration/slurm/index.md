---
title: 自述文件
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/9hjgaba7/
---
# 在 SLURM 环境中工作

除非您很幸运，拥有一个完全在您控制之下的专用集群，否则您很可能不得不使用 SLURM 与他人共享 GPU。但是，通常情况下，如果您在 HPC 上进行训练，并且您被分配了一个专用分区，您仍然需要使用 SLURM。

SLURM 的缩写代表：**简单的 Linux 资源管理工具** - 尽管现在它被称为 Slurm 工作负载管理器。它是一个免费的开源作业调度程序，适用于 Linux 和类 Unix 内核，被世界上许多超级计算机和计算机集群使用。

这些章节不会试图详尽地教您 SLURM，因为有很多手册，但会涵盖一些有助于训练过程的特定细微差别。

- [SLURM 用户指南](./users.md) - 在 SLURM 环境中进行训练所需知道的一切。
- [SLURM 管理](./admin.md) - 如果您不幸除了使用 SLURM 集群外还需要管理它，本文档中有一个不断增长的配方列表，可以帮助您更快地完成工作。
- [性能](./performance.md) - SLURM 性能细微差别。
- [启动器脚本](./launchers) - 如何在 SLURM 环境中使用 `torchrun`、`accelerate`、pytorch-lightning 等启动。
