---
title: 编排
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/uzb76yod/
---
# 编排

有许多容器/加速器编排解决方案 - 其中许多是开源的。

到目前为止，我一直在使用 SLURM：

- [SLURM](slurm/) - 简单的 Linux 资源管理工具，您肯定会在大多数 HPC 环境中找到它，并且通常大多数云提供商都支持它。它已经存在了 20 多年。
- SLURM on Kubernetes: [Slinky](https://github.com/stas00/ml-engineering/pull/99) - 这是一个最近创建的框架，用于在 Kubernetes 之上运行 SLURM。

另一个最受欢迎的编排器是 Kubernetes：

- [Kubernetes](https://kubernetes.io/) - 也称为 K8s，是一个用于自动化容器化应用程序部署、扩展和管理的开源系统。这里有一个很好的 [SLURM 和 K8s 的比较](https://www.fluidstack.io/post/is-kubernetes-or-slurm-the-best-orchestrator-for-512-gpu-jobs)。

以下是其他各种不太流行但仍然非常强大的编排解决方案：

- [dstack](https://github.com/dstackai/dstack) 是一个轻量级的开源替代方案，可替代 Kubernetes 和 Slurm，通过多云和本地支持简化 AI 容器编排。它原生支持 NVIDIA、AMD 和 TPU。
- [SkyPilot](https://github.com/skypilot-org/skypilot) 是一个用于在任何基础设施上运行 AI 和批处理工作负载的框架，提供统一的执行、高成本节省和高 GPU 可用性。
- [OpenHPC](https://github.com/openhpc/ohpc) 提供了部署和管理 HPC Linux 集群所需的各种通用、预构建的组件，包括配置工具、资源管理、I/O 客户端、运行时、开发工具、容器和各种科学库。
- [run.ai](https://www.run.ai/) - 被 NVIDIA 收购，并计划很快开源。
- [Docker Swarm](https://docs.docker.com/engine/swarm/) 是一个容器编排工具。
- [IBM Platform Load Sharing Facility (LSF)](https://www.ibm.com/products/hpc-workload-management) Suites 是一个用于分布式高性能计算 (HPC) 的工作负载管理平台和作业调度程序。
