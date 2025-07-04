---
title: 自述文件
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/lds5jibu/
---
# 使用 SLURM 的单节点和多节点启动器

以下是完整的 SLURM 脚本，演示了如何将各种启动器与使用 `torch.distributed` 的软件集成（但应能轻松适应其他分布式环境）。

- [torchrun](torchrun-launcher.slurm) - 与 [PyTorch distributed](https://github.com/pytorch/pytorch) 一起使用。
- [accelerate](accelerate-launcher.slurm) - 与 [HF Accelerate](https://github.com/huggingface/accelerate) 一起使用。
- [lightning](lightning-launcher.slurm) - 与 [Lightning](https://lightning.ai/) (“PyTorch Lightning” 和 “Lightning Fabric”) 一起使用。
- [srun](srun-launcher.slurm) - 与本地 SLURM 启动器一起使用 - 在这里我们必须手动预设 `torch.distributed` 期望的环境变量。

所有这些脚本都使用 [torch-distributed-gpu-test.py](../../../debug/torch-distributed-gpu-test.py) 作为演示脚本，您只需使用以下命令即可将其复制到此处：
```
cp ../../../debug/torch-distributed-gpu-test.py .
```
假设您克隆了此仓库。但您可以将其替换为您需要的任何其他内容。
