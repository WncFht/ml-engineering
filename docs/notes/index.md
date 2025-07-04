---
title: 所有文档
createTime: 2025/07/03 00:47:40
permalink: /notes/notes/
---
# 机器学习工程开放手册

这是一个开放的方法论、工具和分步说明的集合，旨在帮助成功训练和微调大型语言模型、多模态模型及其推理。

这是一份技术材料，适用于 LLM/VLM 训练工程师和操作员。也就是说，这里的内容包含大量脚本和可复制粘贴的命令，使您能够快速满足需求。

这个仓库是我训练大型语言模型（LLM）和视觉语言模型（VLM）经验的持续性脑力倾倒；其中很多知识是我在 2022 年训练开源 [BLOOM-176B](https://huggingface.co/bigscience/bloom) 模型和 2023 年训练 [IDEFICS-80B](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct) 多模态模型，以及 2024 年在 [Contextual.AI](https://contextual.ai/) 训练 RAG 模型时获得的。

我一直在为自己整理这些信息，以便能快速找到过去研究过且行之有效的解决方案，但像往常一样，我很高兴与更广泛的机器学习社区分享这些笔记。


## 目录


**第一部分：洞见**

1. **[人工智能战场工程](./insights/ai-battlefield.md)** - 成功所需要了解的知识。

1. **[如何选择云提供商](./insights/how-to-choose-cloud-provider.md)** - 这些问题将使您能够成功体验云计算。

**第二部分：硬件**

1. **[计算](compute)** - 加速器、CPU、CPU 内存。

1. **[存储](storage)** - 本地、分布式和共享文件系统。

1. **[网络](network)** - 节点内和节点间网络。


**第三部分：编排**

1. **[编排系统](orchestration)** - 管理容器和资源
1. **[SLURM](orchestration/slurm)** - 简单的 Linux 资源管理工具


**第四部分：训练**

1. **[训练](training)** - 模型训练相关指南


**第五部分：推理**

1. **[推理](inference)** - 模型推理洞见


**第六部分：开发**

1. **[调试和故障排除](debug)** - 如何调试简单和困难的问题

1. **[更多调试技巧](https://github.com/stas00/the-art-of-debugging)**

1. **[测试](testing)** - 使测试编写变得愉快的众多技巧和工具


**第七部分：其他**

1. **[资源](resources)** - LLM/VLM 编年史


## 更新

我会在我的推特频道 [https://twitter.com/StasBekman](https://twitter.com/StasBekman) 上宣布任何重大更新。

## PDF 版本

下载本书的 [PDF](https://huggingface.co/stas/ml-engineering-book/resolve/main/Stas%20Bekman%20-%20Machine%20Learning%20Engineering.pdf?download=true) 版本。

我会尽量每周重建一次，但如果您想要最新的版本，构建说明在[这里](build)。

感谢 HuggingFace 允许我在 [HF hub](https://huggingface.co/) 上托管我的书的 PDF。

## 讨论

如果您想讨论与机器学习工程相关的内容，本仓库提供[社区讨论](https://github.com/stas00/ml-engineering/discussions)功能 - 所以请不要犹豫，分享您的经验或就您热衷的话题发起新的讨论。

## 关键比较表

高端加速器：

- [理论加速器 TFLOPS](compute/accelerator#tflops-comparison-table)
- [加速器内存大小和速度](compute/accelerator#accelerator-memory-size-and-speed)

网络：

- [理论节点间速度](network#inter-node-networking)
- [理论节点内速度](network#intra-node-networking)

## 快捷方式

您可能需要经常快速查找的内容。

工具：

- [all_reduce_bench.py](network/benchmarks/all_reduce_bench.py) - 一种比 nccl-tests 更简单的基准测试网络吞吐量的方法。
- [torch-distributed-gpu-test.py](debug/torch-distributed-gpu-test.py) - 一个快速测试节点间连接性的工具。
- [mamf-finder.py](compute/accelerator/benchmarks/mamf-finder.py) - 您可以从加速器获得的实际 TFLOPS 测量值是多少。

指南：

- [调试 pytorch 应用程序](debug/pytorch.md) - 快速复制粘贴解决方案，以解决挂起或中断的 pytorch 应用程序。
- [slurm 用户指南](orchestration/slurm/users.md) - slurm 备忘单和技巧。
- [制作小型模型/数据集/分词器](debug/make-tiny-models-tokenizers-datasets.md)
- [LLM/VLM 编年史合集](resources#publicly-available-training-llmvlm-logbooks)


## 致谢

没有被委托进行特定的 LLM/VLM 训练，我不可能获得最初的专业知识。这是一个只有少数人能享有的特权，因为租用大型机器学习计算集群的成本高得令人望而却步。所以希望机器学习社区的其他人能通过这些笔记间接地学习。

特别感谢 [Thom Wolf](https://github.com/thomwolf)，在我对大规模训练一无所知时，他建议我领导 BLOOM-176B 的训练。这个项目让我进入了紧张的学习过程。当然，还要感谢 HuggingFace 给我机会全职从事 BLOOM-176B 以及后来的 IDEFICS-80B 的训练。

最近，我在 [Contextual.AI](https://contextual.ai/) 训练模型和构建可扩展的训练/推理系统时，继续扩展我的知识和经验，我感谢 Aman 和 Douwe 给予的这个机会。

我还要感谢众多[贡献者](contributors.md)，他们使得这篇文章变得出色且无错误。

## 贡献

如果您发现错误、拼写错误或想提出改进建议，请随时提交 [Issue](https://github.com/stas00/ml-engineering/issues) 或贡献 PR。


## 许可证

本网站内容根据[知识共享署名-相同方式共享 4.0 国际许可协议](LICENSE-CC-BY-SA)分发。


## 引文

```bibtex
@misc{bekman2024mlengineering,
  author = {Bekman, Stas},
  title = {机器学习工程开放手册},
  year = {2023-2024},
  publisher = {Stasosphere Online Inc.},
  journal = {GitHub 仓库},
  url = {https://github.com/stas00/ml-engineering}
}
```

## 我的仓库地图

✔ **机器学习：**
 [机器学习工程开放手册](https://github.com/stas00/ml-engineering) |
 [ML ways](https://github.com/stas00/ml-ways) |
 [移植](https://github.com/stas00/porting)

✔ **指南：**
 [调试的艺术](https://github.com/stas00/the-art-of-debugging)

✔ **应用：**
 [ipyexperiments](https://github.com/stas00/ipyexperiments)

✔ **工具和备忘单：**
 [bash](https://github.com/stas00/bash-tools) |
 [conda](https://github.com/stas00/conda-tools) |
 [git](https://github.com/stas00/git-tools) |
 [jupyter-notebook](https://github.com/stas00/jupyter-notebook-tools) |
 [make](https://github.com/stas00/make-tools) |
 [python](https://github.com/stas00/python-tools) |
 [tensorboard](https://github.com/stas00/tensorboard-tools) |
 [unix](https://github.com/stas00/unix-tools)
