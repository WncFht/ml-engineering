---
title: 资源
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/ws7s86ie/
---
# 资源

## 有用的汇编

- [@StellaAthena](https://github.com/StellaAthena) 创建了[通用 LLM 设置电子表格](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit#gid=0)，当您即将开始新的 LLM 训练时，这可能是一个非常有用的资源 - 因为它告诉您已知 LLM 训练的创建方式。

- 几年前，我开始整理关于[模型训练中使用的数据类型](https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671)的信息 - 它只包含少数模型，但如果您正在研究数据类型，它仍然可能有用。我当时使用这些信息来尝试编写[一个模型预训练数据类型自动检测](https://github.com/stas00/ml-ways/blob/master/numbers/detect-model-pretrained-in-bf16-fp16-fp32.ipynb)，这里有一个相关的[float16 与 bfloat16 数值属性比较](https://github.com/stas00/ml-ways/blob/master/numbers/bfloat16-vs-float16-study.ipynb)。

## 公开可用的训练 LLM/VLM 日志

训练 LLM/VLM 的日志和编年史是学习处理训练不稳定性和选择良好超参数的最佳来源之一。

如果您知道此列表中没有的公开 LLM/VLM 训练日志，请 kindly 告知我或通过 PR 添加。谢谢！

该列表除按年份分组外，没有特别的顺序。

### 2021

- BigScience pre-BLOOM 108B 训练实验 (2021):
[编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md) |
[完整规格和讨论](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide)
(备份:
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide/chronicles.md) |
[2](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide))


### 2022

- BigScience BLOOM-176B (2022):
[编年史前传](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) |
[完整规格和讨论](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/)
(备份:
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[2](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles.md) |
[3](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/))

- Meta OPT-175B (2022):
 [日志](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) | [视频](https://www.youtube.com/watch?v=p9IxoSkvZ-M) (备份: [1](https://github.com/stas00/metaseq-backup/tree/main/projects/OPT/chronicles))

- THUDM GLM-130B (2022): [英文日志](https://github.com/THUDM/GLM-130B/blob/main/logs/main-log-en.md) | [中文版](https://github.com/THUDM/GLM-130B/blob/main/logs/main-log.md) (备份: [1](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log-en.md) | [2](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log.md))


### 2023

- HuggingFace IDEFICS-80B 多模态 (Flamingo 复现) (2023): [学习日志](https://github.com/huggingface/m4-logs/blob/master/memos/README.md) | [训练编年史](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md) (备份: [1](https://github.com/stas00/m4-logs-backup/blob/master/memos/README.md) | [2](https://github.com/stas00/m4-logs-backup/blob/master/tr-190-80b/chronicles.md))

- BloombergGPT 50B LLM - [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564) 中的 C 部分


### 2024

- [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/abs/2402.15627) - 该论文涵盖了各种训练问题及其解决方案 - 尽管是在专有模型上，但同样具有指导意义/有用。

- Imbue 的[从裸机到 70B 模型：基础设施设置和脚本](https://imbue.com/research/70b-infrastructure/)非常详细的技术文章涵盖了他们在训练一个专有的 70B 参数模型时必须克服的许多与训练相关的问题。




## 硬件设置日志

- Imbue 发布了一份详细的日志，记录了他们如何设置一个 512 节点的 IB-fat-tree 集群并使其工作：[从裸机到 70B 模型：基础设施设置和脚本](https://imbue.com/research/70b-infrastructure/)，他们还开源了他们在此过程中创建的[集群工具](https://github.com/imbue-ai/cluster-health)。

- SemiAnalysis 发表了一篇关于[建立 Neocloud 集群需要做什么](https://semianalysis.com/2024/10/03/ai-neocloud-playbook-and-anatomy/)的精彩详细文章。
