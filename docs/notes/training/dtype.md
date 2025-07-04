---
title: 数据类型
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/s7pwwt0g/
---
# 张量精度 / 数据类型

截至撰写本文时，机器学习中常用的数据类型（通常称为 `dtype`）如下：

浮点格式：
- fp32 - 32 位
- tf32 - 19 位 (NVIDIA Ampere+)
- fp16 - 16 位
- bf16 - 16 位
- fp8 - 8 位 (E4M3 和 E5M2 格式)
- fp6 - 6 位
- fp4 - 4 位

有关视觉比较，请参阅以下表示形式：

![fp32-tf32-fp16-bf16](images/fp32-tf32-fp16-bf16.png)

（[来源](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)）

![fp16-bf16-fp8](images/fp16-bf16-fp8.png)

（[来源](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)）


新硬件正在采用的新格式是：
- fp4: `float4_e2m1fn`
- fp6:`float6_e2m3fn` 和 `float6_e3m2fn`
- fp8: `float8_e3m4`, `float8_e4m3`, `float8_e4m3b11fnuz`, `float8_e4m3fn`, `float8_e4m3fnuz`, `float8_e5m2`, `float8_e5m2fnuz`, `float8_e8m0fnu`

[这里](https://github.com/jax-ml/ml_dtypes?tab=readme-ov-file#specifications-of-implemented-floating-point-formats) 对这些变体中的每一种都有很好的解释。

要解读数字后面的字母：
- `e` 表示指数的长度
- `m` 表示尾数的长度
- `b` 表示偏置

要解读数字后面出现的字母：
- `f` 表示它只有限值（没有无穷大）。
- `n` 表示它包含 NaN，但仅在外部范围内。
- `u` 代表无符号格式。
- `uz` 代表无符号零。

所以，例如：`float8_e4m3b11fnuz` 代表 fp8 + 4 位指数 + 3 位尾数 + 偏置 11 + 仅限有限值 + 包含 NaN，但仅在外部范围内 + 无符号零。


量化中使用的整数格式：

- int8 - 8 位
- int4 - 4 位
- int1 - 1 位

## ML dtype 演进

最初 ML 使用 fp32，但它非常慢。

接下来[使用 fp16 和 fp32 组合的混合精度](https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/)被发明出来，极大地加快了训练速度。

![fp32/fp16 混合精度](images/mixed-precision-fp16.png)

（[来源](https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/)）

但 fp16 被证明不是很稳定，训练 LLM 非常困难。

幸运的是，bf16 问世并使用相同的混合精度协议取代了 fp16。这使得 LLM 训练更加稳定。

然后 fp8 问世，混合精度已切换到[那个](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)，这使得训练速度更快。请参阅论文：[用于深度学习的 FP8 格式](https://arxiv.org/abs/2209.05433)。

为了体会不同格式之间的速度提升，请看下表，了解 NVIDIA A100 TFLOPS 规格（无稀疏性）：

| 数据类型 | TFLOPS |
| :---                   |    --: |
| FP32                   |   19.5 |
| Tensor Float 32 (TF32) |    156 |
| BFLOAT16 Tensor Core   |    312 |
| FP16 Tensor Core       |    312 |
| FP8 Tensor Core        |    624 |
| INT8 Tensor Core       |    624 |

除了 fp32 比其余的慢得多之外，每个后续的 dtype 都比前一个快大约 2 倍。

与混合训练方案并行，ML 社区开始提出各种量化方法。可能最好的例子之一是 Tim Dettmers 的 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)，它提供了许多 4 位和 8 位量化解决方案。Deepspeed 团队也有一些[有趣的量化解决方案](https://www.deepspeed.ai/tutorials/model-compression/)。

## TF32

TF32 是自 Ampere 以来 NVIDIA GPU 上可用的一种神奇的数据类型，它允许以比普通 fp32 `matmul` 快得多的速度执行 fp32 `matmul`，而精度损失很小。

以下是 A100 TFLOPS（无稀疏性）的示例：

| 数据类型 | TFLOPS |
| :---                   |    --: |
| FP32                   |   19.5 |
| Tensor Float 32 (TF32) |    156 |

如您所见，TF32 比 FP32 快 8 倍！

默认情况下它是禁用的。要启用它，请在程序开头添加：

```
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

有关实际精度损失的更多信息，请参阅[此](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)。


## 何时使用 fp32 累加器

每当使用低精度 dtype 时，都必须小心不要在该 dtype 中累积中间结果。

类似 `LayerNorm` 的操作不能在半精度下工作，否则它们可能会丢失大量数据。因此，当这些操作被正确实现时，它们会在输入的 dtype 中高效地进行内部工作，但使用 fp32 累加寄存器，然后它们的输出被下转换为输入的精度。

通常，只有累加是在 fp32 中完成的，因为否则将许多低精度数字相加会非常耗损。

以下是一些例子：

1. 规约集合

* fp16：如果使用了损失缩放，则可以在 fp16 中进行

* bf16：只能在 fp32 中进行

2. 梯度累积

* 对于 fp16 和 bf16，最好在 fp32 中完成，但对于 bf16 绝对是必须的

3. 优化器步骤 / 梯度消失

* 当将一个非常小的梯度添加到一个大数上时，该加法通常会被抵消，因此通常使用 fp32 主权重和 fp32 优化器状态。

* 当使用 [Kahan 求和](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)或[随机舍入](https://en.wikipedia.org/wiki/Rounding)（在[重新审视 BFloat16 训练](https://arxiv.org/abs/2010.06192)中引入）时，可以使用 f16 主权重和优化器状态。

有关后者的示例，请参阅：[AnyPrecision 优化器](https://github.com/pytorch/torchdistx/pull/52)，最新版本可在[此处](https://github.com/facebookresearch/multimodal/blob/6bf3779a064dc72cde48793521a5be151695fc62/torchmultimodal/modules/optimizers/anyprecision.py#L17)找到。


## 训练后更改精度

有时，在模型训练后更改精度是可以的。

- 在 fp16 模式下使用 bf16 预训练模型通常会失败 - 由于溢出（fp16 中可以表示的最大数字是 64k），有关深入讨论和可能的解决方法，请参阅此 [PR](https://github.com/huggingface/transformers/pull/10956)。

- 在 bf16 模式下使用 fp16 预训练模型通常可以工作 - 转换时会损失一些性能，但应该可以工作 - 最好在使用前进行一些微调。
