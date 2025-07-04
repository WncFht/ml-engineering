---
title: 自述文件
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/lyztvg57/
---
# 避免、从不稳定性中恢复和理解不稳定性

子章节：

* [理解训练损失模式](training-loss-patterns.md) - 尖峰、发散、顿悟时刻、恢复等的类型。

## 从训练日志中学习

最好的学习方法是阅读[公开可用的训练 LLM/VLM 日志](../../resources#publicly-available-training-llmvlm-logbooks)，因为在那里你可以确切地看到发生了什么以及问题是如何被克服的。


## STD 初始化

正确初始化张量的初始分布可以对训练的稳定性产生巨大影响。`std` 值不是固定的，取决于隐藏维度的大小。

在我们预 BLOOM 104B 的实验中，这被证明是一个非常关键的设置，在我们弄清楚 Megatron-LM 中 0.02 的默认 `--init-method-std` 对于我们的模型来说太大了之前，我们无法突破最初的几千次迭代。

我们参考了这两个来源：

1. "Transformers without Tears" 论文 https://arxiv.org/abs/1910.05895 规定：`sqrt(2/(NHIDDEN*5))`

2. 530B 训练论文 https://arxiv.org/abs/2201.11990 他们使用了一个更小的初始化公式：`sqrt(1/(NHIDDEN*3))`

并决定使用 530B 的那个，因为它会导致一个更小的初始化值。

为了更容易比较这两个公式，它们可以重写为：
1. `sqrt(0.4000/NHIDDEN)`
2. `sqrt(0.3333/NHIDDEN)`

因此，对于 `NHIDDEN=14336`，计算结果为 `sqrt(1/(14336*3)) = 0.00482`，这就是我们使用的。这肯定不是我们在 BLOOM-176B 训练期间没有稳定性问题的唯一原因，但我认为这是关键原因之一。


## 数值不稳定性

在处理低精度数字时，某些数学运算可能不稳定。

例如，请参阅这个非常有趣的[关于数值稳定性的 PyTorch 指南](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)。

现在让我们看一个这个概念在实践中的具体例子。

在 104B 训练实验中，使用了 fp16 混合精度 - [Corby Rosset](https://github.com/corbyrosset) 提出了以下改进，以使[自注意力更稳定](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118)。

具体来说，这[行](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/c839a8aa30731f71b3738d56009be9668508e366/megatron/model/transformer.py#L303)显示 `norm_factor` 可以在 Query * Key 矩阵乘法之后相乘。如果 Q 和 K 的维度非常大，输出可能会爆炸，`norm_factor` 将无法挽救它。

建议：将 `norm_factor` 向内移动，以便在矩阵乘法之前将 Q 和 K 按比例缩小：
```
        matmul_result = torch.baddbmm(
            matmul_result,
            1.0/math.sqrt(self.norm_factor) * query_layer.transpose(0, 1),   # [b * np, sq, hn]
            1.0/math.sqrt(self.norm_factor) * key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0 if alibi is None else 1.0, alpha=1.0)

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
```

为了使运算在数学上等效，如果 n 是标量，A 和 B 是矩阵，将范数因子向内移动需要再次取平方根：
```
n * (A dot B) === (sqrt(n) * A) dot (sqrt(n) * B)
```

现在 A 和 B 的维度可以大得多。

对于 CUDA 内核编写者，截至撰写本文时，[CuBlas](https://docs.nvidia.com/cuda/cublas/index.html) 的 `GemmStridedBatchedEx` 存在类似问题。它定义为：

```
C+i*strideC=αop(A+i*strideA)op(B+i*strideB)+β(C+i*strideC), for i ∈[0,batchCount−1]
```

问题在于 `alpha` 是在矩阵-矩阵乘法完成后相乘的，因此可能会导致不稳定。

## "坏"数据批次和模型参数状态的组合

PaLM 团队在训练更大的模型时观察到数十次损失峰值，这些峰值出现的"间隔极不规律"。虽然他们无法追查到根本原因，但他们通过从较早的检查点重新启动并跳过可能有问题的数据批次来缓解了这个问题。[第 5.1 节 训练不稳定性](https://arxiv.org/pdf/2204.02311.pdf)


## Adam 中的时域相关性发散

[大规模机器学习中 Adam 不稳定性理论](https://arxiv.org/abs/2304.09871) 对训练高达 546B 参数的 LLM 时的发散峰值进行了严格研究 - 并表明时域相关性导致了 Adam 的发散。这是由 epsilon 值不够小以及梯度估计分量变得与 epsilon 相似所触发的。

在第 7.1 节中，他们提出了实用的建议，其中最有趣的一个是将 epsilon 设置为 0，并可能处理除以零的情况。
