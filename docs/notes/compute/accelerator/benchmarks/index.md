---
title: 自述文件
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/eqbpoog3/
---
# 加速器基准测试

## 最大可实现 Matmul FLOPS 查找器

最大可实现 Matmul FLOPS (MAMF) 基准测试: [mamf-finder.py](./mamf-finder.py)

有关详细讨论和各种加速器的数字，请参阅[最大可实现 FLOPS](../#maximum-achievable-flops)。

虽然一些加速器制造商公布了理论 TFLOPS，但这些通常是无法达到的。因此，当我们尝试优化我们的软件时，我们没有一个现实的性能标准来与自己比较。模型 FLOPS 利用率 (MFU) 指标衡量的是相对于理论 TFLOPS 所达到的 TFLOPS。通常，当 MFU 得分在 50% 左右时，就被认为是胜利了。但这并不能告诉我们距离真正可实现的吞吐量还有多远。

这个基准测试扫描各种大的 matmul 形状，并报告它记录到的最高可实现 TFLOPS。由于 transformers 训练和部分推理工作负载主要由大的 matmul 操作主导，因此可以安全地使用在每个加速器上可以测量的最佳 matmul TFLOPS 作为最大可实现 Matmul FLOPS (MAMF) 的粗略估计。现在，可以使用模型可实现 Matmul FLOPS 利用率 (MAMFU) 来代替之前使用的 MFU。

因此，现在您可以将您为训练或推理测量的 TFLOPS 与一个现实的数字进行比较。由于您现在将更接近 100%，因此更容易知道何时停止优化。

目前支持的高端架构：
- NVIDIA: V100, A100, H100, ...
- AMD: MI250, MI300X, MI325X, ...
- Intel Gaudi2/3

公平性说明：
- 如果您能找到一种更好、更有效的方法来检测最佳 matmul TFLOPS，将每个新加速器视为一个黑匣子，请好心地提交一个包含改进和生成的日志文件的 PR。
- 此外，如果您知道这个基准测试应该在特殊条件下运行以显示最佳结果，例如某些内核设置或类似设置，请提交一个 PR 以添加此类特殊说明。例如，对于 AMD MI300X，我被告知禁用 numa_balancing 应该会有所帮助。

### 特定架构说明：

在运行基准测试之前，请遵循特殊的设置说明以获得最佳结果：

**MI300x, MI325X, etc.**:

1. 关闭 numa_balancing 以获得更好的性能：
```
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```
2. 启用：
```
export PYTORCH_TUNABLEOP_ENABLED=1
```
这将使第一次迭代非常慢，因为它在 BLAS 库中为遇到的每个 `matmul` 形状搜索最佳的 GEMM 算法，但后续操作可能会明显快于基线。请参阅[使用 PyTorch TunableOp 在 ROCm 上加速模型](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html) (需要 `torch>=2.3`) [文档](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md)。

**Intel dGPUs (A770, A750, B580, etc.)**
- 遵循 Intel Extension for Pytorch [安装步骤](https://pytorch-extension.intel.com/installation?platform=gpu)

### 使用示例

在下面的范围中，`K` 是规约维度，因此 `(MxK)*(KxN)=(MxN)`，我们打印最佳测量 TFLOPS 的 MxKxN 形状。

此外，默认情况下，我们对每个形状使用 50 次预热和 100 次测量迭代，然后选择最快的结果（而不是平均值）。您可以通过 `--num_warmup_iterations` 和 `--num_iterations` 参数相应地更改迭代次数。

您可以通过 `--dtype` 参数指定数据类型，它必须是有效的 `torch` 数据类型之一 - 例如 `float8_e4m3fn`, `float16`, `bfloat16`, `float32` 等。如果未指定，则使用 `bfloat16`。

这里我们做 `torch.mm(MxK,KxN) -> MxN`

1. 快速运行（1 分钟以内）- 应该能得到最大可实现结果的 80-90% 左右 - 适合快速试用，但不足以获得高测量值。

```
./mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

2. 更详尽的搜索（15-30 分钟）- 但当它运行足够长时，你可以按 Ctrl-C 并获得迄今为止最好的结果：

```
./mamf-finder.py --m_range 0 16384 1024 --n_range 0 16384 1024 --k_range 0 16384 1024  --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

可以随意将步长从 1024 减小到 512 或 256 - 但运行时间将相应地增加 8 倍或 64 倍。1k 步长应该可以很好地快速覆盖不同的形状范围。

3. 一个超长的详尽搜索（可能需要数小时/数天）- 但当它运行足够长时，你可以按 Ctrl-C 并获得迄今为止最好的结果：

```
./mamf-finder.py --m_range 0 20480 256 --n_range 0 20480 256 --k_range 0 20480 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

4. 如果您想测量训练中使用的特定形状，请使用确切的形状，而不是范围，例如，如果您想测量 1024x1024x1024 - 您可以运行：

```
./mamf-finder.py --m 1024 --n 1024 --k 1024 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

5. 特定加速器范围搜索建议

但是，似乎不同的加速器具有不同的形状范围，可以产生最佳的 TFLOPS，因此很难建议一个对所有加速器都有效的范围 - 相反，这里有一些基于实验和贡献者建议的建议：

- **A100** + **MI300X**

```
./mamf-finder.py --m_range 0 5376 256 --n_range 0 5376 256 --k_range 0 5376 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

- **H100**

```
./mamf-finder.py --m_range 0 20480 256 --n_range 0 20480 256 --k_range 0 20480 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

要更好地了解哪种形状可以为特定加速器提供最高的 matmul FLOPS，请参阅[向量和矩阵大小的可分性](../../../training/performance/README.md#vector-and-matrix-size-divisibility)。


### 结果

到目前为止我收集到的测量结果可以在[最大可实现 Matmul FLOPS 比较表](../#maximum-achievable-matmul-flops-comparison-table)中找到。当我能够访问某个特定加速器时，我会自己运行基准测试，当我没有时，是热心的贡献者投入他们的时间来获取这些数字。所以我非常感谢[那些人](../../../contributors.md)。




## 如何对加速器进行基准测试

### CUDA 基准测试

关于如何执行 CUDA 基准测试，有一些非常出色的详细文章：

1. [如何在 Pytorch 中准确地计时 CUDA 内核](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch)
2. [如何在 CUDA 设备上对代码进行基准测试？](https://salykova.github.io/sgemm-gpu#2-how-to-benchmark-code-on-cuda-devices) - 这篇文章与（1）的不同之处在于，它建议同时设置 GPU 和内存时钟，而（1）只锁定 GPU 时钟。

您可以在 [mamf-finder.py](./mamf-finder.py) 中看到这些指令的应用（除了时钟锁定）

以下是一些优秀的相关读物：

- Horace 的[奇怪的是，当给定"可预测"数据时，GPU 上的矩阵乘法运行得更快](https://www.thonking.ai/p/strangely-matrix-multiplications?utm_source=substack&publication_id=1781836&post_id=142508107)展示了如果使用非正态分布的数据，基准测试如何可能过度报告，以及功耗如何影响性能。
