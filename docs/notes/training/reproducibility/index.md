---
title: 自述文件
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/3888fajb/
---
# 可复现性

## 在基于随机性的软件中实现确定性

在调试时，始终为所有使用的随机数生成器 (RNG) 设置一个固定的种子，以便您在每次重新运行时获得相同的数据/代码路径。

不过，由于系统种类繁多，要涵盖所有系统可能会很棘手。以下是尝试涵盖几种情况的方法：

```
import random, torch, numpy as np
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"使用种子: {seed}")

    random.seed(seed)    # python RNG
    np.random.seed(seed) # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)          # cpu + cuda
    torch.cuda.manual_seed_all(seed) # multi-gpu - 可以在没有 gpu 的情况下调用
    if use_seed: # 速度变慢！ https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    return seed
```
如果您使用这些子系统/框架，则可能还有其他几种方法：
```
    torch.npu.manual_seed_all(seed)
    torch.xpu.manual_seed_all(seed)
    tf.random.set_seed(seed)
```

当您为了解决某个问题而一次又一次地重新运行相同的代码时，请在代码开头使用以下命令设置一个特定的种子：
```
enforce_reproducibility(42)
```
但正如上面提到的，这仅用于调试，因为它会激活各种有助于确定性的 torch 标志，但会降低速度，因此您不希望在生产环境中使用它。

但是，您可以在生产环境中使用以下命令：
```
enforce_reproducibility()
```
即不带显式种子。然后它会选择一个随机种子并记录下来！因此，如果生产环境中发生任何事情，您现在可以重现观察到问题时的相同 RNG。这次没有性能损失，因为只有在您明确提供了种子的情况下才会设置 `torch.backends.cudnn` 标志。比如说它记录了：
```
使用种子: 1234
```
然后您只需将代码更改为：
```
enforce_reproducibility(1234)
```
您将获得相同的 RNG 设置。

正如第一段中提到的，一个系统中可能涉及许多其他 RNG，例如，如果您希望以相同的顺序为 `DataLoader` 提供数据，您需要[也设置它的种子](https://pytorch.org/docs/stable/notes/randomness.html#dataloader)。

其他资源：
- [pytorch 中的可复现性](https://pytorch.org/docs/stable/notes/randomness.html)



## 重现软件和系统环境

当发现结果（例如质量或吞吐量）存在差异时，此方法很有用。

这个想法是记录用于启动训练（或推理）的环境的关键组件，以便如果以后需要完全重现它，就可以做到。

由于使用的系统和组件种类繁多，因此不可能规定一种永远有效的方法。因此，让我们讨论一种可能的方案，然后您可以将其调整到您的特定环境。

这会添加到您的 slurm 启动器脚本中（或您用来启动训练的任何其他方式）- 这是一个 Bash 脚本：

```bash
SAVE_DIR=/tmp # 编辑为真实路径
export REPRO_DIR=$SAVE_DIR/repro/$SLURM_JOB_ID
mkdir -p $REPRO_DIR
# 1. 模块（写入 stderr）
module list 2> $REPRO_DIR/modules.txt
# 2. 环境
/usr/bin/printenv | sort > $REPRO_DIR/env.txt
# 3. pip（这包括开发安装的 SHA）
pip freeze > $REPRO_DIR/requirements.txt
# 4. 安装到 conda 中的 git 克隆中未提交的差异
perl -nle 'm|"file://(.*?/([^/]+))"| && qx[cd $1; if [ ! -z "\$(git diff)" ]; then git diff > \$REPRO_DIR/$2.diff; fi]' $CONDA_PREFIX/lib/python*/site-packages/*.dist-info/direct_url.json
```

如您所见，此方案在 SLURM 环境中使用，因此每个新的训练都会转储特定于 SLURM 作业的环境。

1. 我们保存加载了哪些 `modules`，例如，在云集群/HPC 设置中，您可能会使用它来加载 CUDA 和 cuDNN 库

   如果您不使用 `modules`，则删除该条目

2. 我们转储环境变量。这可能至关重要，因为像 `LD_PRELOAD` 或 `LD_LIBRARY_PATH` 这样的单个环境变量在某些环境中可能会对性能产生巨大影响

3. 然后我们转储 conda 环境包及其版本 - 这应该适用于任何虚拟 python 环境。

4. 如果您使用 `pip install -e .` 进行开发安装，它除了知道其 git SHA 外，对安装自的 git 克隆仓库一无所知。但问题在于，您很可能在本地修改了文件，现在 `pip freeze` 会遗漏这些更改。因此，这部分将遍历所有未安装到 conda 环境中的包（我们通过查看 `site-packages/*.dist-info/direct_url.json` 来找到它们）

另一个有用的工具是 [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md)，它可以帮助您找出 2 个 conda 环境之间的确切差异。

说个趣闻，我和我的同事在云集群上运行完全相同的代码时，训练 TFLOPs 差异很大 -  буквально 启动了来自同一个共享目录的同一个 slurm 脚本。我们首先使用 [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md) 比较了我们的 conda 环境，发现了一些差异 - 我安装了她所拥有的确切软件包以匹配她的环境，但仍然显示出巨大的性能差异。然后我们比较了 `printenv` 的输出，发现我设置了 `LD_PRELOAD` 而她没有 - 这产生了巨大的差异，因为这个特定的云提供商要求将多个环境变量设置为自定义路径才能充分利用他们的硬件。
