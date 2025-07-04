---
title: 自述文件
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/nfzdbgyi/
---
# 容错性

无论您是拥有 ML 训练硬件还是按小时租用，在这个不断加速的 ML 领域，及时完成训练都非常重要。因此，如果您在睡觉时其中一个 GPU 发生故障，或者检查点存储空间不足导致训练崩溃，那么您醒来后会发现许多训练小时都已丢失。

由于 ML 硬件成本高得令人望而却步，因此很难像在 Web 服务中那样提供冗余故障转移解决方案。尽管如此，只需几个简单的方案即可实现训练的容错性。

由于大多数重要的训练任务都是在 SLURM 环境中执行的，因此会经常提到它，但本章的大部分见解都适用于任何其他训练环境。

## 始终计划拥有比所需更多的节点

GPU 设备的现实是它们容易发生故障。有时它们只是过热并关闭，但可以恢复，而其他时候它们会直接损坏并需要更换。

当您使用相同的节点几周/几个月后，情况会趋于好转，因为坏的节点会逐渐被替换掉，但如果您幸运地获得一批新的 GPU，尤其是当该技术刚推出时的早期 GPU，预计会有相当大比例的 GPU 会发生故障。

因此，如果您需要 64 个节点来进行训练，请确保您有几个备用节点，并研究在备用节点不足时如何快速更换发生故障的节点。

很难预测确切的冗余百分比应该是多少，但 5-10% 应该不会不合理。您越是急于按时完成训练，安全边际就应该越高。

一旦您有了备用节点，请验证您的 SLURM 环境会自动从可用节点池中移除任何有问题的节点，以便它可以用好的节点自动替换坏的节点。

如果您使用非 SLURM 调度程序，请验证它也能够进行无人值守的坏节点替换。

您还需要至少一个额外的节点来运行各种预防性看门狗（本章稍后讨论），可能用于卸载检查点和执行清理工作。



## 将多个训练任务排入队列

下一个关键步骤是确保如果训练崩溃，会有一个新的任务排队接替前一个任务。

因此，在启动训练时，不要使用：
```
sbatch train.slurm
```

您需要将其替换为：

```
sbatch --array=1-10%1 train.slurm
```

这告诉 SLURM 预订一个包含 10 个任务的任务数组，如果其中一个任务正常完成或崩溃，它将立即调度下一个任务。

脚注：`--array=1-10%1` 中的 `%1` 告诉 SLURM 串行启动任务数组 - 一次一个任务。

如果您在没有此准备的情况下已经启动了训练，可以通过使用 `--dependency` 参数轻松修复，而无需中止当前任务：
```
sbatch --array=1-10%1 --dependency=CURRENTLY_RUNNING_JOB_ID train.slurm
```
所以如果您启动的任务看起来像这样：

```
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-63]
```
您会注意到当前的 `JOBID=87`，现在您可以在以下命令中使用它：
```
sbatch --array=1-10%1 --dependency=87 train.slurm
```
然后新的状态将显示为：
```
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-63]
 88_[10%1]   prod    my-training-10b  PENDING       0:00 1-16:00:00   64                    N/A (Dependency)
```
所以您可以看到，一个包含 10 个任务的数组 (`88_[10%1]`) 被附加到当前任务 (`87`) 完成或失败后立即启动。

当然，如果导致崩溃的条件仍然存在，后续的任务也会失败。例如，如果存储设备已满，再多的重启也无法让训练继续进行。我们稍后将讨论如何避免这种情况。

但是，由于训练崩溃的主要原因是 GPU 故障，确保故障节点被自动移除并且新任务从一组新的节点开始，可以实现从崩溃中平稳恢复。

在 SLURM 的术语中，被移除的节点被赋予一个名为 `drained` 的新状态。这是一个假设的 SLURM 集群的示例：

```
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
prod*       up   infinite       4  drain node-[0-3]
prod*       up   infinite      47  alloc node-[4-51]
prod*       up   infinite      23   idle node-[52-73]
```

这里我们有 47 个正在使用的节点 (`alloc`)，23 个可用节点 (`idle`) 和 4 个不可用节点 (`drained`)。

系统管理员应定期检查被排空的节点，修复或更换它们，然后通过将其状态更改为 `idle` 使其再次可用。

另一种方法是通过 `--dependency` 将任务链接起来，如[此处](../../orchestration/slurm/users.md#request-allocation-via-dependency)所述。这两种方法也可以结合使用。

您如何知道何时任务数组或菊花链不应恢复 - 嗯，通常训练循环会在知道任务完成后立即退出。但您也可以添加[终止开关](#kill-switch)等功能，这些功能更容易使用以防止任务数组运行。


## 倾向于固定的加速器分配而非动态分配

通常，当获得一组新的加速器节点时，尤其是当它是最近推出的新型加速器时，许多加速器会发生故障，这使得 LLM 训练变得相当成问题。对于新的加速器，早期可能会有多达 10% 的故障率，在后期阶段仍然有相当高的故障率。请记住，如果您有 8 个加速器，即使只有一个加速器发生故障，从训练程序的角度来看，也相当于所有 8 个都发生了故障。

如果您使用固定的节点分配，几个月后，坏的加速器将被淘汰，发生故障的加速器应该会非常少。它仍然会发生，但这将是一个罕见的事件。

确保您的提供商在加速器发生故障时为您提供新的加速器，而不是在让它们冷却后（字面意思）将相同的加速器返还给您。例如，请参阅如何跟踪 [NVIDIA GPU UUID](../../compute/accelerator/nvidia/debug.md#how-to-detect-if-you-get-the-same-broken-node-again-and-again)。这些瞬时故障在重负载下很可能会重复出现，所以您希望它们被真正更换掉。

如果您使用动态分配，即使在新加速器类型发布一年后，也预计会有大量发生故障的加速器，因为您会收到其他用户拒绝的节点。当然，有些云在认真更换坏硬件方面比其他云做得更好，问题在于有许多加速器不会直接发生故障，当有人放弃一个坏节点时，查看它的技术人员在试用时可能看不到任何问题。如果用户只是释放了节点而没有报告它已损坏，如果云提供商在将节点交给下一个用户之前没有重新检查节点是否正常，那么获得一个坏节点的概率就非常高。



## 频繁保存检查点

每当训练任务失败时，都可能损失许多小时的训练时间。通过频繁保存检查点可以缓解这个问题。当训练恢复时，它将从最后保存的检查点继续。如果在最后一次保存检查点后 12 小时发生故障，那么就会损失 12 小时的训练时间，需要重新进行。如果训练使用数百个 GPU，这可能会非常昂贵。

理论上，可以每 10 分钟保存一个检查点，这样最多只会损失 10 分钟的训练时间，但这也会大大延迟到达终点线的时间，因为大型模型无法快速保存，如果保存时间开始成为训练的瓶颈，这种方法就会变得适得其反。

根据您的检查点方法和 IO 存储分区的速度，保存一个大型模型可能需要几十秒到几分钟。因此，最佳的保存频率方法介于两者之间。

计算很简单 - 测量保存检查点所需的时间，乘以您想要保存的次数，看看检查点保存会给总训练时间增加多少额外的延迟。

用例：在训练 BLOOM-176B 时，我们有一个非常快的基于 NVME 的 GPFS 文件系统，在 384 个进程上并发写入一个 2.3TB 的检查点仅需 40 秒。我们大约每 3 小时保存一个检查点。由于我们训练了大约 3 个月，这意味着我们保存了大约 720 个检查点 (`90 天 * 24 小时 / 3 小时`) - 也就是说，仅仅保存检查点就多花了 8 个小时 (`720 次 * 40 秒 / 3600 秒`) - 或者说占总训练时间的约 0.37% (`8 小时 / (90 天 * 24 小时)`)。现在，假设 IO 速度慢 5 倍，这在云上并不少见，除非有人为高级 IO 付费，那么这将占训练时间的 2%，这将是相当可观的。

脚注：如果您没有大的本地存储，必须将检查点卸载到云端，请确保最近的 2 个检查点保留在本地，以便快速恢复。保留 2 个而不是 1 个的原因是，如果在保存过程中发生崩溃，最后一个检查点可能会损坏或未完成保存。

虽然此方法会给训练带来开销，但拥有训练检查点非常有用。因为它们允许您在出现分歧时回滚多个步骤，可用于分析各种事件，并且如今许多训练都从训练中单一损失测量的评估（提供很少有用信号）转向在训练期间对每个检查点应用多个基准的完整数据集评估。后者可以在不减慢训练速度的情况下在额外的节点上进行训练中评估。


## 基于多副本的容错

还有另一种处理加速器崩溃的方法，它不涉及保存检查点。这种方法仅在训练期间至少使用两个模型副本的情况下才有效。

请先查看各种[模型并行](../model-parallelism)技术，以便能够跟上。

- 如果使用 3D 模型并行的一些变体，即您有张量并行（TP）和/或流水线并行（PP）和/或数据并行（DP），则副本数等于 DP 度。
- 如果使用混合 ZeRO-DP 并行，则副本数等于混合副本的度。

例如，假设您的训练设置使用 TP=4, PP=2, DP=2 的 3D 并行 - 那么您有 2 个副本，每个副本使用 8 个 GPU `node0` 和 `node1`（TP=4, PP=2 => `4*2=8`）- 实际上，每个副本使用一个完整的 8-GPU 节点。

此外，您还有一个备用节点 `node2`，有 8 个 GPU 处于空闲状态，但随时准备使用。

现在，假设在训练期间 `node0.gpu0` 发生故障。由于您有第二个副本具有完整的数据，您可以切换到备用 8GPU 节点，从第二个副本的 GPU RDMA 复制数据，然后您可以从中断的地方继续训练。这是一个非常简化的解释，因为根据故障发生在迭代循环的哪个阶段，确定此类恢复存在多个细微差别。换句话说，需要实现一个复杂的算法。

当然，在大型训练中，您可能会有一百个活动节点和少数几个备用节点。

这种方法优于文件系统检查点保存，因为您最多只损失一次迭代，而使用文件系统检查点保存则会损失数百次迭代。

我不知道这种高级容错方法的任何开源实现，但我们知道一些大公司在内部使用这种方法。





## 终止开关

在许多 SLURM 环境中，用户没有 `sudo` 访问权限，当一个用户开始训练然后去睡觉，然后发现了一个问题，其他用户无法轻易地停止训练并重新启动它。

这就是 BLOOM-176B 训练期间的情况，我们实现了一个终止开关来处理这个问题。机制非常简单。训练循环在开始新的迭代之前轮询一个特定文件是否出现，如果文件存在，程序会保存检查点并退出，允许除启动前一个训练的用户之外的其他用户更改内容并重新启动它。在 `main` 的最开始处添加了另一个轮询，这样如果正在睡觉的用户排队了很长的任务数组，可以通过让每个任务在启动时快速退出，从而快速"消耗"掉它们。

这在[这里](../../orchestration/slurm/users.md#overcoming-the-lack-of-group-slurm-job-ownership)也进行了讨论。

这个设施有助于最大限度地减少浪费的训练时间。

## 保存开关

在提到终止开关时，最好快速提一下它的表亲，一个保存开关。与终止开关类似，保存开关是前者的一个变体，不同之处在于，如果训练循环发现出现了一个保存开关文件，它不会停止训练，而是会保存一个检查点，但会继续训练。它还会自动从文件系统中移除保存开关，这样就不会在每次迭代后意外地开始保存检查点。

对于那些观察训练图表的人来说，这个功能非常有用。如果有人在训练损失或其他训练指标中看到一个有趣或关键的情况，他可以快速要求训练程序保存感兴趣的检查点，并能够在以后随意重现当前情况。

此功能的主要用途是观察训练损失峰值和发散。

（给自己备注：最好放在不稳定性章节）

## 预防

避免浪费训练时间的最简单方法是防止某些类型的问题发生。虽然除了确保提供足够的冷却之外，无法防止 GPU 发生故障，但可以肯定地确保有足够的磁盘空间用于未来几天的训练。这通常是通过运行计划的看门狗来完成的，这些看门狗监视各种资源，并在问题发生前很早就向操作员发出警报。

### 计划的看门狗

在我们讨论各种看门狗之前，至关重要的是您有一个允许您运行计划任务的机制。在 Unix 世界中，这是通过 [`crontab` 工具](https://en.wikipedia.org/wiki/Cron) 实现的。

这是一个如何每小时启动 `~/bin/watch-fs.sh` 的示例：
```
0 * * * * ~/bin/watch-fs.sh
```
上面的链接解释了如何配置一个 crontab 任务以各种其他频率运行。

要设置一个 crontab，执行 `crontab -e` 并检查哪些任务已计划 `crontab -l`。

我不详细介绍的原因是许多 SLURM 环境不提供对 `crontab` 工具的访问。因此，需要使用其他方法来调度任务。

关于 [Crontab 仿真](../../orchestration/slurm/users.md#crontab-emulation) 的部分讨论了如何实现类似 crontab 的 SLURM 仿真，以及[自我延续的 SLURM 任务](../../orchestration/slurm/users.md#self-perpetuating-slurm-jobs)。


### 通知设施

然后您需要一个或多个通知设施。

最简单的是使用电子邮件发送警报。要使其工作，您需要确保有办法从 SLURM 任务发送电子邮件。如果尚不可用，您可以向您的系统管理员请求此功能，或者您可以使用外部 SMTP 服务器提供商。

除了电子邮件，您可能还可以设置其他通知，例如短信警报和/或如果您使用 Slack，则可以向您选择的频道发送 slack 通知。

一旦您了解了如何调度看门狗并且您有一个正常工作的通知设施，接下来让我们讨论关键的看门狗。

### 任务是否正在运行的看门狗

最明显的看门狗是检查是否有正在运行的训练 SLURM 任务或更多已计划运行的任务。

这是一个在 BLOOM-176B 训练期间使用的 [slurm-status.py](slurm-status.py) 示例。如果检测到任务既没有运行也没有计划，这个看门狗会发送一封电子邮件，并且它还会将其检查结果通过管道传输到主训练的日志文件中。由于我们使用了[Crontab 仿真](../../orchestration/slurm/users.md#crontab-emulation)，我们只需将 [slurm-status.slurm](slurm-status.slurm) 放入 `cron/cron.hourly/` 文件夹中，先前启动的 SLURM crontab 仿真调度程序就会大约每小时启动一次此检查。

SLURM 任务的关键部分是：
```
tools/slurm-status.py --job-name $WATCH_SLURM_NAME 2>&1 | tee -a $MAIN_LOG_FILE
```
它告诉脚本要监视哪个任务名称，您还可以看到它会记录到一个日志文件中。

例如，如果您使用以下命令启动了脚本：
```
tools/slurm-status.py --job-name my-training-10b
```
并且当前状态报告显示：
```
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
  JOBID    PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
    87     prod      my-training-10b  RUNNING 2-15:52:19 1-16:00:00  64    2023-10-07T01:26:28 node-[1-63]
```
那么一切正常。但如果 `my-training-10b` 任务没有显示，就会发送警报。

您现在可以通过最少的更改（编辑路径和电子邮件地址）来根据您的需求调整这些脚本。如果不是您启动的任务，那么用启动它的用户的名字替换 `whoami`。`whoami` 只有在是您自己启动的情况下才有效。


### 任务是否挂起的看门狗

如果应用程序正在执行 `torch.distributed` 或类似操作，并且在其中一个集合操作期间发生挂起，它最终会超时并抛出异常，这将重新启动训练，并且可以发送一个警报，通知任务已重新启动。

但是，如果挂起发生在另一个可能没有超时的系统调用期间，例如从磁盘读取，应用程序很容易在那里挂起数小时而无人知晓。

大多数应用程序都会定期记录日志，例如，大多数训练每隔几分钟就会记录最后 N 步的统计信息。然后可以检查日志文件是否在预期的时间范围内更新 - 如果没有 - 则发送警报。您可以自己编写，也可以使用 [io-watchdog](https://github.com/grondo/io-watchdog)。



### 低磁盘空间警报

下一个最大的问题是磁盘空间不足。如果您的检查点很大并且保存频繁，而且没有卸载到别处，那么很容易很快就用完磁盘空间。此外，通常多个团队成员共享同一个集群，您的同事可能会很快消耗大量磁盘空间。理想情况下，您应该有一个专门用于您训练的存储分区，但这通常很难实现。无论如何，您需要知道磁盘空间何时不足，以及需要采取什么措施来腾出空间。

现在，触发警报的阈值应该是什么。它们不能太早发出，因为如果您在比如 50% 的使用率时开始发送这些警报，用户会开始忽略它们。但百分比也不总是适用的，因为如果您有一个与他人共享的巨大磁盘空间，该磁盘空间的 5% 可能意味着许多 TB 的可用磁盘空间。但在一个小分区上，即使 25% 也可能只有几 TB。因此，您确实应该知道您写检查点的频率，每天需要多少 TB 的磁盘空间，以及有多少可用磁盘空间。

用例：在 BLOOM 训练期间，我们每 3 小时写入一个 2.3TB 的检查点，因此我们每天消耗 2.6TB！

此外，通常会有多个分区 - 用于写入检查点的更快的 IO 分区，以及用于代码和库的更慢的分区，可能还有正在使用的各种其他分区，如果它们的可用性对训练不崩溃是必需的，那么所有这些都需要被监控。

这里还有另一个需要注意的地方 - 当涉及到分布式文件系统时，并非所有文件系统都能可靠地为您提供 100% 的所购磁盘空间。事实上，对于某些类型的文件系统，您最多只能可靠地使用所分配存储空间的约 80%。问题在于这些系统使用物理磁盘，它们在计划的周期或触发的事件中重新平衡，因此任何这些单独的磁盘都可能达到其容量的 100% 并导致写入失败，这将使训练过程崩溃，即使 `df` 报告分区上只有 80% 的空间使用率。我们在训练 BLOOM-176B 时没有遇到这个问题，但在训练 IDEFICS-80B 时遇到了这个问题 - 在那里，80% 就是新的 100%。您如何知道是否遇到此问题 - 嗯，通常您在为训练做准备时会发现它。

这还不是全部。还有一个 inode 可用性的问题，一些存储分区没有非常大的 inode 配额。Python 包以拥有成百上千个小文件而臭名昭著，这些文件加起来总空间很小，但在一个人的虚拟环境中加起来就是数万个文件，突然之间，虽然一个人有 TB 级的可用磁盘空间，但却用完了免费的 inode，发现他们的训练崩溃了。

最后，许多分布式分区不会实时显示您的磁盘使用情况统计信息，可能需要长达 30 分钟才能更新。

脚注：使用 `df -ih` 查看 inode 配额和当前使用情况。

脚注：一些文件系统使用内部压缩，因此报告的磁盘使用量如果复制到别处可能会小于实际情况，这可能会令人困惑。

所以这里是 [fs-watchdog.py](./fs-watchdog.py)，它在 BLOOM-176B 训练期间使用。如果任何存储要求阈值未得到满足，这个看门狗会发送一封电子邮件，这里是相应的 [fs-watchdog.slurm](./fs-watchdog.slurm) 来驱动它。

如果您研究看门狗代码，您会发现对于每个分区，我们都在监控磁盘使用情况和 inode。我们使用 HPC 提供的特殊配额工具来获取某些分区的即时统计信息，但这些工具并非对所有分区都有效，因此我们不得不回退到使用 `df` 甚至更慢的 `du`。因此，应该很容易根据您的用例进行调整。


### 处理慢速内存泄漏

有些程序会出现微小的内存泄漏，这可能很难调试。不要将这些与 MMAP 的使用混淆，在 MMAP 中，程序使用 CPU 内存来快速读取数据，并且内存使用量可能会随着时间的推移而增加，但这并不是真的，因为当需要时这些内存会被释放。您可以阅读[深入调查 MMAP 不会泄漏内存](https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/)来理解为什么。

当然，理想情况下，人们会分析他们的软件并修复泄漏，但有时泄漏可能来自第三方包，或者很难诊断，而且通常没有时间去做。

当涉及到 GPU 内存时，可能存在内存碎片化的问题，随着时间的推移，越来越多的小的未使用的内存段加起来，使得 GPU 看起来有大量的可用内存，但是当程序试图从这些内存中分配一个大的张量时，它会失败并出现 OOM 错误，例如：

```
RuntimeError: CUDA out of memory. Tried to allocate 304.00 MiB (GPU 0; 8.00 GiB total capacity;
142.76 MiB already allocated; 6.32 GiB free; 158.00 MiB reserved in total by PyTorch)
```
在这个例子中，如果有 6.32GB 的可用空间，为什么 304MB 无法分配呢？

我的团队在 IDEFICS-80B 训练期间开发的一种方法是，在训练循环中安装一个看门狗，检查内存使用情况，如果达到阈值，它会主动退出训练循环。当时有一些微小的 CPU 内存泄漏，通常需要几天时间才会导致 CPU 内存耗尽。然后，下一个训练任务将在所有 CPU 内存被回收后恢复。

脚注：机器学习训练的现实是，并非所有问题都能用有限的资源解决，通常一个可靠的变通办法能更快地到达终点，而不是"停止印刷机"，并可能因为试图找出问题所在而将训练延迟数周。例如，我们用 `CUDA_LAUNCH_BLOCKING=1` 训练了 BLOOM-176B，因为没有它训练就会挂起，在多次诊断失败后，我们再也等不起了，只能按原样进行。幸运的是，这个通常用于调试目的的环境变量，理论上应该会使一些 CUDA 操作变慢，但实际上对我们的吞吐量没有任何影响。但我们从未找出问题所在，今天我们是否找出问题也无关紧要了，因为我们已经转向了其他不受该问题影响的项目。

这个想法类似于前面讨论的终止和保存开关，但这里我们不是轮询特定文件的出现，而是简单地观察使用了多少驻留内存。例如，如果操作系统显示只剩下 5% 的虚拟 CPU 内存，您可以这样自动退出：

```
import psutil
for batch in iterator:
    total_used_percent = psutil.virtual_memory().percent
    if total_used_percent > 0.95:
        print(f"由于 cpu 内存几乎已满，提前退出：({total_used_percent}%)")
        save_checkpoint()
        sys.exit()

    train_step(batch)
```

类似的启发式方法可以用于为 GPU 内存使用设置阈值，只是需要注意 cuda 张量缓存和 python 垃圾回收调度，因此要获取实际的内存使用情况，您需要首先运行垃圾回收器，然后清空 cuda 缓存，只有这样才能获得真实的内存使用情况统计信息，然后在 GPU 太接近满时优雅地退出训练。

```
import gc
import torch

for batch in iterator:
    gc.collect()
    torch.cuda.empty_cache()

    # 以 GB 为单位获取内存使用情况，如果剩余的可用 GPU 内存少于 2GB 则退出
    free, total = map(lambda x: x/2**30, torch.cuda.mem_get_info());
    if free < 2:
        print(f"由于 GPU 内存几乎已满，提前退出：({free}GB 剩余)")
        save_checkpoint()
        sys.exit()

    train_step(batch)
```

脚注：除非您真的必须这样做，否则不要这样做，因为缓存可以加快速度。理想情况下，应该找出碎片化问题。例如，在 [`PYTORCH_CUDA_ALLOC_CONF`](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) 的文档中查找 `max_split_size_mb`，因为它控制着内存的分配方式。像 [Deepspeed](https://github.com/deepspeedai/DeepSpeed) 这样的框架通过在启动时预分配张量，然后一次又一次地重用它们，从而完全避免了碎片化问题。

脚注：这个简化的例子适用于单个节点。对于多个节点，您需要从所有参与的节点收集统计信息，找到剩余内存最少的节点，并据此采取行动。


## 处理强制任务抢占

前面您已经看到了如何使用[终止开关解决方案](#kill-switch)来优雅地停止训练，当您需要按需停止或暂停训练时，这很有用。

在 HPC 集群上，SLURM 任务有最大运行时间。典型的是 20 小时。这是因为在 HPC 上，资源是多个用户/组共享的，因此每个用户/组都被分配一个时间片来进行计算，然后任务被强制停止，以便其他任务可以使用共享资源。

脚注：这也意味着您无法计划训练需要多长时间，除非您的任务在集群上以最高优先级运行。如果您的优先级不是最高的，那么等待数小时甚至数天才能恢复您的任务并不少见。

当然，可以让任务被杀死，并希望自从[上次保存检查点](#frequent-checkpoint-saving)以来没有花费太多周期，然后让任务从这个检查点恢复，但这相当浪费，最好避免。

有效的解决方案是在硬性时间限制到来并且任务被 SLURM 杀死之前优雅地退出。

首先，您需要弄清楚您的程序需要多少时间来优雅地完成。这通常需要 2 个持续时间：

1. 如果您刚刚开始一个新的迭代，完成单个迭代需要多长时间
2. 保存检查点需要多长时间

例如，如果迭代最多需要 2 分钟，保存检查点需要另外 2 分钟，那么您至少需要 4 分钟的宽限时间。为了安全起见，我至少会加倍。提前一点退出没有坏处，因为没有浪费资源。

所以，例如，假设您的 HPC 允许 100 小时的任务，那么您的 slurm 脚本将这样写：
```
#SBATCH --time=100:00:00
```

### 方法 A. 在启动时告诉程序何时应该开始退出过程：
```
srun ... torchrun ... --exit-duration-in-mins 5990
```
100 小时是 6000 分钟，所以这里我们给程序 10 分钟来优雅地退出。

当您启动程序时，您会创建一个计时器，然后在每个新迭代开始之前，您会检查是否达到了时间限制。如果达到了，您就保存检查点并退出。

案例研究：您可以看到这是如何[在 BLOOM 训练任务中设置的](https://github.com/bigscience-workshop/bigscience/blob/58d99c67f643d27b5765a73a2ee2d1ce0a4b2c6b/train/tr11-176B-ml/tr11-176B-ml.slurm#L97-L100)，然后[在这里](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/training.py#L985-L998)执行的：

```
        # 基于持续时间的退出
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             lr_scheduler)
                print_datetime('在 {} 分钟后退出程序'.format(train_time))
                sys.exit()
```

如您所见，由于训练是分布式的，我们必须在所有 rank 之间同步退出事件

您还可以通过检索正在运行的任务的 `EndTime` 来自动化派生：
```
$ scontrol show -d job $SLURM_JOB_ID | grep Time
   RunTime=00:00:42 TimeLimit=00:11:00 TimeMin=N/A
   SubmitTime=2023-10-26T15:18:01 EligibleTime=2023-10-26T15:18:01
   AccrueTime=2023-10-26T15:18:01
   StartTime=2023-10-26T15:18:01 EndTime=2023-10-26T15:18:43 Deadline=N/A
```
然后在程序中与当前时间进行比较，而不是设置优雅的退出周期。从输出中可以看出，还可以检索其他时间戳和持续时间。

### 方法 B.1. 在结束前 X 分钟发送自定义信号

在您的 sbatch 脚本中，您可以设置：

```
#SBATCH --signal=USR1@600
```
然后 SLURM 将在任务结束时间前 10 分钟向您的程序发送一个 `SIGUSR1` 信号。

脚注：通常 SLURM 调度程序在任务时间结束前约 30-60 秒发送一个 `SIGCONT`+`SIGTERM` 信号，并且就在时间结束时，如果任务仍在运行，它将发送一个 `SIGCONT`+`SIGTERM`+`SIGKILL` 信号。`SIGTERM` 可以被捕获并采取行动，但 30 秒不足以让一个大型模型训练程序优雅地退出。

让我们演示一下信号发送和捕获是如何工作的。在终端 A 中，运行：
```
python -c "
import time, os, signal

def sighandler(signum, frame):
    print('信号处理程序被调用，信号为', signum)
    exit(0)

signal.signal(signal.SIGUSR1, sighandler)
print(os.getpid())
time.sleep(1000)
"
```
它将打印进程的 pid，例如 `4034989`，然后进入睡眠状态（模拟实际工作）。现在在终端 B 中向终端 A 中的 python 程序发送 `SIGUSR1` 信号：

```
kill -s USR1 4034989
```

程序将捕获此信号，调用 `sighandler`，该 `sighandler` 现在将打印它已被调用并退出。

```
信号处理程序被调用，信号为 10
```
`10` 是 `SIGUSR1` 的数值。

所以这里是使用 SLURM 设置的同样的事情：

```
$ cat sigusr1.slurm
#SBATCH --job-name=sigusr1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:03:00
#SBATCH --partition=mypartition
#SBATCH --output=%x-%j.out
#SBATCH --signal=USR1@170

srun python -c "
import time, os, signal

def sighandler(signum, frame):
    print('信号处理程序被调用，信号为', signum)
    exit(0)

signal.signal(signal.SIGUSR1, sighandler)
print(os.getpid())
time.sleep(1000)
"
```
在 SLURM 脚本中，我们告诉 SLURM 在任务结束前 170 秒向程序发送一个信号，而任务本身被设置为运行 180 秒（3 分钟）。

当这个任务被调度时：
```
sbatch sigusr1.slurm
```
任务启动后 10 秒 (`180-170`)，它将退出并记录日志：

```
58307
信号处理程序被调用，信号为 10
```

这意味着任务的 pid 是 `58307`，它捕获了 `SIGUSR1` (`10`) 并退出了。

现在您了解了这种机制的工作原理，您可以设置一个 exit-asap 标志，完成当前运行的迭代，检查该标志是否已设置，保存检查点并退出，而不是立即 `exit(0)`。这与上面方法 A 中显示的代码非常相似。


### 方法 B.2. 选择向哪个进程发送信号

现在，如果您的主程序不是用 `srun` 启动的那个，该怎么办 - 如果您使用像 `torchrun` 或 `accelerate` 这样的中间启动器，上面的方法将不起作用，因为 `SIGUSR1` 很可能不会从启动器传播到其子进程。在这种情况下，我们需要一个比

我们必须替换：
```
#SBATCH --signal=USR1@600
```
为：
```
#SBATCH --signal=B:USR1@600
```

添加的 `B:` 告诉 SLURM 不要向 `srun` 进程（启动器）发送信号，而是向 `sbatch` shell 发送。

现在我们必须将 SLURM 脚本的结尾从典型的基于启动器的代码更改为：

```
CMD="python -u -m torch.distributed.run ... train.py ..." # 实际命令在这里
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE

```
到这个：
```
trap 'echo "SIGUSR1 received!"; \
pid=$(pgrep -f "^python.*(accelerate|deepspeed|torchrun|distributed.run)"); \
pgrep -P $pid | xargs -r kill -USR1; \
wait;' SIGUSR1

CMD="python -u -m torch.distributed.run ... train.py ..." # 实际命令在这里
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE &

wait
```

由于之前的 `--signal=B:USR1@600` 现在会向 `sbatch` shell 发送信号，我们可以捕获它并做一些事情，这就是 `trap` 行的作用。

传递给 `trap` 的信号处理程序中的神奇代码会找到作为 `accelerate`、`deepspeed`、`torchrun` 或 `torch.distributed.run` 等任何启动器的直接子进程的所有进程，并向它们发送 `SIGUSR1` 信号。

最后，最后一个更改是，为了让 `trap` 工作，我们需要在后台运行 `srun` - 所以我们在 `srun` 命令的末尾添加了 `&`，并且我们需要添加 `wait`，这样 `sbatch` shell 就不会在 `srun` 完成之前退出。

您的捕获信号处理程序的 python 代码与方法 B.1 中保持相同。

以下是 SLURM 脚本的重要部分：

```
$ cat launch.slurm
#!/bin/bash
[...]
#SBATCH --partition=dev
#SBATCH --signal=B:USR1 # 自定义抢占信号
[...]

trap 'echo "SIGUSR1 received!"; \
pid=$(pgrep -f "^python.*(accelerate|torchrun|deepspeed|distributed.run)"); \
pgrep -P $pid | xargs -r kill -USR1; wait;' SIGUSR1

CMD="python -u -m torch.distributed.run ... train.py ..." # 实际命令在这里
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE &

wait
```

您最初的训练循环可能看起来像这样：
```
$ cat train.py

for batch in dl:
    train_iteration(batch)
```

现在它会变成：
```
$ cat train.py

import signal
import sys

pre_emption_activated = False
def activate_pre_emption(sig, frame):
    global pre_emption_activated
    print("SIGUSR1 received, saving checkpoint")
    pre_emption_activated = True

signal.signal(signal.SIGUSR1, activate_pre_emption)

for batch in dl:
    train_iteration(batch)

    if pre_emption_activated:
        save_checkpoint()
        sys.exit()
```

当然，您可能会在实际软件中在训练器对象中设置一个标志，而不是使用 `global`，但为了简短演示，这已经足够了。

如果您想测试此解决方案，只需将您的 SLURM 脚本头更改为：

```
#SBATCH --time=0:05:00
#SBATCH --signal=B:USR1@60
```

这里我们告诉 SLURM 只运行任务 5 分钟 (`--time=0:05:00`)，并要求它在 5 分钟到期前 `60` 秒向我们的 `sbatch` 脚本发送 `SIGUSR1`，即任务启动后 4 分钟。



### 基于 QoS 的 SLURM 抢占

到目前为止，我们还没有讨论当使用服务质量（QoS）时会发生什么，这也可能强制抢占现有任务。该功能与任务分配时间即将结束的抢占类型相同，只是它可以随时发生，而不是在任务结束前 X 秒。

考虑一个 SLURM 设置，其中您有 `--qos=high`，可以抢占 `--qos=low` 的任务，并且低优先级任务有 10 分钟的宽限时间来关闭：

```
$ sacctmgr show qos format=name,priority,preempt,MaxTRESPerUser,GraceTime,Preempt,Flags
      Name   Priority     MaxTRESPU  GraceTime    Preempt                Flags
---------- ---------- ------------- ---------- ---------- --------------------
       low          0                 00:10:00
      high          0                 00:00:00        low
```

这与基于时间的抢占非常相似，只是这里的宽限时间是硬编码的，用户无法修改。

如果一个任务以 `--qos=high` 启动，并且没有足够的节点，SLURM 将踢出一些低优先级的任务，以便为高优先级的任务提供节点。

默认情况下，`GraceTime` 可能非常短，不足以让您的程序在被抢占时安全地结束 - 在这种情况下，请您的系统管理员将其持续时间提高到能满足您需求的程度。

否则，方法 B.1 和 B.2 中描述的相同解决方案将适用于这种类型的强制抢占。
