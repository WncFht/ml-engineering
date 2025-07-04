---
title: 存储
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/ma83jz65/
---
# 存储：文件系统和 IO

## 3 种机器学习的 IO 需求

在机器学习工作负载中，有 3 种不同的 IO 需求：

1.  你需要能够快速地为 DataLoader 提供数据 - (超快的读取速度，不关心写入速度) - 需要持续数小时乃至数天的可持续负载。
2.  你需要能够快速地写入检查点 - (超快的写入速度，读取速度也要比较快，因为你可能会恢复几次) - 需要突发写入能力 - 你希望速度超快，以免长时间阻塞训练 (除非你使用某种 CPU 卸载来快速解除训练阻塞)。
3.  你需要能够加载和维护你的代码库 - (中等的读写速度) - 这也需要共享，因为你希望所有节点都看到相同的代码库 - 由于这只在启动或恢复时发生，所以频率不高。

如你所见，这 3 种需求在速度和可持续负载方面有非常不同的要求，因此理想情况下，你应该有 3 种不同的文件系统，每种都为所需的使用场景进行优化。

如果你有无限的资金，当然可以购买一个单一的超快读、超快写、并且可以连续几天不间断工作的文件系统。但对我们大多数人来说，这是不可能的，所以选择 2 到 3 种不同类型的分区，最终花费更少，是更明智的选择。




## 术语表

- NAS: 网络附加存储 (Network Attached Storage)
- SAN: 存储区域网络 (Storage Area Network)
- DAS: 直连式存储 (Direct-Attached storage)
- NSD: 网络共享磁盘 (Network Shared Disk)
- OSS: 对象存储服务器 (Object storage server)
- MDS: 元数据服务器 (Metadata server)
- MGS: 管理服务器 (Management server)



## 选择哪个文件系统

**分布式并行文件系统是最快的解决方案**

分布式并行文件系统在数百到数千个客户端可以同时访问共享存储的情况下，显著提高了性能。它们还有助于减少热点 (即某些数据区域的访问频率远高于其他区域)。

我使用过的 3 种性能卓越的并行文件系统是：

- [GPFS](https://en.wikipedia.org/wiki/GPFS) (IBM)，最近更名为 IBM Storage Scale，之前叫做 IBM Spectrum Scale。
- [WekaIO](https://www.weka.io/)
- [Lustre FS](https://www.lustre.org/) (开源) ([维基](https://wiki.lustre.org/Main_Page))

这些解决方案已经存在了 20 多年，并且符合 POSIX 标准。它们也不是轻易就能创建的——你必须建立一个完全独立的集群，其中包含多个专用于这些文件系统的纯 CPU 虚拟机，然后才能挂载它们。相比之下，较弱的云提供商"内置"解决方案只需回答几个屏幕的问题即可激活。在创建存储集群时，选择哪种虚拟机用于哪种功能是一门完整的科学。例如，这里有一份 [GCP 上的 Lustre 指南](https://cloud.google.com/architecture/parallel-file-systems-for-hpc#overview_of_lustre_and_exascaler_cloud)。

案例研究：2021 年在 JeanZay HPC (法国)，我们能在 384 个进程上并行保存一个 2.3TB 的检查点，耗时仅 40 秒！这快得令人难以置信——而且是在 NVME 硬盘上运行的 GPFS。

NASA 的集群在使用 Lustre 时有[一长串的注意事项](https://www.nas.nasa.gov/hecc/support/kb/lustre-best-practices_226.html)。

GPFS 的一些非常有用的优点：
- 如果你有大量的小文件，你很容易用完 inode (`df -i` 来检查)。GPFS 5.x 永远不会用完 inode，它会根据需要动态创建更多。
- GPFS 没有 Lustre 的问题，即如果其中一个子磁盘满了并且没有及时重新平衡，你可能会在 80% 的时候就用完磁盘空间——你可以可靠地使用所有 100% 的已分配存储。
- GPFS 不使用中央元数据服务器 (或其集群)，这在处理小文件时通常会成为瓶颈。就像数据一样，元数据由存储集群中的每个节点处理。
- GPFS 自带一个原生的 NSD 客户端，它优于通用的 NFS 客户端，但两者都可以与它一起使用。
- 可以构建一个多层系统。例如，第 1 层通常由 NVME 驱动器组成，第 2 层通常使用一些云存储系统。因此，当第 1 层的容量变低时，一段时间未被访问的文件会自动移动到云存储。例如，你的第 1 层可能是 100TB，第 2 层可能是 1PB。这种方法可以节省大量资金，因为 1PB 的云存储比 1PB 的 NVME 驱动器便宜得多。
- 数据保护可以使用各种 RAID 方法。通常使用条带化来节省成本。

Weka 在功能和性能上与 GPFS 非常相似。主要区别在于你可以与任一提供商协商的许可成本。你的大部分成本将在于运行系统所需的虚拟机成本——例如，如果你有大量小文件，你将需要许多虚拟机来快速处理元数据。

我还没有直接经验的其他并行文件系统：

- [BeeGFS](https://www.beegfs.io/)
- [DAOS](https://docs.daos.io/) (分布式异步对象存储) (英特尔)
- [NetApp](https://www.netapp.com)
- [VAST](https://www.vastdata.com/)

大多数云服务提供商至少提供其中一种实现，但并非全部。如果你的云服务提供商不提供至少其中一种，并且他们没有足够快的替代方案来满足你的需求，你应该重新考虑。

**尚可的解决方案**

[各种云提供商](#cloud-shared-storage-solutions)提供了许多尚可的解决方案。在承诺使用任何方案之前，请认真对其进行基准测试。这些方案通常在处理大文件方面表现不错，但在处理小文件方面则不尽如人意。

案例研究：在撰写本文时，使用 GCP 的 Zonal FileStore over NFS 解决方案，`python -c "import torch"` 的执行时间长达 20 秒，这非常慢！一旦文件被缓存，它大约需要 2 秒。安装一个包含少量预构建 python 包的 conda 环境，很容易就需要 20-30 分钟！我们最初使用的这个解决方案非常痛苦，并且对我们的工作效率造成了负面影响。这会影响到任何拥有大量 python 包和 conda 环境的人。但是，当然，GCP 也提供了快得多的解决方案。

## 远程文件系统客户端

你需要选择使用哪个客户端将文件系统连接到你的虚拟机。

最常见的选择是：[NFS](https://en.wikipedia.org/wiki/Network_File_System) - 它已经存在了 40 年。它会引入额外的开销并降低速度。因此，如果你的虚拟机支持原生客户端，使用它而不是 NFS 会获得更快的整体性能。例如，GPFS 附带一个 [NSD](https://www.ibm.com/docs/en/linux-on-systems?topic=configurations-network-shared-disk-nsd) 客户端，它优于 NFS。

## 文件块大小

如果你使用的文件系统块大小为 16mb，但文件的平均大小为 16k，那么你将比实际使用多占用 1000 倍的磁盘空间。例如，当实际磁盘空间仅为 100MB 时，你将看到使用了 100TB 的磁盘空间。

脚注：在 Linux 上，原生文件系统通常使用 4k 的块大小。

所以，你通常可能会有两种截然不同的需求，需要两个为不同需求优化的不同分区。

1.  成千上万个小文件 - 4-8k 块大小
2.  少量大文件 - 2-16mb 块大小

案例研究：Python 在处理成千上万个小文件方面表现很差，以至于如果你有很多 conda 环境，在某些情况下很可能会用尽 inode。在 JeanZay HPC，我们不得不请求一个特殊的专用分区来安装所有 conda 环境，因为我们在普通的 GPFS 分区上总是用尽 inode。我想问题在于那些 GPFS 分区配置了 16MB 的块大小，所以这不适合 4KB 大小的文件。

好消息是，现代解决方案开始引入动态块大小。例如，最新的 GPFS 支持子块。因此，例如，可以将 GPFS 配置为 2mb 的块大小，子块为 8k，然后小文件会被打包成子块，从而不会浪费太多磁盘空间。

## 分布式存储服务器与客户端的邻近性

使用共享分布式存储的集群应该将存储服务器放置在靠近使用这些服务器的集群的地方。如果运行存储服务器的虚拟机距离很远（经过许多交换机），IO 延迟可能会很高，并且存储的交互式使用可能会令人沮丧地缓慢。例如，当你尝试运行 `du` 和其他访问许多文件元数据的工具时，与元数据服务器的任何交互都会很慢。

所以，如果你有控制权，请要求云服务提供商给你分配的纯 CPU 存储服务器虚拟机，在网络距离上尽可能靠近你的加速器虚拟机。



## 云共享存储解决方案

以下是各云服务提供商提供的共享文件系统存储解决方案：

- [GCP](https://cloud.google.com/architecture/filers-on-compute-engine)
- [Azure](https://learn.microsoft.com/en-us/azure/virtual-machines/disks-shared)
- [AWS](https://aws.amazon.com/what-is/nas/#seo-faq-pairs#how-can-aws-help-with-storage-solutions)


## 本地存储优于云存储

虽然云存储更便宜，但在训练时动态获取和处理训练数据流的想法问题重重，存在大量潜在问题。

动态将检查点卸载到云端也是如此。

拥有足够的本地磁盘空间用于数据加载要好得多。

对于检查点，应该有足够的本地磁盘空间来快速可靠地保存检查点，然后通过 crontab 作业或 slurm 作业将其卸载到云端。始终在本地保留最近的几个检查点，以便在作业崩溃时快速恢复，因为等待从云端获取检查点进行恢复会非常耗时。

案例研究：我们别无选择，在 IDEFICS-80B 训练期间不得不使用云存储进行数据加载，因为我们几乎没有本地存储，而且由于是多模态数据，数据量高达数 TB。我们花了好几个星期试图使这个解决方案变得健壮，但最终效果很差。最大的问题是当时很难跟踪 DataSampler 的 RNG 状态，因为我们使用的解决方案，嗯，根本没考虑这个问题。所以，花费大量时间创建的大量数据被浪费了（没有使用），而且很多数据被重复了，所以我们没有一个单一 epoch 的唯一数据。

在某些情况下，人们找到了使用基于云的数据集的好方法，我个人还没有顺利的体验，这就是我提倡本地存储的原因。如果你找到了一个能够正确恢复而不会丢失数据和重复相同数据，并且不需要庞大本地工作节点的良好流式解决方案，那么它可能会运行得很好。

## 当心你买的存储实际上只有 80% 可用

在计算节点上使用的分布式共享存储存在一个微妙的问题。由于用于构建大型文件系统的大多数物理磁盘只有 0.3-2TB 大，任何一个物理磁盘都可能在组合存储满之前就满了。因此，它们需要不断地重新平衡，以避免出现一个磁盘 99% 满而其他磁盘只有 50% 满的情况。由于重新平衡是一项耗时的操作，就像大多数编程语言的垃圾回收一样，它不经常发生。因此，如果你运行 `df` 并看到 90% 满，任何程序很可能随时都会失败。

根据与 IO 工程师的交谈，一个公认的现实（但不知为何没有传达给客户）是，只有大约 80% 的分布式大容量存储是可靠的。

这意味着，如果你想拥有 100TB 的可靠云存储，你实际上需要购买 125TB 的存储，因为其中的 80% 是 100TB。所以你需要计划比你为实际需求配置的预算多支付 25%。我不确定为什么客户应该为技术缺陷买单，但事实就是如此。

例如，GCP 指出只有 [89%](https://cloud.google.com/filestore/docs/known-issues#capacity_errors_before_reaching_full_provisioned_capacity) 可以可靠使用，尽管对我来说，存储不止一次在 83% 的时候就失败了。值得称赞的是，谷歌甚至将此作为已知问题披露，尽管不是在用户购买存储的时候。就好比——我们建议你购买比你实际计划使用的多 12% 的存储，因为我们只能可靠地交付其中的 89%。

我还与提供托管 IBM Storage Scale (GPFS) 解决方案的 [Sycomp](https://sycomp.com/) 工程师交谈过，据他们说，GPFS 没有这个问题，整个 100% 都可以可靠使用。

此外，在某些设置中，如果你通过云提供商 API（而不是直接在文件系统上）进行备份，它们最终可能会使用相同的分区，当然也会消耗磁盘空间，但是当你运行 `df` 时，它不会显示真实的磁盘使用情况——它可能显示不包括备份的使用情况。所以，如果你的备份消耗了 50% 的分区。

无论你选择哪种存储解决方案，请询问提供商有多少存储可以可靠使用，以免日后出现意外。


## 当心在某些云提供商上，备份会使用它们所备份的同一个分区

这对我来说毫无意义，但在某些提供商那里，当你使用他们的工具备份一个分区时，备份会使用该分区的空间。而在其中一些提供商那里，你甚至不知道发生了什么，直到你真的只使用了分配分区的 30% 就用完了磁盘空间。在那些提供商那里，运行 `df` 毫无意义，因为它会告诉你可用磁盘空间，但不会包含任何备份。所以你根本不知道发生了什么。

如果你开始备份，然后突然一切都失败了，因为所有进程都无法写入，但 `df` 报告使用了 30%，现在你就知道为什么会这样了。快照也使用同一个分区。

比如说，你为一个 100TB 的分区付费，你使用了 95TB，现在你想备份它——嗯，你不能——即使它压缩了，如果只剩下 5TB 的数据，它能把 95TB 的数据放在哪里呢？

当我发现具有这种不直观行为的具体解决方案时，我将添加指向如何查看实际磁盘使用情况的指针：
- [GCP FileStore](https://cloud.google.com/filestore/docs/monitoring-instances#free-raw-capacity-percent) (但它不适用于 Basic Tier)


## 别忘了校验和

当您将数据同步到云端或从云端同步数据时，请务必研究您使用的工具是否检查校验和，否则您最终可能会得到在传输过程中损坏的数据。有些工具会自动执行此操作，而其他工具则需要您启用此功能（因为它通常会带来额外的计算成本和传输速度减慢）。慢一点但安全总比快一点好。

这些通常是 MD5 和 SHA256 校验和。如果您的环境安全，通常 MD5 就足够了，但如果您需要额外的安全性，请使用 SHA256 校验和。



## 概念

这里有几个你可能需要熟悉的关键存储相关概念：

### 队列深度

**队列深度**（或 **IO 深度**）是存储设备控制器一次可以排队的 IO 请求数量。如果发送的 IO 请求超过控制器可以排队的数量，操作系统通常会将其放入自己的队列中。

在 Linux 上，本地块设备的队列深度通常由内核预先配置。例如，如果你想检查为 `/dev/sda` 设置的最大队列深度，你可以 `cat /sys/block/sda/queue/nr_requests`。要查看本地设备的当前队列深度，请运行 `iostat -x` 并注意 `aqu-sz` 列。（`apt install sysstat` 来获取 `iostat`。）

通常，缓冲的 IO 请求越多，延迟就越大，吞吐量也越好。这是因为如果一个请求不能立即被处理，它将延长响应时间，因为它必须等待才能被服务。但是，让多个请求在设备队列中等待服务通常会加快总吞吐量，因为发出单个请求之间的等待时间更少。

### 直接 IO vs 缓冲 IO

**直接** IO 指的是绕过操作系统缓存缓冲区的 IO。这对应于 `open(2)` 系统调用中的 `O_DIRECT` 标志。

与之相反的是**缓冲** IO，这通常是大多数应用程序进行 IO 的默认方式，因为缓存通常会使事情更快。

当我们运行 IO 基准测试时，关闭缓存/缓冲至关重要，因为否则基准测试的结果很可能无效。你通常不会连续读写同一个文件数百次。因此，你很可能希望在基准测试的标志中打开直接模式（如果它提供的话）。

在某些情况下，使用 `O_DIRECT` 打开文件实际上可能有助于克服延迟。例如，如果训练程序将日志记录到一个日志文件（尤其是在慢速共享文件系统上），如果应用程序和文件系统缓冲都在起作用，你可能在几秒钟内都看不到日志。由写入者使用 `O_DIRECT` 打开日志文件通常有助于读者更快地看到记录的行。


### 同步 vs 异步 IO

在同步 IO 中，客户端提交一个 IO 请求并等待其完成后再向同一目标设备提交下一个 IO 请求。

在异步 IO 中，客户端可以一个接一个地提交多个 IO 请求，而无需等待任何一个完成。这要求目标设备可以[排队多个 IO 请求](#queue-depth)。


### 顺序 vs 随机访问 IO

**顺序访问** IO 是指您按顺序逐块读取数据（就像看电影一样）。以下是一些示例：
- 一次性读取或写入模型的检查点文件
- 加载一个 python 程序
- 安装一个软件包

**随机访问** IO 是指您随机访问文件的部分内容。以下是一些示例：
- 数据库查询
- 以随机方式从预处理的数据集中读取样本
- 使用 `seek` 在文件中移动



## 基准测试

时间就是金钱，无论是开发人员的时间还是模型训练的时间，因此确保存储 IO 不会成为您人力和计算工作流程的瓶颈至关重要。

在以下部分中，我们将讨论各种方法来确定所提议的存储解决方案是否满足您的工作需求。

### 指标

人们通常关心的三个主要存储 IO 指标是：

1.  [吞吐量](https://en.wikipedia.org/wiki/Network_throughput)或带宽（字节/秒 - 可以是 MBps、GBps 等）
2.  [IOPS](https://en.wikipedia.org/wiki/IOPS)（系统每秒可以执行的输入/输出操作数）
3.  [延迟](https://en.wikipedia.org/wiki/Latency_(engineering))（毫秒或微秒）

- *IOPS* 衡量给定的存储设备或集群每秒可以执行多少次输入和/或输出操作。通常读写 IOPS 不会相同。对于许多系统，它还取决于操作是顺序的还是随机的。因此，一个存储系统将有 4 种不同的 IOPS 速率：

1. 随机读取的 IOPS
2. 随机写入的 IOPS
3. 顺序读取的 IOPS
4. 顺序写入的 IOPS

- *吞吐量* 指的是每秒可以处理多少数据。

IOPS vs. 吞吐量

- 当你处理小文件时，高 IOPS 很重要。
- 当你处理大文件时，高吞吐量很重要。

IOPS 通过块大小与吞吐量相关联：`吞吐量 = IOPS * 块大小`

因此，在给定的 IOPS 下，系统可以读或写的块大小越大，吞吐量就越大。

并且由于有 4 个 IOPS 类别，相应地也有 4 个吞吐量值与之匹配。

*延迟*：是指从发出数据传输指令到该指令的响应到达之间的延迟。

通常，数据包需要传输的距离（交换机、中继、实际距离）越远，延迟就越大。

因此，如果你有一个本地 NVME 驱动器，你的读写延迟将比读写位于另一个大陆的存储设备的延迟短得多。



### fio

[fio - Flexible I/O tester](https://fio.readthedocs.io/en/latest/) 是一款常用的 IO 基准测试工具，操作相对简单。它有许多选项，允许你模拟几乎任何类型的负载，并提供非常详细的性能报告。

首先用 `apt install fio` 或你的包管理器的方式安装 `fio`。

这是一个读基准测试的例子：

```
base_path=/path/to/partition/
fio --ioengine=libaio --filesize=16k --ramp_time=2s --time_based --runtime=3m --numjobs=16 \
--direct=1 --verify=0 --randrepeat=0 --group_reporting --unlink=1 --directory=$base_path  \
--name=read-test --blocksize=4k --iodepth=64 --readwrite=read
```

这里 16 个并发的读线程将运行 3 分钟。基准测试使用 4k 的块大小（大多数操作系统的典型值）和 16k 的文件大小（大多数 Python 文件的常见大小），采用顺序读取方式，并使用[非缓冲 IO](#direct-vs-buffered-io)。因此，这组特定的标志将创建一个很好的基准测试，以显示你在 16 个并发进程上导入 Python 模块的速度。

案例研究：在一个 NFS 设置上，我们第一次运行 `python -c "import torch"` 需要 20 秒，这比在普通 NVME 驱动器上进行的相同测试慢了大约 20 倍。当然，一旦文件被缓存，加载速度会快得多，但这造成了非常痛苦的开发过程，因为一切都很慢。

推荐阅读：[Fio 输出解释](https://tobert.github.io/post/2014-04-17-fio-output-explained.html) - 这是一篇老文章，但仍然很好 - 如果你有更新的说明，请发给我链接或 PR。

重要提示：如果你不使用 `--unlink=1` 标志，请确保在不同基准测试之间删除 `fio` 的工作文件 - 否则可能导致严重的报告错误，因为 `fio` 会重用它为不同基准测试准备的文件，而如果基准测试参数已更改，则不得重用这些文件。显然，这种重用是 `fio` 的一个功能，但对我来说这是一个 bug，因为我不知道这个细微之处，并因此得到了大量无效报告，并且花了一段时间才意识到它们是错误的。

回到基准测试 - 参数需要更改以适应你关心的 IO 操作类型 - 是进行大量的 pip 安装，还是在 512 个进程上写入检查点，还是从 parquet 文件中进行随机读取 - 每个基准测试都必须进行调整以衡量正确的事情。

一开始，我手动筛选我需要的信息，所以我自动化了这个过程，得到了 [fio-scan](./fio-scan) 基准测试，它会对 16KB、1MB 和 1GB 文件大小运行一对读/写基准测试，每个都使用固定的 4k 块大小（总共 6 个基准测试）。它使用一个辅助脚本 [fio-json-extract.py](./fio-json-extract.py) 来解析日志文件，并提取平均延迟、带宽和 iops，并以格式良好的 markdown 表格报告它们。

以下是如何运行它：
```
git clone https://github.com/stas00/ml-engineering/
cd ml-engineering
cd storage

path_to_test=/path/to/partition/to/test
./fio-scan $path_to_test
```
修改 `path_to_test` 指向你想要进行基准测试的分区路径。

注意：日志解析器使用 python3。如果 `fio-scan` 失败，很可能是因为你在一个默认安装 python2 的系统上运行它。它期望 `python --version` 是某个 python 3.x 版本。你可以编辑 `fio-scan` 来指向正确的 `python`。

这是我的三星 SSD 980 PRO 2TB NVME 驱动器上的 IO 扫描示例 ([摘要](benchmarks/results/hope-2023-12-20-14-37-02-331702-summary.md)):

* filesize=16k 读取

| 延迟(毫秒) | 带宽(MBps) | IOPS     | 任务数 |
| -------: | ------: | -------: | ---: |
| 4.0      | 1006.3  | 257614   | 16   |

* filesize=16k 写入

| 延迟(毫秒) | 带宽(MBps) | IOPS     | 任务数 |
| -------: | ------: | -------: | ---: |
| 3.2      | 1239.1  | 317200   | 16   |

* filesize=1m 读取

| 延迟(毫秒) | 带宽(MBps) | IOPS     | 任务数 |
| -------: | ------: | -------: | ---: |
| 1.7      | 2400.1  | 614419   | 16   |

* filesize=1m 写入

| 延迟(毫秒) | 带宽(MBps) | IOPS     | 任务数 |
| -------: | ------: | -------: | ---: |
| 2.1      | 1940.5  | 496765   | 16   |

* filesize=1g 读取

| 延迟(毫秒) | 带宽(MBps) | IOPS     | 任务数 |
| -------: | ------: | -------: | ---: |
| 1.4      | 2762.0  | 707062   | 16   |

* filesize=1g 写入

| 延迟(毫秒) | 带宽(MBps) | IOPS     | 任务数 |
| -------: | ------: | -------: | ---: |
| 2.1      | 1943.9  | 497638   | 16   |


如你所见，在撰写本文时，如果你想将其用作基准来对比，例如，网络共享文件系统，这是一个相当快的 NVMe 驱动器。


### 可用性感知 IO 基准测试

除了精心设计的性能基准测试能给你一些你可能欣赏也可能不欣赏的数字之外，还有一种感知基准测试，那就是某个功能或服务给人的感觉如何。例如，访问一个网站时，是否感觉加载网页的时间太长？或者访问一个视频服务时，视频开始播放的时间是否太长，是否每隔几秒钟就停止缓冲？

因此，对于文件系统，问题非常简单——安装或启动程序是否感觉时间太长？由于我们很多人都生活在 Python 世界中，Python 以拥有数千个通常安装在虚拟环境中的小文件而闻名，而 [conda](https://www.anaconda.com/download) 是目前许多人的选择。

在我们遇到的一个环境中，我们注意到开发人员在共享文件系统上的生产力非常低，因为安装一个包含使用某个 ML 训练框架所需各种软件包的 conda 环境需要长达 30 分钟，而且我们还注意到 `python -c "import torch'` 可能需要超过 20 秒。这比快速的本地 NVME 文件系统慢了大约 5-10 倍。显然，这很糟糕。所以我设计了一个使用 `time` 来测量常见活动的感知测试。通过这种方式，我们可以快速判断我们考虑切换到的共享文件系统解决方案是否显著更好。我们不想要一个快 2 倍的解决方案，我们想要一个好 10 倍的解决方案，因为让一个昂贵的开发人员等待 proverbial 油漆干掉对企业来说不是一件好事。

所以这里是我们使用的简陋基准测试，这只是一个例子。当然，如果你考虑一下你开发人员的工作流程，你会很快发现哪里慢，并设计出最适合你需求的基准测试。

注意：为了有一个比较的基准，请在最近制造的本地 NVME 上进行这些计时测试。这样你就知道上限是多少，但要注意许多共享文件系统将无法达到这个水平。

第一步：如果你要测试的共享文件系统上还没有 conda，请先安装。

```
export target_partition_path=/mnt/weka  # 编辑我！！！
mkdir -p $target_partition_path/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $target_partition_path/miniconda3/miniconda.sh
bash $target_partition_path/miniconda3/miniconda.sh -b -u -p $target_partition_path/miniconda3
rm -rf $target_partition_path/miniconda3/miniconda.sh
$target_partition_path/miniconda3/bin/conda init bash
bash
```
注意：
- 如果你不是在 x86 平台上，请调整 `target_partition_path` 和 miniconda 下载链接。
- 最后我们启动一个新的 `bash` shell 让 conda 设置生效，如果你不是 `bash` 用户，你可能需要进一步调整——我相信你会知道该怎么做。

第 2步：测量 conda 安装时间（写入测试）

计时创建一个新的 conda 环境：
```
time conda create -y -n install-test python=3.9
```

```
real    0m29.657s
user    0m9.141s
sys     0m2.861s
```

计时安装一些大的 pip 包：
```
conda deactivate
conda activate install-test
time pip install torch torchvision torchaudio
```

```
real    2m10.355s
user    0m50.547s
sys     0m12.144s
```

请注意，这个测试有些偏差，因为它也包括了包的下载时间，这取决于你的网络速度，可能会非常快也可能非常慢，从而影响结果。但是，一旦下载的包被缓存，对于 conda 来说，它们也被解压了，所以如果你第二次尝试安装这些包，基准测试就不再公平了，因为在慢速的共享文件系统上，解压可能会非常慢，而我们想捕捉到这一点。

我对此并不担心，因为通常当文件系统非常慢的时候，即使下载速度很慢，你也能看出来它非常慢，你只需观察进度就能判断出来。

如果你确实想让这个基准测试更精确，你可能可以保留预下载的 conda 包，然后只删除它们解压后的目录：

```
find $target_partition_path/miniconda3/pkgs -mindepth 1 -type d -exec rm -rf {} +
```

对于 `pip` 来说，它不会解压任何东西，只是缓存它下载的 wheel 文件，所以 `time pip install` 基准测试如果你第二次运行它，肯定会更精确（第一次下载、缓存并安装，第二次从缓存安装）。所以你可以这样做：

```
conda create -y -n install-test python=3.9
conda activate install-test
pip install torch torchvision torchaudio
conda create -y -n install-test2 python=3.9
conda activate install-test2
time pip install torch torchvision torchaudio
```
如你所见，我们只在第二次安装 pip 包时计时。


第 3 步：清空内存和文件系统缓存后测量加载时间（读取测试）

```
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
time python -c "import torch"
```

如您所见，在进行测量之前，我们必须告诉操作系统清空其内存和文件系统缓存。

如果你没有 `sudo` 权限，可以跳过涉及 `sudo` 的命令，有时系统也可以在没有 `sudo` 的情况下工作。如果你无法运行同步和清空文件系统缓存的命令，你将得到不正确的结果，因为基准测试将测量加载已缓存的文件系统对象的时间。为了克服这个问题，要么请你的系统管理员为你做，要么干脆等到早上，希望你的文件系统缓存了其他东西并清除了 python 包，然后再重复 python 单行命令，希望那些文件不再在缓存中。


以下是查看缓存效果的方法：
```
$ time python -c "import torch"

real    0m5.404s
user    0m1.761s
sys     0m0.751s

$ time python -c "import torch"

real    0m1.977s
user    0m1.623s
sys     0m0.519s

$ sudo sync
$ echo 3 | sudo tee /proc/sys/vm/drop_caches
$ time python -c "import torch"

real    0m5.698s
user    0m1.712s
sys     0m0.734s
```

你可以看到，第一次它没有被缓存，花费的时间长了大约 3 倍，然后当我第二次运行它时。然后我告诉系统刷新内存和文件系统缓存，你可以看到它又长了 3 倍。

我认为在写入测试中再次进行内存和文件系统缓存可能是个好主意，因为即使在那里，缓存也会使基准测试看起来比现实世界中第一次安装新软件包时更快。

还有一次我注意到 `git status` 花了好几秒钟。我使用 [bash-git-prompt](https://github.com/magicmonty/bash-git-prompt)，它在 git repo 克隆中每次返回提示符之前都会运行 `git status`，这变得非常迟缓和难以使用。所以我对 `git status` 进行了基准测试：

```
git clone https://github.com/pytorch/pytorch
cd pytorch
time git status
```
在这个慢速的文件系统上，它花了 3.7 秒，需要修复（在本地 SSD 上只需要 0.02 秒）。好在，这个实际的感知基准测试很容易传递给系统管理员，他们可以立即重现问题，然后着手修复，同时将此基准测试作为参考。

然而，还有一次我注意到，`pytest` 启动需要很长时间，所以我测量了它的收集时间，确实非常慢：

```
time pytest --disable-warnings --collect-only -q
```

所以现在你有很多例子可以选择，我相信你会找到你自己的用例，这些用例很容易可靠地重现，并作为参考点，来判断什么感觉好，什么感觉不好，哪些需要修复。




### 其他工具

-
- [HPC IO 基准测试库](https://github.com/hpc/ior) (`mdtest` 已于 2017 年合并到 `ior` 中)
- [DLIO](https://github.com/argonne-lcf/dlio_benchmark)

XXX：当我有机会尝试这些时，会详细说明如何使用它们



### 已发布的基准测试

以下是一些已发布的 IO 基准测试：

- [MLPerf via MLCommons](https://mlcommons.org/) 发布了各种硬件基准测试，用于衡量训练、推理、存储和其他任务的性能。例如，这是截至本文撰写时最新的 [storage v0.5](https://mlcommons.org/benchmarks/storage/) 结果。尽管我发现结果很难理解——列太多，用户没有任何控制权，而且每个测试都使用不同的参数——那么你如何比较事物呢。

然后是各种你可以自己运行的基准测试：




## 既然可以轻松清理，为什么还要为更多存储付费

在与几家存储供应商交谈后，我了解到许多公司并不费心去清理，而是不断地购买越来越多的存储。如果你不是那样的公司，并且想保持整洁，在接下来的部分中，我将分享如何轻松地清理我们 Python/Pytorch 生态圈中许多人使用的各种缓存（其中很多也适用于其他生态圈）。

### HuggingFace Hub 缓存

非常流行的 HuggingFace Hub 使得下载模型和数据集并将其缓存在本地变得非常容易。你可能没有意识到的是，每当模型或数据集的新版本发布时，旧版本仍然保留在你的磁盘上——因此随着时间的推移，你可能会有很多无用的数据。

缓存的文件通常位于 `~/.cache/huggingface`，但也可以使用 `HF_HOME` 环境变量覆盖它们，并将它们放在其他地方，如果你的 `/home/` 没有空间存放巨大的文件。（过去，这些变量是 `HUGGINGFACE_HUB_CACHE` 和 `TRANSFORMERS_CACHE` 等）。

另一个不需要修改环境变量的解决方案是，将你的缓存符号链接到另一个分区。你可以对所有缓存都这样做：
```
mkdir -p ~/.cache
mv ~/.cache /some/path/
ln -s /some/path/.cache ~/.cache
```

或者只针对 HF hub 缓存：
```
mkdir -p ~/.cache/huggingface
mv ~/.cache/huggingface /some/path/
ln -s /some/path/cache/huggingface ~/.cache/cache/huggingface
```

这里的 `mkdir` 调用是为了防止你还没有使用缓存，所以它们不存在，并确保上面的代码不会失败。

现在你知道了缓存在哪里，你当然可以每隔一段时间就清除整个缓存，但如果这些是巨大的模型和数据集，特别是如果后者进行了一些预处理——你真的不希望一遍又一遍地重复那些耗时的任务。所以我将教你如何使用 HuggingFace 提供的特殊工具来进行清理。

HF hub 上的版本工作方式是通过将 `main` 指向文件的最新版本，同时保留旧版本，以备有人因某种原因想使用旧版本。很有可能你总是想要最新的版本，所以这里是如何删除所有旧版本，只保留 `main`，只需几个快速步骤，无需繁琐的手动编辑。

在终端 A 中：
```
$ pip install huggingface_hub["cli"] -U
$ huggingface-cli delete-cache --disable-tui
File to edit: /tmp/tmpundr7lky.txt
0 revisions selected counting for 0.0. Continue ? (y/N)
```
不要回答提示，继续我的指示。

（注意你的临时文件路径会不同，请在下面相应调整）

在终端 B 中：
```
$ cp /tmp/tmpedbz00ox.txt cache.txt
$ perl -pi -e 's|^#(.*\(detached\).*)|$1|' cache.txt
$ cat cache.txt >>  /tmp/tmpundr7lky.txt
```
这个 perl 单行命令取消了所有包含 `(detached)` 的行的注释——所以可以被清除。然后我们把它粘贴回 `huggingface-cli` 期望被编辑的临时文件中。

现在回到终端 A，然后按：N，Y，Y，所以它看起来像：

```
0 revisions selected counting for 0.0. Continue ? (y/N) n
89 revisions selected counting for 211.7G. Continue ? (y/N) y
89 revisions selected counting for 211.7G. Confirm deletion ? (Y/n) y
```
完成。

如果你在回答提示时搞砸了，你仍然有 `cache.txt` 文件，当你再次运行 `huggingface-cli delete-cache --disable-tui` 时，你可以再次将其提供给它将创建的新临时文件。

附上快照，因为它在 twitter 上更容易阅读，但请使用消息来复制粘贴。

请注意，你也可以使用此工具选择要完全删除的模型或数据集。你只需在编辑器中打开 `cache.txt`，然后删除包含 `main` 的行前面的 `#`，以选择要删除的模型/数据集。然后重复上面解释的过程，只是用手动编辑代替了 `perl` 单行命令。

此外，你会发现 HF `datasets` 有一个 `~/.cache/huggingface/datasets/downloads` 目录，其中通常会包含大量数据集下载和预处理的残留物，包括各种锁定文件。在一个设置中，我发现那里有数百万个文件。所以我这样清理它们：

```
sudo find ~/.cache/huggingface/datasets/downloads -type f -mtime +3 -exec rm {} \+
sudo find ~/.cache/huggingface/datasets/downloads -type d -empty -delete
```

第一个命令会保留 3 天以内的文件，以防有人正在下载/处理东西，我们不想把地毯从他们脚下抽走。

像往常一样，如果你把缓存放在其他地方，你可能需要调整路径。

注意：如果你的团队使用 `HF_HOME` 来共享 HF hub 模型/数据集等 - `$HF_HOME/token` 也会被共享，只要使用的是非门控模型，这就可以正常工作。但是如果你想访问门控模型，你可能会遇到问题。因此，你很可能不希望共享访问令牌。你可以通过添加类似以下内容来解决这个问题：

```
export HF_TOKEN_PATH=~/.cache/hf_hub_token
```
（然后将其放入 `~/.bashrc` 中以使其始终有效）

现在每个用户运行一次：
```
huggingface-cli login
```
这将要求他们从 https://huggingface.co/settings/tokens 添加他们的访问令牌 - 它会将其保存在 `~/.cache/hf_hub_token` 下。

现在你的团队的每个成员都会有他们独特的令牌，并且为他们的 HF hub 用户批准的门控模型现在可以被他们访问。


### Python 包管理器清理

conda 和 pip 会随着时间的推移在你的系统上堆积越来越多的文件。conda 是最糟糕的，因为它保留了解压后的文件，这些文件消耗了大量的 inode，并使备份和扫描变慢。pip 至少只缓存 wheel 文件（压缩文件）。

所以你可以安全地删除这些目录：

```
rm -rf ~/.cache/pip
rm -rf ~/anaconda3/pkgs/
```

请确保如果你的 conda 安装在别处，请编辑最后一条命令。


### 在组环境中共享缓存

如果你的系统上有超过 2 个人在工作，你真的不希望每个人都有自己的 `pip`、`conda`、HF 模型、数据集以及可能的其他东西的缓存。让每个用户的设置指向一个共享缓存非常容易。

例如，假设你在 `/data/cache` 下创建了 `pip` 和 `conda` 缓存，像这样：

```
mkdir /data/cache/conda
mkdir /data/cache/pip
chmod a+rwx /data/cache/conda
chmod a+rwx /data/cache/pip
```

现在你只需要从每个用户的本地缓存符号链接到这个共享缓存：
```
mkdir -p ~/.cache

rm -rf ~/.cache/pip
ln -s /data/cache/pip ~/.cache/pip

rm -rf ~/.conda/pkgs
ln -s /data/cache/conda/pkgs ~/.conda/pkgs
```
注意，我们删除了现有的缓存，但你也可以将它们移动到共享缓存中——无论哪种方式都可以，你反正需要定期清理它们。

所以现在当 `pip` 或 `conda` 尝试访问用户缓存时，它们将被重定向到共享缓存。如果你的组里有 20 个人，那就是 20 倍少的文件——这非常重要，因为 conda 包文件是解压的，并且占用了大量的磁盘 inode。

所以这种方法唯一的问题是文件权限。如果用户 A 安装了一些包，用户 B 可能无法读取或写入它们。

如果这是一个没有恶意用户的隔离集群，你可以简单地要求每个人在他们的 `~/.bashrc` 中使用 `umask 000`，甚至可以通过 `/etc/profile` 或 `/etc/bash.bashrc` 以及其他 shell 配置文件在系统范围内配置此设置，如果 `bash` 不是你选择的 shell。

一旦运行 `umask 000`，大多数文件将以读/写权限创建，以便所有用户都可以读/写彼此的文件。

当然，如果你使用的是某种 HPC，许多不相关的组使用同一个集群，这将不起作用，那么你要么使用组而不是让所有人都可读写文件，可能预设 `setgid` 位或使用 ACL。在任何这样的环境中，总是有系统管理员，所以你可以问他们如何为你的团队设置共享缓存，他们会知道该怎么做。

此外，最近一些应用程序添加了清理工具，例如对于 `conda` 和 `pip`：
```
conda clean --all -f -y
pip cache purge
```

### 常规磁盘使用情况

当然，迟早你的分区会变得越来越大，你可能会想知道数据泄露在哪里。通常，你需要找到贡献了大部分数据消耗的用户，并要求他们进行一些清理。

例如，要找出哪些用户消耗了最多的磁盘，请运行：
```
sudo du -ahd1 /home/* | sort -rh
```
它会按最严重的违规者对数据进行排序。如果你想帮助他们，你可以进入他们的目录并更深入地分析数据：

```
sudo du -ahd1 /home/*/* | sort -rh
```
或者针对特定用户 `foo`：
```
sudo du -ahd1 /home/foo/* | sort -rh
```

你也可以设置磁盘使用配额，但这通常效果不佳，因为根据你公司的工作流程，一些用户需要生成比其他用户多得多的数据，所以他们不应该因此而受到无法完成工作的惩罚，导致他们的工作崩溃——这可能已经运行了几个小时，所有这些工作都将丢失——所以最终公司将为丢失的时间买单。

让用户意识到他们使用了太多的磁盘空间可能是一项非常困难的任务。

### 分区 inode 限制

另外，要注意 inode 的使用情况，在 HPC 的一些共享分区上，我不止一次看到作业崩溃不是因为没有磁盘空间了，而是因为作业用完了最后一个 inode，整个事情就崩溃了。

要查看 inode 使用情况，请使用 `df -i`：
```
$ /bin/df -hi
Filesystem     Inodes IUsed IFree IUse% Mounted on
tmpfs             16M  1.9K   16M    1% /run
/dev/sda1         59M  4.1M   55M    7% /
```
 `-h` 将大数字格式化为人类可读的字符串。

 所以在这里你可以看到 `/` 分区正在使用总可能 inode 的 7%。

 根据文件系统的类型，在某些情况下可以添加更多的 inode，而在其他情况下则不可能。

 因此，作为磁盘空间监控的一部分，您还需要监控 inode 使用情况，将其视为一项关键资源。



### 计算节点上的 `/tmp`

通常，计算节点会使用 `/tmp/` 来存放临时文件。问题在于，在大多数设置中，`/tmp` 位于每个节点的微小 `/` 文件系统上（通常小于 100GB），而且由于 `/tmp/` 只有在重启时才会重置，因此在 SLURM 作业之间不会被清理，这会导致 `/tmp` 空间耗尽。因此，当你尝试运行解压文件之类的操作时，很可能会遇到：

```
OSError: [Errno 28] No space left on device
```

解决方案是在你的 SLURM 启动脚本中设置：
```
export TMPDIR=/scratch
```

现在，slurm 作业将使用更大的 `/scratch` 而不是 `/tmp`，这样就有足够的临时空间来写入了。

脚注：虽然 `/scratch` 很常见——但挂载的本地 SSD 磁盘挂载点可以被命名为任何东西，例如 `/localssd`——通过在其中一个计算节点上运行 `df` 应该很容易看到正确的路径。

您还可以安排 SLURM 设置在作业终止时自动清理此类文件夹。


### 如何找到检查点占用大量磁盘空间的用户

你的团队在训练模型时是否遇到问题，因为巨大的模型检查点没有足够快地卸载到桶存储，导致你不得不不断购买更多存储？

这里有一个单行命令，可以递归分析你选择的路径，找到所有检查点，加总它们的大小，并按最大用户排序打印总计，这样你就可以告诉他们清理他们的文件了 :) 只需将 `/mypath` 编辑为实际路径。

```
find /mypath/ -type f -regextype posix-egrep -regex ".*\.(pt|pth|ckpt|safetensors)$" | \
perl -nle 'chomp; ($uid,$size)=(stat($_))[4,7]; $x{$uid}+=$size;
END { map { printf qq[%-10s: %7.1fTB\n], (getpwuid($_))[0], $x{$_}/2**40 }
sort { $x{$b} <=> $x{$a} } keys %x }'
```

给出：
```
user_a    :     2.5TB
user_c    :     1.6TB
user_b   :      1.2TB
```

当然，您可以更改正则表达式以匹配其他模式，或者可以完全删除它以测量所有文件：

```
find /mypath/ -type f | \
perl -nle 'chomp; ($uid,$size)=(stat($_))[4,7]; $x{$uid}+=$size;
END { map { printf qq[%-10s: %7.1fTB\n], (getpwuid($_))[0], $x{$_}/2**40 }
sort { $x{$b} <=> $x{$a} } keys %x }'
```

如果你想高效地排除一些子目录：

```
find /mypath/ -regextype posix-egrep \
-type d -regex "/mypath/(exlude_a|exclude_b|exclude_c)/.*" -prune -o \
-type f -regex ".*\.(pt|pth|ckpt|safetensors)$" | \
perl -nle 'chomp; ($uid,$size)=(stat($_))[4,7]; $x{$uid}+=$size;
END { map { printf qq[%-10s: %7.1fTB\n], (getpwuid($_))[0], $x{$_}/2**40 }
sort { $x{$b} <=> $x{$a} } keys %x }'
```

提示：第二行告诉 `find` 跳过匹配 `/mypath/(exlude_a|exclude_b|exclude_c)/.*` 正则表达式的文件夹。根据您的用例需要进行调整。


### 如何自动删除旧的检查点

接上条，如果你想自动删除旧的检查点（例如超过 30 天的）。

首先尝试确保候选检查点确实可以删除：

```
find /mypath/ -regextype posix-egrep -regex ".*\.(pt|pth|ckpt|safetensors)$" -mtime +30
```

当你觉得删除是安全的，才添加 `rm`
```
find /mypath/ -regextype posix-egrep -regex ".*\.(pt|pth|ckpt|safetensors)$" -mtime +30 -exec rm {} +
```

</rewritten_file>