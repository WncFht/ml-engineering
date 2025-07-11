---
title: 如何选择云提供商
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/gupqe5b5/
---
# 如何选择云提供商

在长期和短期使用过多个计算云，并参加过许多"发现"电话会议后，我了解到以最谨慎和专注的态度来选择云服务是绝对关键的。特别是对于长期合同 - 你可能会陷入一个为期 3 年的锁定，支付数百万美元，最终却得到糟糕的体验，而且无法摆脱合同。

给您一个视角 - 一个 64 节点的集群在 3 年内可能轻易花费 2000-5000 万美元。这通常比初创公司支付的薪水还要多。

我必须强调，选择一份糟糕的 3 年合同可能会阻止你的初创公司成功。

在本文中，我不会告诉您要避开哪些云，而是试图让您有能力避免糟糕的体验，并至少拥有一次体面的体验，这将给您的公司一个成功的机会。

这些笔记假设您已经知道您特定工作负载需要什么样的计算资源。如果您不知道，请浏览 [加速器](../compute/accelerator)、[存储](../storage) 和 [网络](../network) 章节，以了解现在有哪些可用的。大多数时候，您想要云提供的最新产品。

## 术语表

- CSP：云服务提供商
- SLA：服务水平协议
- SLO：服务水平目标
- TCO：总拥有成本

## 合同

如果你是按小时付费，你就不需要担心合同。但这种方法从长远来看并不好，因为你会多付很多倍的钱，而且你不会有一个稳定可靠的加速器基础。一份长期合同，有时加上一个好的谈判者，可以节省 10 倍的总拥有成本（TCO）（和时间）！

### 免费试用

大多数云服务提供商（CSP）都有试用计划，您可以在几个节点上免费"试用"几天/几周。

当然，这并不能说明更大的集群扩展得有多好，但足以运行相当多的基准测试和实验。

它还会给你一个很好的机会来检查提供商的客户支持是如何工作的（如果免费套餐中包含任何支持的话）。

### 半成品解决方案

由于新一代加速器大约每 12-18 个月出现一次，而客户希望"昨天"就拥有这些最新的加速器以获得相对于竞争对手的业务优势 - 这使得 CSP 几乎没有时间来集成新一代硬件、进行测试、调整其软件堆栈并对这些组件进行老化测试。

所以如果你想在最新一代产品一上市就拥有它，你几乎肯定会有一个糟糕的体验，因为，嗯，把事情做好是需要时间的——我们说的是几个月的等待。但客户是上帝——所以 CSP 给他们想要的东西，通常不太会说客户得到的东西还没有完全准备好。

我不确定 CSP 是否应该受到指责，因为他们通常在制造商承诺交货后几个月才收到硬件，当然，到那时他们已经无法兑现对客户的承诺，所以他们就直接交货了……

然后一些 CSP 为了获得更好的利润而开发自己的硬件（例如网络堆栈），然后他们未能按时完成这些定制解决方案，最新的加速器已经有了，但整个系统却在跛行。提供现成的组件要安全得多，因为这些组件很可能是经过良好测试的工作组件（预计成本可能会更高）。

我认为如果客户想早点拿到硬件是可以的，只是应该有一个诚实的披露，比如："看，我们需要再花 3 个月的时间来把事情做好，如果你现在就要节点，你可以拿走，但我们不能保证任何事情。"

### 我们会尽力而为条款

很多长期云合同都可能包含大量的"我们会尽力而为"条款。

然而：

1. 客户不被允许"尽力而为"地付款，他们在法律上有义务按时支付他们同意支付的金额。
2. 客户不被允许在合同期满前违约。

根据我的经验，一级云服务商通过派遣 10 多人参加与客户的会议来展示"我们将尽力而为"。其中一些人会毫无头绪，只是坐在那里让公司看起来资源丰富："看，我们正在为你们遇到的问题分配 10 多个人。你们没什么好担心的"。然而，大多数时候那些人都无法解决你的问题。

您需要的是电话会议上有两名云支持人员 - 一名产品经理和一名直接负责解决手头问题的工程师。根据我的经验，这种会议可能需要数周到数月才能实现，或者根本无法实现。通常需要有良好的人脉才能将问题升级到"高层"。

对于您购买的套餐中每一个关键组件，您都需要一个可量化的交付成果。例如，如果您购买的网络应该在那么多节点上以 X GBps 的速度运行 all-reduce，而您测量的速度明显更低，那么合同中应该规定当这种情况发生时 CSP 会怎么做。他们有多长时间来解决问题，以及如果双方同意的时间内问题没有解决，您是否可以终止合同。

存储、加速器和您计划依赖的任何其他关键组件也是如此。

当然，具体的影响由您来协商，但可能最好的是在问题解决之前停止付款。这样就有很大的动力去解决问题。

唉，不付钱有帮助，但不能使用计算资源仍然是一个巨大的问题。而违约并迁移到另一个提供商是一项巨大的工程，不能掉以轻心。但至少如果得不到你需要的东西，你还有事可做。

我还必须说，这几乎从来都不是工程师的问题，他们通常都是非常棒、经验丰富的人——大多数时候是管理和资源分配的问题。所以请尽可能温和地对待你交往的人，同时坚定地要求解决方案。我知道这很难——不止一次我已到山穷水尽的地步，我不能总是保持冷静。

### 服务水平协议

作为上一节的延续，[服务水平协议](https://en.wikipedia.org/wiki/Service-level_agreement) (SLA) 是服务提供商和客户之间的协议，它定义了关于服务质量和可用性以及各种责任的各种保证和期望。

另一个术语是服务水平目标（SLO），其中 SLA 被量化。例如，一个 SLO 可能将每月正常运行时间百分比定义为 99.5%，如果正常运行时间低于 99.5%，提供商将按一定比例返还客户花费的金额。例如，如果正常运行时间在 99-99.5% 之间，则返还 10%；如果为 95-99%，则返还 25% 等。这里是一个 [GCP SLA](https://cloud.google.com/ai-platform/training-and-prediction/sla?hl=en)。

在租用 ML 集群时，应该关心的主要类别是加速器和/或整个节点故障。如果您为 64 个节点付费，但只能使用 60 个，那么您应该为您无法使用的那些节点获得报销/补偿。您的 SLA 应该定义停机时间，在此之后提供商开始向您支付补偿以及支付多少。

网络和存储也是如此，尽管它们通常比加速器故障的频率低得多，但它们确实会发生故障。

通常，服务的任何关键部分都应该有 SLO，并明确规定如果未达到 SLO 的后果。

大多数一级公司应该已经在合同中包含了他们的标准 SLA。理论上，客户应该能够协商这些以适应他们的需求，尽管这可能并不总是可能的。有时，支付更多费用可能会获得比标准 SLO 更好的服务。

### 讨论合同终止条款

双方都应有权体验互利的商业体验。

因此，至关重要的是，如果您的业务体验因对方未能满足商定的期望而没有益处，您应该能够合法地退出合同。

当然，这意味着不要打官司，因为官司可能非常昂贵，而且一级云服务商有很多钱聘请最好的律师，所以这可能是一场败仗。

在什么情况下可以在合同到期前干净利落地退出合同，这取决于您的谈判。

### 必须包含付费支持

在我工作过的一家公司，我们的云合同不包括付费支持服务，我们唯一的支持是通过客户聊天。跳过付费支持是为了节省成本，但天哪，我们因此损失了好几天的计算资源。

不要试图在这里省钱 - 你最终会损失很多钱、开发人员时间和头发。确保你有一种提交带有优先级的工单的方式，并在合同中明确规定处理速度的期望。

当您试图使用客户聊天来解决紧急问题时，他们没有义务做任何事情，或者至少没有义务及时处理。

如果您正在与产品经理打交道，您需要知道您能多快地直接与终端工程师交谈，同时去掉中间人。

### 非工作时间支持

您在周末/节假日/夜晚遇到紧急情况时能获得人工支持吗？例如，在我使用的一个 HPC 上，人工支持只在周一至周五的 9 点到 5 点提供。

如果这个不可用，至少要确保你的团队可以自己进行集群复苏——并进行演练以确保这实际上是可行的。这意味着你需要有一个 API 来执行所有这些事情，而不需要提供商的支持。

### 下一代加速器迁移

平均而言，新一代加速器每 12-18 个月就会出现一次，但典型的合同为期 3 年。这意味着在大约一半的时间里，您最终将使用劣质产品。

当有更快版本的加速器可用时，没有人愿意使用慢 2-5 倍的加速器，但大多数客户现在在整个 3 年合同期内都被旧加速器所困。

您需要协商在期限结束前迁移到新一代的能力，这显然需要为此支付额外的费用。

## 加速器

这组问题是针对加速器的。

### 加速器需要老化测试

当一批新组件到货时，提供商必须在将其交给客户之前对其进行"老化测试"。这是一个运行大量压力测试以检测任何有故障的加速器和其他系统组件的过程。

如果不这样做，客户最终会在运行其工作负载时艰难地发现"坏苹果"。这会导致计算和开发人员时间的损失。如果工作负载使用几个节点，一个出现故障的加速器大多数时候不是大问题，但如果工作负载使用数十或数百个节点，成本就非常巨大了。

发现坏加速器不应该是客户的责任。虽然不能保证加速器在经过压力测试后不会出现故障 - 但它应该很少发生。

否则，一批新的加速器通常有 3-10% 的故障率，这对客户来说是巨大的并且非常昂贵！

所以问问你的提供商他们为你的加速器/系统老化测试了多长时间，如果他们做了的话。

我还没有找到一个黄金参考点，但是，例如，[SemiAnalysis](https://semianalysis.com/2024/10/03/ai-neocloud-playbook-and-anatomy/#cluster-deployment-and-acceptance-test) 建议 OEM 提供商执行 3-4 周的老化测试，然后 CSP 再进行 2-3 天的老化/验收测试。因此，如果是这种情况，您需要确保系统至少经过 2-3 天的压力测试。

### 处理加速器故障

根据我的经验，虽然其他计算组件偶尔会出故障，但 95% 的时间都是加速器出故障。

因此，您需要有一个非常清晰和快速的途径来更换加速器。

理想情况下，这个过程需要自动化。所以你需要问是否有 API 可以释放一个损坏的节点并获得一个替换品。如果你必须要求人类来做这件事，通常效果不太好。事情越自动化，体验就越高效。

提供商端备用池中有多少加速器可供您使用？他们通常会承诺每月一定数量的快速更换。

话虽如此，如果时间对您的工作流程至关重要，因为大多数时候您无法获得即时更换，您应该始终比您需要的节点多支付约 10% 的费用。额外的节点可用于开发，如果在训练期间有故障节点，您可以立即使用自己的额外节点。

### 确保所有节点都在同一个网络骨干上

除非您租用 1 万个 GPU，否则大多数较小的集群都可以轻松地共置在同一个网络骨干上 - 这样从任何节点到任何其他节点执行节点间网络流量所需的时间都是相同的。

确保您不付费但用于处理故障加速器的任何备用节点与您付费的节点位于同一网络主干上。如果它们不在，如果您进行多节点训练，将会遇到大问题 - 因为那个替换节点将离所有其他节点更远，并且会拖慢整个集群（链条中最薄弱的环节）。

### 确保重启后保留好的加速器

您希望您的集群有固定的分配。这意味着如果您需要重新部署节点，特别是如果您计划停机，其他客户不会抢走那些节点！

一旦您花了数周时间从好节点中筛选出坏节点，将这些节点留给自己并且不再开始痛苦且昂贵的筛选过程至关重要。

### 你认为你需要扩张吗？

这是一个难题，因为很难提前知道您要求的节点数量将来是否需要增长。

理想情况下，您希望与您的提供商讨论这个问题，以防他们可以为您即将到来的扩张做好计划。

因为否则，比如说，你想将你的节点数量增加一倍，但为了获得更多的节点，它们只能分配在另一个网络骨干上——这将是一个问题，因为它会影响训练速度。

很可能你将不得不放弃你目前的分配，并转移到另一个更大的分配——如果他们没有本地容量，甚至可能在不同的地区。而转移到不同的地区可能是一个非常缓慢和昂贵的经历，因为你必须把你的存储转移到你的新集群所在的地方。根据个人经验——不要轻视这一点。

## 存储

大型和快速的存储对于良好的开发人员体验和快速的训练/微调/推理工作负载都非常重要 - 特别是在加载/保存检查点方面。

### 保证最大容量

询问您将支付的存储空间有多少是保证的。

例如，如果使用 Lustre 文件系统，客户需要知道他们必须超额配置 25% 才能获得他们需要的实际存储容量，因为 Lustre 可能会在总存储容量达到 80% 时写入失败，这是因为磁盘平衡设计不佳。而为额外 25% 付费的责任在于客户！

我经历过的大多数其他文件系统通常都能在不失败的情况下达到 100% 的容量，但最好还是问一下你计划使用的特定文件系统。

### 了解您的存储 IO 需求

在我们使用的一个云上，我们使用了一个非并行的分布式文件系统，开发体验非常糟糕。虽然处理大文件还可以接受，但小文件的体验非常慢——安装一个基本的 Conda 环境需要 30 分钟，运行 `python -c "import torch"` 需要 2 分钟。这是因为 Python 有成千上万个 4-16kb 的文件，如果文件系统没有优化来处理这些文件，并且元数据服务器很弱，这将是一次非常令人沮丧的体验。

一般来说，典型的 Python 开发环境需要一个能够处理以下情况的文件系统：
- 成千上万个小文件
- 少量巨大文件

但是，当然，只有您知道您的工作负载的具体要求。还要考虑本地存储和远程（共享）存储之间的关系，因为一些提供商会为了省钱而减少本地驱动器的大小和性能。在许多情况下，开发人员会从共享文件系统中读取可以本地缓存的数据（代码库、模型、数据集）。教人们如何将 [rsync](https://linux.die.net/man/1/rsync) 与本地 NVMe 结合使用可以改善开发人员体验，并减少共享文件系统上的 I/O。

有关存储要求及其基准测试的细微差别，请参阅[存储章节](../storage)中的说明和指南。

### 存储故障时会发生什么

使用先进昂贵的分布式文件系统，故障的几率相对较小，但使用较便宜的存储解决方案则相当大。

但任何系统都可能发生。

你需要知道：
- 谁负责解决问题？
- 恢复需要多长时间？
- 谁为停机时间买单？
- 出现问题时用户该怎么办？

如果解决方案需要很长时间，通常需要添加另一个临时文件系统分区，以便人们能够完成他们的工作。当然，您将不得不为此付费。

### 区域迁移

当升级到下一代加速器或扩展容量时，如果所在区域没有您需要的东西，集群可能会被迫迁移到不同的区域。为了工作流程的快速，存储必须与加速器在同一区域。

迁移事件会引发有时非常痛苦的存储迁移体验。

在迁移开始之前，您需要问一些关键问题。

- 提供商负责移动您的数据还是这是您的责任？
- 您是否检查过所提供的工具是否足以在几小时内移动 TB 级的数据，或者是否需要很多天才能移动？例如，使用存储云进行迁移通常会丢弃所有文件元数据，这可能是一个巨大的问题。如果您有 500 万个小文件，复制可能需要很长时间。除非您使用 `tar`，但创建它可能需要很多小时，而且您是否有 2 倍的存储空间来存放数据的 2 个副本？
- 您是否应该为两个重叠的集群支付存储和计算费用？
- 在文件系统迁移期间，正在编辑和创建的文件会发生什么 - 您是让所有人都回家，同时冻结文件系统吗？

### 备份和归档

许多 CSP 只有一个价位的一个文件存储层。然而，组织可能需要多层存储。例如，您可能希望将旧的模型检查点或微调数据集归档到廉价的冷存储中，例如 HDD 上的 S3 对象。

拥有扩展总存储容量的灵活性，并保持"热"（本地 NVMe）、"温"（共享 NVMe）、"冷"（共享 HDD）和"归档"（磁带）同步，可以帮助提高系统的弹性、节省资金，并便于将来迁移或扩展。

## 网络

这部分主要与计划进行训练和微调的人相关。如果您需要租用加速器用于通过微服务的大规模部署进行推理，或者用于小规模、按需、交互式工作（即笔记本），您可以安全地忽略此信息。唯一的例外是当您计划推理需要多个节点才能容纳单个副本的非常大的模型时。

通常，您需要确保所提供的[节点内](../network#intra-node-networking)和[节点间](../network#intra-node-networking)网络速度与承诺和您的期望相符。

### 询问实际性能数据

计算理论永远与现实不符，而且即使所有提供商都使用相同的组件，现实情况也可能因提供商而异，因为它取决于所有相关组件的质量以及机架的设计和组装情况。

最简单的要求是请求一个在 4-8-16-32-64 个节点（如果您的集群超过 64 个节点，则更多）上的 `all-reduce` 基准测试图。您会期望带宽随着参与节点数量的增加而逐渐变差，但不会急剧下降。一些网络在节点数量较多时会变得非常低效。

更多详情请参考[真实网络吞吐量](../network#real-network-throughput)。

理想情况下，您至少要对几个有效载荷进行基准测试——那些您特别感兴趣的，因为您知道这是您将在工作负载中使用的集合有效载荷。我通常首先要求一个大约 4-16GB 大有效载荷的图（16GB 在最新的最快的节点间网络上会获得最佳带宽），如果性能下降到理论 GBps 的 80% 以下，那么我就知道我们有问题了。

### 网络是否会占用加速器内存？

我在其中一个云上遇到的一个意外是，当我开始使用 GPU 时，我发现每个 GPU 的 5GB 已经被网络软件占用了——我们设法将其减少到一个较低的值，但我们仍然被卖了内存小于其标称大小的 GPU，而且在签合同之前没有人告诉我们。

随着加速器变得越来越大，这可能变得不那么重要，但是当你在 H100 上获得 75GB 的可用内存而不是 80GB 时——每个 GPU 损失的内存量是巨大的。

### Infiniband还是以太网？

通常，CSP 遵循 NVIDIA 的 [DGX SuperPOD 参考架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/abstract.html)，该架构提供了大量关于如何构建轨道优化 InfiniBand 网络的详细信息。轨道优化基本上意味着 8 路系统中的每个 GPU 都连接到自己的叶交换机。其他一切都是标准的胖树结构。

然而，现在世界上最大的许多 GPU 集群都运行 RoCEv2 而不是 Infiniband。Meta 已经[证明](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/)你可以在 RoCEv2 网络上训练前沿级别的 Llama 模型。Semianalysis/Fabricated Knowledge 显示 NVIDIA GPU 的网络附加率[显著下降](https://www.fabricatedknowledge.com/p/nvidia-waiting-on-blackwell-and-whats?utm_source=post-banner&utm_medium=web&utm_campaign=posts-open-in-app&triedRedirect=true)。

由于多节点训练依赖于网络集合（即 NCCL 或 RCCL），网络类型会显著影响性能和用户体验。

## 安全性

尽管有时会被忽略，但 CSP 对安全性的处理方式可能千差万别。仅仅获得 SOC 2 Type 2 合规性认证可能还不够。最好检查一下您将使用的机器是否是虚拟化的。如果您不在虚拟机中，而云提供商为其他租户提供服务，您可能不信任他们在您不在的机器上所做的事情。最好检查您的云提供商在为您配置（或重新配置）服务器之前是否验证了已知良好版本的 BMC 固件、系统和 BIOS 固件。

## 其他

### 一级云与二级云

我还没有明确的建议，是一级云（AWS、GCP、Azure 等）还是新兴的较小二级云更好。我的直觉是，二级云可能会提供更好、更个性化的支持，因为它们必须更努力地争取客户。

价格方面，二级云通常更便宜，否则它们无法与一级云竞争。然而，很明显它们的"利润"会小得多，因为二级云没有一级云的批量购买力。

二级云更有可能更灵活，拥有非主流加速器（例如 AMD 和 Intel），并且可能更愿意以很少或没有成本的方式帮助调整事物。

### 编排

一个运行良好的节点编排对于成功使用多节点集群至关重要。

确保您知道您需要哪种——通常是 [SLURM](../orchestration/slurm/)、Kubernetes 或两者的结合，并确保它得到很好的支持。一些云只支持其中一种，或者对另一种提供非常有限的支持。如今，SLURM 主要用于训练/微调，而 Kubernetes 用于推理。还有其他[新兴的编排平台](../orchestration/)。

与硬件一样，根据您是否计划管理自己的集群，您需要知道谁将处理任何问题。这是您技术栈中非常关键的组成部分，因为如果编排系统损坏，没有人可以使用集群，您就会损失时间/金钱。

### 最新的软件/操作系统版本

确保询问提供商不会强迫您使用某些旧版本的软件和操作系统。

我曾有过这样的经历，我们被迫使用一些非常旧的 Ubuntu 版本，因为我们必须使用的提供商的软件堆栈不支持更新的、最新的操作系统。

### 系统管理

如今，很难找到一个了解 ML 工作负载特定需求的优秀系统管理员，因此最好问问是否可以将部分工作外包给 CSP。一级 CSP 会将服务分包给可以提供不同程度系统管理的服务公司。较小的云可能会提供自己的直接服务。他们通常对 ML 工作负载的需求有很好的把握。

没有经验丰富的人来照管您的集群，您将无法成功。让您的 ML 工程师同时处理系统管理工作可能会非常适得其反，因为它可能是一项非常耗时且具有干扰性的工作。

要么聘请一名系统管理员，要么聘请一家服务公司为您做这件事。

## 结论

这些笔记基于我的直接经验，显然我还没有接触到所有可能出错并对您的集群造成严重破坏或让您的整个团队筋疲力尽并掉光头发的事情。但这应该是一个很好的思考基础。

通过思考对您来说什么是重要的，哪些失败可能会阻止您实现计算目标，来添加您自己的问题。

如果您正在考虑某个特定的 CSP，请向社区询问有关他们的信息，特别是要避免哪些陷阱。

本文的核心信息是，为您选择一个云，在这个云中您的选择权没有被剥夺，您不会被困在一个您的开发人员讨厌的服务中，这很可能会导致人们离开您的公司。

如果您觉得这些笔记对您来说太过繁重，我偶尔会提供咨询，帮助进行尽职调查并参加发现电话会议。您可以通过 [stas@stason.org](mailto:stas@stason.org?subject=Choosing%20cloud%20consulting) 联系我。

## 延伸阅读

- semianalysis.com 创建了一个 ClusterMax CSP 评级系统，并对不同标准进行了出色的解释，并计划继续对许多 CSP 进行排名。[2025](https://semianalysis.com/2025/03/26/the-gpu-cloud-clustermax-rating-system-how-to-rent-gpus/)
