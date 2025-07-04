---
title: 自述文件
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/wyiaqonf/
---
# 网络调试

通常您不需要成为网络工程师就能解决网络问题。通过阅读以下说明，可以解决一些常见问题。



## 术语表

- OOB：带外（通常是较慢的以太网 NIC）
- Bonding：将多个 NIC 绑定在一起以获得更快的速度或作为备份
- IB：InfiniBand（最初由 Mellanox 开发，后被 NVIDIA 收购）
- NIC：网络接口卡


## 如何诊断 NCCL 多 GPU 和多节点连接问题

本节绝对不是详尽无遗的，旨在涵盖我经常遇到的一些最常见的设置问题。对于更复杂的问题，请研究 [NCCL 仓库问题](https://github.com/NVIDIA/nccl/issues) 或如果您找不到与您的情况匹配的问题，请提交新问题。NCCL 还包括一个简短的[故障排除部分](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html)，但通常通过阅读[问题](https://github.com/NVIDIA/nccl/issues)可以学到更多。

对于网络诊断工作，我建议使用这个专门开发的测试脚本：[torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py)，而不是使用可能需要很长时间才能启动并存在无关问题的完整应用程序。

首先，在设置后运行基于 nccl 的程序：

```
export NCCL_DEBUG=INFO
```
这将打印大量关于 NCCL 设置及其网络流量的调试信息。

例如，如果您正在使用前面提到的调试脚本，对于一个有 8 个 GPU 的单节点，您可能会这样做：

```
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 torch-distributed-gpu-test.py
```

要在多个节点上启动它，您必须使用像 SLURM 或 Kubernetes 这样的编排软件，或者在每个节点上手动启动它（`pdsh` 会有很大帮助）- 有关详细信息，请参阅 [torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py) 中的说明。但为了了解工作原理，我建议从 1 个节点开始，然后逐步增加到 2 个，然后再到更多节点。

现在，检查程序的输出并查找以以下开头的行：
```
NCCL INFO NET/
```
然后检查它正在使用哪个协议和哪个接口。

例如，这个输出：
```
NCCL INFO NET/FastSocket : Using [0]ibs108:10.0.19.12<0> [1]ibs109:10.0.19.13<0> [2]ibs110:10.0.19.14<0> [3]ibs111:10.0.19.15<0> [4]ibs112:10.0.19.16<0> [5]ibs113:10.0.19.17<0> [6]ibs114:10.0.19.18<0> [7]ibs115:10.0.19.19<0>
```

告诉我们正在使用 [nccl-fastsocket](https://github.com/google/nccl-fastsocket) 传输层插件，并且它发现了 8 个 `ibs*` 网络接口（NIC 卡）。如果您使用的是 Google Cloud，这是正确的，并且您的 NCCL 很可能已正确设置。但如果您使用的是 InfiniBand (IB) 并且得到上述输出，您很可能会得到非常低的节点间速度，因为这意味着您激活了错误的插件。

在使用 IB 的情况下，您希望看到的是 `NET/IB` 及其 IB 接口：
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/IB [RO]; OOB eno1:101.262.0.9<0>
```

在这里，您可以看到 IB 与 8 个 `mlx5_*` 接口用于集合通信，还有一个 OOB，代表带外，用于引导连接，通常使用较慢的以太网 NIC（有时是[多个 NIC 绑定成一个](https://wiki.linuxfoundation.org/networking/bonding) - 以防您想知道接口名称中的 `bond` 代表什么）。

要知道您的节点有哪些 TCP/IP 接口，您可以在其中一个节点上运行 `ifconfig`（通常所有相似的节点都会有相同的接口名称，但并非总是如此）。

如果您的集合通信网络是 IB，您应该运行 `ibstat` 而不是 `ifconfig`。`NCCL INFO NET` 的最后一个示例将对应于以下输出：

```
$ ibstat | grep mlx5
CA 'mlx5_0'
CA 'mlx5_1'
CA 'mlx5_2'
CA 'mlx5_3'
CA 'mlx5_4'
CA 'mlx5_5'
CA 'mlx5_6'
CA 'mlx5_7'
```

除了快速的节点间连接 NIC 之外，您很可能还有一个慢速的管理以太网 NIC（甚至有几个），用于配置节点、使用共享文件系统、访问互联网，因此几乎可以肯定 `ifconfig` 也会包括额外的 NIC。此外，您很可能还有一个 docker 网络接口、`lo` 环回和其他一些接口。例如，在我的桌面上，我可能会得到以下输出：

```
$ ifconfig
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.99.0.1  netmask 255.255.0.0  broadcast 172.99.255.255
        inet6 f330::42:fe33:f335:7c94  prefixlen 64  scopeid 0x20<link>
        ether 02:42:fe:15:1c:94  txqueuelen 0  (Ethernet)
        RX packets 219909  bytes 650966314 (650.9 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 262998  bytes 20750134 (20.7 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 1147283113  bytes 138463231270 (138.4 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 1147283113  bytes 138463231270 (138.4 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 2601:3108:1c71:600:4224:7e4b:13e4:7b54  prefixlen 64  scopeid 0x0<global>
        ether 04:41:1a:16:17:bd  txqueuelen 1000  (Ethernet)
        RX packets 304675330  bytes 388788486256 (388.7 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 74956770  bytes 28501279127 (28.5 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device memory 0xa3b00000-a3bfffff
```

我之所以提到所有这些，是因为关键部分是确保 NCCL 在其 `Using` 调试行中只报告正确的接口。如果任何像 `docker0` 或 `lo` 或 `eth0` 这样的接口最终被报告，例如：

```
NCCL INFO NET/Socket : Using [0]eth0:10.0.0.23<0>
```

如果您有更快的网络接口可用，这很可能不是您想要的。但是，当然，在某些情况下，以太网 NIC 是您所拥有的全部，在这种情况下，上述情况就可以了 - 只是会很慢。

有时，如果使用了错误的接口，应用程序可能会挂起。

如果您有所有正确的接口，加上一些不正确的接口，NCCL 可能会工作，但速度会变慢。

如果是在云环境中，通常您的云提供商应该会给您如何正确设置的说明。如果他们没有，那么您至少需要询问他们需要使用哪些网络接口来设置 NCCL。

虽然 NCCL 会尽力自动发现它应该使用哪些接口，但如果它无法正确地这样做，您可以通过告诉它要使用或不使用哪些接口来帮助它：

- `NCCL_SOCKET_IFNAME` 可用于指定在不使用 Infiniband 时要包含或排除哪些 `ifconfig` 接口。以下是一些示例：

```
export NCCL_SOCKET_IFNAME=eth:        使用所有以 eth 开头的接口，例如 eth0, eth1, …
export NCCL_SOCKET_IFNAME==eth0:      仅使用接口 eth0
export NCCL_SOCKET_IFNAME==eth0,eth1: 仅使用接口 eth0 和 eth1
export NCCL_SOCKET_IFNAME=^docker:    不使用任何以 docker 开头的接口
export NCCL_SOCKET_IFNAME=^=docker0:  不使用接口 docker0。
```
完整的文档在[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname)。

- 使用 IB RDMA（IB Verbs 接口）时，请使用 `NCCL_IB_HCA` 环境变量代替 `NCCL_SOCKET_IFNAME`，它选择用于集合通信的接口。示例：

```
export NCCL_IB_HCA=mlx5 :               使用所有以 mlx5 开头的卡的所有端口
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1 : 使用卡 mlx5_0 和 mlx5_1 的端口 1。
export NCCL_IB_HCA=^=mlx5_1,mlx5_4 :    不使用卡 mlx5_1 和 mlx5_4。
```
完整的文档在[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)。

例如，通常使用 IB 时，会有额外的接口，例如 `mlx5_bond_0`，您不希望将其包含在 NCCL 通信中。例如，此报告将表明包含了错误的 `[8]mlx5_bond_0:1/RoCE` 接口，这几乎肯定会导致低带宽：
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/I [8]mlx5_bond_0:1/RoCE [RO]; OOB ibp25s0:10.0.12.82<0>
```
在那里，您可以使用以下命令将其排除：
```
export NCCL_IB_HCA=^mlx5_bond_0:1
```
或者，您可以明确列出您想要的接口，例如：
```
export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

如前所述，在与 IB 互连的节点之一上使用 `ibstat` 将向您显示可用的 IB 接口。

由于 NCCL 会尝试自动选择最佳的网络接口，因此只有在 NCCL 不工作或速度慢时才需要执行上述操作。在正常情况下，NCCL 应该开箱即用，用户无需执行任何特殊操作。

此外，根据所使用的云，提供商很可能会给您一系列要设置的环境变量。如果您错误地设置了其中一些，NCCL 可能会工作缓慢或根本不工作。

用户遇到的另一个典型问题是，当他们尝试在云 B 上重用他们在云 A 上工作的 NCCL 设置时。通常情况下，事情无法转换，必须仔细删除任何先前设置的环境变量，并为新云正确地重新设置它们。即使您使用的是同一个云，但不同类型的实例，也可能会出现此问题，因为某些网络设置对于给定的实例非常具体，在其他地方将不起作用。

一旦您认为已正确设置 NCCL，下一步就是对您的连接进行基准测试，并确保其与宣传的速度相符（嗯，大约是其 80%）。请转到[基准测试章节](../benchmarks)。


## NCCL 与 docker 容器

* 通过向 docker `run` 添加这些附加参数来提供足够的资源：`–shm-size=1g –ulimit memlock=-1` ([更多详细信息](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#sharing-data))
* 特权访问：有时您需要向 docker `run` 参数添加 `--privileged`。
* 确保 docker 镜像包含正确的包，例如，如果使用 IB，您至少需要安装 `libibverbs1 librdmacm1`



## 如何检查是否支持 P2P

有时您需要知道计算节点上的 GPU 是否支持 P2P 访问（点对点）。禁用 P2P 通常会导致节点内连接速度缓慢。

您可以看到在这个特定的 8x NVIDIA H100 节点上支持 P2P：

```
$ nvidia-smi topo -p2p r
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
 GPU0   X       OK      OK      OK      OK      OK      OK      OK
 GPU1   OK      X       OK      OK      OK      OK      OK      OK
 GPU2   OK      OK      X       OK      OK      OK      OK      OK
 GPU3   OK      OK      OK      X       OK      OK      OK      OK
 GPU4   OK      OK      OK      OK      X       OK      OK      OK
 GPU5   OK      OK      OK      OK      OK      X       OK      OK
 GPU6   OK      OK      OK      OK      OK      OK      X       OK
 GPU7   OK      OK      OK      OK      OK      OK      OK      X

图例：

  X    = 自身
  OK   = 状态正常
  CNS  = 芯片组不支持
  GNS  = GPU 不支持
  TNS  = 拓扑不支持
  NS   = 不支持
  U    = 未知
```

另一方面，对于这个特定的 2x NVIDIA L4，不支持 P2P：
```
$ nvidia-smi topo -p2p r
        GPU0    GPU1
 GPU0   X       CNS
 GPU1   CNS     X
```

从图例中可以看到，`CNS` 表示"芯片组不受支持"。

如果您使用的是高端数据中心 GPU，这种情况发生的可能性很小。不过，像上面 L4 的例子一样，一些低端数据中心 GPU 可能不支持 P2P。

对于消费级 GPU，您的 GPU 不受支持的原因可能多种多样，通常是启用了 IOMMU 和/或 ACS 功能。其他时候，只是驱动程序版本的问题。如果您花一些时间搜索，您可能会找到有人破解驱动程序以在不应支持 P2P 的 GPU 中启用 P2P，例如这个 [4090 P2P 支持仓库](https://github.com/tinygrad/open-gpu-kernel-modules)。

要检查 PCI 访问控制服务 (ACS) 是否已启用以及如何禁用它们，请遵循[本指南](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#pci-access-control-services-acs)。

可以在 BIOS 中禁用 IOMMU。

您还可以使用 torch 检查特定 GPU 之间的 P2P 支持 - 在这里我们检查 GPU 0 和 1：

```
python -c "import torch; print(torch.cuda.can_device_access_peer(torch.device('cuda:0'), torch.device('cuda:1')))"
```
如果没有 P2P 支持，上面的命令会打印 `False`。



## 如何计算 NCCL 调用次数

为子系统启用 NCCL 调试日志 - 集合：
```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
```

如果您在多节点 slurm 环境中工作，您可能只想在 rank 0 上执行此操作，如下所示：
```
if [[ $SLURM_PROCID == "0" ]]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=COLL
fi
```

假设您的所有日志都已发送到 `main_log.txt`，那么您可以使用以下命令计算每个集合调用的执行次数：
```
grep -a "NCCL INFO Broadcast" main_log.txt     | wc -l
2590
grep -a "NCCL INFO AllReduce" main_log.txt     | wc -l
5207
grep -a "NCCL INFO AllGather" main_log.txt     | wc -l
1849749
grep -a "NCCL INFO ReduceScatter" main_log.txt | wc -l
82850
```

最好先隔离训练的特定阶段，因为加载和保存将具有与训练迭代非常不同的模式。

所以我通常首先切分出一次迭代。例如，如果每次迭代日志都以 `iteration: ...` 开头，那么我首先会这样做：
```
csplit main_log.txt '/iteration: /' "{*}"
```
然后分析与迭代相对应的结果文件之一。默认情况下，它将被命名为 `xx02` 之类的东西。


## 有用的 NCCL 调试环境变量

以下环境变量在调试 NCCL 相关问题（如挂起和崩溃）时最有用。完整的列表可以在[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)找到。


### `NCCL_DEBUG`

这是调试网络问题最常用的环境变量。

值：
- `VERSION` - 在程序开始时打印 NCCL 版本。
- `WARN` - 每当任何 NCCL 调用出错时打印明确的错误消息。
- `INFO` - 打印调试信息
- `TRACE` - 在每次调用时打印可重放的跟踪信息。

例如：

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

这将转储大量与 NCCL 相关的调试信息，如果您发现报告了一些问题，则可以在线搜索这些信息。

当使用 `NCCL_DEBUG` 时，`NCCL_DEBUG_FILE` 应该非常有用，因为信息量很大，尤其是在使用许多节点时。



### `NCCL_DEBUG_FILE`

当使用 `NCCL_DEBUG` 环境变量时，将所有 NCCL 调试日志输出重定向到一个文件。

默认值为 `stdout`。当使用许多 GPU 时，将每个进程的调试信息保存到其自己的日志文件中会非常有用，可以像这样操作：

```
NCCL_DEBUG_FILE=/path/to/nccl-log.%h.%p.txt
```

- `%h` 被主机名替换
- `%p` 被进程 PID 替换。

如果您之后需要一次性分析数百个这样的文件，这里有一些有用的快捷方式：

- grep 查找特定匹配项，并打印找到它的文件和行号：

```
grep -n "Init COMPLETE" nccl-log*
```

- 显示所有 nccl 日志文件的 `tail -1`，后跟每个文件的名称

```
find . -name "nccl*" -exec sh -c 'echo "$(tail -1 "$1") ($1)"' _ {} \;
```



### `NCCL_DEBUG_SUBSYS`

`NCCL_DEBUG_SUBSYS` 与 `NCCL_DEBUG` 结合使用，告诉后者要显示哪些子系统。通常您不必指定此变量，但有时帮助您的开发人员可能会要求将输出限制为仅某些子系统，例如：

```
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING
```



### `NCCL_P2P_DISABLE`

禁用 P2P 通信 - 例如，如果有 NVLink，将不会使用它，因此性能会慢得多。通常您不希望这样，但在紧急情况下，这有时在调试期间会很有用。


### `NCCL_SOCKET_IFNAME`

如果您有多个网络接口并且想要选择一个特定的接口来使用，这个非常有用。

默认情况下，NCCL 将尝试使用最快的接口类型，通常是 `ib` (InfiniBand)。

但是，假设您想改用以太网接口，那么您可以使用以下命令覆盖：

```
NCCL_SOCKET_IFNAME=eth
```

这个环境变量有时可以用来调试连接问题，比如说，如果其中一个接口被防火墙了，而其他接口可能没有，可以尝试使用它们。或者，如果您不确定某个问题是与网络接口有关还是其他原因，那么测试其他接口有助于排除问题来自网络的可能性。
