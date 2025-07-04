---
title: pytorch
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/joyolwoj/
---
# 调试 PyTorch 程序


## 让节点互相通信

一旦您需要使用多个节点来扩展您的训练，例如，如果您想使用 DDP 来加快训练速度，您就必须让节点相互通信，以便通信集合可以将数据相互发送。这通常通过像 [NCCL](https://github.com/nVIDIA/nccl) 这样的通信库来完成。在我们的 DDP 示例中，在训练步骤结束时，所有 GPU 都必须执行一个 `all_reduce` 调用来同步所有等级的梯度。

在本节中，我们将讨论一个非常简单的案例，即只有 2 个节点（每个节点有 8 个 GPU）相互通信，然后可以轻松地扩展到任意数量的节点。假设这些节点的 IP 地址为 10.0.0.1 和 10.0.0.2。

一旦我们有了 IP 地址，我们就需要选择一个通信端口。

在 Unix 中有 64k 个端口。前 1k 个端口是为常用服务保留的，这样互联网上的任何计算机都可以连接到任何其他计算机，并提前知道要连接到哪个端口。例如，端口 22 是为 SSH 保留的。因此，每当您执行 `ssh example.com` 时，实际上程序会打开一个到 `example.com:22` 的连接。

由于有成千上万的服务，保留的 1k 端口是不够的，因此各种服务可以使用几乎任何端口。但不要害怕，当您在云或 HPC 上获得您的 Linux 盒子时，您不太可能有许多预装的服务会使用高编号的端口，因此大多数端口应该是可用的。

因此，我们选择端口 6000。

现在我们有 `10.0.0.1:6000` 和 `10.0.0.2:6000`，我们希望它们能够相互通信。

首先要做的是在两个节点上为传入和传出连接打开端口 `6000`。它可能已经打开，或者您可能需要阅读您特定设置的说明以了解如何打开给定端口。

以下是您可以用来测试端口 6000 是否已打开的多种方法。

```
telnet localhost:6000
nmap -p 6000 localhost
nc -zv localhost 6000
curl -v telnet://localhost:6000
```

这些大多数应该可以通过 `apt install` 或您的包管理器使用的任何方式获得。

让我们在这个例子中使用 `nmap`。如果我运行：

```
$ nmap -p 22 localhost
[...]
PORT   STATE SERVICE
22/tcp open  ssh
```
我们可以看到端口是打开的，并且它告诉我们哪个协议和服务被分配了，这是一个额外的好处。

现在我们运行：
```
$ nmap -p 6000 localhost
[...]

PORT     STATE  SERVICE
6000/tcp closed X11
```
在这里您可以看到端口 6000 是关闭的。

现在您了解了如何测试，您可以继续测试 `10.0.0.1:6000` 和 `10.0.0.2:6000`。

首先在终端 A 中 ssh 到第一个节点，并测试第二个节点上的端口 6000 是否已打开：

```
ssh 10.0.0.1
nmap -p 6000 10.0.0.2
```
如果一切正常，那么在终端 B 中 ssh 到第二个节点，并反向执行相同的检查：

```
ssh 10.0.0.2
nmap -p 6000 10.0.0.1
```

如果两个端口都打开，您现在可以使用此端口。如果一个或两个都关闭，您必须打开这些端口。由于大多数云都使用专有解决方案，因此只需在互联网上搜索“打开端口”和您的云提供商的名称即可。

接下来要理解的重要事情是，计算节点通常会有多个网络接口卡 (NIC)。您可以通过运行以下命令来发现这些接口：

```
$ sudo ifconfig
```

一个接口通常用于用户通过 ssh 连接到节点或用于各种其他与计算无关的服务 - 例如，发送电子邮件或下载一些数据。通常这个接口被称为 `eth0`，其中 `eth` 代表以太网，但它也可以有其他名称。

然后是节点间接口，可以是 Infiniband、EFA、OPA、HPE Slingshot 等。（[更多信息](../network#inter-node-networking)）。可以有一个或几十个这样的接口。

以下是 `ifconfig` 输出的一些示例：

```
$ sudo ifconfig
enp5s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        [...]
```
我删除了大部分输出，只显示了一些信息。这里的关键信息是 `inet` 后面列出的 IP 地址。在上面的例子中是 `10.0.0.23`。这是接口 `enp5s0` 的 IP 地址。

如果有另一个节点，它可能是 `10.0.0.24` 或 `10.0.0.21` 之类的——最后一个段是数字不同的那个。

我们再看一个例子：

```
$ sudo ifconfig
ib0     Link encap:UNSPEC  HWaddr 00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00
        inet addr:172.0.0.50  Bcast: 172.0.0.255  Mask:255.255.255.0
        [...]
```
这里的 `ib` 通常告诉我们它是一个 InfiniBand 卡，但实际上它也可以是任何其他供应商。例如，我见过 [OmniPath](../network#omni-path) 使用 `ib`。同样，`inet` 告诉我们这个接口的 IP 是 `172.0.0.50`。

如果你跟丢了，我们需要 IP 地址，以便我们可以测试在每个相关节点上 ip:port 是否打开。

最后，回到我们的 `10.0.0.1:6000` 和 `10.0.0.2:6000` 对，让我们使用 2 个终端进行 `all_reduce` 测试，我们选择 `10.0.0.1` 作为主主机来协调其他节点。
为了测试，我们将使用这个辅助调试程序 [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py)。

在终端 A 中：

```
$ ssh 10.0.0.1
$ python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py
```

在终端 B 中：

```
$ ssh 10.0.0.2
$ python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py
```

请注意，我在两种情况下都使用相同的 `--master_addr 10.0.0.1 --master_port 6000`，因为我们检查了端口 6000 是打开的，并且我们使用 `10.0.0.1` 作为协调主机。

这种从每个节点手动运行东西的方法很痛苦，所以有一些工具可以自动在多个节点上启动相同的命令。

**pdsh**

`pdsh` 就是这样一种解决方案 - 它就像 `ssh`，但会自动在多个节点上运行相同的命令：

```
PDSH_RCMD_TYPE=ssh pdsh -w 10.0.0.1,10.0.0.2 \
"python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py"
```

您可以看到我如何将 2 组命令折叠成 1 组。如果您有更多节点，只需添加更多节点作为 `-w` 参数即可。


**SLURM**

如果您使用 SLURM，几乎可以肯定的是，设置它的人已经为您打开了所有端口，所以它应该可以正常工作。但如果不行，本节中的信息应该有助于调试。

以下是您如何在 SLURM 中使用它。

```
#!/bin/bash
#SBATCH --job-name=test-nodes        # 名称
#SBATCH --nodes=2                    # 节点
#SBATCH --ntasks-per-node=1          # 关键 - 每个 dist 每个节点只有一个任务！
#SBATCH --cpus-per-task=10           # 每个任务的核心数
#SBATCH --gres=gpu:8                 # gpu 数量
#SBATCH --time 0:05:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
#
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torch-distributed-gpu-test.py'
```
如果您有超过 2 个节点，您只需要更改节点数，上面的脚本就会自动适用于任何数量的节点。


**MPI**:

另一种流行的方法是使用[消息传递接口 (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface)。有几种开源的实现可用。

要使用此工具，您首先要创建一个 `hostfile`，其中包含您的目标节点以及应在每个主机上运行的进程数。在本节的示例中，有 2 个节点，每个节点有 8 个 gpu，它将是：

```
$ cat hostfile
10.0.0.1:8
10.0.0.2:8
```
要运行，只需：
```
$ mpirun --hostfile  -np 16 -map-by ppr:8:node python my-program.py
```

请注意，我在这里使用了 `my-program.py`，因为 [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) 是为与 `torch.distributed.run`（也称为 `torchrun`）一起工作而编写的。使用 `mpirun`，您将必须检查您的特定实现，以查看它使用哪个环境变量来传递程序的秩，并用它替换 `LOCAL_RANK`，其余的应该基本相同。

细微差别：
- 您可能需要通过添加 `--mca btl_tcp_if_include 10.0.0.0/24` 来明确告诉它使用哪个接口以匹配我们的示例。如果您有许多网络接口，它可能会使用一个未打开的或只是错误的接口。
- 您也可以反过来排除一些接口。例如，假设您有 `docker0` 和 `lo` 接口 - 要排除它们，请添加 `--mca btl_tcp_if_exclude docker0,lo`。

`mpirun` 有大量的标志，我建议阅读它的手册页以获取更多信息。我的目的只是向您展示如何使用它。此外，不同的 `mpirun` 实现可能会使用不同的 CLI 选项。



### 解决多个节点之间的 Infiniband 连接问题

在 Azure 的一个情况下，我在一个共享子网上获得了 2 个节点，当我尝试运行 2 节点 NCCL 测试时：

```
NCCL_DEBUG=INFO python -u -m torch.distributed.run --nproc_per_node=1 --nnodes 2 --rdzv_endpoint 10.2.0.4:6000  --rdzv_backend c10d torch-distributed-gpu-test.py
```
我在调试消息中看到检测到了 Infiniband 接口：
```
node-2:5776:5898 [0] NCCL INFO NET/IB : Using [0]ibP111p0s0:1/IB [1]rdmaP1111p0s2:1/RoCE [RO]; OOB eth0:10.2.0.4<0>
```
但是连接随后会超时，并显示以下消息：
```
node-2:5776:5902 [0] transport/net_ib.cc:1296 NCCL WARN NET/IB : Got completion from peer 10.2.0.5<33092> with error 12, opcode 0, len
0, vendor err 129 (Recv)
node-2:5776:5902 [0] NCCL INFO transport/net.cc:1134 -> 6
node-2:5776:5902 [0] NCCL INFO proxy.cc:679 -> 6
node-2:5776:5902 [0] NCCL INFO proxy.cc:858 -> 6 [Proxy Thread]
```
并且什么也做不了。所以这里两个节点之间的以太网连接正常，但 IB 接口不正常。

这失败的原因可能有很多，但最可能的一个是当您在云上并且这两个节点没有被配置为它们的 IB 连接时。所以您的以太网节点间连接正常，但速度太慢。很可能您需要重新配置节点，以便将它们一起分配。例如，在 Azure 上，这意味着您必须在特殊的[可用性集](https://learn.microsoft.com/en-us/azure/virtual-machines/availability-set-overview?source=recommendations)中分配节点。

回到我们的案例研究，一旦节点被删除并在可用性集中重新创建，测试就可以正常工作了。

单个节点通常不用于节点间通信，云通常有集群的概念，集群是为将多个节点作为一个组进行分配而设计的，并且已经预先配置为可以协同工作。




## 使用 `node:rank` 为日志添加前缀，交错断言

在本节中，我们将在演示期间使用 `torchrun` (`torch.distributed.run`)，在本节末尾将列出其他启动器的类似解决方案。

当您有警告和回溯（或调试打印）时，在每行日志前加上其 `hostname:rank` 前缀会很有帮助，这可以通过向 `torchrun` 添加 `--role $(hostname -s): --tee 3` 来完成：

```
python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 1 --nproc_per_node 2 \
torch-distributed-gpu-test.py
```

现在每行日志都会以 `[hostname:rank]` 为前缀。

注意冒号很重要。

如果您在 SLURM 环境中，上面的命令行将变为：

```
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
--role $(hostname -s): --tee 3 \
torch-distributed-gpu-test.py'
```

当然，请根据需要调整您的环境变量，这只是一个例子。

重要！请注意，我正在使用传递给 `bash -c` 的单引号字符串命令。这样，`hostname -s` 命令会延迟到在每个节点上运行时才执行。如果您在上面使用双引号，`hostname -s` 将在启动节点上执行，然后所有节点都将获得相同的主机名作为前缀，这违背了使用这些标志的目的。因此，如果您使用双引号，则需要像这样重写上面的内容：

```
srun --jobid $SLURM_JOBID bash -c "python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank \$SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
--role \$(hostname -s): --tee 3 \
torch-distributed-gpu-test.py"
```

`$SLURM_PROCID` 也被转义了，因为它需要特定于每个节点，并且在主节点上启动 slurm 作业时是未知的。所以在这个版本的命令中有 2 个 `\$` 转义。

当分布式程序失败时，这种前缀功能也非常有用，因为它通常会导致交错的回溯，很难解释。因此，通过 `grep` 查找一个选择的 `node:rank` 字符串，现在可以重建真正的错误消息。

例如，如果您得到如下所示的回溯：

```
  File "/path/to/training/dataset.py", line 785, in __init__
  File "/path/to/training/dataset.py", line 785, in __init__
    if self.dataset_proba.sum() != 1:
AttributeError: 'list' object has no attribute 'sum'
    if self.dataset_proba.sum() != 1:
  File "/path/to/training/dataset.py", line 785, in __init__
  File "/path/to/training/dataset.py", line 785, in __init__
    if self.dataset_proba.sum() != 1:
    if self.dataset_proba.sum() != 1:
AttributeError: 'list' object has no attribute 'sum'
AttributeError: 'list' object has no attribute 'sum'
AttributeError: 'list' object has no attribute 'sum'
```

当它在 8 个节点上有几十个帧时，很难理解，但是上面的 `-tee` + `--role` 添加将生成：

```
[host1:0]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:1]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:0]    if self.dataset_proba.sum() != 1:
[host1:0]AttributeError: 'list' object has no attribute 'sum'
[host1:1]    if self.dataset_proba.sum() != 1:
[host1:2]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:3]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:3]    if self.dataset_proba.sum() != 1:
[host1:2]    if self.dataset_proba.sum() != 1:
[host1:1]AttributeError: 'list' object has no attribute 'sum'
[host1:2]AttributeError: 'list' object has no attribute 'sum'
[host1:3]AttributeError: 'list' object has no attribute 'sum'
```
你可以 `grep` 这个输出只找一个 `host:rank` 前缀，这会给我们：

```
$ grep "[host1:0]" log.txt
[host1:0]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:0]    if self.dataset_proba.sum() != 1:
[host1:0]AttributeError: 'list' object has no attribute 'sum'
```

瞧，你现在可以知道到底发生了什么。正如我之前提到的，那里很容易有一百到一千个交错的回溯行。

另外，如果您只有一个节点，您可以只传递 `-tee 3`，而不需要传递 `--role`。

如果 `hostname -s` 太长，但您的每个主机都有自己的序列号，例如：
```
[really-really-really-long-hostname-5:0]
[really-really-really-long-hostname-5:1]
[really-really-really-long-hostname-5:2]
```
您当然可以通过将 `hostname -s` 替换为 `hostname -s | tr -dc '0-9'` 来使其更短，这将导致更短的前缀：
```
[5:0]
[5:1]
[5:2]
```

当然，如果您正在进行调试打印，那么要解决这个问题，您可以使用 [`printflock`](./torch-distributed-hanging-solutions.md#good-old-print)。

以下是您如何使用其他启动器实现相同功能的方法：

- `srun` 在 SLURM 中：添加 `--label`
- `openmpi`：添加 `--tag-output`
- `accelerate`：您可以像在 `torchrun` 中一样传递相同的 `-tee` + `--role` 标志


## 处理异步 CUDA 错误

当使用 CUDA 时，失败的 pytorch 程序通常会产生一个毫无意义或无法操作的 python 回溯。这是因为 CUDA 的异步特性——当执行 CUDA 内核时，程序已经继续运行，当错误发生时，程序的上下文已经不存在了。异步功能是为了让事情更快，这样当 GPU 在处理一些 `matmul` 时，CPU 上的程序已经可以开始做其他事情了。

在其他时候，系统的某些部分实际上会告诉您它们无法生成正确的回溯，如以下错误所示：

```
[E ProcessGroupNCCL.cpp:414] Some NCCL operations have failed or timed out. Due to the
asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/
incomplete data. To avoid this inconsistency, we are taking the entire process down.
```

有几种解决方案。

如果故障是即时的，并且可以在 CPU 上重现（并非所有程序都可以在 CPU 上工作），只需在隐藏您的 GPU 后重新运行它。您可以这样做：

```
CUDA_VISIBLE_DEVICES="" python my-pytorch-program.py
```

环境变量 `CUDA_VISIBLE_DEVICES` 用于手动限制 GPU 对执行程序的可见性。因此，例如，如果您有 8 个 gpu，并且您想用前 4 个 gpu 运行 program1.py，用剩余的 2 个 gpu 运行 program2.py，您可以这样做：

```
CUDA_VISIBLE_DEVICES="0,1,2,3" python my-pytorch-program1.py
CUDA_VISIBLE_DEVICES="4,5,6,7" python my-pytorch-program2.py
```
第二个程序不会知道它没有使用 GPU 0-3。

但是在调试的情况下，我们通过设置 `CUDA_VISIBLE_DEVICES=""` 来隐藏所有 GPU。

现在程序在 CPU 上运行，您将得到一个非常好的回溯，并可以立即解决问题。

但是，当然，如果您的程序需要多个 GPU，这将无法工作。所以这里有另一个解决方案。

在设置此环境变量后重新运行您的程序：

```
CUDA_LAUNCH_BLOCKING=1 python my-pytorch-program.py
```

这个变量告诉 pytorch（或任何其他基于 CUDA 的程序）关闭其所有地方的异步特性，现在所有操作都将是同步的。所以当程序崩溃时，您现在应该会得到一个完美的回溯，并且您会确切地知道是什么困扰着您的程序。

理论上，启用此变量应该会让一切运行得非常慢，但实际上这真的取决于您的软件。我们用 `CUDA_LAUNCH_BLOCKING=1` 和 [`Megatron-Deepspeed`](https://github.com/bigscience-workshop/Megatron-DeepSpeed) 进行了整个 BLOOM-176B 训练，并且没有减速——我们必须使用它，因为没有它 pytorch 就会挂起，我们没有时间去弄清楚挂起的原因。

所以，是的，当你从异步切换到同步时，它通常可以隐藏一些微妙的竞争条件，所以有时挂起会像我上面分享的例子一样消失。所以用和不用这个标志来测量你的吞吐量，有时它实际上不仅有助于获得上下文回溯，而且实际上可以完全解决你的问题。

注意：当使用 `CUDA_LAUNCH_BLOCKING=1` 时，[NCCL==2.14.3 附带的 `pytorch==1.13` 会挂起](https://github.com/NVIDIA/nccl/issues/750)。所以不要在那个版本的 pytorch 中使用它。这个问题已在 `nccl>=2.17` 中修复，该版本应包含在 `pytorch==2.0` 中。




## 段错误和从核心文件中获取回溯

对于复杂的 pytorch 程序来说，段错误并生成核心文件并不少见。特别是如果您
正在使用像 NCCL 这样的复杂扩展。

核心文件是程序在低级别崩溃时生成的——例如，当使用 python 扩展时——例如 CUDA 内核或实际上任何用 C 或其他语言的变体直接编码并通过某些绑定 API 在 python 中可访问的库。段错误最常见的原因是当此类软件访问其未分配的内存时。例如，程序可能会尝试释放其未分配的内存。但可能还有许多其他原因。

当段错误事件发生时，Python 无能为力，因为 proverbial 地毯被从它脚下抽走了，所以它无法生成异常，甚至无法向输出写入任何内容。

在这种情况下，必须去分析导致段错误的 libC 级调用，幸运的是，这些调用保存在核心文件中。

如果您的程序崩溃了，您通常会找到一个看起来像这样的文件：`core-python-3097667-6`


在我们继续之前，请确保您已安装 `gdb`：
```
sudo apt-get install gdb
```

现在，请确保您知道用于运行崩溃程序的 python 可执行文件的路径。如果您有多个 python 环境，您必须首先激活正确的环境。否则 `gdb` 可能无法解压核心文件。

所以我通常会这样做：

```
conda activate my-env
gdb python core-python-3097667-6
```
- 根据您使用的环境调整 `my-env`，或者如果您使用其他方式来激活您的 python 环境，请使用该方式——也许您使用的是系统范围的 python，那么您就不需要激活任何东西。
- 将核心文件的名称调整为您获得的文件——可能有很多——然后选择最新的。

现在 `gdb` 会运行一会儿，然后给你一个提示，你在那里输入：`bt`。我们将在这里使用一个实际的核心文件：

```
(gdb) bt
#0  0x0000147539887a9f in raise () from /lib64/libc.so.6
#1  0x000014753985ae05 in abort () from /lib64/libc.so.6
#2  0x000014751b85a09b in __gnu_cxx::__verbose_terminate_handler() [clone .cold.1] () from /lib64/libstdc++.so.6
#3  0x000014751b86053c in __cxxabiv1::__terminate(void (*)()) () from /lib64/libstdc++.so.6
#4  0x000014751b860597 in std::terminate() () from /lib64/libstdc++.so.6
#5  0x000014751b86052e in std::rethrow_exception(std::__exception_ptr::exception_ptr) () from /lib64/libstdc++.so.6
#6  0x000014750bb007ef in c10d::ProcessGroupNCCL::WorkNCCL::handleNCCLGuard() ()
   from .../python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so
#7  0x000014750bb04c69 in c10d::ProcessGroupNCCL::workCleanupLoop() ()
   from.../python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so
#8  0x000014751b88cba3 in execute_native_thread_routine () from /lib64/libstdc++.so.6
#9  0x000014753a3901cf in start_thread () from /lib64/libpthread.so.0
#10 0x0000147539872dd3 in clone () from /lib64/libc.so.6
```

就这样。你怎么理解它呢？

嗯，你从堆栈的底部到顶部。你可以看出在 `libc` 中进行了一个 `clone` 调用，然后它调用了 `libpthread` 中的 `start_thread`，然后如果你继续看，你会发现在 torch 库中有一些调用，最后我们可以看到程序自己终止了，最终以 `libc` 的 `raise` 结束，它告诉 Linux 内核杀死程序并创建核心文件。

这不是一个容易理解的回溯。

脚注：是的，python 称之为 *traceback*，其他地方称之为 *backtrace* - 这很令人困惑，但它们或多或少是同一回事。

实际上，我不得不向 pytorch 开发人员寻求帮助，并收到了：

- PyTorch `ProcessGroup` 看门狗线程从 NCCL 捕获了一个异步错误
- 此错误是一个“未处理的系统错误”，在这种特殊情况下，它原来是一个 IB-OPA 错误
- `ProcessGroup` 的 `WorkCleanUp` 线程重新抛出了该错误，以便主进程会崩溃，并通知用户（否则此异步错误不会出现）

相信我，有时候即使你没有经验，回溯也可以给你足够的提示，让你知道应该去哪里进行故障排除。

但是不要害怕 - 大多数时候您不需要理解回溯。理想情况下，您只需将核心文件附加到您提交的 Issue 中。但它可能很容易有 5GB 大。所以试图帮助您的开发人员会要求您生成一个 `gdb` 回溯，现在您知道该怎么做了。

我没说这会很容易，我只是告诉你从哪里开始。

现在另一个有用的细节是，如今许多程序都运行多个线程。而 `bt` 只显示进程的主线程。但是，通常，当段错误发生时，查看进程中其他线程的位置会很有帮助。为此，您只需在 `(gdb)` 提示符下键入 2 个命令：

```
(gdb) thread apply all bt
(gdb) bt
```

这一次，您通常会得到一份巨大的报告，每个线程一个回溯。




## py-spy

这是一个非常有用的工具，用于分析挂起的程序。例如，当您遇到资源死锁或网络连接问题时。

您可以在[此处](./torch-distributed-hanging-solutions.md#py-spy)找到对此工具的详尽介绍。


## strace

与 [py-spy](./torch-distributed-hanging-solutions.md#py-spy) 类似，`strace` 是一个非常有用的工具，它可以在低级系统调用（例如 `libC` 等）上跟踪任何正在运行的应用程序。

例如，运行：
```
strace python -c "print('strace')"
```
您将看到上述程序运行时在系统调用级别上完成的所有操作。

但通常它更有用的是当你有一个卡住的程序，它让所有 CPU 核心都以 100% 的速度旋转，但什么也没发生，而你想看看它在做什么。在这种情况下，你只需像这样附加到正在运行的程序上：

```
strace --pid PID
```
其中，您可以从 `top` 或 `ps` 的输出中获取 PID。通常，我只是复制并粘贴消耗最多 CPU 的程序的 PID - `top` 通常会将其显示在列表的最顶部。

与 `py-spy` 一样，您可能需要 `sudo` 权限才能附加到已经运行的进程——这完全取决于您的系统设置。但是您可以像我在原始示例中展示的那样，始终使用 `strace` 启动程序。

让我们看一下 `strace python -c "print('strace')"` 输出的一个小子片段

```
write(1, "strace\n", 7strace
)                 = 7
```
在这里我们可以看到在文件描述符 `1` 上执行了一个写调用，它几乎总是 `stdout`（`stdin` 是 0，`stderr` 是 2）。

如果您不确定文件描述符指向什么，通常您可以从 `strace` 的输出本身看出来。但您也可以这样做：

```
ls -l /proc/PID/fd
```
其中 PID 是您正在尝试调查的当前正在运行的程序的 pid。

例如，当我在运行一个带有 gpu 的 pytest 测试时运行上面的命令，我得到了（部分输出）：
```
l-wx------ 1 stas stas 64 Mar  1 17:22 5 -> /dev/null
lr-x------ 1 stas stas 64 Mar  1 17:22 6 -> /dev/urandom
lrwx------ 1 stas stas 64 Mar  1 17:22 7 -> /dev/nvidiactl
lrwx------ 1 stas stas 64 Mar  1 17:22 8 -> /dev/nvidia0
lr-x------ 1 stas stas 64 Mar  1 17:22 9 -> /dev/nvidia-caps/nvidia-cap2
```
所以你可以看到一个设备 `/dev/null` 作为 FD（文件描述符）5 打开，`/dev/urandom` 作为 FD 6，等等。

现在让我们看另一个来自我们的 `strace` 运行的片段。

```
access("/etc/ld.so.preload", R_OK)      = -1 ENOENT (No such file or directory)
```
在这里，它试图查看文件 `/etc/ld.so.preload` 是否存在，但正如我们所见，它不存在 - 如果缺少某个共享库，这可能很有用 - 您可以看到它试图从哪里加载它。

我们再试一个：
```
openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libpthread.so.0", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\0\0\0\0\0\0\0\0"..., 832) = 832
newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=21448, ...}, AT_EMPTY_PATH) = 0
mmap(NULL, 16424, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f8028807000
mmap(0x7f8028808000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1000) = 0x7f8028808000
mmap(0x7f8028809000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f8028809000
mmap(0x7f802880a000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7f802880a000
close(3)
```
在这里我们可以看到它打开了 `/lib/x86_64-linux-gnu/libpthread.so.0` 并将其分配给 FD 3，然后它从 FD 3 读取 832 个字符（我们还可以看到第一个字符是 ELF - 这代表共享库格式），然后内存映射它并关闭该文件。

在下面的例子中，我们看到一个 python 缓存文件被打开，它的文件指针被移动到 0，然后它被读取并关闭。
```
openat(AT_FDCWD, "/home/stas/anaconda3/envs/py38-pt113/lib/python3.8/__pycache__/abc.cpython-38.pyc", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0664, st_size=5329, ...}) = 0
lseek(3, 0, SEEK_CUR)                   = 0
lseek(3, 0, SEEK_CUR)                   = 0
fstat(3, {st_mode=S_IFREG|0664, st_size=5329, ...}) = 0
brk(0x23bf000)                          = 0x23bf000
read(3, "U\r\r\n\0\0\0\0\24\216\177c\211\21\0\0\343\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"..., 5330) = 5329
read(3, "", 1)                          = 0
close(3)
```
需要注意的是，文件描述符是重复使用的，所以我们看到了两次相同的文件描述符 3，但每次它都打开了不同的文件。

例如，如果您的程序试图连接到互联网，您也可以从 `strace` 中看到这些调用，因为程序将从一个套接字文件描述符中读取。

所以让我们在一个从 HF hub 下载文件的程序上运行一个例子：
```
strace python -c 'import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1])' t5-small
```

以下是与本讨论相关的一些片段：
```
socket(AF_INET6, SOCK_STREAM|SOCK_CLOEXEC, IPPROTO_TCP) = 3
setsockopt(3, SOL_TCP, TCP_NODELAY, [1], 4) = 0
ioctl(3, FIONBIO, [1])                  = 0
connect(3, {sa_family=AF_INET6, sin6_port=htons(443), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1f18:147f:e850:e203:c458:10cd:fc3c
", &sin6_addr), sin6_scope_id=0}, 28) = -1 EINPROGRESS (Operation now in progress)
poll([{fd=3, events=POLLOUT|POLLERR}], 1, 10000) = 1 ([{fd=3, revents=POLLOUT}])
getsockopt(3, SOL_SOCKET, SO_ERROR, [0], [4]) = 0
[...]
write(3, "\26\3\3\0F\20\0\0BA\4\373m\244\16\354/\334\205\361j\225\356\202m*\305\332\275\251\17J"..., 126) = 126
read(3, 0x2f05c13, 5)                   = -1 EAGAIN (Resource temporarily unavailable)
poll([{fd=3, events=POLLIN}], 1, 9903)  = 1 ([{fd=3, revents=POLLIN}])
read(3, "\24\3\3\0\1", 5)               = 5
read(3, "\1", 1)                        = 1
read(3, "\26\3\3\0(", 5)                = 5
read(3, "\0\0\0\0\0\0\0\0\344\v\273\225`\4\24m\234~\371\332%l\364\254\34\3472<\0356s\313"..., 40) = 40
ioctl(3, FIONBIO, [1])                  = 0
poll([{fd=3, events=POLLOUT}], 1, 10000) = 1 ([{fd=3, revents=POLLOUT}])
write(3, "\27\3\3\1.\0\374$\361\217\337\377\264g\215\364\345\256\260\211$\326pkR\345\276,\321\221`-"..., 307) = 307
ioctl(3, FIONBIO, [1])                  = 0
read(3, 0x2ef7283, 5)                   = -1 EAGAIN (Resource temporarily unavailable)
poll([{fd=3, events=POLLIN}], 1, 10000) = 1 ([{fd=3, revents=POLLIN}])
```

您可以看到它再次使用 FD 3，但这次它打开的是一个 INET6 套接字而不是文件。您可以看到它然后连接到该套接字，轮询，并从中读取和写入。

使用这个工具还可以得出许多其他非常有用的理解。

顺便说一句，如果您不想上下滚动，也可以将输出保存到文件中：
```
strace -o strace.txt python -c "print('strace')"
```

现在，由于您可能想从一开始就跟踪程序，例如为了解决分布式文件系统上的某些竞争条件，您需要告诉它跟踪任何派生的进程。这就是 `-f` 标志的作用：


```
strace -o log.txt -f python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --tee 3 test.py
```

所以在这里我们启动 4 个进程，最终将在至少 5 个进程上运行 `strace` - 启动器加上 4 个进程（每个进程都可能产生更多的子进程）。

它会方便地在每行前加上程序的 pid，所以应该很容易区分哪个系统是由哪个进程创建的。

但如果您想为每个进程单独记录日志，请使用 `-ff` 代替 `-f`。

`strace` 手册页有大量其他有用的选项。


## 在多节点训练中对特定 rank 调用 pdb

一旦 pytorch 2.2 发布，您将拥有一个方便的新调试功能：

```
import torch.distributed as dist
[...]

def mycode(...):

   dist.breakpoint(0)

```

这与 `ForkedPdb`（下面）相同，但会自动在您选择的 rank 上为您设置断点 - 在上面的示例中是 rank0。只需确保在断点命中时立即调用 `up;;n` 即可进入您的正常代码。

以下是其底层实现：

```
import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    用于调试多进程代码的 PDB 子类
    建议见：https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def mycode():

    if dist.get_rank() == 0:
        ForkedPdb().set_trace()
    dist.barrier()

```
所以你也可以自己编写代码。

您也可以将 `ForkedPdb` 代码用于普通的分叉应用程序，减去 `dist` 调用。



## 不同设备上的浮点数学差异

重要的是要理解，根据执行浮点数学的设备，结果可能会有所不同。例如，在 CPU 和 GPU 上执行相同的浮点运算可能会导致不同的结果，同样，当使用两种不同的 GPU 架构时，尤其是在使用两种不同类型的加速器（例如 NVIDIA 与 AMD GPU）时更是如此。

以下是我在 11 代 Intel i7 CPU 和 NVIDIA A100 80GB (PCIe) GPU 上执行相同简单浮点数学运算时得到的差异示例：

```
import torch

def do_math(device):
    inv_freq = (10 ** (torch.arange(0, 10, device=device) / 10))
    print(f"{inv_freq[9]:.20f}")
    return inv_freq.cpu()

a = do_math(torch.device("cpu"))
b = do_math(torch.device("cuda"))

torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
```
当我们运行它时，我们得到 10 个元素中有 2 个不匹配：
```
7.94328212738037109375
7.94328308105468750000
[...]
AssertionError: 张量不相等！

不匹配的元素: 2 / 10 (20.0%)
最大绝对差: 9.5367431640625e-07 在索引 (9,)
最大相对差: 1.200604771156577e-07 在索引 (9,)
```


这是一个简单的低维示例，但实际上张量要大得多，通常最终会有更多的不匹配。

现在您可能会说 `1e-6` 的差异可以安全地忽略。只要这是最终结果，通常是这样。如果上面示例中的这个张量现在通过 100 层的 `matmul`，这个微小的差异将会复合并扩散，从而影响许多其他元素，最终结果与在另一类设备上执行的相同操作大不相同。

例如，请参阅此[讨论](https://github.com/deepspeedai/DeepSpeed/issues/4932) - 用户报告说，在进行 Llama-2-7b 推理时，他们得到的 logits 会根据模型的初始化方式而有很大差异。需要澄清的是，最初的讨论是关于 Deepspeed 可能是问题所在，但在后面的评论中，您可以看到它被简化为仅取决于模型的缓冲区是在哪个设备上初始化的。训练好的权重不是问题，它们是从检查点加载的，但是缓冲区在加载模型时是从头开始重新创建的，所以问题就出在这里。

小的变化通常不会产生太大的影响，但有时差异会很明显，就像在这个例子中，同一张图片在 CPU 和 MPS 设备上产生。

![](images/math-fp-discrepancy-outcome-lizard.png)

这张快照和评论来自这个 [PyTorch Issue 讨论串](https://github.com/pytorch/pytorch/issues/84936#issuecomment-1246084645)。

如果您好奇我从哪里获取这段代码 - 这是 [modeling_llama.py](https://github.com/huggingface/transformers/blob/3f69f415adcbdaedec154ba8eac220ef3276975d/src/transformers/models/llama/modeling_llama.py#L130) 中原始代码的简化版本：

```
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
```

</rewritten_file>