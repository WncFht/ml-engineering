---
title: torch-distributed-hanging-solutions
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/dqx8hu5n/
---
# 诊断多节点多 GPU Python 程序中的挂起和死锁

虽然本文中的方法论是在使用基于 pytorch 的多节点多 GPU 训练时开发的，但它们当然可以帮助解决任何多进程多节点 Python 程序的问题。

## 辅助工具

尝试使用以下脚本 [torch-distributed-gpu-test.py](torch-distributed-gpu-test.py) 来诊断情况。

这主要有助于发现与网络相关的问题。并且也能快速了解多 GPU 通信的工作原理。

对于与代码相关的问题，请阅读本文档的其余部分。

## 诊断多 GPU 挂起/死锁的方法

### py-spy

首先执行 `pip install py-spy`。

现在您可以使用以下命令附加到每个进程：

```
py-spy dump -n -p PID
```
它会告诉您进程挂起的位置（通常是 nccl 集合函数或 `barrier`）。

- `PID` 是挂起的 python 进程的进程 ID。
- `-n` 在您想查看用 C、C++ 等编写的 python 扩展的堆栈跟踪时很有用，因为程序可能会在其中一个扩展中挂起。
- 您可能需要在命令前添加 `sudo` - 更多详细信息请参阅[此说明](https://github.com/benfred/py-spy/blob/master/README.md#when-do-you-need-to-run-as-sudo)。

如果您没有 `sudo` 权限，您的系统管理员可能会为您执行此操作：
```
sudo echo 0 > /proc/sys/kernel/yama/ptrace_scope
```
这将允许您在不需要 `sudo` 的情况下运行 `py-spy`（和 `strace`）。请注意可能的[安全隐患](https://wiki.ubuntu.com/SecurityTeam/Roadmap/KernelHardening#ptrace_Protection) - 但通常如果您的计算节点无法从互联网访问，则风险较小。

要使此更改永久生效，请编辑 `/etc/sysctl.d/10-ptrace.conf` 并设置：
```
kernel.yama.ptrace_scope = 0
```

以下是 `py-spy dump` python 堆栈跟踪的示例：
```
Thread 835995 (active): "MainThread"
    broadcast (torch/distributed/distributed_c10d.py:1191)
    _aggregate_total_loss (deepspeed/runtime/pipe/engine.py:540)
    train_batch (deepspeed/runtime/pipe/engine.py:330)
    train_step (megatron/training.py:436)
    train (megatron/training.py:851)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)
```
第一行是程序卡住的地方。

如果挂起发生在 CPP 扩展内部，请添加 `--native` `py-spy`，它将显示任何非 python 代码。

#### 多进程 py-spy

现在，您如何为多个进程执行此操作。一个一个地做太慢了。所以我们一次性完成。

如果启动命令是 `python`，您可以这样做：
```
pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}
```

如果是 `deepspeed`：
```
pgrep -P $(pgrep -o deepspeed) | xargs -I {} py-spy dump --pid {}
```

对于 `accelerate`：
```
pgrep -P $(pgrep -o accelerate) | xargs -I {} py-spy dump --pid {}
```

你明白了。

这种特殊的方法只会分析主进程，而不会分析这些进程产生的各种其他子进程/线程。因此，如果您有 8 个 gpu 和 8 个进程，上述命令将生成 8 个堆栈跟踪。

然后，您可以将输出通过管道传输到这个附加的有用过滤器中：
```
pgrep -P $(pgrep -o deepspeed) | xargs -I {} py-spy dump --pid {} | grep -A5 MainThread
```
所以它会显示 `MainThread` 的每个回溯的前 5 个条目。

如果您有先前运行进程的僵尸进程，并且它们已失效且无法被杀死，您很可能需要切换到 `pgrep -n` 来 grep 最新的进程，而不是最旧的进程 (`pgrep -o`)。
```
pgrep -P $(pgrep -n deepspeed) | xargs -I {} py-spy dump --pid {}
```

在某些情况下，当添加一个额外的启动器包装器，比如说调用一个 `deepspeed` 启动器时，我会看到你最终会得到一个额外的 Python 父进程，所以你需要再添加一级 `pgrep -P`：
```
pgrep -P $(pgrep -P $(pgrep -n deepspeed)) | xargs -I {} py-spy dump --pid {}
```

如果您想要所有进程及其子进程，那么您只需运行：
```
pgrep -f python | xargs -I {} py-spy dump --pid {}
```
（并且如前所述，如果启动程序不是 `python`，请用其名称替换 `python`）


#### 通过 srun 进行多节点 py-spy

如果您有多个节点怎么办？

您当然可以交互式地 `ssh` 到每个节点并转储堆栈跟踪。

如果您正在使用 SLURM 环境，您可以使用 `srun` 在所有节点上为您执行此操作。

现在在另一个控制台中获取 `SLURM_JOBID`（或从 `salloc` 日志中获取）：
```
squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"
```

现在使用以下 `srun` 命令，并根据上面命令的结果调整 jobid 和 `SLURM_JOBID`：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```

注意：
- 必须为监视器 `srun` 使用 `--gres=gpu:0`，否则它将阻塞直到主 `srun`（运行训练的那个）退出。
- 每个节点都将生成其唯一的日志文件，名为 `trace-nodename.out` - 因此这将有助于识别哪个节点有问题。如果您希望所有内容都转储到 stdout，可以删除 `--output=trace-%N.out`
- 在某些 SLURM 版本中，您可能还需要添加 `--overlap`
- 在某些 SLURM 版本中，jobid 可能与 `squeue` 中报告的不匹配，因此您必须从您要"附加"到的作业的日志中获取正确的 `SLURM_JOB_ID` - 即您分配了 GPU 的 `srun` 作业。
- 有时 `bash` 不起作用，但 `sh` 可以。我想这与 `source` 了哪些点文件有关
- 您可能还需要激活一个自定义的 python 环境，您可以像这样操作：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'conda activate myenvname; ps auxc | ... ' || echo "failed"
```
或者您可以在 `~/.bashrc` 或您决定使用的任何 shell 的 rc 文件中执行此操作。

如前所述，如果您只想要主进程，则应使用此命令：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}' || echo "failed"
```
如果需要，如上面的多 GPU 部分所述，调整 `python`。

之前的较长命令将为所有 python 进程提供跟踪。

如果您什么也没得到，请从基本的调试开始，例如：

```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'date'
```
一旦您知道您正在与所有节点通信，那么您就可以逐步展开调用深度，如下所示：

```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'date'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -o python'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) '
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}'
```
并且在每个阶段检查输出是否有意义——例如，在第 2 次和第 3 次调用时，您应该得到进程的 PID。



#### 通过 pdsh 进行多节点 py-spy

`pdsh` 似乎是一个很好的易于使用的工具，可以完成多个节点上的远程工作。比如说，您正在两个主机名为 `nodename-5` 和 `nodename-8` 的节点上运行，那么您可以通过以下命令快速测试远程执行是否正常，只需获取这些主机上的 `date` 即可：
```
$ PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] "date"
nodename-5: Wed Oct 25 04:32:43 UTC 2023
nodename-8: Wed Oct 25 04:32:45 UTC 2023
```

脚注：`pdsh` 应该可以通过常规的操作系统包安装程序获得。

一旦您测试 `date` 可以正常工作，就可以开始使用 `py-spy` 了。

要在所有 python 子进程上执行 `py-spy`，应该是：
```
PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}'
```
但是由于您很可能需要运行 `~/.bashrc`，因此需要将其克隆到 `~/.pdshrc`，将该克隆减少到需要运行的内容（例如，修改 `PATH`、`activate conda`），然后 `source` 它，例如：

```
PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] 'source ~/.pdshrc; pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}"'
```

您需要启动脚本的原因是，通常 `~/.bashrc` 以以下内容开头：
```
# 如果不是交互式运行，则不执行任何操作
case $- in
    *i*) ;;
      *) return;;
esac
```
因此，当您运行此类非交互式工作流时，Bash 通常不会处理其 `~/.bashrc`（提前退出），因此任何依赖于此启动脚本的内容都将不起作用。因此，您可以删除上面的非交互式退出代码，或者将 `~/.bashrc` 分叉到一个仅包含远程命令成功所需内容的启动文件中。


脚注：`~/.pdshrc` 没有什么特别之处——任何其他名称都可以，因为您是手动 `source` 它的。


如果您的系统没有像前面几节所述那样在没有 `sudo` 的情况下运行 `py-spy`，您将需要类似以下内容：

```
PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] 'sudo bash -c "source ~/.pdshrc; pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}"'
```

当然，您可能需要编辑 `pgrep` 部分以缩小您要监视的进程范围。

此外，为了避免被提示：
```
您确定要继续连接吗 (yes/no/[fingerprint])?
```
对于每个您尚未登录的新节点，您可以使用以下命令禁用此检查：
```
echo "Host *" >> ~/.ssh/config
echo "  StrictHostKeyChecking no" >> ~/.ssh/config
```
在这里，我假设您在一个隔离的集群上，因此您不必担心安全问题，因此绕过此类检查很可能是可以的。



#### 通过 ds_ssh 进行多节点 py-spy

这是另一种方法，但请确保先阅读上面的 `pdsh` 部分。

以下说明需要 `pip install deepspeed`。

在一个 SLURM 环境中，我也尝试通过 `ds_ssh` 使用 `pdsh`，但不知何故我无法远程运行 `py-spy` - 主要问题是远程 `ssh` 命令没有提供与我通过 `ssh` 交互式登录时相同的环境。但是，如果您在计算节点上有 `sudo` 权限，那么您可以这样做：

首先准备 `hostfile`：
```
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=8 if $slots==0; # 解决方法，8 个 gpu 的机器
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile
```
将 `$slots` 调整为每个节点的 gpu 数量。如果您的 `scontrol` 产生不同的输出，您可能需要调整此脚本。

现在在所有参与的节点上运行 `py-spy` 提取命令：
```
ds_ssh -f hostfile "source ~/.pdshrc; ps aux | grep python | grep -v grep | grep `whoami` | awk '{print \$2}' | xargs -I {} sudo py-spy dump --pid {} "
```

注意：
- 将您可能需要运行的任何初始化代码放入 `~/.pdshrc` 中。如果您不需要任何东西，可以从命令行中删除 `source ~/.pdshrc;`。
- 如果您还没有安装 `ds_ssh`，当您执行 `pip install deepspeed` 时会安装它。
- 如果您收到 `rcmd: socket: Permission denied` 错误，您可能需要 `export PDSH_RCMD_TYPE=ssh`




### 网络级挂起

挂起可能发生在网络级别。`NCCL_DEBUG=INFO` 在这里可以提供帮助。

使用 `NCCL_DEBUG=INFO` 环境变量运行脚本，并尝试研究输出以查找明显的错误。它会告诉您正在使用哪个设备，例如：
```
DeepWhite:21288:21288 [0] NCCL INFO NET/Socket : Using [0]enp67s0:192.168.50.21<0>
```
所以它正在使用接口 `enp67s0`，地址为 `192.168.50.21`

您的 `192.168.50.21` 是否被防火墙了？或者它是否是某种配置错误的网络设备？

如果使用环回设备 `127.0.0.1` 是否可以工作？
```
NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=lo python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 torch-distributed-gpu-test.py
```

如果没有，请通过 `ifconfig` 查看您还有哪些本地网络设备 - 如果有的话，请尝试使用它们而不是 `lo`。

在上面的例子中，它当前正在使用 `enp67s0`。


### 隔离有问题的 GPU

您也可以尝试看看是否只有某些 GPU 会失败

例如，如果使用前 2 个或后 2 个 gpu 是否可以工作：

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```
然后是第二对：
```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```


### python `trace`

现在，当训练不仅仅是挂起，而且挂起的进程停止响应时会发生什么？例如，当出现严重的硬件问题时会发生这种情况。但是，如果它是经常性的，并且 `py-spy` 在这里帮不上忙，因为它无法附加到一个没有响应的进程上。

所以下一个想法是像 `strace(1)` 那样跟踪所有调用，我研究了 python 调用跟踪工具，并发现 python 有一个 `trace` 子系统。

以下代码将跟踪所有 python 调用，并将它们记录到控制台和一个专用的每个进程的日志文件中，通过我添加的一个自定义 `Tee` 模块。

这可以帮助理解某些进程在何处停止响应，因为我们将拥有最后一个调用和在它变得无响应之前的所有先前调用的日志。

```
$ cat train.py
[...]

def main():
    # [...]
    train()

import re
class Tee:
    """
    一个辅助类，用于将 print 的输出同时输出到文件。
    用法：
    sys.stdout = Tee(filename)
    """

    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, "a")

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

if __name__ == "__main__":

    import sys
    import trace
    import socket
    import os

    # 启用跟踪
    if 0:
        cwd = os.path.realpath('.')
        pid = os.getpid()
        hostname = socket.gethostname()
        local_rank = int(os.environ["LOCAL_RANK"])
        trace_output_file = f"{cwd}/trace-{hostname}-{local_rank}-{pid}.txt"

        # 创建一个 Trace 对象，告诉它要忽略什么，以及是否
        # 进行跟踪或行计数或两者兼有。
        tracer = trace.Trace(
            ignoredirs=[sys.prefix, sys.exec_prefix],
            trace=1,
            count=1,
            timing=True,
        )

        # 使用给定的跟踪器运行新命令
        sys.stdout = Tee(trace_output_file)
        tracer.run('main()')
    else:
        main()

```

此代码除了通过将 `if 0` 更改为 `if 1` 来启用跟踪外，不需要任何特殊处理。

如果您不设置 `ignoredirs`，这将转储所有 python 调用。这意味着预计会记录大量 GB 的数据，特别是如果您有数百个 GPU。

当然，您不必从 `main` 开始跟踪 - 如果您怀疑某个特定区域，您可以从那里开始跟踪，这样会快得多，并且需要保存的数据也更少。

我希望我能告诉 `trace` 要跟踪哪些包，但可惜它只支持要忽略的目录，这更难设置，因此您最终会得到比需要的多得多的数据。但这仍然是调试挂起进程的超级有用的工具。

此外，您的代码现在会运行得慢得多，并且您跟踪的包越多，它就会变得越慢。

#### NicerTrace

由于 `Trace` 在调试复杂的多节点多小时运行崩溃时证明其可用性非常有限，我开始着手开发一个更好版本的 `trace` python 模块。

您可以在这里找到它：[NicerTrace](./NicerTrace.py)

我在构造函数中添加了多个附加标志，并使输出更有用。您将在同一个文件中找到一个完整的工作示例，只需运行：

```
python trace/NicerTrace.py
```
你应该会看到：

```
        trace/NicerTrace.py:1 <module>
0:00:00 <string>:     1:         trace/NicerTrace.py:185 main
0:00:00 NicerTrace.py:   186:     img = Image.new("RGB", (4, 4))
        PIL.Image:2896 new
0:00:00 Image.py:  2912:     _check_size(size)
        PIL.Image:2875 _check_size
0:00:00 Image.py:  2883:     if not isinstance(size, (list, tuple)):
0:00:00 Image.py:  2886:     if len(size) != 2:
0:00:00 Image.py:  2889:     if size[0] < 0 or size[1] < 0:
```
正如您将在示例中看到的，我设置了：

```
            packages_to_include=["PIL"],
```
所以它将跟踪 `PIL` 以及任何不在 `site-packages` 下的东西。如果您需要跟踪另一个包，只需将其添加到该列表中即可。

这是一个非常新的正在进行中的包，所以随着我们试图让它帮助我们解决一个非常复杂的崩溃情况，它正在不断发展。


#### 使用生成的跟踪文件

当每个节点级别的跟踪文件生成后，以下内容可能有助于快速分析情况：


- grep 查找特定匹配项，并打印找到它的文件和行号：

```
grep -n "backward" trace*
```

- 显示所有跟踪文件的 `tail -1`，后跟每个文件的名称：

```
find . -name "trace*" -exec sh -c 'echo "$1: $(tail -3 "$1")"' _ {} \;
```

- 或者与上述类似，但打印 5 个最后一行，并带有前导文件名和一些垂直空白，以便于阅读：

```
find . -name "trace*" -exec sh -c 'echo; echo $1; echo "$(tail -5 "$1")"' _ {} \;
```

- 计算 grep 在每个文件中匹配给定模式的次数，并打印匹配的文件（在此示例中匹配模式 `backward`）：

```
find . -name "trace*" -exec sh -c 'echo "$1: $(grep "backward" $1 | wc -l)"' _ {} \;
```


### 老掉牙的 `print`

现在，一旦你发现挂起发生在哪里，为了进一步理解为什么会发生这种情况，理想情况下会使用调试器，但通常情况下，调试多进程（多节点）问题可能非常困难。

在这种情况下，一个老掉牙的 `print` 就派上用场了。你只需要在挂起的地方的调用之前添加一些调试打印，这些打印将有助于理解导致死锁的原因。例如，某个 `barrier` 丢失了，一个或几个进程跳过了一些代码，而其余的进程仍然阻塞等待每个人发送一些数据（例如在像 `gather` 或 `reduce` 这样的 NCCL 集合函数中）。

当然，您希望在每个打印前加上进程的秩，以便您能区分哪个是哪个。例如：

```
import torch.distributed as dist
print(f"{dist.get_rank()}: passed stage 0")
```

您会很快发现，如果您有多个 GPU，这些打印会严重交错，您将很难理解调试数据。所以我们来解决这个问题。我们将用一个自定义版本的 `print` 来覆盖它，但它使用 `flock` 来确保一次只有一个进程可以写入 stdout。

辅助模块 `printflock.py` 包含在[此处](../training/tools/printflock.py)。要激活它，只需在您要调试的模块顶部运行此命令：

```
from printflock import printflock as print
```

现在您在该模块中的所有 `print` 调用都将神奇地不再交错。您当然也可以直接使用 `printflock`：

```
from printflock import printflock
import torch.distributed as dist
printflock(f"{dist.get_rank()}: passed stage 0")
```

### 核心文件

如果挂起发生在非 python 代码中，并且由于某种原因 `py-spy --native` 不够用，您可以让挂起的程序转储一个核心文件，这可以通过以下方法之一完成：

```
gcore <pid>
kill -ABRT <pid>
```

然后您可以像[此处](pytorch.md#segfaults-and-getting-a-backtrace-from-a-core-file)解释的那样内省核心文件。

如果您没有得到核心文件转储，您需要配置您的系统以允许这样做，并指定核心文件应该保存到哪里。

要确保文件在 bash 中被转储，请运行（其他 shell 可能使用不同的命令）：
```
ulimit -c unlimited
```
要使其持久化，请运行：
```
echo '* soft core unlimited' >> /etc/security/limits.conf
```


在像 Ubuntu 这样的某些系统上，核心文件被 `apport` 劫持，请检查 `/proc/sys/kernel/core_pattern` 的内容以查看它们被发送到哪里。您可以使用以下命令覆盖它们被发送到的位置：

```
sudo sysctl -w kernel.core_pattern=/tmp/core-%e.%p.%h.%t
```

如果您愿意，可以更改目录，但请确保运行程序的用户可以写入该目录。
要使此更改永久生效，请编辑 `/etc/sysctl.conf` 并添加 `kernel.core_pattern=/tmp/core-%e.%p.%h.%t`（如果它已经存在，则修改它）。

脚注：有关所有可用模板，请参阅 `man core`

如果在 Ubuntu 上，默认情况下它会将核心文件发送到 `apport`，它可能会将核心保存到 `/var/lib/apport/coredump` 或
`/var/crash`。但您可以如上所述进行更改。

快速测试您的设置是否可以生成核心文件的方法是：
```
sleep 10 &
killall -SIGSEGV sleep
```

通常 `SIGSEGV` 不建议用于诊断挂起程序的真实情况，因为 `SIGSEGV` 很可能会启动一个信号处理程序，但对于这个测试来说，它已经足够好了。


### 代码循环

在挂起的情况下，调试代码循环可能很棘手。如果您有类似以下的代码：

```
for i, d in enumerate(data):
    some_hanging_call(d)
```

一个进程可能在第一次迭代中挂起，而另一个进程在第二次迭代中挂起，这会使事情非常混乱。但是堆栈跟踪不会给出这样的指示，因为行号是相同的，即使进程在代码进展方面不在同一个位置。

在这种情况下，将循环展开为：
```
d_iter = iter(data)
some_hanging_call(next(d_iter)
some_hanging_call(next(d_iter)
```
现在当你运行 `py-spy` 时，行号将是正确的。挂在第一次迭代中的进程将报告第一个 `some_hanging_call`，而在第二次迭代中的进程将报告第二个调用 - 因为现在每个都有自己的行。


## 硬件特定问题

### AMD/ROCm 在启用 IOMMU 时挂起或变慢

AMD Instinct 用户可能需要[禁用 IOMMU](https://github.com/stas00/toolbox/issues/1#issuecomment-1076830400) 或将其设置为：
```
GRUB_CMDLINE_LINUX_DEFAULT="iommu=soft"
```
在 `/etc/default/grub` 中（grub 配置文件可能因操作系统而异）。

禁用是 `GRUB_CMDLINE_LINUX="amd_iommu=off"`
