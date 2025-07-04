---
title: users
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/cphxnh8t/
---
# SLURM 用户指南

## 快速入门

只需复制此 [example.slurm](./example.slurm) 并根据您的需求进行调整。

## SLURM 分区

在本文档中，我们将使用一个示例设置，其中包含以下 2 个集群名称：

- `dev`
- `prod`

要查找节点的名称及其可用性，请使用：

```
sinfo -p dev
sinfo -p prod
```

Slurm 配置位于 `/opt/slurm/etc/slurm.conf`。

要查看所有分区的配置：

```
scontrol show partition
```

## 资源授予等待时间

```
squeue -u `whoami` --start
```
将显示任何待处理作业计划何时开始。

如果其他人在预订结束前取消预订，它们可能会提前开始。



## 通过依赖关系请求分配

当一个或多个当前计划的作业结束时（无论它是否仍在运行或尚未开始），要安排一个新作业，请使用依赖机制，通过告诉 `sbatch` 在当前运行的作业成功后启动新作业，使用：

```
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID tr1-13B-round1.slurm
```

使用 `--dependency` 可能会比使用 `--begin` 导致更短的等待时间，因为如果传递给 `--begin` 的时间允许自上一个作业停止以来有几分钟的延迟，调度程序可能已经开始了一些其他作业，即使它们的优先级低于我们的作业。这是因为调度程序会忽略任何带有 `--begin` 的作业，直到指定的时间到来。


## 在预定时间进行分配

要将分配推迟到给定时间，请使用：
```
salloc --begin HH:MM MM/DD/YY
```

`sbatch` 也是如此。

它只会在请求的时间将作业放入队列，就好像您此时执行此命令一样。如果那时有可用资源，则会立即分配。否则，它将被排队。

有时相对开始时间很有用。也可以使用其他格式。示例：

```
--begin now+2hours
--begin=16:00
--begin=now+1hour
--begin=now+60  # 默认秒
--begin=2010-01-20T12:34:00
```

时间单位可以是 `seconds` (默认)、`minutes`、`hours`、`days` 或 `weeks`：

## 无时间限制的预分配节点（60分钟）

这对于运行重复的交互式实验非常有用——这样就不需要等待分配的进展。因此，策略是为一段延长的时间一次性分配资源，然后使用此分配运行交互式 `srun` 作业。

将 `--time` 设置为期望的时间窗口（例如 6 小时）：
```
salloc --partition=dev --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --gres=gpu:8 --time=6:00:00 bash
salloc: Pending job allocation 1732778
salloc: job 1732778 queued and waiting for resources
salloc: job 1732778 has been allocated resources
salloc: Granted job allocation 1732778
```
现在使用这个保留的节点多次运行一个作业，通过传递 `salloc` 的作业 ID：
```
srun --jobid $SLURM_JOBID --pty bash
```
如果从通过 `salloc` 启动的 `bash` 内部运行。但它也可以从另一个 shell 启动，但那时需要明确设置 `--jobid`。

如果这个 `srun` 作业超时或手动退出，您可以在同一个保留节点上再次重新启动它。

`srun` 当然可以直接调用真正的训练命令，而不仅仅是 `bash`。

重要提示：当分配单个节点时，分配的 shell 不在节点上（它从来都不是）。您必须找出节点的名称（在分配时报告或通过 `squeue` 和 `ssh` 到它）。

完成后，要释放资源，请退出在 `salloc` 中启动的 shell 或 `scancel JOBID`。

这个保留的节点在分配的整个时间内都会计入小时使用量，所以一旦用完就立即释放。

实际上，如果这只是一个节点，那么首先不使用 `salloc` 而使用 `srun` 会更容易，它既会分配又会给你使用的 shell：
```
srun --pty --partition=dev --nodes=1 --ntasks=1 --cpus-per-task=96 --gres=gpu:8 --time=60 bash
```

## 超线程

默认情况下，如果 CPU 启用了[超线程](https://en.wikipedia.org/wiki/Hyper-threading) (HT)，SLURM 将会使用它。如果您不想使用 HT，则必须指定 `--hint=nomultithread`。

脚注：HT 是英特尔特有的命名，通用概念是同步多线程 (SMT)

例如，对于一个每个节点有 2 个 CPU，每个 CPU 有 24 个核心和 2 个超线程的集群，总共有 96 个超线程或 48 个 CPU 核心可用。因此，要充分利用该节点，您需要配置：

```
#SBATCH --cpus-per-task=96
```
或者如果您不想使用 HT：
```
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
```

最后一种方法将为每个核心分配一个线程，在这种模式下，只有 48 个 CPU 核心可用。

请注意，根据您的应用程序，这两种模式之间的性能可能会有很大差异。因此，请尝试两种模式，看看哪一种能给您带来更好的结果。

在像 AWS 这样的一些设置上，当使用 `--hint=nomultithread` 时，all-reduce 吞吐量会急剧下降！而在其他一些设置上，情况正好相反——没有 HT 的吞吐量更差！

要检查您的实例是否启用了 HT，请运行：

```
$ lscpu | grep Thread
Thread(s) per core: 2
```

如果是 `2`，则表示启用了 HT，如果是 `1`，则表示未启用。


## 重用分配

例如，当希望在相同的节点分配上运行各种作业时。

在一个 shell 中：
```
salloc --partition=prod --nodes=16 --ntasks=16 --cpus-per-task=96 --gres=gpu:8 --time=3:00:00 bash
echo $SLURM_JOBID
```

在另一个 shell 中：
```
export SLURM_JOBID=<上面得到的 JOB ID>
srun --jobid=$SLURM_JOBID ...
```

您可能需要设置 `--gres=gpu:0` 以在节点上运行一些诊断作业。例如，让我们检查所有主机的共享内存：
```
srun --jobid 631078 --gres=gpu:0 bash -c 'echo $(hostname) $(df -h | grep shm)'
```


## 特定节点选择

要排除特定节点（当您知道某些节点已损坏但仍处于 IDLE 状态时很有用）：

```
sbatch --exclude nodeA,nodeB
```
或通过： `#SBATCH --exclude ...`

要使用特定节点：

```
sbatch --nodelist= nodeA,nodeB
```
也可以使用短的 `-w` 代替 `--nodelist`


管理员也可以在 `slurm.conf` 中定义一个 `feature=example`，然后用户可以通过 `--constraint=example` 请求该节点子集。


## 通知正在运行的作业完成

由于每个 SLURM 运行都有一个有限的时间跨度，它可以被配置为在分配的时间结束前所需的时间量向程序发送一个选择的信号。
```
--signal=[[R][B]:]<sig_num>[@<sig_time>]
```
TODO：需要对此进行实验，以帮助训练优雅地完成，而不是在保存最后一个检查点后开始新的周期。



## 详细的作业信息

虽然最有用的信息预设在各种 `SLURM_*` 环境变量中，但有时会缺少一些信息。在这种情况下，请使用：
```
scontrol show -d job $SLURM_JOB_ID
```
然后解析出需要的内容。


对于一个已完成运行的作业，请使用：
```
sacct -j JOBID
```

此命令对于发现您在该分配上是否已有任何 `srun` 作业正在运行（包括那些已完成或已取消的作业）也很有用。例如，您可以通过 `scancel <jobid>.<step-id>` 终止某个失控的 `srun` 步骤，并通过上述命令找到该 `<step-id>`。即使您取消了所有步骤作业，主作业（如果是交互式作业）也将继续运行。

查看更多详情：
```
sacct -ojobid,start,end,state,exitcode --format nodelist%300  -j JOBID
sacct -j JOBID --long
```

或者，要查看所有作业及其子步骤，同时将列表限制为特定分区且仅限于您自己的用户：

```
sacct -u `whoami` --partition=dev  -ojobid,start,end,state,exitcode --format nodelist%300
sacct -u `whoami` --partition=prod -ojobid,start,end,state,exitcode --format nodelist%300
```

要查看特定作业是如何启动的以及其所有 `srun` 子步骤的命令行：
```
sacct -j JOBID -o submitline -P
```

## 显示作业


只显示我的工作：
```
squeue -u `whoami`
```

按作业 ID 显示作业：
```
squeue -j JOBID
```

显示特定分区的作业：
```
squeue --partition=dev
```


## 别名

方便的别名

```
alias myjobs='squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"'
alias groupjobs='squeue -u foo,bar,tar -o "%.16i %u %9P %26j %.8T %.10M %.8l %.6D %.20S %R"'
alias myjobs-pending="squeue -u `whoami` --start"
alias idle-nodes="sinfo -p prod -o '%A'"
```




## 僵尸进程

如果节点上留下了任何僵尸进程，发送一个命令来杀死它们。

```
srun pkill python
```

## 详细访问 SLURM 账户

`sacct` 显示 Slurm 作业记账日志或 Slurm 数据库中所有作业和作业步骤的记账数据。

所以这是一个分析过去事件的好工具。

例如，要查看最近的 gpu 作业使用了哪些节点：

```
sacct -u `whoami` --partition=dev -ojobid,start,end,state,exitcode --format nodelist%300
```

这里的 `%300` 告诉它为输出使用 300 个字符的宽度，这样它就不会被截断。

有关更多字段和信息字段，请参阅 `man sacct`。



## 队列


### 取消作业

取消作业：
```
scancel [jobid]
```

取消所有你的工作：
```
scancel -u <userid>
```

取消特定分区上的所有作业：
```
scancel -u <userid> -p <partition>
```

### 提示

- 如果您看到 `salloc` 分配的交互式作业计划在您需要的时间之后很久才运行，请尝试取消该作业并请求更短的时间——通常对于更短的时间分配，可能会有更近的时间窗口。


## 日志记录

如果我们需要将日志分离到每个节点的不同日志文件中，请添加 `%N`（用于短主机名），以便我们有：

```
#SBATCH --output=%x-%j-%N.out
```

这样我们就可以判断某个特定节点是否行为异常——例如，GPU 是否损坏。这是因为目前 pytorch 不会记录是哪个节点/GPU 等级触发了异常。

希望它能成为 pytorch 的内置功能 https://github.com/pytorch/pytorch/issues/63174，这样就不需要在日志记录方面把事情搞复杂了。


## 显示节点状态
```
sinfo -p PARTITION
```

非常有用的命令是：
```
sinfo -s
```

然后查看主要统计数据，例如：

```
NODES(A/I/O/T) "allocated/idle/other/total".
597/0/15/612
```
所以这里 612 个节点中有 597 个被分配了。0 个空闲，15 个因其他原因不可用。

```
sinfo -p gpu_p1 -o "%A"
```

给出：
```
NODES(A/I)
236/24
```

所以你可以看到在 4x v100-32g 分区（`gpu_p1`）上是否有任何可用的节点。

要检查特定分区：

```
sinfo -p gpu_p1 -o "%A"
```

有关哪个分区是哪个，请参阅本文档顶部的表格。


### sinfo 状态


- idle: 无作业运行
- alloc: 节点被分配给当前正在执行的作业
- mix: 节点的部分 CPU 已分配，而其他 CPU 则处于空闲状态
- drain: 由于管理原因，节点不可用
- drng: 节点正在运行一个作业，但在完成后由于管理原因将不可用


### 节点状态码

节点状态后面可能跟一个有特殊含义的单个字符。它是以下之一：

- `*`: 节点当前没有响应，不会分配任何新工作。如果节点仍然没有响应，它将被置于 DOWN 状态（除了 COMPLETING、DRAINED、DRAINING、FAIL、FAILING 节点）。
- `~`: 节点当前已关机。
- `#`: 节点当前正在启动或配置中。
- `!`: 节点待关机。
- `%`: 节点当前正在关机。
- `$`：节点当前处于一个标志值为"维护"的预留中。
- `@`: 节点等待重启。
- `^`: 节点重启已发出。
- `-`：该节点由回填调度程序计划用于更高优先级的作业。

### 作业状态码

- `CD` | 已完成：作业已成功完成。
- `CG` | 正在完成：作业正在结束，但一些进程仍在活动。
- `F` | 失败：作业以非零退出代码终止并执行失败。
- `PD` | 待处理：作业正在等待资源分配。它最终会运行。
- `PR` | 已抢占：作业因被另一个作业抢占而终止。
- `R` | 运行中：作业当前已分配给一个节点并正在运行。
- `S` | 已暂停：一个正在运行的作业已被停止，其核心已释放给其他作业。
- `ST` | 已停止：一个正在运行的作业已被停止，其核心被保留。


### 排空节点

要查看所有排空的节点以及排空原因（编辑 `%50E` 以使原因字段更长/更短）
```
% sinfo -R -o "%50E %12U %19H %6t %N"
```

或者如果您想简短一些，就只用 `-R`：

```
% sinfo -R
```



## 作业数组


要运行一系列作业，以便当前运行的作业在 20 小时内结束后立即安排下一个 slurm 作业，我们使用作业数组。

让我们从 10 个这样的工作开始：

```
sbatch --array=1-10%1 array-test.slurm
```

`%1` 将此作业数组中同时运行的任务数限制为 1。没有它，它将尝试一次运行所有作业，这有时可能是我们想要的（在这种情况下删除 %1），但在训练时，我们需要一次一个作业。

或者，像往常一样，这个参数可以是脚本的一部分：
```
#SBATCH --array=1-10%1
```

这是一个玩具 slurm 脚本，可以用来看看它是如何工作的：

```
#!/bin/bash
#SBATCH --job-name=array-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # 关键 - 每个节点每个 dist 只有一个任务！
#SBATCH --cpus-per-task=1            # 每个任务的核心数
#SBATCH --time 00:02:00              # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --error=%x-%j.out            # 错误文件名（相同以方便只看一个文件）
#SBATCH --partition=dev

echo $SLURM_JOB_ID
echo "我是作业 ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
date
sleep 10
date
```

注意 `$SLURM_ARRAY_JOB_ID` 和 `$SLURM_JOB_ID` 是一样的，而 `$SLURM_ARRAY_TASK_ID` 是作业的索引。

要查看正在运行的作业：
```
$ squeue -u `whoami` -o "%.10i %9P %26j %.8T %.10M %.6D %.20S %R"
     JOBID PARTITION                       NAME    STATE       TIME  NODES           START_TIME NODELIST(REASON)
591970_[2-   dev             array-test  PENDING       0:00      1  2021-07-28T20:01:06 (JobArrayTaskLimit)
```
现在作业 2 正在运行。

要取消整个数组，像平常一样取消作业 ID（`_` 前面的数字）：
```
scancel 591970
```

要取消特定作业：
```
scancel 591970_2
```

如果日志文件包含数组 ID 很重要，请添加 `%A_%a`：

```
#SBATCH --output=%x-%j.%A_%a.log
```

更多细节 https://slurm.schedmd.com/job_array.html


## 作业数组训练及其暂停和释放

在这个方法中，我们完成了两件事：

1. 允许修改下一个作业的 slurm 脚本
2. 允许暂停和恢复作业数组，而不会在没有准备好继续运行作业时丢失队列中的位置

SLURM 是一个非常不宽容的环境，一个小错误就可能导致数天的等待时间。但是有一些策略可以减轻这种严酷性。

SLURM 作业在队列中有一个"年龄"的概念，除了项目优先级外，它还决定了作业何时被安排运行。如果你刚刚安排了一个新作业，它没有"年龄"，通常会被放在最后运行，与较早进入队列的作业相比。当然，除非这个新作业来自高优先级的项目，在这种情况下它会进展得更快。

所以，这里有一种方法，可以在需要修复运行脚本中的某些内容时，或者例如切换到另一个脚本时，保持"年龄"而不丢失它。

这个想法是：

1. `sbatch` 一个长作业数组，例如 `-array=1-50%1`
2. 在 slurm 脚本内部，除了 `source another-script.slurm` 之外，不要有任何代码——这样你就可以在下一个作业开始之前修改目标脚本或符号链接到另一个脚本
3. 如果你需要停止作业数组列车——不要取消它，而是暂停它，而不会丢失你在队列中的位置
4. 准备好继续时——取消暂停作业数组——只有暂停的时间不计入其年龄，但之前的所有年龄都保留下来。

正在运行的作业的节点数、时间和硬件以及分区不能被修改，但是您可以通过 `scontrol update jobid=<desired_job_id> numnodes=<new number> partition=<new partition>` 来更改作业数组中待处理的作业。

如果你有 `sudo` 权限，那么你也可以更改当前作业的作业时间。

这是一个例子：

创建一个作业脚本：

```
$ cat train-64n.slurm
#!/bin/bash
#SBATCH --job-name=tr8-104B
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1          # 关键 - 每个节点每个 dist 只有一个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核心数
#SBATCH --gres=gpu:8                 # gpu 数量
#SBATCH --time 20:00:00              # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=dev

source tr8-104B-64.slurm
```
这样启动它：
```
sbatch --array=1-50%1 train-64.slurm
```

现在，您可以在下一个作业运行之前轻松编辑 `tr8-104B-64.slurm`，如果需要，可以让当前作业完成，或者如果需要中止它，只需终止当前正在运行的作业，例如 `1557903_5`（而不是作业数组 `1557903`），然后让列车从中断的地方继续，但使用编辑后的脚本。

好处是这不需要对原始脚本（本例中为 `tr8-104B-64.slurm`）进行任何更改，并且后者仍然可以单独启动。

现在，如果出了什么问题，你需要 10 分钟或 10 小时来修复某些东西怎么办？在这种情况下，我们使用以下命令暂停列车：

```
scontrol hold <jobid>
```

`jobid` 可以是"普通"作业、作业数组的 ID 或作业数组步骤的 ID

然后在准备好继续时释放作业：

```
scontrol release <jobid>
```


## 如何在退出其 shell 的同时保持 salloc 分配的活动状态

如果您像这样运行分配的节点：

```
salloc --partition=dev --nodes=1 --ntasks-per-node=1 --time=1:00:00 bash
```
并且您退出了 shell，或者您的 ssh 连接断开了，分配将会丢失。

如果您想打开一个在退出 shell 后仍能存活的分配，请使用 `--no-shell` 并且不要使用 `bash`，像这样：

```
salloc --no-shell --partition=dev --nodes=1 --ntasks-per-node=1 --time=1:00:00
```
现在如果您需要加入会话，请参阅[如何以交互方式重新加入已分配的节点](#how-to-rejoin-the-allocated-node-interactively)。

但请注意，如果您 `ssh` 到已分配的节点并正常启动某些程序，然后关闭连接，该作业将会丢失，因为连接的 shell 会向其子进程发送 `SIGHUP`。为避免这种情况并保持作业运行，请在使用 `&` 将程序置于后台进程的同时使用 `nohup`。示例：

```
nohup my-program &
```

`nohup` 将忽略 `SIGHUP`，并将 stderr 重定向到 stdout，并将 stdout 附加到一个特殊文件 `nohup.out`。如果您想控制 std 流应该写入的位置，请使用普通的 stdout 重定向 `>` 或 `>>`，例如：
```
nohup my-program >> some-file.txt &
```

如前所述，程序也用 `&` 发送到后台。

现在您可以安全地断开连接，当您回来时，程序将继续运行。

这个解决方案可以防止程序退出，但是当你再次连接时，你将无法正常地与它交互，因为标准流将被重定向。你当然仍然可以通过它的 pid 杀死程序，改变它的 `nice` 状态等，就像你对任何其他进程所做的那样。

但是，如果你想使用某种可以断开连接再重新连接并继续正常使用程序的工具，你就必须使用[终端多路复用器](https://en.wikipedia.org/wiki/Terminal_multiplexer)程序，如[`tmux`](https://github.com/tmux/tmux)或[GNU `screen`](https://www.gnu.org/software/screen/)，它们在节点上运行一个守护进程，并允许你在重新连接时重新获得对程序的正常控制。还有[`mosh`](https://github.com/mobile-shell/mosh)和其他类似的工具，可以进一步辅助这个过程。



## 如何以交互方式重新加入已分配的节点

要让多个交互式 shell 进入同一个作业，应该使用 `--overlap`。

例如，在控制台 A 中，让我们分配一个节点：
```
$ salloc --partition=dev --nodes=1 --ntasks-per-node=1 --cpus-per-task=26 --gres=gpu:1 --time=2:00:00 bash
salloc: Granted job allocation 1916
salloc: Nodes my-node-1 are ready for job
```

在控制台 B 中：
```
$ srun --overlap --pty --jobid 101 bash
```
并且上面的操作可以在任意多的控制台中重复。

如果这是第一个伪终端 shell，你甚至不需要 `--overlap`，但你需要它来用于额外的 shell。

如果你最初是通过 `srun --pty` 分配节点的，它的工作方式也是一样的
```
srun --pty -p dev --gpus 8 --time=2:00:00 bash
```

当然，你也可以通过 `ssh` 访问节点，但是如果你的 SLURM 被设置为进行各种虚拟化（例如，只给每个用户几个 GPU，或者虚拟化 `/tmp/` 或 `/scratch` 并在退出时自动清理），那么从 `ssh` 看到的视图将是不同的。例如，如果一个作业分配了 2 个 GPU，ssh shell 将显示所有的 GPU，而不仅仅是 2 个——所以如果你与他人共享节点，这将无法很好地工作。

这适用于多节点分配，默认情况下，您将在分配的第一个节点上获得一个交互式 shell。如果要进入特定节点，请使用 `-w` 来指定它。例如，假设您分配了 `node-[1-4]`，并且您想进入 `node-3`，则指定：
```
srun --pty -p dev --gpus 8 --time=2:00:00 -w node-3 bash
```
如果它失败并显示：
```
srun: error: Unable to create step for job 1930: Invalid generic resource (gres) specification
```
请重新添加 `--gres=gpu:8` 设置。如果您最初的分配命令已经使用了这个标志，则不需要这样做。




## 故障排除



### `SLURM_PROCID` 提前插值

当使用带有-多节点设置的 SLURM 时，正确设置以下内容至关重要：
```
"--machine_rank \$SLURM_PROCID"
```
它不能提前被插值，因为如果设置为 `"--machine_rank $SLURM_PROCID"`，启动器将会挂起。

最好将启动器与程序隔离，像这样：

```
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3333
ACCELERATE_CONFIG_FILE=path/to/accelerate.config.yaml # 编辑我
LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): --tee 3 \
    "
PROGRAM="myprogram.py"

CMD="$LAUNCHER $PROGRAM"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --unbuffered \
    --jobid $SLURM_JOBID \
    "

srun $SRUN_ARGS bash -c "$CMD" 2>&1 | tee -a main_log.txt
```

现在启动器将始终工作，用户只需要调整 `PROGRAM` 变量。

使用 `torchrun`：

```
export $GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3333
LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank \$SLURM_PROCID
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`:--tee 3 \
    "
```

有关完整的工作示例，请参见[使用 SLURM 的单节点和多节点启动器](launchers/)。


### 节点数量不匹配

如果 pytorch 启动器失败，通常意味着 SLURM 节点数和启动器节点数不匹配，例如：

```
grep -ir nodes= tr123-test.slurm
#SBATCH --nodes=40
NNODES=64
```

这行不通。它们必须匹配。

您可以在脚本中添加一个健全性检查：

```
#!/bin/bash
#SBATCH --job-name=test-mismatch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # 关键 - 每个 dist 每个节点只有一个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核心数
#SBATCH --gres=gpu:8                 # gpu 数量
#SBATCH --time 0:05:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=prod

[...]

NNODES=2

# 检查 NNODES 和 `#SBATCH --nodes` 是否匹配，假设您使用 NNODES 变量
if [ "$NNODES" != "$SLURM_NNODES" ]; then
    echo "脚本配置错误：NNODES=$NNODES != SLURM_NNODES=$SLURM_NNODES"
    exit 1
fi

[...]
```

或者你可以直接这样做：

```bash
#SBATCH --nodes=2
[...]
NNODES=$SLURM_NNODES
```

然后它就总是正确的了



### 查找故障节点并排除它们

有时一个节点坏了，这会妨碍训练，特别是因为重启作业通常会遇到同一组节点。所以需要能够隔离坏节点并将其从 `sbatch` 中排除。

要找到故障节点，请编写一个小脚本来报告所需检查的状态。

例如，要测试所有节点上的 cuda 是否可用：
```
python -c 'import torch, socket; print(f"{socket.gethostname()}: {torch.cuda.is_available()}")'
```

并且只报告失败的节点：
```
python -c 'import torch, socket; torch.cuda.is_available() or print(f"坏掉的节点: {socket.gethostname()}") '
```

当然，问题可能不同——例如，GPU 无法分配内存，所以更改测试脚本以在 cuda 上进行少量分配。这是一种方法：

```
python -c "import torch; torch.ones(1000,1000).cuda()"
```

但是由于我们需要在所有节点上运行测试脚本，而不仅仅是第一个节点，所以 slurm 脚本需要通过 `srun` 来运行它。所以我们的第一个诊断脚本可以写成：

```
srun --jobid $SLURM_JOBID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'
```

由于引号问题，我稍微修改了一下。

你总是可以把单行命令转换成一个真正的脚本，这样就不会有引号问题了。

```
$ cat << EOT >> test-nodes.py
#!/usr/bin/env python
import torch, socket
print(socket.gethostname(), torch.cuda.is_available())
EOT
$ chmod a+x ./test-nodes.py
```

现在让我们创建一个驱动 slurm 脚本。为此测试使用几分钟的时间，以便 SLURM 更快地产生它：
```
#!/bin/bash
#SBATCH --job-name=test-nodes
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # 关键 - 每个 dist 每个节点只有一个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核心数
#SBATCH --gres=gpu:8                 # gpu 数量
#SBATCH --time 0:05:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=prod

source $six_ALL_CCFRWORK/start-prod
srun --jobid $SLURM_JOBID ./test-nodes.py
```
一旦它运行，检查日志，看看是否有任何报告 `False`，这些是您想要排除的节点。

现在一旦找到故障节点，就将其提供给 `sbatch`：
```
sbatch --exclude=hostname1,hostname2 ...
```
并且 `sbatch` 将从分配中排除坏节点。

此外，请将故障节点报告给 `#science-support`，以便更换。

这里还有一些情况以及在这些情况下如何找到坏节点：

### 损坏的 NCCL

如果您正在测试需要分布式设置的东西，情况会稍微复杂一些。这里有一个 slurm 脚本，用于测试 NCCL 是否正常工作。它设置 NCCL 并检查屏障是否正常工作：

```
#!/bin/bash
#SBATCH --job-name=test-nodes-nccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # 关键 - 每个 dist 每个节点只有一个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核心数
#SBATCH --gres=gpu:8                 # gpu 数量
#SBATCH --time 0:05:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=prod

source $six_ALL_CCFRWORK/start-prod

NNODES=2

GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export SCRIPT=test-nodes-nccl.py

cat << EOT > $SCRIPT
#!/usr/bin/env python
import torch.distributed as dist
import torch
import socket
import os
import fcntl

def printflock(*msgs):
    """ print """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
header = f"{socket.gethostname()}-{local_rank}"
try:
    dist.barrier()
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} 正常")
except:
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} 损坏")
    raise
EOT

echo $LAUNCHER --node_rank $SLURM_PROCID $SCRIPT

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $SCRIPT'
```
该脚本使用 `printflock` 来解决交错打印输出的问题。


### GPU 内存检查


这测试了分配节点上的每个 GPU 是否可以成功分配 77Gb（例如，测试 80GB A100）（必须减去几 GB 用于 cuda 内核）。


```python
import torch, os
import time
import socket
hostname = socket.gethostname()

local_rank = int(os.environ["LOCAL_RANK"]);

gbs = 77
try:
    torch.ones((gbs*2**28)).cuda(local_rank).contiguous() # 在 cpu 上分配，然后移动到 gpu
    print(f"{local_rank} {hostname} 正常")
except:
    print(f"{local_rank} {hostname} 分配 {gbs}GB DRAM 失败")
    pass

time.sleep(5)


```


### 网络中断

节点的另一个问题是当其网络中断，其他节点无法连接到它时。

您可能会遇到类似以下的错误：
```
work = default_pg.barrier(opts=opts)
RuntimeError: NCCL error in: /opt/conda/conda-bld/pytorch_1616554793803/work/torch/lib/c10d/ProcessGroupNCCL.cpp:825, unhandled system error, NCCL version 2.7.8
ncclSystemError: System call (socket, malloc, munmap, etc) failed.
```
以下是如何调试此问题：

1. 添加：
```
export NCCL_DEBUG=INFO
```
在 `srun` 命令之前，然后重新运行您的 slurm 脚本。

2. 现在研究日志。如果你发现：
```
r11i6n2:486514:486651 [1] include/socket.h:403 NCCL WARN Connect to 10.148.3.247<56821> failed : Connection refused
```
让我们看看是哪个节点拒绝接受连接。我们从上面的错误中获取 IP 地址，并将其反向解析为其名称：
```
nslookup 10.148.3.247
247.3.148.10.in-addr.arpa       name = r10i6n5.ib0.xa.idris.fr.
```

将 `--exclude=r10i6n5` 添加到您的 `sbatch` 命令中，并将其报告给 JZ 管理员。


### 在所有节点上运行 py-spy 或任何其他监控程序

在处理挂起问题时，以下是如何自动为每个进程记录 `py-spy` 跟踪的方法。

当然，同样的过程可以用来为给定作业的所有节点运行某个命令。也就是说，它可以用来在正常运行期间运行某些东西——例如，通过 `nvidia-smi` 或任何其他需要运行的程序，转储每个进程中的所有内存使用情况。



```
cd ~/prod/code/tr8b-104B/bigscience/train/tr11-200B-ml/

salloc --partition=prod --nodes=40 --ntasks-per-node=1 --cpus-per-task=96 --gres=gpu:8 --time 20:00:00

bash 200B-n40-bf16-mono.slurm
```

在另一个 shell 中获取上述 `salloc` 的 JOBID：
```
squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"
```
根据上面的信息调整 jobid 和节点数（XXX：可能可以完全删除 `--nodes=40` 并依赖 `salloc` 配置）：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```
现在所有的 `py-spy` 跟踪都进入了 `cwd` 下的 `trace-$nodename.out` 文件。

关键是使用 `--gres=gpu:0`，否则第二个 `srun` 将会阻塞，等待第一个 `srun` 释放 GPU。

此外，假设在 `~/.bashrc` 中激活了某个已安装 `py-spy` 的 conda 环境。如果您的环境中没有这样做，请在 `py-spy` 命令之前将加载环境的指令添加到上述命令中——否则它将找不到它。

当此过程完成后，不要忘记手动释放分配。

## 将 SLURM_JOB_NODELIST 转换为 hostfile

一些多节点启动器需要一个 `hostfile` —— 以下是如何生成一个：

```
# 自动生成 deepspeed 的 hostfile
# 1. 处理两种格式的 SLURM_JOB_NODELIST：
# r10i1n8,r10i2n0
# r10i1n[7-8]
# 2. 并依赖 SLURM_STEP_GPUS=0,1,2... 来获取每个节点的 gpu 插槽数
#
# 用法：
# makehostfile > hostfile
function makehostfile() {
perl -le '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"}; $_=$ENV{"SLURM_JOB_NODELIST"}; if (/^(.*?)\[(\d+)-(\d+)\]/) { print map { "$1$_ slots=$slots\n" } $2..$3} elsif (/,/) { print map { "$1$_ slots=$slots\n" } split /,/ } '
}
```

## 环境变量

你总是可以这样做：

```
export SOMEKEY=value
```
从 slurm 脚本中将所需的环境变量传递给从中启动的程序。

您还可以在 slurm 脚本的顶部添加：
```
#SBATCH --export=ALL
```
启动的程序将看到在其启动的 shell 中可见的所有环境变量。



## Crontab 仿真

最重要的 Unix 工具之一是 crontab，它对于能够安排各种作业至关重要。然而，它通常在 SLURM 环境中不存在。因此，必须对其进行仿真。以下是方法。

对于此演示，我们将使用 `$WORK/cron/` 作为基本目录。并且您有一个已导出的环境变量 `WORK` 指向文件系统上的某个位置——如果您使用 Bash，可以在 `~/.bash_profile` 中进行设置，如果使用其他 shell，则使用相应的启动文件。


### 1. 一个自我延续的调度器作业

我们将使用 `$WORK/cron/scheduler` 目录来存放调度器作业，`$WORK/cron/cron.daily` 存放每日作业，`$WORK/cron/cron.hourly` 存放每小时作业：

```
$ mkdir -p $WORK/cron/scheduler
$ mkdir -p $WORK/cron/cron.daily
$ mkdir -p $WORK/cron/cron.hourly
```

现在将这两个 slurm 脚本复制到 `$WORK/cron/scheduler` 中：
- [cron-daily.slurm](cron-daily.slurm)
- [cron-hourly.slurm](cron-hourly.slurm)

在编辑它们以适应您特定环境的帐户和分区信息后。

现在您可以启动 crontab 调度程序作业：

```
$ cd $WORK/cron/scheduler
$ sbatch cron-hourly.slurm
$ sbatch cron-daily.slurm
```

这是它，这些作业现在将自我延续，通常您不需要再考虑它，除非发生使 SLURM 丢失所有作业的事件。


### 2. 每日和每小时 Cronjobs

现在，每当您希望某个作业每天运行一次时，您只需创建一个 slurm 作业并将其放入 `$WORK/cron/cron.daily` 目录中。

这是一个每天运行以更新 `mlocate` 文件索引的示例作业：
```
$ cat $WORK/cron/cron.daily/mlocate-update.slurm
#!/bin/bash
#SBATCH --job-name=mlocate-update    # 作业名称
#SBATCH --ntasks=1                   # MP 任务数
#SBATCH --nodes=1
#SBATCH --hint=nomultithread         # 我们得到物理核心而不是逻辑核心
#SBATCH --time=1:00:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=PARTITION     # 编辑我
#SBATCH --account=GROUP@PARTITION # 编辑我

set -e
date
echo "正在更新 mlocate 数据库"
/usr/bin/updatedb -o $WORK/lib/mlocate/work.db -U $WORK --require-visibility 0
```

这将构建 `$WORK` 下文件的索引，然后您可以使用以下命令快速查询：
```
/usr/bin/locate -d $WORK/lib/mlocate/work.db pattern
```

要停止运行此作业，只需将其移出 `$WORK/cron/cron.daily` 目录。

同样的原则也适用于放在 `$WORK/cron/cron.hourly` 目录中的作业。这些作业对于每小时运行某些任务很有用。

请注意，由于 SLURM 调度中的各种延迟，这个 crontab 实现在时间上是近似的，它们将大约每小时和每天运行一次。如果您必须在更精确的时间启动某些任务，您可以重新编写代码以要求 SLURM 在更精确的时间启动，但大多数时候，刚刚介绍的方法效果很好。

此外，您可以编写自己的变体以满足项目的特定需求，例如，每 30 分钟或每 12 小时的作业。


### 3. 清理

最后，由于每个 cron 启动器作业都会留下一个日志文件（如果由于某种原因事情不起作用，这很有用），您需要创建一个 cronjob 来清理这些日志。否则，您可能会用尽 inode——这些日志文件很小，但可能有成千上万个。

你可以在每日任务中使用类似这样的东西。

```
find $WORK/cron -name "*.out" -mtime +7 -exec rm -f {} +
```
请注意，它被设置为只删除超过 7 天的文件，以防您需要最新的日志进行诊断。


### 细微差别

调度程序以启动 SLRUM cron 调度程序作业的人的 Unix 权限运行，因此该 cron 作业启动的所有其他 SLURM 脚本也以该权限运行。

## 自我延续的 SLURM 作业

在[构建调度程序](#1-a-self-perpetuating-scheduler-job)中使用的相同方法可用于创建独立的自我延续作业。

例如：

```
#!/bin/bash
#SBATCH --job-name=watchdog          # 作业名称
#SBATCH --ntasks=1                   # MP 任务数
#SBATCH --nodes=1
#SBATCH --time=0:30:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=PARTITION        # 编辑我

# 确保首先在 1 小时后重新启动自己
RUN_FREQUENCY_IN_HOURS=1
sbatch --begin=now+${RUN_FREQUENCY_IN_HOURS}hour watchdog.slurm

... 在这里做看门狗的工作 ...
```
然后你用以下命令启动它一次：
```
sbatch watchdog.slurm
```
然后，它将立即安排自己在启动后 1 小时运行，然后将完成正常的作业工作。无论作业的其余部分是成功还是失败，该作业都将继续大约每小时重新启动一次。由于调度程序作业启动开销和节点可用性问题，这是不精确的。但是，如果至少有一个备用节点可用，并且作业本身可以快速完成，那么以近似频率运行的要求应该是足够的。

由于大多数 SLURM 环境除了昂贵的 GPU 节点外，还提供便宜得多的纯 CPU 节点，因此您应该为任何不需要 GPU 运行的作业选择一个纯 CPU 的 SLURM 分区。


## 获取有关作业的信息

在 slurm 文件中，可以访问有关当前作业分配的信息。

获取分配的主机名以及基于此的有用派生：
```
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
```



## 将紧凑节点列表转换为扩展节点列表

有时你会得到像 `node-[42,49-51]` 这样的 SLURM 工具给出的字符串，这需要一些编码才能将其扩展为 `node-42,node-49,node-50,node-51`，但是有一个专门的工具可以处理这个问题：

```
$ scontrol show hostnames node-[42,49-51]
node-42
node-49
node-50
node-51
```
瞧！

案例研究：例如，如果您想获取因作业退出太慢而被排空但实际上节点没有问题的节点列表，这很有用。因此，这个单行命令将以扩展格式为您提供此类节点的列表，然后您可以编写脚本循环此列表以取消排空这些节点，或许在检查此时进程已死亡之后：
```
sinfo -R | grep "Kill task failed" | perl -lne '/(node-.*[\d\]]+)/ && print $1' | xargs -n1 scontrol show hostnames
```

## 克服缺乏组 SLURM 作业所有权的问题

SLURM 在 Unix 上运行，但令人惊讶的是，其设计者在 SLURM 作业方面没有采用组所有权的概念。因此，如果您的团队成员启动了一个包含 10 个作业的数组，每个作业 20 小时，然后去度假了——除非您有 `sudo` 权限，否则如果出现问题，您现在无法停止这些作业。

我还没有找到为什么会这样，但到目前为止，我们一直在使用一个终止开关的解决方法。你必须在你的框架中编码它。例如，看看它是如何在 [Megatron-Deepspeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/training.py#L104) (Meg-DS) 中实现的。程序轮询文件系统上一个在启动时预先配置的路径，如果它在那里找到一个文件，它就会退出。

所以，如果我们用 `--kill-switch-path $WORK/tmp/training17-kill-switch` 启动 Meg-DS，然后在任何时候我们需要杀死 SLURM 作业，我们只需这样做：

```
touch $WORK/tmp/training17-kill-switch
```
下次程序检查此文件时，它将检测到该事件并自愿退出。如果您有一个作业数组，那么，您将不得不等到每个作业启动，检测到终止开关并退出。

当然，当你停止工作后，别忘了删除它。
```
rm $WORK/tmp/training17-kill-switch
```

现在，这并不总是有效。如果作业挂起，它永远不会到检查终止开关的地步，唯一的解决方案是联系系统管理员为您终止作业。有时，如果挂起是一个简单的情况，pytorch 的分布式设置通常会在预设的 30 分钟超时后自动退出，但这并不总是有效。


## 如何在 SLURM 作业抢占时优雅退出

有几种方法可以优雅地处理基于时间和 QoS 的 SLURM 抢占，这些方法在本节中有深入介绍：[处理强制作业抢占](../../training/fault-tolerance/#dealing-with-forced-job-preemption)。


## 一个作业使用多少个 GPU

要弄清楚一个已经在运行的作业使用了多少个 gpu，请解析 `show job -d` 输出中的 `JOB_GRES=gpu:` 条目。例如，如果作业是这样启动的：

```
srun --pty --partition=dev --nodes=2 --ntasks-per-node=1 --gres=gpu:8 --time=8:00:00 bash
```
也就是说我们分配了 16 个 GPU，我们现在可以通过编程方式获取该数字：

```
$ TOTAL_JOB_GPUS=$(scontrol show job -d $SLURM_JOBID | perl -ne 'm|JOB_GRES=gpu:(\d+)| && print $1')
$ echo $TOTAL_JOB_GPUS
16
```

如果您运行命令的 shell 中尚未设置 `$SLURM_JOBID`，请用 SLURM 作业 ID 替换它（[`squeue`](#show-jobs)）。


## 作业运行了多长时间

虽然通常 `squeue` 会显示当前正在运行的作业的持续时间，但要查看作业完成时运行了多长时间，您需要知道作业 ID，然后可以像这样查询它：

```
$ sacct -j 22171 --format=JobID,JobName,State,Elapsed
JobID           JobName      State    Elapsed
------------ ---------- ---------- ----------
22171          example   COMPLETED   00:01:49
```

所以我们知道这个工作在不到 2 分钟内就完成了。

## 公平共享

许多 SLURM 集群使用 FairShare 系统，即使用集群越多的人，运行作业的优先级就越低，或者如果存在抢占，他们更有可能被抢占。

要查看您的 FairShare 分数，请运行：
```
sshare
```

例子：

```
Account                    User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare
-------------------- ---------- ---------- ----------- ----------- ------------- ----------
root                                          0.000000   711506073      1.000000
 all                                     1    0.500000   711506073      1.000000
  all                      stas          1    0.022727    14106989      0.019827   0.288889
```

如果您的 FairShare 分数超过 0.5，这意味着您使用集群的次数少于您被分配的次数，如果低于 0.5，则意味着您使用集群的次数多于被分配的次数。

随着时间的推移，这个分数会衰减，所以如果你有一个非常低的分数，并且你使用集群的次数少得多，那么你的分数会随着时间的推移而提高。

要查看特定用户的分数：
```
sshare -u username
```

要查看所有人的分数，按 FairShare 排序：
```
sshare --all | sort -nk7 -r
```

这是最重要的输出，因为您单独的分数并不重要。重要的是您相对于所有其他用户的分数。每个分数比您高的人都有更高的机会首先获得他们的工作，并且他们的工作被抢占的机会更低。

除了 FairShare，优先级通常是根据多个指标的组合来配置的，通常包括作业在队列中等待的时间长度、作业大小、服务质量 (QOS) 设置、分区特性等。具体细节将取决于您的系统管理员如何配置 slurm。

</rewritten_file>


