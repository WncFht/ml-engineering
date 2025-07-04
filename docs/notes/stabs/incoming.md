---
title: 待办
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/20zwt2lj/
---
# 需要添加/集成的内容

# pdf 书籍笔记

来自 Sam 的想法：https://github.com/saforem2: https://github.com/stas00/ml-engineering/pull/17#discussion_r1439912709
https://quarto.org/, https://quarto.org/docs/gallery/, https://kevinheavey.github.io/modern-polars/, https://quarto.org/docs/output-formats/pdf-basics.html

# 性能

https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

Dirk Groeneveld 检查节点对速度的脚本 https://github.com/allenai/OLMo/commit/f91cebdfa299bf55e815d496c367de8b59881c2e
```bash
#!/bin/bash

NCCL_LIB_DIR=/var/lib/tcpxo/lib64 source /var/lib/tcpxo/lib64/nccl-env-profile.sh

set -euxo pipefail

HOST_VARS=$(sed 's/ \{1,\}/ -x /g' <<<"${!NCCL*} LD_LIBRARY_PATH")
FIRST_HOST=$(( echo "$1" && echo "$2" ) | sort | head -1)
mpirun \
  --mca btl self,tcp \
  --mca btl_tcp_if_include enp0s12 \
  --mca orte_base_help_aggregate 0 \
  -H $1,$2 \
  -np 2 \
  --bind-to none \
  -npernode 1 \
  -tag-output \
  -x ${HOST_VARS} \
  -x NCCL_NET=FasTrak \
  -x GLOO_SOCKET_IFNAME=enp0s12 \
  -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -x OMP_NUM_THREADS=16 \
  bash -c "source ~/venv/OLMo/bin/activate && torchrun --nproc_per_node 8 --nnodes=2 --rdzv-backend=c10d --rdzv-endpoint=$FIRST_HOST ~/OLMo/scripts/augusta/all_reduce_bench.py"
```
运行命令：
```
# 检查所有节点对的 reduce 性能
fgrep -hv \# ~/hostfiles/hosts | \
parallel -N2 'echo {} $(./check_node_pair.sh {} 2>&1 | fgrep busbw)'
```


# 存储章节

### 存储基准测试：

https://github.com/argonne-lcf/dlio_benchmark


Ross Wightman 待集成的建议：

- 我会尝试按工作负载分离卷，所以将"大量小文件"、高流失率的环境、代码与数据集、检查点等批量存储分开。甚至可能也拆分那些，因为数据集基本上是静态的，而检查点一直在轮换

- 当数据集在网络存储上时，就像桶存储一样，它们应该由大文件组成，并且应该作为大文件读取（以大块顺序读取，而不是 mmap！）。避免在数据集中进行寻址

-像 HF 数据集这样的设置可能具有欺骗性，可能看起来像一个大文件，但通常是 mmap 并且 IO 读取模式很疯狂，比将它们作为单个文件读取多 3-4 倍的 iops。
  可以关闭 Mmap 加载，但如果是这样，对于许多数据集，您会将问题转移到 DataLoader 进程中，需要一次性将太多数据读入内存。更好地了解不同用例的权衡，尤其是在适当时使用 Iterable 流式传输。

- 在某种程度上，像 s3 这样的桶存储，通过接口限制，强制执行了对此类存储后端合理的模式。哦，它被挂载为一个文件夹，我可以做任何我想做的事情（mmap 文件、写大量小文件、删除所有文件等等），这才是问题所在。

- 人们也不能指望像对待本地磁盘一样对待分布式文件系统。如果你按工作负载分离卷，你可能能够利用更高百分比的总存储空间。不要将高流失率、小文件与低流失率、大文件混合在一起。

- 另外，请注意，一旦您的数据集对大型分布式网络文件系统进行了优化，它们通常就可以直接从具有该选项的云系统中的桶存储中进行流式传输。因此，在这种情况下，最好将它们移出网络文件系统。

# 调试

内存泄漏检查

```
cuda-memcheck --leak-check full python program.py
```


竞争条件检测：
```
cuda-memcheck --tool racecheck
```
使用额外选项：
 --save 将输出保存到磁盘
 --print-level 控制输出

```
cuda-memcheck --tool racecheck --racecheck-report analysis
```

使用 cuda 的 gdb

```
cuda-gdb
```

- 集成 debug_utils.py


# 模型并行

这里有一个很好的表格，列出了每种并行类型的扩展方程。
https://www.cerebras.net/blog/cerebras-sets-record-for-largest-ai-models-ever-trained-on-single-device#summary


# 网络

创建一个新的基准测试部分：

1. nccl-tests
2. `all_reduce_bench.py`
3. https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/communication
4. 像 nccl-tests 一样，HPC 站点使用的另一组常见基准测试是 OSU 微基准测试，如 osu_lat、osu_bw 和 osu_bibw。

https://mvapich.cse.ohio-state.edu/benchmarks/

这些是基于 MPI 的基准测试。这些可以使用 GPUDirect RDMA 运行，因此您可以测量 GPU 之间（无论是在同一节点上还是在节点之间）的 MPI 性能。


## Infiniband

参考资料：
- [系统管理员袖珍生存指南 - InfiniBand](https://tin6150.github.io/psg/infiniband.html)


### 诊断

非 IB 特定
- `ifconfig` - 显示当前活动接口的状态
- `ip addr show` - 显示系统上配置的每个链路的地址

显示本地主机的 IB 设备状态（3 种不同视图）。
- `ibstat`
- `ibstatus`
- `ibv_devinfo`

扫描 IB 网络：
- `ibnetdiscover` - 扫描拓扑
- `ibroute` - 显示交换机的单播和多播转发表
- `ibdiagnet` - IB 诊断网络

检查网络错误：
- `ibcheckerrors` - 检查端口/节点的错误计数器是否在预定义阈值内
- `ibchecknet` - 对子网执行端口/节点/错误检查。

测试 IB 网络配置：
- `ibcheckport` - 对指定端口执行一些基本测试
- `ibchecknode` - 对指定节点执行一些基本测试
- `ibclearcounters` - 清除 InfiniBand 子网的端口计数器

其他检查：
- `iblinkinfo`
- `ibcheck`
- `wwibcheck`
- `ibswitch` - 验证机架中是否安装了 IB-QNEM
- `ibhosts` - 列出 IB 网络中的所有主机。
`ibswitches` - 列出所有 ib 交换机

追踪：
- `ibping` - 在 InfiniBand 节点之间进行 ping/pong
- `ibsysstat` - 获取远程节点的基本信息（主机名、cpu、内存、利用率）
- `ibswitches` - 扫描网络或使用现有的网络拓扑文件并列出所有交换机
- `ibhosts` - 扫描网络或使用现有的网络拓扑文件并列出所有主机

显示网络拓扑：
- `iblinkinfo -R`

使用 `ifconfig` 发现 `IPoIB` 网络，例如，如果你得到 `ib0` 设备，其 `inet addr:100.1.1.102`，你就可以连接到它 - 例如 `ping 100.1.1.102`

找到控制器：
`lspci | grep Mellanox`

打印驱动程序配置（接口名称来自 `ifconfig`）：
`ethtool -i enP49239s1`

### 性能

`perftest` 包包括：
- `ib_send_bw`
- `ib_send_lat`
- `ib_write_bw`
- `ib_write_lat`
- `ib_read_bw`
- `ib_read_lat`
- `ib_atomic_bw`
- `ib_atomic_lat`

示例：`ib_send_bw -a address` - 测试带宽

`qperf` 测量两个节点之间的带宽和延迟（TCP/IP 和 RDMA 传输）



如果网络比应有的速度慢得多，可能需要指定使用哪个 HCA（使用 `ibv_devinfo` 获取 HCA）
```
export NCCL_IB_HCA=mlx5
```

可能需要在虚拟机上安装 ib 包：

```
sudo apt-get install -y automake dh-make git libcap2 libnuma-dev libtool make pkg-config udev curl librdmacm-dev rdma-core \
    libgfortran5 bison chrpath flex graphviz gfortran tk dpatch quilt swig tcl ibverbs-utils infiniband-diags
sudo sed -i -e 's/# OS.EnableRDMA=y/OS.EnableRDMA=y/g' /etc/waagent.conf
```

- Verbs：允许在功能丰富的 IB 交换机上执行命令。


# SLURM

要探索的仓库：
https://github.com/OleHolmNielsen/Slurm_tools


# 测试

- 集成 testing_utils.py 的功能


# 来自 LLNL 的 Adam Moody 团队


- NUMA 亲和性

https://github.com/LLNL/mpibind/tree/master/python
mpibind for Python 使得在任意 Python 程序中使用 mpibind 算法成为可能。

- 训练挂起检测工具：

这是为了扩展：
https://github.com/stas00/ml-engineering/tree/master/fault-tolerance#is-job-hanging-watchdog


来自 Adam 的笔记：

https://github.com/LLNL/STAT - 堆栈跟踪分析工具
https://hpc.llnl.gov/software/development-environment-software/stat-stack-trace-analysis-tool

https://github.com/grondo/io-watchdog

你可以在这个页面下方看到我们是如何集成 STAT 的：

https://hpc.llnl.gov/software/development-environment-software/stat-stack-trace-analysis-tool

有一些"动作"脚本需要编写，当 io-watchdog 检测到挂起时会执行这些脚本。页面上没有显示这些脚本的内容，但如果你好奇，我可以查一下。用户会创建一个像这样的配置文件：

```
search /usr/local/tools/io-watchdog/actions
timeout = 20m
actions = STAT, kill
```

这将 io-watchdog 配置为，如果它在 20 分钟内没有看到任何输出（来自 rank 0），就认为作业卡住了，然后运行"STAT"来收集堆栈跟踪，并运行"kill"来 scancel 作业。我们还有其他一些脚本，比如一个在 io-watchdog 检测到挂起时给用户发邮件的脚本。然后这样启动：
```
srun --io-watchdog mpi_application
```

SCR 的一个快速演示。使用它的 python 代码非常简洁。

安装 SCR 库（C + MPI）
https://scr.readthedocs.io/en/v3.0/users/build.html#cmake

安装 scr.py 模块：
https://github.com/LLNL/scr/tree/develop/python#installing-the-scr-python-module

python 中的检查点示例：
https://github.com/LLNL/scr/blob/1878de8756c2b51882a7cda7b97b142eae4e3995/python/scr_example.py#L64-L105



  396  dmesg | grep -i 'limited by'
  397  sudo dmesg | grep -i 'limited by'
  398  nvidia-smi nvlink -e


在研究问题时，GPU VBIOS 版本可能很重要。让我们将名称和总线 id 添加到查询中，我们得到：

```
$ nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv

$ nvidia-smi -q | grep "VBIOS Version"
    VBIOS Version                         : 96.00.89.00.01
    [...]
    VBIOS Version                         : 96.00.89.00.01
```


检查 NVLink 链接的错误计数器

```
$ nvidia-smi nvlink -e
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abcdefab-cdef-abdc-abcd-abababababab)
         Link 0: Replay Errors: 0
         Link 0: Recovery Errors: 0
         Link 0: CRC Errors: 0

         Link 1: Replay Errors: 0
         Link 1: Recovery Errors: 0
         Link 1: CRC Errors: 0

         [...]

         Link 17: Replay Errors: 0
         Link 17: Recovery Errors: 0
         Link 17: CRC Errors: 0
```

另一个有用的命令是：
```
$ nvidia-smi nvlink --status
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abcdefab-cdef-abdc-abcd-abababababab)
         Link 0: 26.562 GB/s
         [...]
         Link 17: 26.562 GB/s
```
这个告诉你每个链接的当前速度

运行 `nvidia-smi nvlink -h` 以发现更多功能（报告、重置计数器等）。

nvidia-smi --query-remapped-rows=gpu_name,gpu_bus_id,remapped_rows.failure,remapped_rows.pending,\
remapped_rows.correctable,remapped_rows.uncorrectable \
--format=csv gpu_name, gpu_bus_id, remapped_rows.failure,remapped_rows.pending,\
remapped_rows.correctable, remapped_rows.uncorrectable


nvidia-smi --query-remapped-rows=gpu_name,gpu_bus_id,remapped_rows.failure,remapped_rows.pending,remapped_rows.correctable,remapped_rows.uncorrectable --format=csvgpu_name, gpu_bus_id, remapped_rows.failure, remapped_rows.pending, remapped_rows.correctable,remapped_rows.uncorrectable
