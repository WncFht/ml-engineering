---
title: 模拟多节点
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/3p1hzd3h/
---
# 仅使用单个节点模拟多节点设置

目标是使用具有 2 个 GPU 的单个节点来模拟一个 2 节点环境（用于测试目的）。当然，这可以进一步扩展到[更大的设置](#更大的设置)。

我们在这里使用 `deepspeed` 启动器。实际上不需要使用任何 deepspeed 代码，只是使用其更高级的功能会更容易。您只需要安装 `pip install deepspeed`。

完整的设置说明如下：

1. 创建一个 `hostfile`：

```bash
$ cat hostfile
worker-0 slots=1
worker-1 slots=1
```

2. 在您的 ssh 客户端中添加匹配的配置

```bash
$ cat ~/.ssh/config
[...]

Host worker-0
    HostName localhost
    Port 22
Host worker-1
    HostName localhost
    Port 22
```

如果端口不是 22 或者主机名不是 `localhost`，请进行相应调整。


3. 由于您的本地设置可能受密码保护，请确保将您的公钥添加到 `~/.ssh/authorized_keys`

`deepspeed` 启动器明确使用无密码连接，例如在 worker0 上它会运行：`ssh -o PasswordAuthentication=no worker-0 hostname`，所以您总是可以使用以下命令来调试 ssh 设置：

```bash
$ ssh -vvv -o PasswordAuthentication=no worker-0 hostname
```

4. 创建一个测试脚本来检查两个 GPU 是否都被使用。

```bash
$ cat test1.py
import os
import time
import torch
import deepspeed
import torch.distributed as dist

# 使用第二个 gpu 的关键技巧（否则两个进程都会使用 gpu0）
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dist.init_process_group("nccl")
local_rank = int(os.environ.get("LOCAL_RANK"))
print(f'{dist.get_rank()=}, {local_rank=}')

x = torch.ones(2**30, device=f"cuda:{local_rank}")
time.sleep(100)
```

运行：

```bash
$ deepspeed -H hostfile test1.py
[2022-09-08 12:02:15,192] [INFO] [runner.py:415:main] 使用 IP 地址 192.168.0.17 作为节点 worker-0
[2022-09-08 12:02:15,192] [INFO] [multinode_runner.py:65:get_cmd] 在以下 worker 上运行：worker-0,worker-1
[2022-09-08 12:02:15,192] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test1.py
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:158:main] 设置 CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:158:main] 设置 CUDA_VISIBLE_DEVICES=0
worker-1: torch.distributed.get_rank()=1, local_rank=0
worker-0: torch.distributed.get_rank()=0, local_rank=0
worker-1: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
worker-0: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
```

如果 ssh 设置工作正常，您可以并行运行 `nvidia-smi` 并观察到两个 GPU 都从 `torch.ones` 调用中分配了约 4GB 的内存。

请注意，该脚本通过修改 `CUDA_VISIBLE_DEVICES` 来告知第二个进程使用 gpu1，但在两种情况下，它都会被视为 `local_rank==0`。

5. 最后，我们测试一下 NCCL 集合操作是否也正常工作

脚本改编自 [torch-distributed-gpu-test.py](../debug/torch-distributed-gpu-test.py)，只调整了 `os.environ["CUDA_VISIBLE_DEVICES"]`

```bash
$ cat test2.py
import deepspeed
import fcntl
import os
import socket
import time
import torch
import torch.distributed as dist

# 一个关键技巧，让第二个进程使用第二个 GPU（否则两个进程都会使用 gpu0）
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def printflock(*msgs):
    """ 解决多进程交错打印问题 """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
hostname = socket.gethostname()

gpu = f"[{hostname}-{local_rank}]"

try:
    # 测试分布式
    dist.init_process_group("nccl")
    dist.all_reduce(torch.ones(1).to(device), op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f'{dist.get_rank()=}, {local_rank=}')

    # 测试 cuda 是否可用并可以分配内存
    torch.cuda.is_available()
    torch.ones(1).cuda(local_rank)

    # 全局排名
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    printflock(f"{gpu} 正常 (全局排名: {rank}/{world_size})")

    dist.barrier()
    if rank == 0:
        printflock(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        printflock(f"设备计算能力={torch.cuda.get_device_capability()}")
        printflock(f"pytorch 计算能力={torch.cuda.get_arch_list()}")

except Exception:
    printflock(f"{gpu} 损坏")
    raise
```

运行：

```bash
$ deepspeed -H hostfile test2.py
[2022-09-08 12:07:09,336] [INFO] [runner.py:415:main] 使用 IP 地址 192.168.0.17 作为节点 worker-0
[2022-09-08 12:07:09,337] [INFO] [multinode_runner.py:65:get_cmd] 在以下 worker 上运行：worker-0,worker-1
[2022-09-08 12:07:09,337] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test2.py
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] 设置 CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] 设置 CUDA_VISIBLE_DEVICES=0
worker-0: dist.get_rank()=0, local_rank=0
worker-1: dist.get_rank()=1, local_rank=0
worker-0: [hope-0] 正常 (全局排名: 0/2)
worker-1: [hope-0] 正常 (全局排名: 1/2)
worker-0: pt=1.12.1+cu116, cuda=11.6, nccl=(2, 10, 3)
worker-0: 设备计算能力=(8, 0)
worker-0: pytorch 计算能力=['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
worker-1: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] 进程 576485 成功退出。
worker-0: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] 进程 576484 成功退出。
```

瞧，任务完成。

我们测试了 NCCL 集合操作可以工作，但它们使用的是本地 NVLink/PCIe，而不是像真实多节点那样的 IB/ETH 连接，因此根据需要测试的内容，这可能足够也可能不够。


## 更大的设置

现在，假设您有 4 个 GPU，并且想要模拟 2x2 节点。那么只需将 `hostfile` 更改为：

```bash
$ cat hostfile
worker-0 slots=2
worker-1 slots=2
```
并将 `CUDA_VISIBLE_DEVICES` 的技巧更改为：

```bash
if os.environ["RANK"] in ["2", "3"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
```

其他所有内容都应保持不变。


## 自动化过程

如果您想要一种自动处理任何拓扑形状的方法，您可以使用类似这样的东西：

```python
def set_cuda_visible_devices():
    """
    通过调整 CUDA_VISIBLE_DEVICES 环境变量，为每个模拟节点自动分配正确的 GPU 组
    """

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    emulated_node_size = int(os.environ["LOCAL_SIZE"])
    emulated_node_rank = int(global_rank // emulated_node_size)
    gpus = list(map(str, range(world_size)))
    emulated_node_gpus = ",".join(gpus[emulated_node_rank*emulated_node_size:(emulated_node_rank+1)*emulated_node_size])
    print(f"设置 CUDA_VISIBLE_DEVICES={emulated_node_gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = emulated_node_gpus

set_cuda_visible_devices()
```


## 使用单个 GPU 模拟多个 GPU

以下是与本文档中讨论的需求正交的需求，但它相关，所以我认为在这里分享一些见解会很有用：

对于 NVIDIA A100，您可以使用 [MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/) 在一个真实 GPU 上模拟多达 7 个 GPU 实例，但遗憾的是，您不能将这些实例用于除独立使用之外的任何其他用途 - 例如，您不能在这些 GPU 上执行 DDP 或任何 NCCL 通信。我曾希望可以使用我的 A100 模拟 7 个实例，并再添加一个真实 GPU，从而拥有 8 个 GPU 进行开发 - 但这行不通。询问 NVIDIA 工程师后，他们表示没有计划支持这种用例。


## 致谢

非常感谢 [Jeff Rasley](https://github.com/jeffra/) 帮助我完成此设置。
