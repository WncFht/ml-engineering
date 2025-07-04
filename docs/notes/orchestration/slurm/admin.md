---
title: admin
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/gr91lu6r/
---
# SLURM 管理


## 在多个节点上运行命令

1. 为了避免被提示：
```
您确定要继续连接吗 (yes/no/[fingerprint])?
```
对于每个您尚未登录的新节点，您可以使用以下命令禁用此检查：
```
echo "Host *" >> ~/.ssh/config
echo "  StrictHostKeyChecking no" >> ~/.ssh/config
```

当然，请检查这是否足够安全以满足您的需求。我假设您已经在使用 SLURM 集群，并且没有在集群外进行 ssh 连接。您可以选择不设置此项，然后您将需要手动批准每个新节点。

2. 安装 `pdsh`

您现在可以在多个节点上运行所需的命令。

例如，我们运行 `date`：

```
$ PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26] date
node-25: Sat Oct 14 02:10:01 UTC 2023
node-21: Sat Oct 14 02:10:02 UTC 2023
node-23: Sat Oct 14 02:10:02 UTC 2023
node-24: Sat Oct 14 02:10:02 UTC 2023
node-26: Sat Oct 14 02:10:02 UTC 2023
```

我们来做一些更有用和复杂的事情。让我们杀死所有在 SLURM 作业被取消时没有退出的与 GPU 绑定的进程：

首先，这个命令将给我们所有占用 GPU 的进程 ID：

```
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq
```

所以我们现在可以一次性杀死所有这些进程：

```
 PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26]  "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq | xargs -n1 sudo kill -9"
```


## Slurm 设置

显示 slurm 设置：

```
sudo scontrol show config
```

配置文件位于 slurm 控制器节点的 `/etc/slurm/slurm.conf`。

一旦 `slurm.conf` 更新后，要重新加载配置，请运行：
```
sudo scontrol reconfigure
```
从控制器节点运行。



## 自动重启

如果节点需要安全地重启（例如，如果镜像已更新），请调整节点列表并运行：

```
scontrol reboot ASAP node-[1-64]
```

对于每个非空闲节点，此命令将等到当前作业结束后，然后重新启动该节点并将其恢复到 `idle` 状态。

请注意，您需要在控制器节点的 `/etc/slurm/slurm.conf` 中设置：
```
RebootProgram = "/sbin/reboot"
```
才能使其工作（如果您刚刚将此条目添加到配置文件中，则需要重新配置 SLURM 守护进程）。


## 更改节点状态

更改由 `scontrol update` 执行

示例：

要取消一个准备好使用的节点的排空状态：
```
scontrol update nodename=node-5 state=idle
```

要从 SLURM 池中删除一个节点：
```
scontrol update nodename=node-5 state=drain
```


## 因进程退出缓慢而被杀死的节点的排空

有时，当作业被取消时，进程会很慢才退出。如果 SLURM 配置为不永远等待，它会自动排空这些节点。但没有理由让这些节点对用户不可用。

所以这里是如何自动化它的。

关键是获取因 `"Kill task failed"` 而被排空的节点列表，这可以通过以下命令检索：

```
sinfo -R | grep "Kill task failed"
```

现在提取并展开节点列表，检查节点确实没有用户进程（或者先尝试杀死它们），然后取消排空。

之前您学习了如何[在多个节点上运行命令](#run-a-command-on-multiple-nodes)，我们将在此脚本中使用它。

这里有一个脚本可以为您完成所有这些工作：[undrain-good-nodes.sh](./undrain-good-nodes.sh)

现在您只需运行此脚本，任何基本上已准备好服务但当前被排空的节点都将切换到 `idle` 状态，并可供用户使用。


## 修改作业的时间限制

要为作业设置新的时间限制，例如 2 天：
```
scontrol update JobID=$SLURM_JOB_ID TimeLimit=2-00:00:00
```

要为之前的设置增加额外的时间，例如再增加 3 小时。
```
scontrol update JobID=$SLURM_JOB_ID TimeLimit=+10:00:00
```

## 当 SLURM 出现问题时

分析 SLURM 日志文件中的事件日志：
```
sudo cat /var/log/slurm/slurmctld.log
```

例如，这可以帮助理解为什么某个节点的作业会提前被取消，或者该节点被完全移除。
