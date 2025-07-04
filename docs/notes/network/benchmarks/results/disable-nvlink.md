---
title: disable-nvlink
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/qaiwjh4o/
---
# 禁用 NVLink 基准测试

让我们比较一个 gpt2 语言模型在一个小的 wikitext 样本上的训练。

结果如下：

| NVlink | 时间 |
| -----  | ---: |
| Y      | 101s |
| N      | 131s |

您可以看到，NVLink 完成训练的速度快了约 23%。在第二个基准测试中，我们使用 `NCCL_P2P_DISABLE=1` 来告诉 GPU 不要使用 NVLink，这将改用 PCIe。

我们将使用 [HF Transformers 示例](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/examples/pytorch/language-modeling/run_clm.py)。

以下是完整的基准测试代码和输出：

```bash
# DDP w/ NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

硬件：2x TITAN RTX 24GB each + NVlink with 2 NVLinks (`NV2` in `nvidia-smi topo -m`)
软件：`pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`
