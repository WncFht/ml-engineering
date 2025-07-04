---
title: 自述文件
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/9lnozt9d/
---
# 检查点

- [torch-checkpoint-convert-to-bf16](./torch-checkpoint-convert-to-bf16) - 将现有的 fp32 torch 检查点转换为 bf16。如果找到 [safetensors](https://github.com/huggingface/safetensors/)，也会一并转换。应该可以轻松适应其他类似的用例。

- [torch-checkpoint-shrink.py](./torch-checkpoint-shrink.py) - 此脚本修复了由于某种原因在保存时存储的张量其存储空间大于其视图的检查点。它会克隆当前视图并仅使用当前视图的存储空间重新保存它们。
