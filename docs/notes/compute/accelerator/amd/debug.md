---
title: debug
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/gh4bo2pl/
---
# AMD GPU 故障排除

XXX：这部分内容还很初步 - 正在收集各种工具/笔记

由于我们大多数人对 NVIDIA 工具非常熟悉，我将尽可能提供与熟悉工具的映射关系。

## 工具

### ROCR_VISIBLE_DEVICES

要选择特定的 GPU（相当于 `CUDA_VISIBLE_DEVICES`）：

```
ROCR_VISIBLE_DEVICES=0,1 python my-program.py
```

### rocm-smi

`rocm-smi`（相当于 `nvidia-smi`）显示所有 ROCm 加速器的简明状态。

例如，这是一个 8xMI300X 节点：
```
$ rocm-smi
========================================= ROCm 系统管理界面 =========================================
=================================================== 简明信息 ===================================================
设备  [型号 : 版本]    温度        功率     分区      SCLK    MCLK    风扇  性能  功耗上限  VRAM%  GPU%
        名称 (20 字符)       (结点)  (插槽)  (内存, 计算)
====================================================================================================================
0       [0x74a1 : 0x00]       45.0°C      173.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
1       [0x74a1 : 0x00]       41.0°C      179.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
2       [0x74a1 : 0x00]       47.0°C      180.0W    NPS1, SPX       131Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
3       [0x74a1 : 0x00]       45.0°C      178.0W    NPS1, SPX       131Mhz  900Mhz  0%   auto  750.0W   17%   0%
        AMD Instinct MI300X
4       [0x74a1 : 0x00]       45.0°C      175.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
5       [0x74a1 : 0x00]       43.0°C      175.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
6       [0x74a1 : 0x00]       45.0°C      175.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
7       [0x74a1 : 0x00]       43.0°C      176.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
====================================================================================================================
=============================================== ROCm SMI 日志结束 ================================================
```

奇怪的是，它没有显示真实的内存使用情况——只显示了百分比，这不太实用。

一个方便的别名可以实时查看更新：
```
alias wr='watch -n 1 rocm-smi'
```

### rocminfo

`rocminfo`（相当于 `nvidia-smi -q`）显示每个加速器的详细信息。

这个会同时显示 CPU 和 GPU 的信息

以下是 cpu0 和 gpu0 的代码片段（注意它从节点 0..1 开始计算 CPU，然后从节点 2..9 开始计算 GPU）：
```
$ rocminfo
ROCk 模块已加载
=====================
HSA 系统属性
=====================
运行时版本:         1.1
系统时间戳频率:  1000.000000MHz
信号最大等待时间:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (时间戳计数)
机器型号:           LARGE
系统字节序:       LITTLE
Mwaitx:                  禁用
DMAbuf 支持:          是

==========
HSA 代理
==========
*******
代理 1
*******
  名称:                    AMD EPYC 9534 64-Core Processor
  Uuid:                    CPU-XX
  营销名称:          AMD EPYC 9534 64-Core Processor
  供应商名称:             CPU
  特性:                 未指定
  配置文件:                 FULL_PROFILE
  浮点舍入模式:        NEAR
  最大队列数:        0(0x0)
  队列最小尺寸:          0(0x0)
  队列最大尺寸:          0(0x0)
  队列类型:              MULTI
  节点:                    0
  设备类型:             CPU
  缓存信息:
    L1:                      32768(0x8000) KB
  芯片 ID:                 0(0x0)
  ASIC 版本:           0(0x0)
  缓存行大小:          64(0x40)
  最大时钟频率 (MHz):   2450
  BDFID:                   0
  内部节点 ID:        0
  计算单元:            128
  每个 CU 的 SIMD 数:            0
  着色器引擎:          0
  每个引擎的着色器数组数:   0
  地址范围上的观察点:1
  特性:                无
  池信息:
    池 1
      段:                 GLOBAL; FLAGS: FINE GRAINED
      大小:                    792303268(0x2f3996a4) KB
      可分配:             TRUE
      分配粒度:           4KB
      分配对齐:         4KB
      所有代理可访问:       TRUE
    池 2
      段:                 GLOBAL; FLAGS: KERNARG, FINE GRAINED
      大小:                    792303268(0x2f3996a4) KB
      可分配:             TRUE
      分配粒度:           4KB
      分配对齐:         4KB
      所有代理可访问:       TRUE
    池 3
      段:                 GLOBAL; FLAGS: COARSE GRAINED
      大小:                    792303268(0x2f3996a4) KB
      可分配:             TRUE
      分配粒度:           4KB
      分配对齐:         4KB
      所有代理可访问:       TRUE
  ISA 信息:
[...]

  名称:                    gfx942
  Uuid:                    GPU-ababaeeffecddc50
  营销名称:          AMD Instinct MI300X
  供应商名称:             AMD
  特性:                 KERNEL_DISPATCH
  配置文件:                 BASE_PROFILE
  浮点舍入模式:        NEAR
  最大队列数:        128(0x80)
  队列最小尺寸:          64(0x40)
  队列最大尺寸:          131072(0x20000)
  队列类型:              MULTI
  节点:                    2
  设备类型:             GPU
  缓存信息:
    L1:                      16(0x10) KB
    L2:                      8192(0x2000) KB
  芯片 ID:                 29857(0x74a1)
  ASIC 版本:           1(0x1)
  缓存行大小:          64(0x40)
  最大时钟频率 (MHz):   2100
  BDFID:                   50688
  内部节点 ID:        7
  计算单元:            304
  每个 CU 的 SIMD 数:            4
  着色器引擎:          32
  每个引擎的着色器数组数:   1
  地址范围上的观察点:4
  一致性主机访问:    FALSE
  特性:                KERNEL_DISPATCH
  快速 F16 操作:      TRUE
  波前大小:          64(0x40)
  工作组最大大小:      1024(0x400)
  每个维度的工作组最大大小:
    x                        1024(0x400)
    y                        1024(0x400)
    z                        1024(0x400)
  每个 CU 的最大波前数:        32(0x20)
  每个 CU 的最大工作项数:    2048(0x800)
  网格最大大小:           4294967295(0xffffffff)
  每个维度的网格最大大小:
    x                        4294967295(0xffffffff)
    y                        4294967295(0xffffffff)
    z                        4294967295(0xffffffff)
  最大 fbarriers/Workgrp:   32
  数据包处理器 uCode:: 132
  SDMA 引擎 uCode::      19
  IOMMU 支持::          无
  池信息:
    池 1
      段:                 GLOBAL; FLAGS: COARSE GRAINED
      大小:                    201310208(0xbffc000) KB
      可分配:             TRUE
      分配粒度:           4KB
      分配对齐:         4KB
      所有代理可访问:       FALSE
    池 2
      段:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
      大小:                    201310208(0xbffc000) KB
      可分配:             TRUE
      分配粒度:           4KB
      分配对齐:         4KB
      所有代理可访问:       FALSE
    池 3
      段:                 GLOBAL; FLAGS: FINE GRAINED
      大小:                    201310208(0xbffc000) KB
      可分配:             TRUE
      分配粒度:           4KB
      分配对齐:         4KB
      所有代理可访问:       FALSE
    池 4
      段:                 GROUP
      大小:                    64(0x40) KB
      可分配:             FALSE
      分配粒度:           0KB
      分配对齐:         0KB
      所有代理可访问:       FALSE
  ISA 信息:
    ISA 1
      名称:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
      机器型号:          HSA_MACHINE_MODEL_LARGE
      配置文件:                HSA_PROFILE_BASE
      默认舍入模式:   NEAR
      默认舍入模式:   NEAR
      快速 f16:                TRUE
      工作组最大大小:      1024(0x400)
      每个维度的工作组最大大小:
        x                        1024(0x400)
        y                        1024(0x400)
        z                        1024(0x400)
      网格最大大小:           4294967295(0xffffffff)
      每个维度的网格最大大小:
        x                        4294967295(0xffffffff)
        y                        4294967295(0xffffffff)
        z                        4294967295(0xffffffff)
      FBarrier 最大大小:       32
```
