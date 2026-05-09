---
date: '2026-05-09T12:00:00+08:00'
draft: false
title: '大模型面试精讲（三）：训练与推理加速'
tags: ["LLM", "面试", "KV Cache", "vLLM", "FlashAttention", "并行训练", "ZeRO"]
summary: '从 Prefill/Decode 到并行训练，系统整理大模型训练与推理优化的面试考点。'
---

这一章混合了推理阶段的问题、训练阶段的问题、内存管理问题、并行系统问题和算子级优化问题。理解顺序：先看推理在做什么 -> 推理瓶颈落在 KV Cache 和访存 -> vLLM / PagedAttention / Continuous Batching -> 训练显存和并行优化 -> FlashAttention、MQA/GQA。

参考教材：大模型面试题精讲教材

<!--more-->

## 一、推理阶段到底在做什么：Prefill 和 Decode

Transformer 推理分成两个阶段：

### 1.1 Prefill

把整段 prompt 一次性送进模型。对这 $n$ 个 token 做完整前向计算，建立每层的 hidden states，建立每层每个位置的 K/V，为后续生成准备 KV Cache。

- 输入长度通常较长
- attention 要处理整段上下文两两关系
- 更像一次大矩阵并行计算
- 吞吐更重要
- 复杂度受 $O(n^2 d_{model})$ 主导

### 1.2 Decode

进入自回归生成阶段后，每次只生成一个新 token。读取历史 KV Cache，只计算当前新 token 的 Q/K/V，用当前 Q 去和历史 K 做匹配。

- 一次只生成一个 token
- 算力不一定最贵，访存常常更关键
- 更关注单步延迟

### 1.3 为什么要把 Prefill 和 Decode 分开讲

因为它们的瓶颈不一样：Prefill 更像大计算吞吐问题，Decode 更像 KV Cache 读取和内存带宽问题。

## 二、KV Cache：推理优化的核心对象

### 2.1 为什么需要 KV Cache

自回归生成时，第 $t$ 步输出依赖前 $t-1$ 步上下文。如果每次生成一个新 token 都把整个历史序列重新算一遍 K/V，会有大量重复计算。实际做法是历史 token 的 K/V 只计算一次、存起来、后续步骤直接复用。

### 2.2 收益与代价

**收益**：避免重复计算历史 K/V，把 decode 的单步开销从"重算整段前缀"降成"只处理新 token + 读取历史缓存"。

**代价**：占显存，上下文越长占用越大，batch 越大占用越大，decode 时频繁读取历史 K/V，访存压力非常高。

一句话：KV Cache 用显存换时间。

### 2.3 为什么大模型推理经常卡在 KV Cache

decoder-only LLM 在线服务常常是上下文很长、并发请求很多、每步都要读取历史 K/V。真正的瓶颈往往不再是模型参数本身，而是 KV Cache 太大、读写太频繁、显存带宽压力太高。

## 三、vLLM：为什么它会快

vLLM 的强项不只是"attention kernel 写得好"，而是在两个层面上做了关键优化：

- **PagedAttention**：KV Cache 的内存管理
- **Continuous Batching**：请求级调度

## 四、PagedAttention：解决 KV Cache 的碎片问题

### 4.1 问题背景

在线推理里，请求长度和完成时间都不一致。KV Cache 会随着生成过程动态增长。如果每个请求都要求一大块连续显存，就很容易出现显存碎片——明明总空闲空间够，但连续空闲空间不够。

### 4.2 核心思想

借用"分页"思想：一个请求的 KV Cache 在逻辑上是连续的，但在物理显存上不要求连续。系统把 KV Cache 切成固定大小 block / page，再通过映射关系管理这些 block。

核心不是"改 attention 数学公式"，而是"改 KV Cache 的存储组织方式"。

### 4.3 一句话记忆

PagedAttention = 用分页方式管理 KV Cache。减少显存碎片，让 KV Cache 动态增长更灵活，提高显存利用率。

## 五、Continuous Batching：解决请求调度问题

### 5.1 传统 batching 的问题

如果系统总是"凑一批请求，一起开始，一起结束"，短请求被长请求拖住，GPU 某些时刻空闲，新请求不能及时插入。

### 5.2 核心思想

- 正在运行的 batch 中，可以动态插入新请求
- 已完成的请求及时退出
- 未完成的请求继续留在批里

这让系统更接近"流式调度"，而不是"整批整批处理"。

### 5.3 和 PagedAttention 的关系

- PagedAttention：偏内存管理
- Continuous Batching：偏调度管理

两者结合，才是 vLLM 在线推理强的原因。

## 六、MHA、MQA、GQA、MLA：从结构上减轻推理负担

这组方法的核心问题是：怎样降低 attention 尤其是 KV Cache 的成本。

### 6.1 标准 MHA

每个 head 有自己的 Q、K、V。推理时需要缓存多组 K/V，KV Cache 开销最大。

### 6.2 MQA（Multi-Query Attention）

保留多个 Q head，但所有 head 共享同一组 K/V。显著降低 KV Cache，但也可能带来更大的表达能力折损。

### 6.3 GQA（Grouped-Query Attention）

query 头很多，但按组共享 K/V。是 MHA 和 MQA 之间的折中：比 MHA 更省 KV Cache，比 MQA 通常保留更多效果。

### 6.4 MLA

更进一步压缩 attention 状态表示的思路，目标仍然是降低 KV Cache、降低访存、提升大模型推理效率。

### 6.5 真正优化的是哪里

主要优化推理阶段 attention 的存储和访存成本，特别是 KV Cache 的大小和读取带宽压力。不是主要在解决序列长度的 $n^2$ 注意力复杂度本身或 FFN 的计算复杂度。

## 七、FlashAttention：算子级优化

### 7.1 标准 attention 的瓶颈

很多人以为 attention 慢是因为算力不够，但现代 GPU 上常见的真正瓶颈是：attention 中间结果太大、显存读写太频繁、HBM 带宽成为限制因素。

### 7.2 FlashAttention 做了什么

没有改 attention 的数学定义，而是：

- 分块处理 Q/K/V
- 避免显式 materialize 完整 attention matrix
- 尽量减少 HBM 读写
- 重新组织计算顺序，让更多工作在更快的片上存储完成

关键词：IO-aware。

### 7.3 收益

更快、更省显存、长序列时收益更明显。优化的是 attention 的访存路径，而不是理论公式。

## 八、训练阶段怎么省显存：梯度检查点和梯度累计

训练除了参数，还要存激活值、梯度、优化器状态，显存压力通常更大。

### 8.1 梯度检查点

本质：用额外计算换显存。

- 前向时不保存全部中间激活
- 反向时需要用到时再重算
- 显著降低激活显存，但训练更慢

### 8.2 梯度累计

本质：用多个小 batch 近似一个大 batch。

- 多个 micro-batch 连续反向
- 暂不更新参数
- 累积若干步后再统一更新
- 在显存有限时扩大等效 batch size

## 九、并行训练：DP、TP、PP、ZeRO

当模型变得很大时，单卡通常放不下，需要并行训练。

### 9.1 DP / DDP

每张卡保存完整模型，数据按 batch 切分，反向后同步梯度。逻辑最简单、易扩展，但单卡仍必须装得下完整模型。

### 9.2 TP（Tensor Parallelism）

把同一层的权重矩阵切到多张卡，一层内部由多卡一起算。单层太大时也能训练，但通信频繁，对高速互联要求高。

### 9.3 PP（Pipeline Parallelism）

按层切模型，不同设备负责不同层段。参数显存分担明显，但有 pipeline bubble，调度复杂。

### 9.4 ZeRO

核心不是按层切模型，而是把训练状态分片：

- **Stage 1**：分优化器状态
- **Stage 2**：再分梯度
- **Stage 3**：连参数也分

显著降低每卡状态开销，让大模型更容易训得动。

## 十、通信原语：系统问题的基本语言

并行训练里很多实现细节，本质上都落在这些通信操作上：

- Broadcast、Reduce、AllReduce、AllGather、Reduce-Scatter

最常记住的关系：DDP 里常见 AllReduce，参数分片恢复常见 AllGather，通信优化里常见 Reduce-Scatter。

## 十一、显存分析：训练和推理为什么都很贵

### 11.1 训练显存主要包括

- 参数
- 梯度
- 优化器状态
- 激活值

常见重灾区：Adam 类优化器状态、长序列下的激活值。

### 11.2 推理显存主要包括

- 模型参数
- KV Cache
- 中间激活
- 运行时框架开销

长上下文在线推理时，KV Cache 很容易成为第一瓶颈。

## 十二、混合精度为什么几乎是标配

混合精度训练常把一部分计算放到 FP16 / BF16：

- 降低显存
- 提高吞吐
- 更好利用 Tensor Core

FP16 更容易数值溢出，BF16 动态范围更大，工程上通常更稳。

## 十三、面试作答框架

如果面试官问"怎么做大模型训练和推理优化"，比较清晰的回答顺序是：

1. 推理上，先区分 Prefill 和 Decode
2. Decode 阶段核心瓶颈常在 KV Cache 和访存
3. 系统上用 PagedAttention 和 Continuous Batching 提高利用率
4. 结构上用 MQA/GQA 降低 KV Cache
5. 算子上用 FlashAttention 减少 IO
6. 训练上再用梯度检查点、梯度累计、并行策略和混合精度解决显存与吞吐问题

一句话总结：推理瓶颈常在 KV Cache 和访存，训练瓶颈常在激活、优化器状态和并行通信，所有优化手段本质都在围绕"算力、显存、带宽、调度"这四件事做权衡。
