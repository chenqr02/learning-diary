---
date: '2026-05-09T12:00:00+08:00'
draft: false
title: '大模型面试精讲（三）：训练与推理加速'
tags: ["LLM", "面试", "KV Cache", "vLLM", "FlashAttention", "并行训练", "ZeRO", "量化", "投机解码", "MoE", "RoPE", "推理系统"]
summary: '从 Prefill/Decode 到并行训练，从量化到投机解码，系统整理大模型训练与推理优化的面试考点。'
---

这一章混合了推理阶段的问题、训练阶段的问题、内存管理问题、并行系统问题和算子级优化问题。理解顺序：先看推理在做什么 -> 推理瓶颈落在 KV Cache 和访存 -> 量化/投机解码/长上下文优化 -> vLLM / PagedAttention / Continuous Batching -> 训练显存和并行优化 -> FlashAttention、MQA/GQA。

参考教材：大模型面试题精讲教材、Efficient LLM Inference (AI Engineering Insider)

<!--more-->

## 一、推理阶段到底在做什么：Prefill 和 Decode

Transformer 推理分成两个阶段：

### 1.1 Prefill

把整段 prompt 一次性送进模型。对这 $n$ 个 token 做完整前向计算，建立每层的 hidden states，建立每层每个位置的 K/V，为后续生成准备 KV Cache。

- 输入长度通常较长
- attention 要处理整段上下文两两关系
- 更像一次大矩阵并行计算（GEMM：$[n, d] \times [d, 4d]$）
- 吞吐更重要，计算密集（compute-bound）
- 复杂度受 $O(n^2 d_{model})$ 主导
- GPU Tensor Core 利用率高

### 1.2 Decode

进入自回归生成阶段后，每次只生成一个新 token。读取历史 KV Cache，只计算当前新 token 的 Q/K/V，用当前 Q 去和历史 K 做匹配。

- 一次只生成一个 token（GEMV：$[1, d] \times [d, 4d]$）
- 算力不一定最贵，**访存常常更关键**（memory-bandwidth-bound）
- 更关注单步延迟
- Tensor Core 利用率通常 <10%

### 1.3 为什么要把 Prefill 和 Decode 分开讲

因为它们的瓶颈不一样：Prefill 更像大计算吞吐问题，Decode 更像 KV Cache 读取和内存带宽问题。现代系统如 Splitwise、Mooncake 甚至把 Prefill 和 Decode 物理分离到不同的 GPU 池。

### 1.4 Roofline 模型：判断计算瓶颈

Roofline 模型提供了一个判断操作是 compute-bound 还是 memory-bound 的框架：

$$\text{Achieved Performance (FLOP/s)} = \min(\text{Peak FLOP/s}, \text{Arithmetic Intensity} \times \text{Peak Bandwidth})$$

其中 Arithmetic Intensity = FLOP / byte（每读取一个字节要做多少次浮点运算）。

**关键洞察**：

| 操作 | 算术强度 | 瓶颈类型 |
|------|---------|---------|
| 矩阵-向量乘（decode, batch=1） | 1-2 FLOP/byte | memory-bound |
| 矩阵-矩阵乘（prefill, 长 prompt） | 高（$\sim 2n/d$） | compute-bound |

**内存带宽才是真正的瓶颈**：一个 7B 参数模型 FP16 占 14 GB，A100 HBM 带宽 2 TB/s，仅加载一次权重就需要 7 ms——这还没算任何计算。batch size 1 时这个开销无法摊销。

## 二、KV Cache：推理优化的核心对象

### 2.1 为什么需要 KV Cache

自回归生成时，第 $t$ 步输出依赖前 $t-1$ 步上下文。如果每次生成一个新 token 都把整个历史序列重新算一遍 K/V，会有大量重复计算。实际做法是历史 token 的 K/V 只计算一次、存起来、后续步骤直接复用。

### 2.2 KV Cache 显存公式

$$\text{Memory} = 2 \times L \times n \times n_{heads} \times d_{head} \times \text{bytes\_per\_element}$$

以 LLaMA-2 70B（80 层，64 头，head_dim=128，FP16，上下文 4096）为例：

$$2 \times 80 \times 4096 \times 64 \times 128 \times 2 \approx 10.7 \text{ GB}$$

### 2.3 收益与代价

**收益**：避免重复计算历史 K/V，把 decode 的单步开销从"重算整段前缀"降成"只处理新 token + 读取历史缓存"。

**代价**：占显存，上下文越长占用越大，batch 越大占用越大，decode 时频繁读取历史 K/V，访存压力非常高。

一句话：KV Cache 用显存换时间。

### 2.4 为什么大模型推理经常卡在 KV Cache

decoder-only LLM 在线服务常常是上下文很长、并发请求很多、每步都要读取历史 K/V。真正的瓶颈往往不再是模型参数本身，而是 KV Cache 太大、读写太频繁、显存带宽压力太高。

## 三、vLLM：为什么它会快

vLLM 的强项不只是"attention kernel 写得好"，而是在两个层面上做了关键优化：

- **PagedAttention**：KV Cache 的内存管理
- **Continuous Batching**：请求级调度

## 四、PagedAttention：解决 KV Cache 的碎片问题

### 4.1 问题背景

在线推理里，请求长度和完成时间都不一致。KV Cache 会随着生成过程动态增长。如果每个请求都要求一大块连续显存，就很容易出现显存碎片——明明总空闲空间够，但连续空闲空间不够。传统方案预先分配最大长度的连续内存，严重浪费。

### 4.2 核心思想

借用操作系统"分页"思想：一个请求的 KV Cache 在逻辑上是连续的，但在物理显存上不要求连续。系统把 KV Cache 切成固定大小 block / page（通常每块 16 tokens），再通过 block table 映射逻辑位置到物理块。

核心不是"改 attention 数学公式"，而是"改 KV Cache 的存储组织方式"。

### 4.3 关键特性

| 特性 | 说明 |
|------|------|
| 按需分配 | 不预分配，生成过程中动态分配 block |
| 无内部碎片 | 每请求最多浪费 $B-1$ tokens 的空间 |
| Copy-on-Write | 共享前缀的请求可共享物理 block，直到路径分叉 |
| 灵活换出 | block 可换出到 CPU/磁盘，为其他请求腾空间 |

PagedAttention 需要修改 attention kernel 以支持非连续 block 的 gather 操作，vLLM 通过自定义 CUDA kernel 实现。

### 4.4 一句话记忆

PagedAttention = 用分页方式管理 KV Cache。减少显存碎片，让 KV Cache 动态增长更灵活，提高显存利用率。通常能让同等 GPU 内存下的 batch size 提升 2-4 倍。

## 五、Continuous Batching：解决请求调度问题

### 5.1 传统 batching 的问题

如果系统总是"凑一批请求，一起开始，一起结束"，短请求被长请求拖住，GPU 某些时刻空闲，新请求不能及时插入。padding 浪费大量算力。

### 5.2 核心思想

由 Orca（Yu et al., 2022）提出，vLLM 实现：

- 在每个 decode step 级别做调度（而非请求级别）
- 正在运行的 batch 中，可以动态插入新请求
- 已完成的请求及时退出
- 未完成的请求继续留在批里

这让系统更接近"流式调度"，而不是"整批整批处理"。实际吞吐比 static batching 提升 5-23 倍。

### 5.3 Chunked Prefill：解决 Prefill 抢占问题

Sarathi-Serve（Agrawal et al., 2024）发现长 prompt 的 prefill 会独占 GPU 数百毫秒，导致 decode 请求卡住（"prefill piracy"）。Chunked Prefill 把长 prompt 的 prefill 切成小块（如 512 tokens），与 decode 步骤交错执行，显著降低 decode 端的 TPOT 抖动。

### 5.4 和 PagedAttention 的关系

- PagedAttention：偏内存管理
- Continuous Batching：偏调度管理

两者结合，才是 vLLM 在线推理强的原因。

## 六、KV Cache 进阶优化

### 6.1 Prefix Caching（Prompt Caching）

很多生产场景中，大量请求共享相同的前缀（系统提示词、文档、few-shot 示例）。Prefix Caching 把共享前缀的 KV Cache 缓存起来复用，避免每次请求都重新计算。vLLM、SGLang、Anthropic/OpenAI 均已支持。

- 可减少 90%+ 的 prefill 开销
- 对固定系统提示词的 API 产品收益最大

### 6.2 StreamingLLM：Attention Sink 现象

对于需要无限长上下文的场景（流式对话、长运行 Agent），无法存储所有 token 的 KV Cache。朴素的滑动窗口（只保留最近 W tokens）会导致性能灾难性下降。

原因：LLM 会发展出"attention sink"——序列最前面的几个 token（无论语义内容如何）在所有层都收到不成比例的高 attention score。移除它们会破坏模型内部表示。

StreamingLLM（Xiao et al., 2023）的解决方案：

$$\text{KV retained} = \text{sink tokens (通常4个)} + \text{recent window}$$

### 6.3 KV Cache 量化

KV Cache 本身可以独立于模型权重进行量化。因为每个 attention step 只读取一次，量化误差不会跨步累积。

- INT8 KV Cache 量化已是标准做法（TensorRT-LLM、vLLM），KV 内存减半
- 更激进的方案用 INT4 或 FP8，需要 per-head 或 per-channel scaling

## 七、MHA、MQA、GQA、MLA、RoPE：从结构上减轻推理负担

这组方法的核心问题是：怎样降低 attention 尤其是 KV Cache 的成本。

### 7.1 标准 MHA

每个 head 有自己的 Q、K、V。推理时需要缓存多组 K/V，KV Cache 开销最大。

### 7.2 MQA（Multi-Query Attention）

保留多个 Q head，但所有 head 共享同一组 K/V。显著降低 KV Cache（减少 $H$ 倍），但也可能带来更大的表达能力折损。

### 7.3 GQA（Grouped-Query Attention）

query 头很多，但按组共享 K/V。是 MHA 和 MQA 之间的折中：比 MHA 更省 KV Cache，比 MQA 通常保留更多效果。LLaMA-3、Mistral、Gemma 均采用。

### 7.4 MLA（Multi-head Latent Attention）

DeepSeek-V2 提出。将 K/V 投影到低维潜在空间 $c \ll d$，KV Cache 只存储潜在向量。比 MHA 减少 5-13 倍 KV Cache，质量损失极小。

**KV Cache 大小比例**：MHA : GQA : MLA $\approx$ 8 : 2 : 1

### 7.5 RoPE（旋转位置编码）

RoPE 是当前 decoder-only LLM 的主流位置编码方案。通过在复数平面上旋转 Q/K 向量来编码位置：

$$q_m = q \cdot e^{im\theta}, \quad k_n = k \cdot e^{in\theta}$$

使得 $\langle q_m, k_n \rangle$ 只依赖相对位置 $m-n$。

**推理时的上下文外推**：RoPE 允许通过 NTK-aware scaling 或 YaRN 进行上下文长度外推，使在 4K 上下文训练的模型能泛化到 128K+。

### 7.6 真正优化的是哪里

主要优化推理阶段 attention 的存储和访存成本，特别是 KV Cache 的大小和读取带宽压力。不是主要在解决序列长度的 $n^2$ 注意力复杂度本身或 FFN 的计算复杂度。

## 八、FlashAttention：算子级优化

### 8.1 标准 attention 的瓶颈

很多人以为 attention 慢是因为算力不够，但现代 GPU 上常见的真正瓶颈是：attention 中间结果太大、显存读写太频繁、HBM 带宽成为限制因素。

标准 attention 需要把完整的 $n \times n$ attention matrix 写入 HBM。$n=16384$、FP16 时，这个矩阵约 537 MB。读写它在 A100 上需要 0.5 ms。

### 8.2 FlashAttention 做了什么

没有改 attention 的数学定义，而是：

- 分块处理 Q/K/V，让计算在 SRAM 中完成
- 避免显式 materialize 完整 attention matrix
- 尽量减少 HBM 读写
- 利用 online softmax trick 实现增量 softmax 计算

关键词：IO-aware。

| 对比 | 标准 Attention | FlashAttention |
|------|---------------|----------------|
| HBM 读写 | $O(n^2)$ | $O(n^2/M)$（更少 pass） |
| SRAM 使用 | $O(n^2)$ | $O(M)$（tile 大小） |
| 速度 | 基线 | 2-4x 加速 |

FlashAttention-2 改进了并行性，FlashAttention-3 利用 H100 特性（WGMMA、FP8）达到 H100 峰值 75% 的 FLOP/s 利用率。

### 8.3 Online Softmax Trick

标准 softmax 需要看到整行所有值才能计算，无法分块。Online softmax 维护运行中的最大值 $m$ 和指数和 $\sigma$，分块处理时逐步更新：

$$m_i = \max(m_{i-1}, \max_j s_{ij}), \quad \sigma_i = e^{m_{i-1}-m_i} \sigma_{i-1} + \sum_j e^{s_{ij}-m_i}$$

最终 $O = O_{running} / \sigma$，数学上与标准 softmax 完全等价。

### 8.4 Triton：Pythonic Kernel 编程

CUDA kernel 开发需要 C++ 和 GPU 底层知识。Triton 是基于 Python 的 DSL，自动处理 shared memory 分配、warp 同步和向量化，生成高度优化的 GPU kernel。开发时间从 CUDA 的数周缩短到 Triton 的数天，性能通常能达到手写 CUDA 的 80-90%。

## 九、量化（Quantization）

### 9.1 为什么量化至关重要

70B 参数模型 FP32 需要 280 GB，FP16 需要 140 GB，INT8 需要 70 GB（一张 H100 装得下），INT4 只需 35 GB。量化不是学术研究——它是"能不能部署"的分水岭。

### 9.2 数值格式

| 格式 | 位数 | 动态范围 | 特点 |
|------|------|---------|------|
| FP32 | 32 | $\pm 3.4\times10^{38}$ | 全精度，训练标准 |
| BF16 | 16 | $\pm 3.4\times10^{38}$ | 与 FP32 相同指数范围，更少尾数精度 |
| FP16 | 16 | $\pm 65504$ | 常见推理格式，有溢出风险 |
| FP8 E4M3 | 8 | $\pm 448$ | H100 原生支持 |
| INT8 | 8 | -128~127 | 广泛支持，2x 尺寸压缩 |
| INT4 | 4 | -8~7 | 4x 压缩，需仔细校准 |

### 9.3 训练后量化（PTQ）

**GPTQ**（Frantar et al., 2022）：逐层二阶 PTQ 方法。逐行量化权重，用 Hessian 逆矩阵补偿已量化权重的误差。对 30B+ 参数模型可实现接近无损的 4-bit 量化。

**AWQ**（Lin et al., 2023）：观察到并非所有权重同等重要——对应大激活值的权重对量化误差更敏感。AWQ 在量化前对这些显著通道进行缩放保护：

$$Q(W \cdot \text{diag}(s)^{-1}) \cdot \text{diag}(s) \approx W$$

AWQ 在 4-bit 精度上通常优于 GPTQ，且更快，是生产中广泛使用的选择。

**QuIP#** 和 **AQLM**：更激进的方法。QuIP# 通过不相干处理（随机旋转权重和激活）将分布展平，实现接近无损的 2-bit 量化。AQLM 使用码本方法，2-bit 压缩质量接近 4-bit 方法。

### 9.4 量化感知训练（QAT）

PTQ 在小数据集上校准后应用量化，精度恢复有限。QAT 在微调的前向传播中引入伪量化算子：

$$\hat{x} = \text{round}\left(\frac{x}{s}\right) \cdot s$$

梯度通过 Straight-Through Estimator（STE）反向传播：将 round 函数的梯度近似为 1。QAT 在极端位宽（2-3 bit）下精度优于 PTQ，但需要大量算力。

### 9.5 激活量化挑战

权重量化相对简单（静态分布，可离线分析）。激活量化更难：LLM 存在"离群通道"——少量通道的激活幅度比典型通道大 100 倍以上。朴素 INT8 激活量化被迫用大 scale 适应离群值，导致正常范围的激活坍缩为几个离散值。

**解决方案**：

| 方法 | 思路 |
|------|------|
| LLM.int8() | 将离群通道保持 FP16，其余 INT8 |
| SmoothQuant | 通过 per-channel 缩放将量化难度从激活迁移到权重：$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$ |
| FP8 激活 | FP8 比 INT8 动态范围更大，天然适应离群值 |

### 9.6 Weight-only vs Weight-and-Activation 量化

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| W4A16 / W8A16 | 只量化权重，激活保持 FP16 | decode 阶段（memory-bound） |
| W8A8 / FP8 | 权重和激活都量化 | prefill 或大 batch（compute-bound） |

## 十、投机解码（Speculative Decoding）

### 10.1 核心洞察：Draft and Verify

验证一段 token 序列是否由大模型生成，可以在一次并行前向传播中完成；而自回归生成这些 token 需要每 token 一次前向传播。利用这种不对称性：

1. 用小 draft 模型（3-7B）便宜地自回归生成 $K$ 个候选 token
2. 用大 target 模型对所有 $K$ 个 token 做一次并行前向传播
3. 用拒绝采样验证每个 draft token，接受匹配的，拒绝不匹配的
4. 结果数学上等价于从 target 模型分布中采样

**期望加速**：$\frac{1}{1-\alpha}$，其中 $\alpha$ 是每 token 接受率。$\alpha=0.8, K=5$ 时，期望每步接受 3.3 个 token。

### 10.2 Token Tree 验证

基本投机解码只生成一条 draft 序列。Tree-based 方法在每个位置生成多个候选 token，形成树结构。Target 模型用 tree-masked attention 一次验证所有分支，大幅提高高接受率的概率。

### 10.3 Medusa：多头投机解码

Medusa（Cai et al., 2024）不需要独立 draft 模型。在 target 模型上加 $K$ 个轻量线性头，每个头预测第 $t+k$ 个 token：

$$\text{logits}_{t+k} = \text{MedusaHead}_k(h_t)$$

实现自投机（self-speculative），Medusa-2 加入自蒸馏，达到 2-3x 加速。

### 10.4 EAGLE

EAGLE（Li et al., 2024）解决了 Medusa 的关键限制：仅从最后隐藏状态预测未来 token 丢失了序列信息。EAGLE 训练轻量 draft 模型，输入同时包含隐藏状态和已生成 token 的特征嵌入，产生更准确的预测。EAGLE-2 引入动态 draft 树，根据预测置信度自适应调整树结构，达到 3-4x 加速。

### 10.5 投机解码的适用场景

| 适合 | 不适合 |
|------|--------|
| 单流低延迟服务（batch=1） | 高 batch 生产服务（GPU 已饱和） |
| 高接受率任务（代码补全、结构化生成） | 低接受率任务（高温度创意生成） |
| GPU 显存充足 | 显存紧张 |
| 输出较长 | 输出极短（1-5 token） |

**注意**：投机解码是延迟优化，不是吞吐优化。

## 十一、长上下文与内存管理

### 11.1 二次方注意力问题

标准 attention 的 $O(n^2)$ 复杂度是长上下文的核心障碍。$n=128K$ 时，attention matrix 约 32 GB；$n=1M$ 时约 2 PB——完全不可行。

### 11.2 FlashAttention 与长上下文

FlashAttention 不降低 $O(n^2)$ 计算复杂度，但将 HBM 内存需求从 $O(n^2)$ 降到 $O(n)$，使 128K token 的 attention 在当前硬件上可行。

### 11.3 Ring Attention

Ring Attention（Liu et al., 2023）将长上下文 attention 分布到多个设备上。每个设备持有序列的一个 chunk 的 QKV 张量，设备间以环形传递 KV chunk。利用 online softmax trick，每个设备只看到一个 KV chunk 也能正确归一化。每设备内存从 $O(n^2)$ 降到 $O(n/p)$，支持百万 token 上下文。

### 11.4 上下文压缩

不计算全部上下文的 attention，而是选择性压缩：

| 方法 | 思路 |
|------|------|
| LLMLingua | 用小模型对每个 token 打困惑度分，低困惑度（高可预测性）token 被丢弃，可压缩 4-20 倍 |
| AutoCompressor | 微调模型将长上下文总结为固定数量的"摘要向量"，作为压缩 KV Cache |

### 11.5 状态空间模型（SSM / Mamba）

Mamba（Gu & Dao, 2023）通过学习的循环状态处理序列，而非 attention：

$$h_t = A h_{t-1} + B x_t, \quad y_t = C h_t$$

- 推理时 $O(n)$ 时间、$O(1)$ 内存（隐藏状态固定大小，无 KV Cache）
- 弱点：对需要精确回溯特定上下文的任务不如 attention
- 混合模型（Jamba、Mamba-2 + attention）结合两者优势

## 十二、推理系统架构

### 12.1 生产推理服务器组成

```
请求 → API Gateway（认证、限流、入队）
     → Scheduler（决定当前 batch、管理 GPU 内存、SLO 约束）
     → Execution Engine（执行 GPU 前向传播、管理 KV Cache block）
     → Tokenizer（text ↔ token ID，通常 CPU）
     → Sampler（temperature、top-k、top-p）
     → 响应
```

### 12.2 分离式 Prefill/Decode

Prefill（compute-bound）和 Decode（memory-bound）的瓶颈不同，应在不同硬件上运行：

- **Prefill 节点**：最大化计算利用率，可用较旧的计算优化 GPU
- **Decode 节点**：最大化内存带宽，受益于高 HBM 带宽的新 GPU

挑战：KV Cache 迁移——prefill 完成后需将 KV Cache 从 prefill 节点传输到 decode 节点（通过 NVLink/InfiniBand/RDMA）。Mooncake 使用分布式 KV Cache 池解耦这一过程。

### 12.3 SLO-Aware 调度

生产系统必须遵守 SLO（p95/p99 TTFT 和 TPOT 目标）：

| 策略 | 说明 |
|------|------|
| 优先级队列 | 短请求/高优先级请求插队 |
| 抢占 | 暂停长请求，释放 KV Cache block，服务更高优先级请求 |
| 请求路由 | 按预估长度和服务器负载分流 |
| 准入控制 | 过载时拒绝或排队，防止 SLO 违规 |

## 十三、推理时计算扩展（Inference-Time Compute Scaling）

### 13.1 新的 Scaling 范式

传统 scaling law：更多训练计算 → 更好模型。2024-2025 出现新范式：推理时计算扩展——在推理时花更多计算来提升输出质量。

核心洞察：对很多任务（推理、数学、代码），一个"思考更久"的小模型可以超过一个"立刻回答"的大模型。

### 13.2 Chain-of-Thought 的推理成本

CoT 让模型在最终答案前生成中间推理步骤，大幅提升多步推理能力，但成本乘以数十倍：一个需要 10 token 答案的问题可能需要 500 token 推理。

**核心问题**：给定固定计算预算，是用大模型跑一次，还是用小模型生成更多推理 token？

结论取决于问题难度：难题用更多推理 token 更高效，简单题过度推理是浪费。

### 13.3 过程奖励模型（PRM）与搜索

PRM 对中间推理步骤打分（不只是最终答案），支持搜索：

| 策略 | 说明 |
|------|------|
| Best-of-N | 生成 N 个独立解，选最高分。$N=64$ 可匹配 10x 大模型 |
| Beam Search | 维护 K 条部分推理路径，扩展最有前途的 |
| MCTS | 用 PRM 作为树搜索的价值函数 |

### 13.4 DeepSeek-R1 与 o1 风格架构

OpenAI o1 和 DeepSeek-R1 代表质变：不是通过 prompt 引导 CoT，而是通过 RL 训练模型推理。模型生成扩展的内部独白（"thoughts"），然后给出最终答案。通过 GRPO（Group Relative Policy Optimization）训练：

- 不需要独立的价值网络（critic）
- 优势函数通过组内归一化计算：$A_i = (r_i - \text{mean}(r)) / \text{std}(r)$
- DeepSeek-R1-Zero 证明纯 RL + 可验证奖励就能涌现推理行为

## 十四、边缘推理（Edge Inference）

### 14.1 约束

旗舰手机 8-16 GB 统一内存（CPU/GPU 共享），功耗预算 10W。对比 H100：80 GB HBM，700W。边缘推理就是在这些约束下获取最大智能。

### 14.2 量化方案选择

| 方案 | 7B 模型大小 | 适用设备 |
|------|-----------|---------|
| FP16 | 14 GB | 服务器 |
| INT4 (AWQ/GPTQ) | 3.5 GB | 高端手机 |
| INT2 (QuIP#) | 1.8 GB | 中端设备（有质量损失） |

### 14.3 运行时框架

| 框架 | 特点 |
|------|------|
| llama.cpp | 纯 C/C++，CPU 推理（AVX-512/NEON），GPU offload（Metal/CUDA/Vulkan），k-quant 格式 |
| Apple MLX | 利用 Apple Silicon 统一内存，M3 Ultra 192 GB 可跑 4-bit 70B，30 token/s |
| ONNX Runtime | 跨平台推理，支持多种硬件加速 |

### 14.4 k-quant 系统（llama.cpp）

不是均匀量化，而是混合精度块量化：权重分组（通常 32 个一组），组内部分权重用更高精度存储。Q4_K_M 是 llama.cpp 中公认的最佳 4-bit 格式。

## 十五、训练阶段怎么省显存：梯度检查点和梯度累计

训练除了参数，还要存激活值、梯度、优化器状态，显存压力通常更大。

### 15.1 梯度检查点

本质：用额外计算换显存。

- 前向时不保存全部中间激活
- 反向时需要用到时再重算
- 显著降低激活显存，但训练更慢

### 15.2 梯度累计

本质：用多个小 batch 近似一个大 batch。

- 多个 micro-batch 连续反向
- 暂不更新参数
- 累积若干步后再统一更新
- 在显存有限时扩大等效 batch size

## 十六、并行训练：DP、TP、PP、ZeRO、MoE

当模型变得很大时，单卡通常放不下，需要并行训练。

### 16.1 DP / DDP

每张卡保存完整模型，数据按 batch 切分，反向后同步梯度。逻辑最简单、易扩展，但单卡仍必须装得下完整模型。

### 16.2 TP（Tensor Parallelism）

把同一层的权重矩阵切到多张卡，一层内部由多卡一起算。Megatron-LM 的实现：

- **Column Parallelism**：$W$ 沿列切分，各卡独立计算，结果通过 AllGather 合并
- **Row Parallelism**：$W$ 沿行切分，各卡计算部分和，通过 AllReduce 合并
- MLP 块：第一层 column parallel + 第二层 row parallel = 每层一次 AllReduce

单层太大时也能训练，但通信频繁，对高速互联（NVLink）要求高。实际扩展到 8 GPU（单节点 NVLink 连接）。

### 16.3 PP（Pipeline Parallelism）

按层切模型，不同设备负责不同层段。参数显存分担明显，但有 pipeline bubble：

$$\text{Bubble fraction} = \frac{p-1}{m+p-1}$$

$p=8, m=32$ 时 bubble $\approx 18\%$。通信高效（只在层边界传激活），可跨节点 InfiniBand 扩展。

### 16.4 ZeRO

核心不是按层切模型，而是把训练状态分片：

- **Stage 1**：分优化器状态
- **Stage 2**：再分梯度
- **Stage 3**：连参数也分

显著降低每卡状态开销，让大模型更容易训得动。

### 16.5 MoE（Mixture of Experts）推理

MoE 模型（Mixtral、DeepSeek-V2）用 E 个专家 FFN 替代密集 FFN，路由器为每个 token 选择 top-K 专家：

$$y = \sum_{k \in \text{Top-K}} g_k \cdot \text{Expert}_k(x)$$

Mixtral 8x7B 总参数 47B，但每 token 只激活 13B。推理挑战：

- **Expert Parallelism**：每个 GPU 存部分专家，token 通过 AllToAll 路由
- **负载不均**：某些专家收到的 token 远多于其他
- **Expert Offloading**：大 MoE 模型的专家可卸载到 CPU/NVMe，按需加载

## 十七、通信原语：系统问题的基本语言

并行训练里很多实现细节，本质上都落在这些通信操作上：

- Broadcast、Reduce、AllReduce、AllGather、Reduce-Scatter

最常记住的关系：DDP 里常见 AllReduce，参数分片恢复常见 AllGather，通信优化里常见 Reduce-Scatter。

## 十八、显存分析：训练和推理为什么都很贵

### 18.1 训练显存主要包括

- 参数
- 梯度
- 优化器状态
- 激活值

常见重灾区：Adam 类优化器状态、长序列下的激活值。

### 18.2 推理显存主要包括

- 模型参数
- KV Cache
- 中间激活
- 运行时框架开销

长上下文在线推理时，KV Cache 很容易成为第一瓶颈。

## 十九、混合精度为什么几乎是标配

混合精度训练常把一部分计算放到 FP16 / BF16：

- 降低显存
- 提高吞吐
- 更好利用 Tensor Core

FP16 更容易数值溢出，BF16 动态范围更大（与 FP32 相同指数范围），工程上通常更稳。H100 原生支持 FP8，进一步翻倍推理吞吐。

## 二十、推理可观测性与生产工程

### 20.1 核心指标

| 指标 | 含义 | 关注点 |
|------|------|--------|
| TTFT | Time To First Token | 用户感知的"思考时间"，>1s 感觉慢 |
| TPOT | Time Per Output Token | 流式输出流畅度，<200ms 体验好 |
| Goodput | 同时满足所有 SLO 的请求比例 | 比单独指标更有意义 |
| MFU | Model FLOP Utilization | 硬件利用效率，prefill 30-60%，decode <10% |
| Cost/1M tokens | 每百万 token 成本 | 业务可行性的关键 |

### 20.2 常见故障模式

| 症状 | 可能原因 | 诊断方法 |
|------|---------|---------|
| TTFT p99 尖峰 | 排队时间长；prefill 被抢占 | 队列深度监控、prefill 延迟分解 |
| TPOT 回退 | KV Cache 换出；batch 增大 | KV Cache 命中率、decode 步延迟 |
| OOM | KV Cache 溢出；请求突增 | 内存监控、PagedAttention block 统计 |
| 吞吐断崖 | TP 通信瓶颈 | GPU 利用率 vs NVLink 带宽 |

### 20.3 成本建模

$$\text{Cost per 1M tokens} = \frac{\text{GPU cost/hour}}{\text{tokens/hour}}$$

降本手段：最大化 GPU 利用率（大 batch、continuous batching）、量化/蒸馏/小模型、Spot 实例（便宜 60-80%）、精简输出。

## 二十一、面试作答框架

如果面试官问"怎么做大模型训练和推理优化"，比较清晰的回答顺序是：

1. 推理上，先区分 Prefill 和 Decode，用 Roofline 模型判断瓶颈
2. Decode 阶段核心瓶颈常在 KV Cache 和访存
3. 结构上用 MQA/GQA/MLA 降低 KV Cache，用 RoPE 支持长上下文
4. 算子上用 FlashAttention 减少 IO，利用 online softmax 和 kernel fusion
5. 系统上用 PagedAttention 和 Continuous Batching 提高利用率，Chunked Prefill 解决公平性
6. 量化降成本：AWQ/GPTQ 4-bit 权重量化，FP8/INT8 激活量化
7. 投机解码降延迟：draft-verify、Medusa、EAGLE
8. 长上下文：Ring Attention、上下文压缩、SSM 混合架构
9. 推理时计算扩展：CoT + PRM + Best-of-N，用小模型+更多推理计算替代大模型
10. 训练上再用梯度检查点、梯度累计、并行策略（DP/TP/PP/ZeRO）和混合精度解决显存与吞吐问题
11. 生产上关注 TTFT/TPOT/Goodput/MFU，做好 SLO 调度和成本建模

一句话总结：推理瓶颈常在 KV Cache 和访存，训练瓶颈常在激活、优化器状态和并行通信，所有优化手段本质都在围绕"算力、显存、带宽、调度"这四件事做权衡。
