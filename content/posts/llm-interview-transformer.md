---
date: '2026-05-09T11:00:00+08:00'
draft: false
title: '大模型面试精讲（二）：Transformer 架构'
tags: ["LLM", "面试", "Transformer", "Attention", "位置编码", "Normalization"]
summary: '从 Attention 到位置编码，系统整理 Transformer 架构面试中的高频考点。'
---

Transformer 面试里最容易被追问的，不是死记概念，而是"为什么这么设计"。这部分按 attention -> norm -> 激活 -> 位置编码 -> 架构细节的顺序展开。

参考教材：大模型面试题精讲教材

<!--more-->

## 一、为什么 Attention 要除以 $\sqrt{d_k}$

标准 scaled dot-product attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 1.1 从方差角度理解

设 query 向量 $q$ 和 key 向量 $k$ 的每一维独立同分布，均值为 0、方差为 1，则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的总方差大致随 $d_k$ 线性增长。

### 1.2 为什么这会带来问题

softmax 对输入尺度很敏感。如果输入特别大，概率分布会变得非常尖锐，大部分位置概率接近 0，梯度会集中在极少数位置，训练更不稳定。

除以 $\sqrt{d_k}$ 的核心目的，是把 score 控制在合适尺度，让 softmax 不至于过早饱和。不是"为了让数变小"，而是"为了让 softmax 的工作区间更健康"。

### 1.3 如果不除会怎样

小模型可能还能训练，维度大了之后 attention score 波动变大，softmax 更尖，梯度更差，容易影响收敛速度和稳定性。

## 二、为什么要分成 Q、K、V 三个矩阵

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

- **Q**：当前 token 想查询什么
- **K**：每个 token 暴露什么"可匹配信号"
- **V**：每个 token 真正携带什么内容

attention 的过程可以理解为两步：先用 $Q$ 和 $K$ 算匹配程度，再用这个匹配权重去加权求和 $V$。

如果只用一套表示同时承担"匹配"和"承载内容"两个职责，表达会更受限。拆成 Q/K/V 以后，模型可以在一个子空间里判断"谁重要"，在另一个子空间里决定"取出什么"。

- **self-attention**：Q、K、V 都来自同一个序列
- **cross-attention**：Q 来自 decoder，K/V 来自 encoder

## 三、多头注意力为什么有效

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

如果只有单头，模型只能在一个注意力模式下工作。多头注意力允许模型在不同子空间并行关注不同关系：语法依赖、指代关系、位置邻近、长距离语义关联。

每个头都有各自的投影矩阵，所以不同头学到的并不是同一件事，而是不同的匹配模式。

## 四、Encoder 和 Decoder 的区别

- **Encoder**：使用双向 self-attention，每个 token 可以看到整个序列，更适合理解任务（BERT）。
- **Decoder**：使用因果掩码 self-attention，当前 token 只能看见自己之前的位置，更适合生成任务（GPT、Llama）。
- **Encoder-decoder**：encoder 负责理解输入，decoder 负责基于 encoder 表示生成输出，常见于翻译、摘要。

因果掩码为什么必要：因为训练目标是自回归预测下一个 token，如果当前位置能偷看到未来 token，训练目标就被破坏了。

## 五、BatchNorm、LayerNorm、RMSNorm

### 5.1 为什么归一化能帮助训练

如果层与层之间的激活尺度漂移很大，后续层就会不断适应前面层分布变化。归一化能控制中间表示尺度、稳定梯度传播、让优化器面对更平滑的优化地形。

### 5.2 BatchNorm

$$
y = \gamma \cdot \frac{x - \mu_{batch}}{\sqrt{\sigma_{batch}^2 + \varepsilon}} + \beta
$$

统计量来自 batch 维度。在 CV 里表现很好，但在 NLP / LLM 里不主流：序列长度不一、micro-batch 常很小、多卡训练时 batch 统计更复杂。

### 5.3 LayerNorm

$$
y = \gamma \cdot \frac{x - \mu_{layer}}{\sqrt{\sigma_{layer}^2 + \varepsilon}} + \beta
$$

对单个样本自身的 hidden dimension 做归一化。不依赖 batch 大小，训练和推理行为一致，特别适合 Transformer。

$\gamma$ 控制缩放，$\beta$ 控制平移，相当于让网络在"先标准化，再学合适尺度和偏移"。

### 5.4 RMSNorm

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}, \quad y = \gamma \cdot \frac{x}{\text{RMS}(x)}
$$

和 LayerNorm 的主要区别：LayerNorm 先减均值再除标准差，RMSNorm 不减均值，只按均方根做尺度归一化。计算更简单，实践中效果通常足够好，在很多 LLM 里成为主流选择。

## 六、Pre-Norm、Post-Norm、DeepNorm

Norm 放在残差块的前面还是后面，会直接影响梯度传播路径和深层训练稳定性。

### 6.1 Post-Norm

$$
x_{l+1} = \text{LN}(x_l + F(x_l))
$$

先经过子层变换，再和残差相加，最后做归一化。原始 Transformer 论文中常见，但深层时更容易训练不稳——残差主干本来是给梯度提供"短路径"的，但 Post-Norm 里残差相加后马上又经过一次 LayerNorm。

### 6.2 Pre-Norm

$$
x_{l+1} = x_l + F(\text{LN}(x_l))
$$

先对输入做归一化，再送进子层，最后和原始残差直接相加。主残差路径更接近恒等映射，梯度传播路径更顺，深层 Transformer 中通常更稳。

Pre-Norm 更像"先把输入整理好，再学一个残差修正"；Post-Norm 更像"先算完再统一归一化"。

### 6.3 从梯度路径角度理解

在 Pre-Norm 里：

$$
\frac{\partial x_{l+1}}{\partial x_l} \approx I + \frac{\partial F(\text{LN}(x_l))}{\partial x_l}
$$

残差主干保留了一个接近 $I$ 的项，梯度不完全依赖子层 $F$ 的导数链式连乘。

在 Post-Norm 里，梯度必须再经过一层 LN 的 Jacobian，残差主干不再是那么"裸露"的恒等路径。

### 6.4 DeepNorm

面向超深 Transformer 的训练稳定性问题。不是简单替换 LayerNorm，而是通过更合理的残差缩放设计，控制深层网络中的信号放大与衰减。

### 6.5 常见模型家族里的 Norm 选择

- 现代大模型常见选择是 Pre-Norm
- 很多 LLM 进一步配 RMSNorm
- LLaMA 系列典型特征是 Pre-Norm + RMSNorm

## 七、激活函数：sigmoid、tanh、ReLU、GELU、GLU、Swish、SwiGLU

- **sigmoid / tanh**：平滑，有明确概率或中心化解释，但容易饱和，深层训练时容易导致梯度消失。
- **ReLU**：计算简单、稀疏性好、训练效率高，但负半轴梯度为 0，可能出现 dying ReLU。
- **GELU**：更平滑的门控版本激活，BERT 和大量 Transformer 变体中常见。
- **SwiGLU**：把 Swish 和门控结构结合，经验效果通常更好，FFN 表达能力更强，在同等参数预算下常能带来更优性能。

## 八、位置编码：正余弦、可学习位置编码、RoPE、ALiBi

### 8.1 为什么一定要有位置信息

self-attention 对输入 token 的排列本身没有天然顺序偏好。如果不额外注入位置信息，模型无法区分"猫咬狗"和"狗咬猫"。

### 8.2 正余弦位置编码

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

固定函数，不需要学习，某种程度上有长度外推能力。但表达形式固定，不够灵活。

### 8.3 可学习位置编码

为每个位置分配一个可训练向量。灵活，在固定长度任务中常能学得不错，但训练没见过的更长位置时外推能力通常较差。

### 8.4 RoPE

RoPE 的关键不是简单"加一个位置向量"，而是通过旋转变换把位置信息编码到 Q、K 中。不同位置的向量会发生不同角度的旋转，attention score 里的内积因此带有相对位置信息。

RoPE 把"位置"直接耦合进了注意力匹配机制本身。在大模型和长上下文场景里表现通常更好。

### 8.5 ALiBi

不去显式旋转向量，而是在 attention score 上加入和距离相关的线性偏置。距离越远，给一个越强的惩罚或偏置。实现更简洁，长度外推也很强。

### 8.6 对比

- **RoPE**：把位置信息编码进 Q/K，与 attention 更深度耦合，目前大模型里非常主流
- **ALiBi**：直接改 attention score，实现更简洁

## 九、Transformer 计算复杂度

约定：序列长度 $n$，hidden size $d_{model}$，head 数 $h$，每个 head 维度 $d_k = d_{model}/h$，FFN 中间层维度 $d_{ff}$。

### 9.1 单头 attention 的时间复杂度

- 线性投影得到 Q、K、V：$O(n d_{model} d_k)$
- 计算 attention score $QK^T$：$O(n^2 d_k)$
- attention 权重乘以 $V$：$O(n^2 d_k)$

当序列长度很长时，真正主导的是 $O(n^2 d_k)$——这就是为什么常说 attention 对序列长度是二次复杂度。

### 9.2 多头注意力的复杂度

关键：每个 head 的维度变小了，$d_k = d_{model}/h$。

$$
h \cdot O(n^2 d_k) = O(n^2 h d_k) = O(n^2 d_{model})
$$

**多头注意力并不会把复杂度额外乘成 $O(h n^2 d_{model})$，因为每个 head 的维度缩小了。**

多头注意力整体：

$$
O(n d_{model}^2 + n^2 d_{model})
$$

### 9.3 Transformer Block 的整体复杂度

- Attention：$O(n d_{model}^2 + n^2 d_{model})$
- FFN：$O(n d_{model} d_{ff}) \approx O(n d_{model}^2)$（当 $d_{ff} \approx 4 d_{model}$）

整个 block：$O(n^2 d_{model} + n d_{model}^2)$

### 9.4 什么时候 attention 主导，什么时候 FFN 主导

- 长上下文：attention 更痛
- 大 hidden size：线性层和 FFN 也很痛

### 9.5 推理时 prefill 和 decode 的复杂度差异

**Prefill**：一次性处理整段 prompt，attention 仍要计算所有位置之间的两两关系，复杂度 $O(n^2 d_{model})$。

**Decode**：有 KV Cache 时，新生成一个 token 只需要拿当前 Q 去和历史 K 做匹配，单步 attention 更接近 $O(n d_{model})$。这也是 KV Cache 能极大加速自回归推理的根本原因。

### 9.6 常见误区

- **误区 1**：head 数越多，复杂度线性乘以 head 数暴涨。不对，因为每个 head 的维度相应减小。
- **误区 2**：attention 一定比 FFN 更贵。不总是，取决于序列长度和 hidden size。
- **误区 3**：训练和推理的复杂度完全一样。不对，训练要保存更多激活，推理有 KV Cache。

## 十、Dropout 训练和推理时的区别

- **训练时**：随机把一部分神经元输出置零，防止 co-adaptation，减少过拟合。
- **推理时**：关闭 dropout，使用完整网络。
- **为什么训练时要做缩放**：为了保持训练和推理时激活值期望一致。

## 十一、一个完整的 Transformer Block 该怎么讲

1. 输入先经过归一化
2. 进入多头注意力模块
3. 做残差连接
4. 再归一化
5. 进入前馈网络 FFN
6. 再做残差连接

一句话：Transformer block 的骨架是"注意力 + FFN"，稳定训练的关键是"残差 + 归一化"。
