---
date: '2026-06-02T11:20:00+08:00'
draft: false
title: 'dLLM 基础架构：从 Masked Diffusion 到 LLaDA'
tags: ["dLLM", "Diffusion", "Language Model", "Masked Diffusion", "LLaDA"]
summary: '整理 diffusion language model 的基本架构：为什么文本扩散需要离散化，D3PM 和 absorbing mask 前向过程怎么定义，denoising transformer 如何训练，LLaDA/MDLM 风格的预训练、SFT 和迭代采样如何工作。'
---

`dLLM` 通常指 `Diffusion Large Language Model`，也就是把扩散生成思想用到语言建模上的大语言模型。它和传统 `autoregressive LLM` 最大的区别是：AR 模型从左到右一次生成一个 token，而 dLLM 通常从大量 `[MASK]` 或噪声 token 开始，多轮并行去噪，逐步把整段文本补出来。

这篇笔记重点写基础架构，不追逐每个最新变体。主线是：

- 文本为什么不能直接照搬图像 DDPM。
- 离散扩散如何用转移矩阵定义前向加噪。
- `absorbing mask` 为什么特别适合语言。
- denoising transformer 的输入、输出和训练目标是什么。
- LLaDA / MDLM 这类 masked diffusion LLM 如何做预训练、SFT 和推理。

参考资料：

- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
- [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)
- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- [LLaDA project page](https://ml-gsai.github.io/LLaDA-demo/)
- [MDLM code repository](https://github.com/kuleshov-group/mdlm)

<!--more-->

## 一、dLLM 想解决什么问题

传统 LLM 通常使用自回归分解：

{{< rawhtml >}}
$$
p_{\theta}(x_1,\dots,x_L)
=
\prod_{i=1}^{L}
p_{\theta}(x_i|x_{\lt i})
$$
{{< /rawhtml >}}

这个分解非常自然，也非常有效。它的核心约束是：第 $i$ 个 token 只能依赖左侧上下文。生成时必须先生成 $x_1$，再生成 $x_2$，一直到 $x_L$。

dLLM 换了一个建模视角。它不把文本生成看成严格从左到右的过程，而是看成一个从“高度破坏的序列”逐步恢复成“干净文本”的过程：

{{< rawhtml >}}
$$
x_T
\rightarrow
x_{T-1}
\rightarrow
\cdots
\rightarrow
x_0
$$
{{< /rawhtml >}}

其中 $x_0$ 是真实文本，$x_T$ 是几乎全噪声或全 `[MASK]` 的序列。

所以 dLLM 的核心问题是：

> 给定一个带噪文本序列，模型能不能同时预测多个位置原本应该是什么 token？

这使它天然具有几个特点：

- 可以使用双向上下文，而不是只看左侧。
- 每轮可以并行预测多个 token。
- 推理过程中可以修改已经预测过的 token。
- 更接近“先草稿，再反复细化”的生成方式。

LLaDA 项目页里的方法图可以作为直观参考：

![LLaDA method diagram](https://ml-gsai.github.io/LLaDA-demo/static/images/method.svg)

图片来源：[LLaDA project page](https://ml-gsai.github.io/LLaDA-demo/)

## 二、文本扩散为什么比图像扩散麻烦

图像 DDPM 处理的是连续变量。像素或 latent 可以直接加高斯噪声：

{{< rawhtml >}}
$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$
{{< /rawhtml >}}

但文本 token 是离散的。一个 token 不是连续向量空间里的自然点，而是词表里的类别：

{{< rawhtml >}}
$$
x_i \in \{1,2,\dots,V\}
$$
{{< /rawhtml >}}

这里 $V$ 是词表大小。

如果直接对 token id 加高斯噪声，结果没有语义意义。例如 token id `1024` 加一点噪声变成 `1024.7`，并不对应一个合法 token。于是文本扩散有两条主要路线：

- `continuous diffusion over embeddings`：把 token 映射到 embedding，在连续向量空间里扩散，最后再 round 回 token。
- `discrete diffusion over tokens`：直接在离散词表上定义噪声过程，比如随机替换 token 或把 token 变成 `[MASK]`。

Diffusion-LM 属于第一类：它在连续 word embedding 上做 diffusion，再把生成的向量映射回词表。D3PM、MDLM、LLaDA 更接近第二类：它们直接把文本当作离散状态序列，前向过程逐渐破坏 token，反向过程逐步恢复 token。

现在的大语言 diffusion 模型通常更偏向离散 masked diffusion，因为它和 tokenizer、Transformer、BERT 式 masked prediction 的结合更自然。

## 三、离散扩散的状态空间

设一句话长度为 $L$，每个位置的 token 属于词表：

{{< rawhtml >}}
$$
x_0 = (x_0^1,\dots,x_0^L),
\quad
x_0^i \in \{1,\dots,V\}
$$
{{< /rawhtml >}}

为了做 masked diffusion，通常给词表额外加一个特殊状态：

{{< rawhtml >}}
$$
m = \text{[MASK]}
$$
{{< /rawhtml >}}

于是每个位置的状态空间变成：

{{< rawhtml >}}
$$
\mathcal{V}_{m}
=
\{1,\dots,V,m\}
$$
{{< /rawhtml >}}

前向扩散过程定义为一个离散马尔可夫链：

{{< rawhtml >}}
$$
q(x_t|x_{t-1})
=
\prod_{i=1}^{L}
q(x_t^i|x_{t-1}^i)
$$
{{< /rawhtml >}}

这里通常假设不同位置独立加噪。位置之间的语言依赖关系不是由前向过程建模，而是交给反向 denoising transformer 学。

对单个位置来说，前向转移可以用矩阵表示：

{{< rawhtml >}}
$$
q(x_t^i=b|x_{t-1}^i=a)
=
\left[Q_t\right]_{a,b}
$$
{{< /rawhtml >}}

其中 $Q_t$ 是一个离散状态转移矩阵。

如果把 token $x$ 写成 one-hot 向量，经过 $t$ 步后的边缘分布可以写成：

{{< rawhtml >}}
$$
q(x_t|x_0)
=
\mathrm{Cat}
\left(
x_t;
x_0 \bar{Q}_t
\right)
$$
{{< /rawhtml >}}

其中：

{{< rawhtml >}}
$$
\bar{Q}_t
=
Q_1Q_2\cdots Q_t
$$
{{< /rawhtml >}}

这就是 D3PM 的基本形式：把连续 DDPM 里的高斯噪声核，替换成离散状态空间里的转移矩阵。

## 四、几种常见离散噪声

### 4.1 Uniform corruption

最简单的方式是随机替换成词表中的任意 token：

{{< rawhtml >}}
$$
q(x_t^i=b|x_{t-1}^i=a)
=
(1-\beta_t)\mathbf{1}[b=a]
+
\beta_t \frac{1}{V}
$$
{{< /rawhtml >}}

它的含义是：以 $1-\beta_t$ 的概率保持原 token，以 $\beta_t$ 的概率随机替换。

这个方式数学上简单，但对语言不一定理想。因为随机替换出来的 token 往往是无意义噪声，模型需要从大量不自然的 token 组合里恢复原句。

### 4.2 Absorbing mask corruption

语言里更常用的是 absorbing mask。每个 token 只会发生两种情况：

- 保持原 token。
- 变成 `[MASK]`。

一旦变成 `[MASK]`，后续前向过程里就保持 `[MASK]`，不会再变回普通 token。这就是 `absorbing` 的含义。

单步转移可以写成：

{{< rawhtml >}}
$$
q(x_t^i=b|x_{t-1}^i=a)
=
\begin{cases}
1-\beta_t, & b=a,\ a\ne m \\
\beta_t, & b=m,\ a\ne m \\
1, & b=m,\ a=m \\
0, & \text{otherwise}
\end{cases}
$$
{{< /rawhtml >}}

从 $x_0$ 直接采样 $x_t$ 时，可以写成：

{{< rawhtml >}}
$$
q(x_t^i|x_0^i)
=
\alpha_t \delta(x_t^i=x_0^i)
+
(1-\alpha_t)\delta(x_t^i=m)
$$
{{< /rawhtml >}}

这里 $\alpha_t$ 表示到时间 $t$ 时该位置仍然没有被 mask 的概率。

当 $t=0$ 时：

{{< rawhtml >}}
$$
\alpha_0 = 1
$$
{{< /rawhtml >}}

当 $t=1$ 或 $T$ 时：

{{< rawhtml >}}
$$
\alpha_t \approx 0
$$
{{< /rawhtml >}}

也就是说，前向过程会逐渐把句子从完整文本变成全 `[MASK]`。

### 4.3 为什么 mask 特别适合语言

Mask 噪声有三个好处。

第一，它不会制造奇怪的随机词。被破坏的位置是 `[MASK]`，没有被破坏的位置仍然是真实 token。

第二，它和 BERT 的 masked language modeling 很接近。区别在于，BERT 通常只训练一个固定比例的 mask，而 masked diffusion 会覆盖从低 mask ratio 到高 mask ratio 的全时间范围。

第三，它让生成过程可以并行。模型每轮看到一个部分填好的序列，同时预测所有 `[MASK]` 位置的分布。

这也是 LLaDA 使用 masked diffusion 的关键原因。

## 五、dLLM 的反向过程

前向过程是把干净文本逐渐 mask 掉。反向过程则是从 mask 状态逐渐恢复文本：

{{< rawhtml >}}
$$
p_{\theta}(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_{\theta}(x_{t-1}|x_t)
$$
{{< /rawhtml >}}

对 absorbing mask 来说，$x_T$ 通常接近全 `[MASK]`：

{{< rawhtml >}}
$$
x_T \approx (m,m,\dots,m)
$$
{{< /rawhtml >}}

模型需要学习：

{{< rawhtml >}}
$$
p_{\theta}(x_{t-1}|x_t)
$$
{{< /rawhtml >}}

但在实践里，很多 dLLM 不直接让网络输出复杂的反向转移矩阵，而是让网络预测干净文本 $x_0$：

{{< rawhtml >}}
$$
p_{\theta}(x_0|x_t,t)
$$
{{< /rawhtml >}}

然后根据这个干净文本预测，构造从 $x_t$ 到 $x_{t-1}$ 的采样分布。

这和图像 DDPM 里“预测噪声 $\epsilon$ 或预测 $x_0$”很类似。区别是，文本里预测的是每个位置上的 token 分布。

## 六、关键后验：q(x_{t-1}|x_t,x_0)

和 DDPM 一样，训练时我们知道原始文本 $x_0$。因此可以写出真实后验：

{{< rawhtml >}}
$$
q(x_{t-1}|x_t,x_0)
\propto
q(x_t|x_{t-1})q(x_{t-1}|x_0)
$$
{{< /rawhtml >}}

在 D3PM 的矩阵形式里，对单个位置：

{{< rawhtml >}}
$$
q(x_{t-1}=a|x_t=b,x_0)
=
\frac{
q(x_t=b|x_{t-1}=a)q(x_{t-1}=a|x_0)
}{
q(x_t=b|x_0)
}
$$
{{< /rawhtml >}}

用转移矩阵表示：

{{< rawhtml >}}
$$
q(x_{t-1}=a|x_t=b,x_0)
=
\frac{
\left[Q_t\right]_{a,b}
\left[x_0\bar{Q}_{t-1}\right]_a
}{
\left[x_0\bar{Q}_{t}\right]_b
}
$$
{{< /rawhtml >}}

这就是离散扩散里的后验公式。它和高斯 DDPM 的后验均值/方差对应，只不过连续高斯换成了离散类别分布。

有了这个后验，训练目标可以仍然写成变分下界里的 KL：

{{< rawhtml >}}
$$
D_{\mathrm{KL}}
\left(
q(x_{t-1}|x_t,x_0)
\Vert
p_{\theta}(x_{t-1}|x_t)
\right)
$$
{{< /rawhtml >}}

问题是，如果词表很大，直接建完整的反向转移会比较重。所以现代 masked diffusion language model 常把目标化简成更接近 masked language modeling 的交叉熵。

## 七、从 ELBO 到 masked token 预测

离散扩散同样可以写出负 ELBO：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{vlb}}
=
\mathbb{E}_{q}
\left[
D_{\mathrm{KL}}
\left(
q(x_T|x_0)
\Vert
p(x_T)
\right)
+
\sum_{t=2}^{T}
D_{\mathrm{KL}}
\left(
q(x_{t-1}|x_t,x_0)
\Vert
p_{\theta}(x_{t-1}|x_t)
\right)
-
\log p_{\theta}(x_0|x_1)
\right]
$$
{{< /rawhtml >}}

但在 absorbing mask 场景里，核心训练信号可以理解为：

> 随机 mask 一部分 token，让模型根据未 mask 的上下文预测被 mask 的原 token。

设 mask 集合为 $M_t$，也就是在时间 $t$ 被 mask 的位置。一个直观的训练损失是：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{mask}}
=
\mathbb{E}_{x_0,t,x_t}
\left[
\sum_{i \in M_t}
-
\log
p_{\theta}
\left(
x_0^i|x_t,t
\right)
\right]
$$
{{< /rawhtml >}}

如果写成对所有位置求和，可以加一个 mask 指示函数：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{mask}}
=
\mathbb{E}_{x_0,t,x_t}
\left[
\sum_{i=1}^{L}
\mathbf{1}[x_t^i=m]
\left(
-
\log
p_{\theta}
\left(
x_0^i|x_t,t
\right)
\right)
\right]
$$
{{< /rawhtml >}}

LLaDA 的预训练形式可以这样理解：随机采样 mask ratio，对输入文本做 mask，然后让模型预测所有 masked tokens。MDLM 则进一步从 masked diffusion 的 ELBO 出发，把 loss 做成更简单、有效的 masked LM 风格目标。

## 八、dLLM 的网络结构

dLLM 的主干通常仍然是 Transformer。关键变化不在于完全发明一个新网络，而在于 attention mask、输入噪声形式、时间条件和输出目标不同。

一个基础 dLLM denoiser 可以写成：

{{< rawhtml >}}
$$
\mathrm{logits}
=
f_{\theta}(x_t,t,c)
$$
{{< /rawhtml >}}

其中：

- $x_t$ 是带噪或带 mask 的 token 序列。
- $t$ 是扩散时间或 mask ratio。
- $c$ 是可选条件，例如 instruction prompt、系统提示、多模态输入等。
- 输出 logits 的形状是 $L \times V$，每个位置输出一个词表分布。

典型模块如下。

### 8.1 Token embedding

输入 token 先变成向量：

{{< rawhtml >}}
$$
h_i^{(0)}
=
E[x_t^i] + P_i + T(t)
$$
{{< /rawhtml >}}

其中：

- $E$ 是 token embedding。
- $P_i$ 是位置编码，例如 RoPE 或 learned position embedding。
- $T(t)$ 是时间 embedding，可以加到每个位置上，也可以通过 AdaLN 等方式注入每层。

### 8.2 Bidirectional self-attention

AR LLM 使用 causal attention：

{{< rawhtml >}}
$$
x_i
\text{ only attends to }
x_{\le i}
$$
{{< /rawhtml >}}

dLLM 通常使用双向 attention：

{{< rawhtml >}}
$$
x_i
\text{ attends to }
x_1,\dots,x_L
$$
{{< /rawhtml >}}

这点非常重要。因为 dLLM 在预测某个 masked token 时，可以同时看左侧和右侧上下文。这也是它和 BERT 接近的地方。

对于指令生成，prompt 部分通常作为条件保留，response 部分被 mask 并逐步生成。此时 attention 可以允许 response token 看 prompt，也允许 response token 之间双向交互。

### 8.3 Time conditioning

扩散时间 $t$ 告诉模型当前噪声强度。低噪声时，模型只需要修补少量缺口；高噪声时，模型要从很少上下文里构造全局语义。

时间 embedding 可以写成：

{{< rawhtml >}}
$$
\tau = \mathrm{MLP}(\mathrm{SinusoidalEmbedding}(t))
$$
{{< /rawhtml >}}

然后注入 Transformer：

{{< rawhtml >}}
$$
h^{(\ell+1)}
=
\mathrm{TransformerBlock}
\left(
h^{(\ell)},\tau
\right)
$$
{{< /rawhtml >}}

在大模型实现里，时间条件常通过归一化层调制、残差缩放或简单加法进入网络。

### 8.4 Output head

最后每个位置输出词表 logits：

{{< rawhtml >}}
$$
p_{\theta}(x_0^i=a|x_t,t)
=
\mathrm{softmax}
\left(
W_o h_i
\right)_a
$$
{{< /rawhtml >}}

训练时只对被 mask 或被噪声破坏的位置计算 loss；未破坏位置通常不需要预测，或者只作为辅助训练项。

## 九、dLLM 和 BERT 的区别

dLLM 很像 BERT，但不能简单说它就是 BERT。

BERT 的 masked language modeling 通常是：

{{< rawhtml >}}
$$
\mathbb{E}
\left[
\sum_{i \in M}
-
\log p_{\theta}(x_0^i|x_{\mathrm{masked}})
\right]
$$
{{< /rawhtml >}}

其中 mask 比例一般固定在一个较窄范围。BERT 主要用于理解任务，不定义一个完整的从全 mask 到文本的生成过程。

dLLM 的 masked diffusion 则有明确的时间路径：

{{< rawhtml >}}
$$
x_0
\rightarrow
x_t
\rightarrow
x_T
$$
{{< /rawhtml >}}

训练时覆盖不同噪声强度：

{{< rawhtml >}}
$$
t \sim U(0,1)
$$
{{< /rawhtml >}}

推理时反向走：

{{< rawhtml >}}
$$
x_T
\rightarrow
x_{T-1}
\rightarrow
\cdots
\rightarrow
x_0
$$
{{< /rawhtml >}}

因此 dLLM 是一个真正的生成模型。它不仅学“填空”，还学在不同噪声等级下如何逐步恢复完整文本。

## 十、预训练流程

一个基础 masked dLLM 预训练 batch 可以这样构造。

给定干净文本：

{{< rawhtml >}}
$$
x_0 = (x_0^1,\dots,x_0^L)
$$
{{< /rawhtml >}}

采样时间：

{{< rawhtml >}}
$$
t \sim U(0,1)
$$
{{< /rawhtml >}}

根据噪声日程得到保留概率 $\alpha_t$，对每个位置采样 mask：

{{< rawhtml >}}
$$
m_i \sim \mathrm{Bernoulli}(1-\alpha_t)
$$
{{< /rawhtml >}}

构造带噪输入：

{{< rawhtml >}}
$$
x_t^i
=
\begin{cases}
\text{[MASK]}, & m_i=1 \\
x_0^i, & m_i=0
\end{cases}
$$
{{< /rawhtml >}}

模型输出：

{{< rawhtml >}}
$$
p_{\theta}(x_0^i|x_t,t)
$$
{{< /rawhtml >}}

训练损失：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{pretrain}}
=
\mathbb{E}
\left[
\frac{1}{|M_t|}
\sum_{i\in M_t}
-
\log p_{\theta}(x_0^i|x_t,t)
\right]
$$
{{< /rawhtml >}}

这里除以 $|M_t|$ 是常见实现细节，用来避免不同 mask ratio 下 loss 尺度差异太大。不同论文会使用不同的时间权重、loss normalization 和参数化方式，但直觉都是：让模型在任意缺失比例下恢复原文。

## 十一、SFT：只扩散 response

对指令微调数据：

{{< rawhtml >}}
$$
(p, r)
$$
{{< /rawhtml >}}

其中 $p$ 是 prompt，$r$ 是 response。

AR SFT 的目标是：

{{< rawhtml >}}
$$
\prod_{i=1}^{L_r}
p_{\theta}(r_i|p,r_{\lt i})
$$
{{< /rawhtml >}}

dLLM SFT 通常保留 prompt，只对 response 做 mask diffusion：

{{< rawhtml >}}
$$
x_0 = [p; r]
$$
{{< /rawhtml >}}

构造 $x_t$ 时：

{{< rawhtml >}}
$$
x_t^i
=
\begin{cases}
p_i, & i \in \mathrm{prompt} \\
\text{diffuse}(r_i,t), & i \in \mathrm{response}
\end{cases}
$$
{{< /rawhtml >}}

loss 只算 response 中被 mask 的位置：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{SFT}}
=
\mathbb{E}
\left[
\sum_{i\in M_t \cap \mathrm{response}}
-
\log
p_{\theta}
\left(
r_i|x_t,t
\right)
\right]
$$
{{< /rawhtml >}}

这样模型学到的是：在给定完整 prompt 的条件下，通过多步去噪生成完整 response。

LLaDA 项目页也强调了这个思路：预训练时随机 mask 所有 token；SFT 时通常只 mask 需要生成的 response 部分。

## 十二、推理：从全 mask 到完整回答

dLLM 生成时不是 next-token decoding，而是 iterative denoising。

给定 prompt，先初始化 response 区域为 `[MASK]`：

{{< rawhtml >}}
$$
x_T = [p; m,m,\dots,m]
$$
{{< /rawhtml >}}

然后做 $K$ 轮反向采样。第 $k$ 轮有当前序列 $x^{(k)}$，模型输出所有 mask 位置的 token 分布：

{{< rawhtml >}}
$$
p_{\theta}(x_0^i|x^{(k)},t_k)
$$
{{< /rawhtml >}}

最简单的策略是每轮填入一部分最高置信度 token。定义置信度：

{{< rawhtml >}}
$$
c_i
=
\max_a
p_{\theta}(x_0^i=a|x^{(k)},t_k)
$$
{{< /rawhtml >}}

选择置信度最高的一批位置：

{{< rawhtml >}}
$$
S_k
=
\mathrm{TopK}
\left(
\{c_i: x_i^{(k)}=m\}
\right)
$$
{{< /rawhtml >}}

更新：

{{< rawhtml >}}
$$
x_i^{(k+1)}
=
\begin{cases}
\arg\max_a p_{\theta}(x_0^i=a|x^{(k)},t_k), & i\in S_k \\
x_i^{(k)}, & \text{otherwise}
\end{cases}
$$
{{< /rawhtml >}}

直到没有 mask 或达到最大步数。

LLaDA 的采样过程示意可以看这个动图：

![LLaDA diffusion sampling](https://ml-gsai.github.io/LLaDA-demo/static/images/diff_normal_150ms.gif)

图片来源：[LLaDA project page](https://ml-gsai.github.io/LLaDA-demo/)

## 十三、Remasking：为什么生成后还能改

有些 dLLM 采样器不只是单向 unmask，还会允许 remask。也就是某个位置即使已经填了 token，如果后续发现置信度低或和全局上下文不一致，也可以重新 mask 再预测。

可以把采样写成：

{{< rawhtml >}}
$$
x^{(k+1)}
=
\mathrm{Denoise}
\left(
\mathrm{Remask}(x^{(k)})
\right)
$$
{{< /rawhtml >}}

这就是 dLLM 相比 AR 解码的一个重要差异。AR 模型一旦生成前面的 token，后面只能在这个前缀后继续写；dLLM 在原则上可以回头修改任何 response 位置。

当然，这不代表 dLLM 一定更好。反复修改需要更多采样轮数，也会带来稳定性、长度控制和一致性问题。

## 十四、长度建模

AR 模型的长度自然由 EOS token 决定：

{{< rawhtml >}}
$$
p(x)
=
\prod_i p(x_i|x_{\lt i})
$$
{{< /rawhtml >}}

生成到 EOS 就停止。

dLLM 常需要先确定一个 response 长度或最大长度，然后在这个长度范围内做 denoising。这会引入一个实际问题：如果输出长度事先不知道，模型要么需要预测长度，要么需要使用 block diffusion / semi-autoregressive 的方式逐块生成。

一种简单方式是先给定最大长度 $L$：

{{< rawhtml >}}
$$
x_T = [p; \underbrace{m,\dots,m}_{L}]
$$
{{< /rawhtml >}}

然后模型在生成中预测 EOS，EOS 后的位置被截断。

另一类方式是 block diffusion：把 response 拆成多个块，每个块内部用 diffusion 并行生成，块与块之间保持一定顺序。这样可以在 AR 和纯 dLLM 之间折中：

- 块内并行去噪。
- 块间顺序推进。
- 更容易处理变长输出和 KV cache。

## 十五、dLLM 的推理复杂度

AR 生成长度为 $L$ 的回答，需要 $L$ 次前向：

{{< rawhtml >}}
$$
N_{\mathrm{AR}} = L
$$
{{< /rawhtml >}}

dLLM 如果用 $K$ 轮 denoising，则需要：

{{< rawhtml >}}
$$
N_{\mathrm{dLLM}} = K
$$
{{< /rawhtml >}}

如果 $K \ll L$，看起来 dLLM 可以更快。但实际情况更复杂：

- AR 每次只新增一个 token，KV cache 很成熟。
- dLLM 每轮可能要重新处理整段序列。
- dLLM 可以并行更新多个 token，但每轮 attention 代价可能较高。
- block diffusion 和 cache-friendly sampler 会影响真实速度。

所以不能简单说 dLLM 一定比 AR 快。更准确的说法是：dLLM 把串行 token 生成，改成了少量轮次的并行迭代生成；它的优势能否兑现，取决于采样轮数、序列长度、cache 设计和硬件并行度。

## 十六、dLLM 与 AR LLM 的架构对比

| 维度 | AR LLM | dLLM |
| --- | --- | --- |
| 生成顺序 | 从左到右 | 多轮并行去噪 |
| Attention | causal attention | 通常是 bidirectional attention |
| 训练目标 | next-token prediction | masked denoising / diffusion ELBO |
| 输入状态 | 干净前缀 | 带 mask / 带噪序列 |
| 输出 | 下一个 token 分布 | 每个位置的 token 分布 |
| 是否能回改 | 通常不能 | 可以通过 remasking 回改 |
| 长度处理 | EOS 自然停止 | 常需长度策略或分块 |
| 推理瓶颈 | token 串行 | denoising 轮数和整段计算 |

从架构上看，dLLM 不是“完全不使用 Transformer”。相反，它通常仍然使用大规模 Transformer，只是把 causal decoder 改成更接近 bidirectional denoising decoder 的形式。

## 十七、Continuous Diffusion-LM 分支

前面主要讲离散 masked diffusion。还有一条历史路线是 Diffusion-LM：先把 token 变成 embedding，再在连续空间里做 diffusion。

设 token embedding 为：

{{< rawhtml >}}
$$
e(x_0^i) \in \mathbb{R}^d
$$
{{< /rawhtml >}}

整句 embedding 序列为：

{{< rawhtml >}}
$$
z_0
=
\left(
e(x_0^1),\dots,e(x_0^L)
\right)
$$
{{< /rawhtml >}}

然后像图像 DDPM 一样加高斯噪声：

{{< rawhtml >}}
$$
z_t
=
\sqrt{\bar{\alpha}_t}z_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$
{{< /rawhtml >}}

模型预测噪声或 $z_0$，最后需要把连续向量映射回离散 token：

{{< rawhtml >}}
$$
x^i
=
\arg\min_{a\in \mathcal{V}}
\left\|
z_0^i - e(a)
\right\|^2
$$
{{< /rawhtml >}}

这条路线的优点是可以直接复用连续 diffusion 的理论和很多控制生成技巧；缺点是 embedding 到 token 的 rounding 会带来误差，而且连续空间里的小扰动不一定对应自然语言里的合理变化。

因此，现在大规模 dLLM 更常讨论离散 token diffusion，尤其是 masked diffusion。

## 十八、SEDD：用离散 score 建模

SEDD 的思路更接近 score-based modeling。连续扩散里 score 是：

{{< rawhtml >}}
$$
\nabla_x \log p_t(x)
$$
{{< /rawhtml >}}

但离散 token 没有普通意义上的梯度。SEDD 改为估计离散状态之间的数据分布比值，也就是类似：

{{< rawhtml >}}
$$
s_{\theta}(x,t)_y
\approx
\frac{p_t(y)}{p_t(x)}
$$
{{< /rawhtml >}}

这里 $x$ 和 $y$ 是离散状态。直觉是：如果从当前 token 状态 $x$ 改成另一个状态 $y$，数据分布概率相对变高还是变低。

这种方法把“score”的概念搬到了离散空间。它和 masked diffusion 的交叉熵式目标不同，但都是为了学会反向去噪。

在架构层面，SEDD 也需要一个网络读取带噪 token 序列和时间 $t$，输出每个位置对词表状态的分数。

## 十九、MDLM：把 masked diffusion 做简单

MDLM 的核心价值在于，它把 absorbing-state masked diffusion 的训练和采样做得更直接。它使用一种 substitution 参数化，使得 masked diffusion 的 loss 可以简化为一组 masked language modeling 风格的损失。

从工程视角看，MDLM 很重要，因为它说明：

- dLLM 不一定需要复杂的完整转移矩阵实现。
- masked diffusion 可以和标准 Transformer LM 训练管线结合。
- 通过合适的 sampler，可以显著降低离散 diffusion 的采样成本。

MDLM repo 里的代码组织也很能体现 dLLM 的基本组件：

- `noise_schedule.py`：定义 mask / noise schedule。
- `diffusion.py`：定义前向和反向扩散。
- `models/`：denoising network，可以是 Transformer、DiT、Mamba 等。
- sampler：决定每一步如何从模型输出转成新的 token 序列。

这说明 dLLM 的架构不是单个模块，而是一套组合：

{{< rawhtml >}}
$$
\text{Tokenizer}
+
\text{Noise Schedule}
+
\text{Denoiser}
+
\text{Sampler}
$$
{{< /rawhtml >}}

## 二十、LLaDA：大规模 masked diffusion LLM

LLaDA 是 dLLM 里很典型的现代代表。它的关键点可以概括为：

- 使用 masked diffusion 建模文本。
- 从头进行大规模预训练，而不是只做小规模文本生成实验。
- 预训练和 SFT 都沿用 LLM 常见训练范式。
- 采样时多轮预测所有 mask，并可以灵活 remask。

LLaDA 的预训练目标可以粗略写成：

{{< rawhtml >}}
$$
\mathbb{E}_{x_0,t,x_t}
\left[
\sum_{i:x_t^i=m}
-
\log p_{\theta}(x_0^i|x_t,t)
\right]
$$
{{< /rawhtml >}}

它挑战的是一个常见假设：强语言模型一定要靠 next-token prediction。LLaDA 的观点是，智能能力更多来自大规模生成建模、数据、模型规模和训练范式，而不是必须来自自回归分解本身。

这不意味着 dLLM 已经全面替代 AR LLM。更准确地说，LLaDA 证明了 diffusion-style 的语言建模在大规模上是可行的，并且具备进一步研究价值。

## 二十一、一个最小 dLLM 伪代码

预训练可以写成：

```python
for batch in dataloader:
    x0 = batch.tokens
    t = sample_uniform_time(batch_size)
    alpha = noise_schedule(t)

    mask = bernoulli(1 - alpha, shape=x0.shape)
    xt = where(mask, MASK_ID, x0)

    logits = model(xt, t)
    loss = cross_entropy(logits[mask], x0[mask])
    loss.backward()
    optimizer.step()
```

推理可以写成：

```python
x = concat(prompt_tokens, repeat(MASK_ID, max_new_tokens))

for step in range(num_denoising_steps):
    t = schedule[step]
    logits = model(x, t)
    probs = softmax(logits)

    masked_pos = where(x == MASK_ID)
    confidence, token = probs[masked_pos].max(dim=-1)

    chosen = select_positions(confidence, step)
    x[chosen] = token[chosen]

    if no_mask_left(x):
        break
```

真实系统会复杂很多，例如加入 temperature、top-p、classifier-free guidance、remasking、block-wise generation、EOS 处理和 KV cache 优化，但核心结构就是这两段。

## 二十二、dLLM 的优势与限制

优势：

- 双向上下文，适合全局一致性修正。
- 多 token 并行生成，理论上可减少串行步数。
- 可以回改，缓解 AR 生成里的早期错误累积。
- 对 infilling、编辑、受控生成比较自然。
- 和 BERT / masked LM 的训练直觉兼容。

限制：

- 推理流程比 AR decoding 更复杂。
- 长度建模不如 EOS 自回归自然。
- KV cache 和流式输出不如 AR 成熟。
- 采样步数、remasking 策略、noise schedule 对质量影响很大。
- 现阶段生态、推理框架、对齐方法仍在快速发展。

因此 dLLM 更像是一条正在成型的并行生成路线，而不是 AR LLM 的直接平替。

## 二十三、总结

dLLM 的基础架构可以压缩成一条公式链：

{{< rawhtml >}}
$$
\begin{aligned}
x_0
&=
\text{clean text}
\\
q(x_t|x_0)
&=
\text{mask or corrupt tokens with noise level } t
\\
p_{\theta}(x_0|x_t,t)
&=
\text{denoising transformer prediction}
\\
\mathcal{L}
&=
\mathbb{E}
\left[
\sum_{i\in M_t}
-
\log p_{\theta}(x_0^i|x_t,t)
\right]
\end{aligned}
$$
{{< /rawhtml >}}

生成时反过来：

{{< rawhtml >}}
$$
[p;m,m,\dots,m]
\rightarrow
\text{partially filled response}
\rightarrow
\text{complete response}
$$
{{< /rawhtml >}}

如果用一句话记：

> AR LLM 是从左到右续写；dLLM 是从模糊到清晰地反复填充和修正。

从架构上看，dLLM 仍然离不开 tokenizer、embedding、Transformer、logits head 这些 LLM 基础组件。真正变化的是训练目标和生成路径：它把 next-token prediction 换成了 masked denoising，把单向串行解码换成了多轮并行去噪。
