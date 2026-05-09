---
date: '2026-05-09T10:00:00+08:00'
draft: false
title: '大模型面试精讲（一）：基础概念'
tags: ["LLM", "面试", "KL散度", "交叉熵", "梯度", "优化器", "Python并发"]
summary: '从 KL 散度、交叉熵到优化器，系统整理大模型面试中最常考的基础概念。'
---

面试复习第一轮，先把基础概念过一遍。这部分涵盖的范围比较广：信息论基本量、损失函数选择、模型幻觉与复读、decoder-only 架构、梯度问题、Python 并发模型、以及优化器演进。

参考教材：大模型面试题精讲教材

<!--more-->

## 一、KL 散度、交叉熵，以及它们的关系

### 1.1 三个基本定义

设真实分布为 $P(x)$，模型预测分布为 $Q(x)$。

**熵**

$$
H(P) = - \sum_x P(x) \log P(x)
$$

分布 `P` 自身的不确定性。分布越均匀，熵通常越大；分布越集中，熵越小。

**交叉熵**

$$
H(P, Q) = - \sum_x P(x) \log Q(x)
$$

如果样本真实来自 `P`，但我们用 `Q` 去编码它，平均需要多少信息量。

**KL 散度**

$$
D_{KL}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \sum_x P(x)[\log P(x) - \log Q(x)]
$$

用 `Q` 近似 `P` 时，多付出的那部分"编码代价"。

### 1.2 关系推导

从 KL 散度出发：

$$
D_{KL}(P \parallel Q) = \sum_x P(x)[\log P(x) - \log Q(x)] = -H(P) + H(P, Q)
$$

因此：

$$
H(P, Q) = H(P) + D_{KL}(P \parallel Q)
$$

这是面试里最关键的一条关系。

### 1.3 直觉理解

- $H(P)$ 是数据本身的难度，固定不变。
- $D_{KL}(P \parallel Q)$ 是模型和真实分布之间的差距。
- $H(P, Q)$ 是总损失，等于"问题本身的难度 + 你模型带来的额外损失"。

所以在监督学习里，如果真实分布 $P$ 已经固定，那么最小化交叉熵就等价于最小化 KL 散度。

### 1.4 one-hot 标签下的交叉熵

如果分类任务中真实标签是 one-hot，且正确类别为 `y`，则交叉熵化简为：

$$
H(P, Q) = -\log Q(y)
$$

只看模型给正确类别分配的概率，概率越大，损失越小。

### 1.5 为什么 KL 散度不是对称的

$$
D_{KL}(P \parallel Q) \ne D_{KL}(Q \parallel P)
$$

因为它衡量的是"用 Q 去近似 P"的代价，而不是一个普通距离。在生成模型中：

- $D_{KL}(P \parallel Q)$ 更怕模型漏掉真实模式。
- $D_{KL}(Q \parallel P)$ 更怕模型生成不存在的模式。

## 二、为什么大模型常用交叉熵损失

### 2.1 语言模型的任务本质

语言模型做的是条件概率建模：

$$
P(x_1, x_2, \ldots, x_T) = \prod_t P(x_t \mid x_{<t})
$$

训练时，目标是最大化训练语料的似然：

$$
\min -\sum_t \log P_{\theta}(x_t \mid x_{<t})
$$

而这正是 token 级别的交叉熵。

### 2.2 从最大似然到交叉熵

关键点在于，真实分布并没有消失，而是被 one-hot 标签"隐含"进去了。设在第 $t$ 个位置，真实分布为 $P_t(\cdot)$，模型预测分布为 $Q_t(\cdot) = P_{\theta}(\cdot \mid x_{<t})$。

在 next-token prediction 里，监督信号通常是 one-hot 标签。假设真实 token 是 $y$，代入交叉熵定义：

$$
H(P_t, Q_t) = - \sum_i P_t(i)\log Q_t(i) = - \log P_{\theta}(x_t = y \mid x_{<t})
$$

对整个序列：

$$
L = \sum_t H(P_t, Q_t)
$$

因此，这里的训练目标既可以叫 token-level cross entropy，也可以叫 negative log-likelihood (NLL)。在 one-hot 标签的分类建模里，这两者是等价的。

### 2.3 和 perplexity 的关系

$$
CE = -\frac{1}{T}\sum_t \log P_{\theta}(x_t \mid x_{<t}), \quad PPL = \exp(CE)
$$

交叉熵越小，困惑度越低，表示模型越擅长预测下一个 token。

## 三、为什么回归常用 MSE，分类常用交叉熵

### 3.1 从概率分布假设来理解

**回归**：假设 $y = f_{\theta}(x) + \varepsilon$，其中 $\varepsilon \sim \mathcal{N}(0, \sigma^2)$，那么最大化对数似然等价于最小化 $\sum (y - f_{\theta}(x))^2$，这就是 MSE。

**分类**：分类预测的是离散分布，最自然的似然模型是 categorical / Bernoulli 分布，因此对应的负对数似然就是交叉熵。

### 3.2 为什么分类不用 MSE

- 输出更像数值拟合，而不是概率分布拟合。
- 配合 sigmoid / softmax 时，梯度在饱和区更弱。
- 错分样本的惩罚方式往往不如交叉熵合理。

## 四、大模型中的幻觉、复读机现象

### 4.1 幻觉的成因

- 预训练目标是拟合文本分布，不是事实校验。
- 参数里存的是统计相关性，不是显式知识库。
- 训练数据有噪声、冲突、过期信息。
- 长上下文里，模型可能没抓到真正相关的证据。
- 对齐阶段如果奖励模型偏爱"自信流畅"，会放大错误答案。

### 4.2 复读的成因

- 解码时温度太低、top-k/top-p 太保守。
- repetition penalty 不合理。
- 模型进入局部高概率循环。
- 长上下文注意力退化，模型失去全局规划能力。

### 4.3 常见解决方式

**数据层**：清洗预训练语料、增加高质量指令数据、构造拒答和不确定性表达数据。

**模型层**：RAG 接入外部证据、改进长上下文建模、提升对齐质量。

**推理层**：合理设置 temperature / top-p / top-k、使用 repetition penalty / n-gram blocking、对事实任务增加检索或工具调用。

## 五、为什么使用 decoder-only 架构

- 训练目标统一：next-token prediction。
- 训练数据容易获取：海量无标注文本即可。
- 结构简单，扩展性强。
- 自回归生成任务和对话任务天然匹配。
- in-context learning 在大规模 decoder-only 模型中表现突出。

**对比**：Encoder-only (BERT) 适合理解任务；Encoder-decoder (T5) 适合标准 seq2seq；Decoder-only (GPT、Llama) 统一处理生成和对话。

## 六、GPT、BERT、CLIP、Llama 的区别

- **GPT**：decoder-only，自回归生成。
- **BERT**：encoder-only，以 MLM 为主，擅长理解。
- **CLIP**：图像编码器 + 文本编码器，对比学习对齐图文空间。
- **Llama**：Meta 的 decoder-only 开源大模型系列。

## 七、梯度爆炸、梯度消失、梯度饱和

深层网络中的梯度传播，本质上是很多局部导数的连乘：

$$
\frac{\partial \mathcal{L}}{\partial W_1}
= \frac{\partial \mathcal{L}}{\partial h_L}
\cdot \frac{\partial h_L}{\partial h_{L-1}}
\cdot \frac{\partial h_{L-1}}{\partial h_{L-2}}
\cdots
\frac{\partial h_1}{\partial W_1}
$$

连乘结果长期大于 1，梯度会爆炸；长期小于 1，梯度会消失。

### 7.1 梯度爆炸

反向传播过程中梯度范数迅速变大，导致参数更新过猛，训练不稳定。

**典型表现**：loss 突然变大、参数更新剧烈震荡、梯度范数非常大、混合精度训练里出现 `inf` 或 `nan`。

**解决方法**：Gradient clipping（最直接的"安全阀"）、降低学习率、合理初始化、残差连接、规范化层。

### 7.2 梯度消失

反向传播时梯度在传到前面层的过程中越来越小，导致早期层几乎收不到有效学习信号。

**典型表现**：浅层参数几乎不更新、训练很慢、loss 长期降不下去。

**解决方法**：使用 ReLU / GELU / SwiGLU 等激活函数、残差连接、Pre-Norm、合理初始化（Xavier / Kaiming）、对 RNN 使用 LSTM / GRU。

### 7.3 梯度饱和

激活函数进入饱和区，导致局部导数接近 0。以 sigmoid 为例：

$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

导数最大值只有 $0.25$。一旦输入落到两端，反向梯度很难通过它传播。

饱和是局部导数问题，消失和爆炸是连乘后的整体梯度传播问题。

### 7.4 为什么 Transformer 比早期 RNN 更好训练

- 残差连接让梯度有短路径传播
- LayerNorm / RMSNorm 控制表示尺度
- Pre-Norm 改善深层梯度稳定性
- 没有传统 RNN 那种沿时间步的长链路重复乘法

## 八、Python 进程、线程、协程、GIL、async

### 8.1 进程（Process）

操作系统分配资源的基本单位，资源隔离强，适合真正的多核并行。创建和切换开销较大，进程间通信比线程麻烦。

### 8.2 线程（Thread）

进程内部的执行单元，共享地址空间。切换成本比进程低，但在 CPython 中受 GIL 影响，难以真正并行执行 Python 字节码。

### 8.3 协程（Coroutine）

用户态的轻量级并发单元，切换通常由程序显式让出控制权，切换开销更低，特别适合大量 I/O 等待场景。协程的关键不在"更快执行 CPU 计算"，而在"等待 I/O 时别闲着，切去做别的事"。

### 8.4 并发和并行的区别

- **并发**：同一时间段内处理多个任务，强调调度与重叠
- **并行**：同一时刻在多个核心上真正同时执行，强调物理同时性

协程通常是并发，不是并行。多线程在 CPython 中常是并发，不一定并行。多进程更容易实现真正并行。

### 8.5 GIL 是什么

GIL 是 CPython 里的全局解释器锁。同一个 Python 进程内，任意时刻通常只允许一个线程执行 Python 字节码。它不是"为了限制性能"，而是解释器实现层面的工程权衡。

### 8.6 CPU 密集和 I/O 密集怎么选

- **CPU 密集**（数值计算、图像编解码）：多进程，或把重计算放到 C/C++ / NumPy / PyTorch 这类释放 GIL 的底层实现中。
- **I/O 密集**（网络请求、数据库访问、文件读写）：多线程或协程。

### 8.7 async / await 本质是什么

不是魔法线程，也不是自动多核并行。本质是把任务写成可挂起、可恢复的状态机，由事件循环负责调度，在等待 I/O 时切换到其他任务。

简洁对比：

- **进程**：重，隔离强，适合 CPU 并行
- **线程**：中等，共享内存，适合 I/O 并发
- **协程**：轻，用户态调度，适合高并发 I/O

## 九、深拷贝和浅拷贝

- **浅拷贝**：只拷贝最外层对象，内部引用共享。
- **深拷贝**：递归复制内部对象，结构独立。

## 十、智能指针

- `unique_ptr`：独占所有权。
- `shared_ptr`：引用计数共享。
- `weak_ptr`：弱引用，解决循环引用。

## 十一、SGD、Momentum、Adam、AdamW、Muon

### 11.1 从最基本的梯度下降开始

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

沿负梯度方向更新。但真正训练深度网络时会遇到问题：不同方向曲率差异大、梯度噪声大、某些方向震荡严重、稀疏梯度场景下更新不均衡。

### 11.2 SGD

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

简单、内存开销小、泛化能力好。但对学习率敏感，在狭长谷底里容易来回震荡。

### 11.3 Momentum

在 SGD 基础上加一个"速度项"：

$$
v_{t+1} = \beta v_t + (1-\beta) g_t, \quad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

如果某个方向上长期梯度一致，Momentum 会不断积累速度；如果梯度来回震荡，历史平均会起到平滑作用。

### 11.4 Adam

Adam 同时维护梯度的一阶矩和二阶矩估计：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
$$

- 用 $m_t$ 做动量平滑
- 用 $v_t$ 感知每个参数维度的梯度尺度
- 对不同维度做自适应缩放

### 11.5 AdamW

在 Adam 里，把 L2 正则直接加进梯度，会被二阶矩缩放，行为就变了。AdamW 把权重衰减从梯度更新里解耦：

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} - \eta \lambda \theta_t
$$

优化器负责按梯度更新，weight decay 负责单独收缩参数，二者职责分开。

AdamW 的核心不是"更快"，而是"weight decay 的语义更正确"。

### 11.6 按这条线记

- **SGD**：只看当前梯度
- **Momentum**：在 SGD 上加历史速度
- **Adam**：在 Momentum 基础上再加按维度自适应缩放
- **AdamW**：把 Adam 里的 weight decay 从梯度项里拆出来

### 11.7 优化器状态为什么占显存

- SGD 主要只需要参数和梯度
- Momentum 额外要存一个速度项 $v_t$
- Adam / AdamW 额外要存 $m_t$ 和 $v_t$

这也是大模型训练里 ZeRO、优化器分片等策略很重要的原因。

### 11.8 Muon

较新的优化思路，关注矩阵参数更新的几何结构，而不是只把参数看成一长串标量。核心动机是改进传统逐元素自适应更新在大规模矩阵参数上的行为。
