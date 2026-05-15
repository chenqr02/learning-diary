---
date: '2026-05-09T10:00:00+08:00'
draft: false
title: '大模型面试精讲（一）：基础概念'
tags: ["LLM", "面试", "KL散度", "交叉熵", "梯度", "优化器", "Python并发", "正则化", "Tokenizer", "Scaling Law", "BPE", "采样策略", "评估指标", "CoT", "ICL"]
summary: '从 KL 散度、交叉熵到优化器，从偏差-方差到 Scaling Law，系统整理机器学习面试和大模型面试中的基础必考概念。'
---

面试复习第一轮，把基础概念过一遍。这部分涵盖的范围比较广：信息论基本量、损失函数选择、模型幻觉与复读、decoder-only 架构、梯度问题、Python 并发模型、优化器演进，以及偏差-方差权衡、正则化、采样策略、Tokenizer、Scaling Law、评估指标、ICL/CoT 等机器学习和大模型面试中的高频基础考点。

参考教材：大模型面试题精讲教材

<!--more-->

## 一、KL 散度、交叉熵，以及它们的关系

### 1.1 三个基本定义

设真实分布为 $P(x)$，模型预测分布为 $Q(x)$。

**熵**

{{< rawhtml >}}
$$
H(P) = - \sum_x P(x) \log P(x)
$$
{{< /rawhtml >}}

分布 `P` 自身的不确定性。分布越均匀，熵通常越大；分布越集中，熵越小。

**交叉熵**

{{< rawhtml >}}
$$
H(P, Q) = - \sum_x P(x) \log Q(x)
$$
{{< /rawhtml >}}

如果样本真实来自 `P`，但我们用 `Q` 去编码它，平均需要多少信息量。

**KL 散度**

{{< rawhtml >}}
$$
D_{KL}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \sum_x P(x)[\log P(x) - \log Q(x)]
$$
{{< /rawhtml >}}

用 `Q` 近似 `P` 时，多付出的那部分"编码代价"。

### 1.2 关系推导

从 KL 散度出发：

{{< rawhtml >}}
$$
D_{KL}(P \parallel Q) = \sum_x P(x)[\log P(x) - \log Q(x)] = -H(P) + H(P, Q)
$$
{{< /rawhtml >}}

因此：

{{< rawhtml >}}
$$
H(P, Q) = H(P) + D_{KL}(P \parallel Q)
$$
{{< /rawhtml >}}

这是面试里最关键的一条关系。

### 1.3 直觉理解

- $H(P)$ 是数据本身的难度，固定不变。
- $D_{KL}(P \parallel Q)$ 是模型和真实分布之间的差距。
- $H(P, Q)$ 是总损失，等于"问题本身的难度 + 你模型带来的额外损失"。

所以在监督学习里，如果真实分布 $P$ 已经固定，那么最小化交叉熵就等价于最小化 KL 散度。

### 1.4 one-hot 标签下的交叉熵

如果分类任务中真实标签是 one-hot，且正确类别为 `y`，则交叉熵化简为：

{{< rawhtml >}}
$$
H(P, Q) = -\log Q(y)
$$
{{< /rawhtml >}}

只看模型给正确类别分配的概率，概率越大，损失越小。

### 1.5 为什么 KL 散度不是对称的

{{< rawhtml >}}
$$
D_{KL}(P \parallel Q) \ne D_{KL}(Q \parallel P)
$$
{{< /rawhtml >}}

因为它衡量的是"用 Q 去近似 P"的代价，而不是一个普通距离。在生成模型中：

- $D_{KL}(P \parallel Q)$ 更怕模型漏掉真实模式。
- $D_{KL}(Q \parallel P)$ 更怕模型生成不存在的模式。

## 二、为什么大模型常用交叉熵损失

### 2.1 语言模型的任务本质

语言模型做的是条件概率建模：

{{< rawhtml >}}
$$
P(x_1, x_2, \ldots, x_T) = \prod_t P(x_t \mid x_{<t})
$$
{{< /rawhtml >}}

训练时，目标是最大化训练语料的似然：

{{< rawhtml >}}
$$
\min -\sum_t \log P_{\theta}(x_t \mid x_{<t})
$$
{{< /rawhtml >}}

而这正是 token 级别的交叉熵。

### 2.2 从最大似然到交叉熵

关键点在于，真实分布并没有消失，而是被 one-hot 标签"隐含"进去了。设在第 $t$ 个位置，真实分布为 $P_t(\cdot)$，模型预测分布为 $Q_t(\cdot) = P_{\theta}(\cdot \mid x_{<t})$。

在 next-token prediction 里，监督信号通常是 one-hot 标签。假设真实 token 是 $y$，代入交叉熵定义：

{{< rawhtml >}}
$$
H(P_t, Q_t) = - \sum_i P_t(i)\log Q_t(i) = - \log P_{\theta}(x_t = y \mid x_{<t})
$$
{{< /rawhtml >}}

对整个序列：

{{< rawhtml >}}
$$
L = \sum_t H(P_t, Q_t)
$$
{{< /rawhtml >}}

因此，这里的训练目标既可以叫 token-level cross entropy，也可以叫 negative log-likelihood (NLL)。在 one-hot 标签的分类建模里，这两者是等价的。

### 2.3 和 perplexity 的关系

{{< rawhtml >}}
$$
CE = -\frac{1}{T}\sum_t \log P_{\theta}(x_t \mid x_{<t}), \quad PPL = \exp(CE)
$$
{{< /rawhtml >}}

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

{{< rawhtml >}}
$$
\frac{\partial \mathcal{L}}{\partial W_1}
= \frac{\partial \mathcal{L}}{\partial h_L}
\cdot \frac{\partial h_L}{\partial h_{L-1}}
\cdot \frac{\partial h_{L-1}}{\partial h_{L-2}}
\cdots
\frac{\partial h_1}{\partial W_1}
$$
{{< /rawhtml >}}

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

{{< rawhtml >}}
$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$
{{< /rawhtml >}}

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

{{< rawhtml >}}
$$
\theta_{t+1} = \theta_t - \eta g_t
$$
{{< /rawhtml >}}

沿负梯度方向更新。但真正训练深度网络时会遇到问题：不同方向曲率差异大、梯度噪声大、某些方向震荡严重、稀疏梯度场景下更新不均衡。

### 11.2 SGD

{{< rawhtml >}}
$$
\theta_{t+1} = \theta_t - \eta g_t
$$
{{< /rawhtml >}}

简单、内存开销小、泛化能力好。但对学习率敏感，在狭长谷底里容易来回震荡。

### 11.3 Momentum

在 SGD 基础上加一个"速度项"：

{{< rawhtml >}}
$$
v_{t+1} = \beta v_t + (1-\beta) g_t, \quad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$
{{< /rawhtml >}}

如果某个方向上长期梯度一致，Momentum 会不断积累速度；如果梯度来回震荡，历史平均会起到平滑作用。

### 11.4 Adam

Adam 同时维护梯度的一阶矩和二阶矩估计：

{{< rawhtml >}}
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
$$
{{< /rawhtml >}}

- 用 $m_t$ 做动量平滑
- 用 $v_t$ 感知每个参数维度的梯度尺度
- 对不同维度做自适应缩放

### 11.5 AdamW

在 Adam 里，把 L2 正则直接加进梯度，会被二阶矩缩放，行为就变了。AdamW 把权重衰减从梯度更新里解耦：

{{< rawhtml >}}
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} - \eta \lambda \theta_t
$$
{{< /rawhtml >}}

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

## 十二、偏差-方差权衡（Bias-Variance Tradeoff）

这是机器学习最核心的概念之一，几乎所有面试都会以不同形式考察。

### 12.1 核心来源

假设真实模型 $y = f(x) + \varepsilon$，其中 $\varepsilon$ 是均值为 0、方差为 $\sigma^2$ 的噪声。我们训练得到模型 $\hat{f}(x)$，用 MSE 评估。

期望泛化误差可以分解为：

{{< rawhtml >}}
$$
\mathbb{E}[(y - \hat{f})^2] = \underbrace{(\mathbb{E}[\hat{f}] - f)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}
$$
{{< /rawhtml >}}

- **Bias（偏差）**：模型的平均预测与真实值的偏离程度。高 bias 意味着模型对数据做了过于简化的假设。
- **Variance（方差）**：模型对训练集波动的敏感程度。高 variance 意味着训练集稍微变化，模型预测就有很大差异。
- **Irreducible Error（不可约误差）**：数据本身的噪声，任何模型都无法消除。

### 12.2 Tradeoff 的本质

{{< rawhtml >}}
$$\text{总误差} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$
{{< /rawhtml >}}

模型越复杂，bias 越低但 variance 越高。最佳点是在二者之间取得平衡。过拟合就是模型太复杂（低 bias、高 variance），欠拟合就是模型太简单（高 bias、低 variance）。

### 12.3 面试回答框架

如果面试官问"这个模型为什么过拟合了"，本质是在问"模型 variance 太高了"。解决方法可以分为：
- 降低 variance：增加数据量、正则化、简化模型、集成学习
- 降低 bias：增加模型复杂度、更多特征、更少正则化
- 有时增加数据量可以同时降低 bias 和 variance

### 12.4 在 LLM 中的体现

大模型本质上是低 bias（容量大、拟合能力强）、高 variance（对训练数据分布敏感）的模型。这也是为什么指令微调需要高质量数据——低 bias 的模型会忠实学到数据中的偏差。

## 十三、过拟合、欠拟合与正则化

### 13.1 过拟合

模型学到了训练数据中的噪声和偶然模式，在训练集上表现很好但在新数据上泛化差。典型表现就是训练 loss 持续下降，验证 loss 开始上升。

**为什么大模型更容易过拟合**：参数量远大于训练样本的独立信息量，模型有足够容量去记忆而非泛化。

### 13.2 欠拟合

模型太简单，连训练数据的主要模式都没抓住。训练 loss 和验证 loss 都居高不下。

### 13.3 正则化方法

正则化的本质是给优化目标加约束，限制模型复杂度，防止过拟合。

**L1 正则化（Lasso）**

{{< rawhtml >}}
$$
L = L_{data} + \lambda \sum |w_i|
$$
{{< /rawhtml >}}

在 loss 里加上参数绝对值之和。L1 鼓励稀疏解，很多不重要的参数被推到 0，天然有特征选择能力。在优化上，L1 导数 $\partial |w|/\partial w = \pm 1$，在 0 处不可导，依赖次梯度。

**L2 正则化（Ridge / Weight Decay）**

{{< rawhtml >}}
$$
L = L_{data} + \frac{\lambda}{2} \sum w_i^2
$$
{{< /rawhtml >}}

在 loss 里加上参数平方和。L2 鼓励小参数但不强制到 0，导数是 $\lambda w$，相当于每次更新时让参数往 0 方向缩一点——这就是 weight decay 的本质。

在 AdamW 里把 weight decay 从梯度里拆出来，原因这里已经埋下了：L2 正则原本和梯度混合会被 Adam 的二阶矩缩放，语义就不对了。

**Dropout**

训练时随机将一部分神经元输出置零（按概率 $p$），迫使网络不能过度依赖某些特征。推理时关闭，并用 $1-p$ 缩放保持期望一致。

**数据增强**

通过对训练数据做合理变换来扩大有效数据量，本质上也是一种正则化。在 LLM 场景包括：回译、指令改写、句子打乱等。

**Early Stopping**

在验证集 loss 开始上升时停止训练，本质上控制了模型的有效迭代步数，限制了参数空间的大小。

### 13.4 正则化的统一理解

所有正则化方法都是在"拟合数据"和"限制复杂度"之间做权衡：Weight decay 直接限制参数范数；Dropout 限制特征 co-adaptation；数据增强和数据配比间接约束模式学习；Early stopping 限制迭代步数。在 LLM 里甚至数据混入比例也是一种隐式正则。

## 十四、参数初始化方法

初始化会影响模型训练速度、收敛质量和最终效果。以下是面试最高频的几种。

### 14.1 为什么初始化重要

如果初始化太大，激活值进入饱和区，梯度消失；如果初始化太小，信号在深层网络中逐层衰减。合理的初始化应该让各层激活值和梯度的方差保持稳定。

### 14.2 Xavier / Glorot 初始化

{{< rawhtml >}}
$$
W \sim \mathcal{U}\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]
$$
{{< /rawhtml >}}

假设激活函数在 0 附近近似线性（如 tanh、sigmoid 未饱和区）。让前向和反向传播中信号方差保持稳定。

### 14.3 Kaiming / He 初始化

{{< rawhtml >}}
$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$
{{< /rawhtml >}}

专门针对 ReLU 类激活函数设计。因为 ReLU 会把一半神经元输出置零，方差天然折半，所以初始化时需要补偿这一损失。PyTorch 里 `nn.Linear` 默认常常接近 Kaiming 均匀分布。

### 14.4 大模型中的初始化

大模型规模很大，初始化不当会导致训练直接崩溃。常见做法包括：
- 使用小的标准差（如 $0.01$ 或 $\sqrt{2/(n_{in} \cdot \text{num_layers})}$）初始化
- 对残差分支的最后一层做 zero-init，让一开始残差路径为主干
- 用 Pre-Norm 配合合理初始化，减少激活值随深度暴涨

## 十五、采样与解码策略

语言模型生成时，模型输出的是词表上的 logits，需要采样策略决定最终输出哪个 token。

### 15.1 Greedy Decoding

每一步选概率最高的 token。简单、确定，但容易重复、缺乏多样性，错过高概率续写路径。

### 15.2 Temperature Scaling

{{< rawhtml >}}
$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$
{{< /rawhtml >}}

- $T \to 0$：分布变得极端尖锐，等价于 greedy
- $T = 1$：保持原始分布
- $T > 1$：分布变平滑，低概率词也有机会被选到
- $T \to \infty$：逼近均匀分布

Temperature 不改变相对排序，只改变分布的尖锐程度。

### 15.3 Top-k Sampling

每一步只从概率最高的 k 个 token 中采样。优点是避免采样到极低概率的离谱 token。缺点是 k 是固定的，不同上下文的最佳 k 可能差异很大。

### 15.4 Top-p (Nucleus) Sampling

每一步选累积概率达到 p 的最小 token 集合，再从集合内重新归一化后采样。核心思想是让采样范围随分布的置信度自适应调整——分布尖锐时选很少的 token，分布平缓时选更多的 token。

### 15.5 Temperature + Top-k + Top-p 的典型组合

通常先调 temperature，再用 top-p 或 top-k 做后处理剪枝。它们不是互斥的，可以联合使用。常见默认值：temperature 0.7-0.9，top-p 0.9-0.95，top-k 40-50。

### 15.6 Repetition Penalty

{{< rawhtml >}}
$$
p_i \propto \exp(z_i / (T \cdot \alpha^{I_i}))
$$
{{< /rawhtml >}}

其中 $I_i$ 是该 token 在已生成序列中出现的次数，$\alpha > 1$ 时重复越多的 token 被惩罚越重。注意它是修改 logits 而非后处理。

### 15.7 Beam Search

在每一步维护 k 个最可能的部分序列（beams），扩展时每个 beam 生成下一个 token 的候选再从中选 top-k。用于机器翻译、语音识别等需要找到全局较优序列的任务。生成式对话中较少用——beam search 倾向于找到高概率但更短更通用的回答。

### 15.8 关键面试点

- Temperature 越大越"随机"，越小越"确定"
- Top-p 是自适应裁剪，Top-k 是固定裁剪
- Repetition penalty 和 temperature 是在 logits 级别上操作的，top-k/top-p 是在概率分布上做后处理
- 推理时不能同时使用 beam search 和采样——它们对"下一个 token 怎么选"的假设冲突

## 十六、Tokenizer 原理

Tokenizer 把文本映射成 token ID 序列，是所有 LLM 的入口。面试常问的是不同分词方法的区别和为什么 LLM 需要分词。

### 16.1 为什么需要分词

语言模型在 token 级别上建模，token 是模型的最小处理单元。词级别词表太大（几十万），字符级别信息太碎片化序列太长，subword 是最实用折中。

### 16.2 BPE（Byte Pair Encoding）

GPT 系列使用的 tokenizer。

**过程**：
1. 把每个词拆成字符序列
2. 统计所有相邻字符对的频率
3. 合并最高频的 pair，加入词表
4. 重复直到词表大小达到目标

**特点**：贪心的、频率驱动的合并策略。常见词保持完整，罕见词拆成 subword。

### 16.3 WordPiece

BERT 使用的 tokenizer。

与 BPE 的区别在于：BPE 合并最高频的相邻 token pair，WordPiece 合并能最大提高训练数据似然的 pair：

{{< rawhtml >}}
$$
\text{Score}(a, b) = \frac{\text{count}(a, b)}{\text{count}(a) \cdot \text{count}(b)}
$$
{{< /rawhtml >}}

本质是互信息驱动的合并，而非纯频率驱动。

### 16.4 Unigram

SentencePiece 支持的另一种方法，从大词表开始逐步剪枝。初始词表很大，然后通过 EM 算法评估每个 token 对似然的贡献，逐步移除最不重要的 token。

**和 BPE 的方向相反**：BPE 从小词表往上加，Unigram 从大词表往下剪。

### 16.5 SentencePiece

不是一种分词算法，是 Google 开源的 tokenizer 训练工具包。关键特性：
- 把原始文本视为 Unicode 序列，不依赖预分词（不要求先按空格分好词）
- 支持 BPE 和 Unigram 两种算法
- 原生处理多语言不用空格划分的问题

### 16.6 LLM 面试高频问题

**中文为什么不一定需要单独的分词器**：SentencePiece 或 BPE 可以直接在 byte/Unicode 级别上操作，不需要显式的中文分词预处理。

**词表大小怎么选**：太小 → 每个 token 信息量太少，序列太长，计算量增加；太大 → embedding 矩阵太大，罕见 token 学不好。典型值 16k-128k。

**OOV 问题为什么在 BPE 里基本不存在**：任何单词都可以被拆成 BPE 已有的 subword 组合，没有真正的"未登录词"。

**tokenizer 对性能的影响**：中文场景一个 token 编码的信息量比英文少，同等序列长度中文的"实际信息密度"更低，这也是中文场景下上下文效率偏低的一个原因。

## 十七、Scaling Law

### 17.1 核心发现

模型性能与模型参数量 $N$、数据量 $D$、计算量 $C$ 之间存在可预测的幂律关系：

{{< rawhtml >}}
$$
L(N, D) \propto N^{-\alpha} + D^{-\beta}
$$
{{< /rawhtml >}}

最关键的结论：**模型性能主要由计算预算（参数量 × 训练数据量）决定，同等算力下最优分配是同时增大模型和数据，而非只堆参数。**

### 17.2 Chinchilla 法则

DeepMind 发现许多模型其实训练数据不足。对于给定算力预算，最优的模型大小和数据量的比例大约是 **每 1B 参数对应 ~20B token**。

这直接动摇了早期"大参数 + 少数据"的范式，后来的 LLM 普遍使用更多数据进行训练（LLaMA 系列就是典型）。

### 17.3 三个关键量

- **参数量 N**：模型容量
- **数据量 D**：训练数据规模
- **计算量 C**：≈ 6ND（前向 + 反向一次）

### 17.4 在面试中怎么回答

如果面试官问"为什么大模型要增大数据量而不是一味增大参数量"：
1. Scaling Law 表明模型性能受 N 和 D 共同约束
2. 单独增大 N 而不增大 D，模型会过拟合到有限数据上
3. Chinchilla 给出了最优配比的参考
4. 所以在计算预算固定的情况下，需要平衡模型大小和数据规模

### 17.5 Emergent Abilities 讨论

规模增大到某个阈值后，模型突然具备了小模型没有的能力。对于涌现，学界存在不同看法——部分认为涌现是小模型量化指标可预测的外推，部分认为存在真正的能力质变。面试中记住核心观点：scale 确实带来能力变化，但涌现的"突然性"部分可能来自评测指标的离散性质。

## 十八、评估指标

### 18.1 Perplexity（PPL）

{{< rawhtml >}}
$$
PPL = \exp\left(-\frac{1}{T}\sum_t \log P_{\theta}(x_t \mid x_{<t})\right)
$$
{{< /rawhtml >}}

语言模型最核心的内在评估指标。越低说明模型对测试数据的预测能力越强。但 PPL 低不等于实际任务表现好——它只能衡量模型对 token 分布的拟合程度，不能衡量事实性、有用性和安全性。

### 18.2 BLEU

机器翻译评估指标，基于 n-gram 精确率（Precision）：

{{< rawhtml >}}
$$
BLEU = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)
$$
{{< /rawhtml >}}

$p_n$ 是 n-gram 精确率，$BP$ 是长度惩罚。偏向参考翻译中出现的词组，对词汇多样性和语义关注不足。

### 18.3 ROUGE

文本摘要评估指标，基于 n-gram 召回率（Recall）：

{{< rawhtml >}}
$$
ROUGE\text{-}N = \frac{\sum \text{count}_{match}(n\text{-gram})}{\sum \text{count}(n\text{-gram})}
$$
{{< /rawhtml >}}

衡量生成结果覆盖了多少参考摘要中的内容。ROUGE-L 额外考虑最长公共子序列。

### 18.4 精确率、召回率、F1

{{< rawhtml >}}
$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
F1 = \frac{2 \cdot P \cdot R}{P + R}
$$
{{< /rawhtml >}}

- **Precision**：预测为正类的样本中，有多少真是正类（"我挑的对不对？"）
- **Recall**：真实正类样本中，有多少被正确捡出来了（"我漏了没有？"）
- **F1**：二者的调和平均

在 LLM 评估中，精确率对应回答的事实正确密度，召回率对应是否覆盖了全部必要信息。

### 18.5 Accuracy 什么时候失效

当类别严重不平衡时，Accuracy 会给出误导性信号——99% 负类、1% 正类时，全判负也有 99% accuracy。此时应该看 Precision / Recall / F1 或 AUC-ROC。

### 18.6 AUC-ROC

ROC 曲线的横轴是 FPR（假正率），纵轴是 TPR（真正率）。AUC 是曲线下面积。

- AUC = 1：完美分类器
- AUC = 0.5：随机猜
- AUC 衡量的是排序质量——给定一个正样本和一个负样本，模型把正样本排在前面的概率

**AUC 对不平衡不敏感**：因为它是基于排序的，不依赖分类阈值。这是它比 accuracy 更适合不平衡场景的根本原因。

## 十九、In-context Learning 与 Chain-of-Thought

### 19.1 In-context Learning（ICL）

大模型通过上下文中的示例来学习完成任务，不需要梯度更新。

**关键发现**：当模型规模足够大时，ICL 能力显著增强。小模型几乎不具备 ICL 能力，这是 emergent ability 的典型例子。

**ICL 为什么有效（主流解释）**：在预训练中，模型见过大量"前文 + 后文"的模式。当 prompt 中给出示例时，激活了模型隐式的模式匹配能力——本质上是把预训练阶段学到的"按示例执行"能力在推理时复用。

### 19.2 Few-shot / One-shot / Zero-shot

- **Zero-shot**：不给示例，直接给出指令让模型完成
- **One-shot**：给一个示例
- **Few-shot**：给几个示例

Few-shot 的效果通常优于 zero-shot，但示例数量增加到一定程度后收益递减。示例的选择和排序对结果有显著影响。

### 19.3 Chain-of-Thought（CoT）

通过在 prompt 中加入推理步骤的示例或直接要求"逐步思考"，让模型显式输出推理过程。

**为什么 CoT 有效**：
- 把多步推理分解成中间步骤，降低了单步推理的负担
- 中间步骤提供了可验证的推理过程
- 本质上把复杂推理转化为更小的子问题

**Zero-shot CoT**：直接在 prompt 后加"Let's think step by step"，不需要 CoT 示例就能提升推理能力。这最有力地说明 CoT 的效果不仅来自示例格式的复制，而是触发了模型的推理机制。

### 19.4 CoT 的局限

- 不保证推理过程正确（模型可能编造看似合理的推理路径）
- 增加推理延迟和 token 消耗
- 对不需要推理的任务引入不必要的冗余

## 二十、损失函数补充：对比损失与难例挖掘

### 20.1 Contrastive Loss / InfoNCE

对比学习的核心是"拉近正样本、推远负样本"：

{{< rawhtml >}}
$$
L = -\log \frac{\exp(q \cdot k^+ / \tau)}{\sum_{i=0}^K \exp(q \cdot k_i / \tau)}
$$
{{< /rawhtml >}}

本质是一个 K+1 类的 softmax 分类问题：从 K+1 个样本中选出正确的那个。$\tau$ 是 temperature，控制分布的尖锐程度。

**在 LLM 中的应用**：
- CLIP 的图文对比学习
- Sentence-BERT 的句向量学习
- 检索模型的 embedding 训练

### 20.2 Triplet Loss

{{< rawhtml >}}
$$
L = \max(0, d(a, p) - d(a, n) + margin)
$$
{{< /rawhtml >}}

对每个锚点 $a$，拉近正样本 $p$、推远负样本 $n$。要求正例距离 + margin < 负例距离。

### 20.3 Focal Loss

{{< rawhtml >}}
$$
L = -\alpha (1 - p_t)^\gamma \log(p_t)
$$
{{< /rawhtml >}}

标准交叉熵对容易分类的样本也产生损失，Focal Loss 通过 $(1-p_t)^\gamma$ 降低易分样本的权重，让模型更关注难分类样本。在类别不平衡和目标检测中效果显著。

### 20.4 面试作答框架

"什么时候用什么损失"：
- **分类、语言模型**：交叉熵
- **回归**：MSE / MAE
- **对比学习 / 检索 / 表征学习**：InfoNCE / Contrastive Loss / Triplet Loss
- **类别不平衡 / 难例挖掘**：Focal Loss
- **排序任务**：Pairwise Ranking Loss / ListNet

## 二十一、模型容量、泛化与奥卡姆剃刀

### 21.1 模型容量

模型拟合复杂函数的能力。参数越多、网络越深，容量越大。但容量大不等于好——太大容量如果不配合足够数据和正则化，会导致过拟合。

### 21.2 奥卡姆剃刀在 ML 中的含义

在同等效果下，更简单的模型通常更好。不是"简单一定正确"，而是"简单模型在有限数据上更容易泛化，因为它的假设空间更小"。

### 21.3 Double Descent 现象

传统认知认为"模型越复杂、过拟合越严重"，但现代深度学习观察到 double descent：模型容量达到某个临界点时测试误差反而再次下降。这挑战了经典 bias-variance tradeoff 的预测，说明现代深度网络在过参数化区间有特殊的泛化行为。

### 21.4 对 LLM 的启示

- 大模型处于 overparameterized 区间，double descent 解释了为什么其在训练数据之外仍有良好的泛化能力
- 但过参数化不保证泛化——数据质量、训练方法仍然关键
- 模型容量大意味着它对数据中的偏见和噪声更敏感

## 二十二、面试高频概念对比

### 22.1 判别模型 vs 生成模型

- **判别模型**：直接建模 $P(y|x)$ 或学决策边界。关注"区分"而非"生成"。如 SVM、逻辑回归、CRF、大多数分类 CNN/Transformer。
- **生成模型**：建模 $P(x, y)$ 或 $P(x)$。能生成新样本。如朴素贝叶斯、HMM、GAN、VAE、扩散模型、语言模型。

**关键区别**：生成模型可以做判别任务（通过 $P(y|x) \propto P(x|y)P(y)$），但判别模型不能做生成。语言模型本质上是生成模型。

### 22.2 参数模型 vs 非参数模型

- **参数模型**：参数数量固定，不随数据量增长（如线性回归、逻辑回归）
- **非参数模型**：有效参数数量随数据增长（如 KNN、决策树、高斯过程）

**关键点**：非参数模型不是说不需要参数，而是模型的复杂度随数据自适应。大模型介于二者之间——参数固定但容量极大，在足够数据下接近非参数模型的行为。

### 22.3 监督、无监督、自监督、半监督、弱监督

- **监督学习**：输入-输出对训练
- **无监督学习**：只有输入，发现数据结构（聚类、降维）
- **自监督学习**：从数据自身构造监督信号（BERT 的 MLM、GPT 的 next-token prediction）
- **半监督学习**：少量标注 + 大量未标注
- **弱监督**：使用不完美、有噪声的监督信号

**LLM 的特点**：预训练是自监督，指令微调是监督，RLHF 是强化学习。

### 22.4 Bagging vs Boosting

- **Bagging**（如随机森林）：并行训练多个独立模型，取平均降低 variance。对过拟合敏感模型效果显著。
- **Boosting**（如 XGBoost、AdaBoost）：串行训练，每个新模型纠正前一个的错误。主要降低 bias，但容易过拟合噪声。

### 22.5 距离度量

- **欧氏距离**：$L_2 = \sqrt{\sum (x_i - y_i)^2}$
- **余弦相似度**：$\cos(x, y) = \frac{x \cdot y}{\|x\| \|y\|}$，关注方向而非幅度
- **点积**：$x \cdot y = \|x\| \|y\| \cos \theta$，同时考虑方向和幅度
- **曼哈顿距离**：$L_1 = \sum |x_i - y_i|$

**在 LLM 中的使用**：embedding 检索常用余弦相似度（因为 embedding 经过了归一化），attention score 用点积（因为 Q、K 已经过投影且存在 scale 设计）。
