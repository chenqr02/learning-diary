---
date: '2026-06-05T15:30:00+08:00'
draft: false
title: '状态空间语言模型：从 SSM 到 Mamba-3 与 ParaRNN'
tags: ["LLM", "State Space Model", "Mamba", "Mamba-3", "ParaRNN", "RNN"]
summary: '从线性状态空间模型出发，理解状态空间语言模型为何能并行训练、常数状态解码，以及 Mamba-3 和 ParaRNN 分别怎样改进表达能力与训练并行性。'
---

Transformer 用注意力直接比较序列中的 token，效果很好，但长序列下的计算和缓存成本也很明显。状态空间语言模型选择了另一条路线：不断把已经读过的信息压缩进一个固定大小的状态，再用这个状态预测后续 token。

这条路线并不等于简单地回到传统 RNN。现代状态空间模型最关键的能力是：

- 推理时像 RNN 一样逐 token 更新，只保留固定大小的状态。
- 训练时利用线性递推的结构，对整段序列做并行计算。
- 通过输入相关的参数，让固定大小的状态有选择地记忆和遗忘。

本文先建立状态空间模型的基础直觉，再介绍两种不同的前沿方向：

- `Mamba-3`：保持线性状态更新，继续提升状态的表达能力和硬件效率。
- `ParaRNN`：不限制递推必须是线性的，而是用 Newton 迭代把非线性 RNN 的训练过程并行化。

参考资料：

- [Efficiently Modeling Long Sequences with Structured State Spaces（S4）](https://arxiv.org/abs/2111.00396)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality（Mamba-2）](https://arxiv.org/abs/2405.21060)
- [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)
- [Mamba 官方代码仓库](https://github.com/state-spaces/mamba)
- [ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models](https://arxiv.org/abs/2510.21450)
- [Apple ParaRNN 介绍](https://machinelearning.apple.com/research/large-scale-rnns)
- [ParaRNN 官方代码仓库](https://github.com/apple/ml-pararnn)

<!--more-->

## 一、从“保存全部历史”到“维护一个状态”

语言模型在第 $t$ 步预测下一个 token 时，需要利用此前的信息。Transformer 的做法是保留过去 token 的 Key 和 Value，让当前 token 通过注意力重新读取历史。

状态空间模型则维护一个状态 $h_t$：

{{< rawhtml >}}
$$
h_t = f(h_{t-1}, x_t)
$$
{{< /rawhtml >}}

其中，$x_t$ 是当前 token 的表示，$h_{t-1}$ 是此前历史的压缩结果。模型再从状态中读出当前输出：

{{< rawhtml >}}
$$
y_t = g(h_t, x_t)
$$
{{< /rawhtml >}}

可以把两种方法粗略理解为：

- Attention 像保留一本不断变厚的原始笔记，需要时回头查找。
- SSM / RNN 像维护一页不断更新的摘要，后续主要读取摘要。

因此，在自回归解码时，Transformer 的 KV cache 会随上下文长度增长，而递归模型只需要保留固定大小的状态。代价也很直接：如果重要信息在压缩时丢失，后面无法再回头读取原始 token。

## 二、状态空间模型的基础形式

### 2.1 连续时间状态空间模型

经典线性状态空间模型来自控制理论。给定随时间变化的输入 $x(t)$，系统内部状态 $h(t)$ 按微分方程演化，并产生输出 $y(t)$：

{{< rawhtml >}}
$$
\begin{aligned}
\frac{d h(t)}{dt}
&= A h(t) + B x(t) \\
y(t)
&= C h(t) + D x(t)
\end{aligned}
$$
{{< /rawhtml >}}

这些矩阵的作用是：

- $A$：状态自身如何随时间演化，也决定信息衰减或振荡的方式。
- $B$：当前输入如何写入状态。
- $C$：如何从状态中读出信息。
- $D$：输入直接传到输出的捷径。

语言是离散 token 序列，不能直接使用连续时间微分方程。因此需要选择步长 $\Delta$，把连续系统离散化：

{{< rawhtml >}}
$$
\begin{aligned}
h_t
&= \overline{A} h_{t-1} + \overline{B} x_t \\
y_t
&= C h_t + D x_t
\end{aligned}
$$
{{< /rawhtml >}}

这已经很像一个没有非线性激活函数的 RNN。

### 2.2 为什么线性递推可以并行训练

递归公式看起来必须先算 $h_1$，再算 $h_2$。但线性变换可以组合。把每一步写成一对参数：

{{< rawhtml >}}
$$
T_t = (\overline{A}_t, \overline{B}_t x_t)
$$
{{< /rawhtml >}}

定义组合运算：

{{< rawhtml >}}
$$
(A_2, b_2) \circ (A_1, b_1)
=
(A_2 A_1,\; A_2 b_1 + b_2)
$$
{{< /rawhtml >}}

这个运算满足结合律，因此可以使用 `parallel scan`，像并行前缀和一样分层合并相邻区间。顺序计算需要沿序列走 $L$ 步；并行 scan 的依赖深度可以降到 $O(\log L)$。

这解释了现代线性 SSM 的核心优势：

- **训练**：对完整序列使用卷积、scan 或分块矩阵乘法并行计算。
- **解码**：继续使用递推形式，只更新当前状态。

这也是为什么“线性”非常重要。普通非线性 RNN 的两步变换通常不能被压缩成同一种简单形式，因此很难直接使用 parallel scan。

## 三、从固定 SSM 到 Mamba

早期 SSM 的 $A$、$B$、$C$ 通常对所有 token 固定。固定系统很适合建模稳定的长程模式，却不容易根据内容决定应该记住什么。

例如，模型读到：

> 密码是 7391。后面的描述很长……请重复密码。

理想模型应该在看到“密码是 7391”时强力写入，在处理中间无关内容时尽量保持状态，最后再读出密码。固定参数较难实现这种与内容相关的选择。

Mamba-1 的关键改变是让部分 SSM 参数依赖当前输入：

{{< rawhtml >}}
$$
\begin{aligned}
\Delta_t &= s_{\Delta}(x_t) \\
B_t &= s_B(x_t) \\
C_t &= s_C(x_t)
\end{aligned}
$$
{{< /rawhtml >}}

这里的 `selective` 可以从三个角度理解：

- $\Delta_t$ 控制当前状态更新的步幅，可理解为遗忘速度。
- $B_t$ 控制当前 token 怎样写入状态。
- $C_t$ 控制当前时刻从状态中读取什么。

参数变成输入相关以后，普通全局卷积不再适用。Mamba 使用硬件感知的 selective scan，在 GPU 上高效计算这种时变递推。

Mamba-2 又从 `State Space Duality` 出发，说明一类结构化 SSM 与因果注意力可以写成相近的矩阵形式，并使用更适合现代 GPU 的分块算法。它把 SSM 的训练进一步转化为大块矩阵乘法。

## 四、Mamba-3：让线性状态更会表达

Mamba-3 没有放弃线性递推，而是从状态空间模型本身继续改进。论文的三个核心变化是：

1. `exponential-trapezoidal discretization`：更精确、表达力更强的离散化。
2. `complex-valued state update`：用复数状态实现数据相关的旋转。
3. `MIMO SSM`：让多个输入、输出共享状态动力学。

![Mamba-2 与 Mamba-3 block 对比（来源：Mamba 官方仓库）](https://raw.githubusercontent.com/state-spaces/mamba/main/assets/mamba3.png)

### 4.1 从 Euler 到指数梯形离散化

离散化不是无关紧要的实现细节。连续系统在一个 token 间隔中持续受到输入影响，而离散模型必须决定怎样近似这段时间里的输入。

Mamba-1/2 的更新可以从 `exponential-Euler` 规则解释：主要使用当前端点的输入。Mamba-3 使用更一般的 `exponential-trapezoidal` 规则，同时考虑区间两端：

{{< rawhtml >}}
$$
\begin{aligned}
h_t
&= e^{\Delta_t A_t} h_{t-1} \\
&\quad + (1-\lambda_t)\Delta_t e^{\Delta_t A_t} B_{t-1}x_{t-1} \\
&\quad + \lambda_t \Delta_t B_t x_t
\end{aligned}
$$
{{< /rawhtml >}}

当 $\lambda_t = 1$ 时，它退化成 Mamba-2 对应的 Euler 形式；当 $\lambda_t = \tfrac{1}{2}$ 时，它对应经典梯形规则，对区间两个端点取平均。

直觉上，旧更新只看当前写入信号，新的更新还显式考虑上一个 token 的写入信号。这在递推中引入了一个很轻量的局部卷积效果，使模型能表达更丰富的短程动态，同时仍保留可并行计算的结构。

### 4.2 复数状态：衰减之外还可以旋转

许多线性递推模型的状态更新主要表现为缩放：

{{< rawhtml >}}
$$
h_t = a_t h_{t-1} + b_t
$$
{{< /rawhtml >}}

如果 $a_t$ 是正实数，旧状态主要只能被放大或衰减。这种动态适合“记住多少”，但不擅长精确追踪顺序、位置和周期。

Mamba-3 把状态转移扩展到复数：

{{< rawhtml >}}
$$
\widetilde{A}_t
=
\rho_t e^{i\theta_t}
$$
{{< /rawhtml >}}

其中 $\rho_t$ 控制状态的衰减，$\theta_t$ 控制状态在复平面上的旋转。论文证明，这种更新可以等价地实现为一种数据相关的旋转位置编码，因此不必真的依赖低效的复数计算。

可以把它类比为：

- 实数衰减回答“这条信息还应该保留多少”。
- 复数旋转额外记录“状态经历了怎样的相位或顺序变化”。

这让模型在需要状态追踪的合成任务上明显更强，也补足了很多线性模型只擅长模糊记忆、不擅长精确跟踪的问题。

### 4.3 从 SISO 到 MIMO

以往 Mamba 主要使用 `SISO`，即每个状态动力系统处理单个输入和单个输出。Mamba-3 引入 `MIMO`，让同一组状态动力学同时接收多个输入、产生多个输出。

如果把状态看作一组共享记忆，SISO 更像每次只允许一名读者写入和读取；MIMO 则允许多个通道协作使用同一组记忆。它提升了每个状态维度所承载的计算和表达能力。

这里还有一个硬件层面的动机：自回归解码常常受显存带宽限制，而不是算力限制。MIMO 在不增大状态大小的前提下增加计算量，可以更充分地使用 GPU 算力。论文报告，在固定状态大小下，MIMO 最多增加约 $4\times$ FLOPs，但 wall-clock 解码延迟仍与 Mamba-2 接近。

### 4.4 Mamba-3 应该怎样理解

Mamba-3 的重点不是“完全换一种语言模型”，而是证明 SSM 的经典设计选择仍有很大改进空间：

- 更好的数值离散化，可以带来更强的递推形式。
- 复数状态可以提升顺序和状态追踪能力。
- MIMO 可以用更多计算换表达力，同时不明显增加解码延迟。

它仍然保持 Mamba 路线的核心特征：训练时并行、解码时常数状态、整体计算量随序列长度近似线性增长。

## 五、ParaRNN：让非线性 RNN 也能并行训练

线性 SSM 容易并行，但线性也限制了状态更新的表达能力。GRU、LSTM 等传统 RNN 在每一步内部使用非线性门控，状态更新更灵活，却必须顺序计算。

ParaRNN 的问题意识是：

> 能不能不把递推限制在线性形式，而是换一种数学方法，并行求出非线性 RNN 的整条状态轨迹？

![顺序递推与 parallel scan（来源：Apple Machine Learning Research）](https://mlr.cdn-apple.com/media/figure4_9120f99899.png)

### 5.1 把递推改写成方程组

一般 RNN 的递推写作：

{{< rawhtml >}}
$$
h_t = f_t(h_{t-1}, x_t)
$$
{{< /rawhtml >}}

把所有时间步的状态放在一起，并把等式移到一边：

{{< rawhtml >}}
$$
F_t(h)
=
h_t - f_t(h_{t-1}, x_t)
=
0
$$
{{< /rawhtml >}}

整段序列的正确隐藏状态，就是非线性方程组 $F(h)=0$ 的解。这样，计算 RNN 不再只是“从左到右运行程序”，也可以被看成“求一个大型方程组的根”。

### 5.2 用 Newton 方法反复线性化

Newton 方法从一个状态轨迹猜测 $h^{(k)}$ 开始，在当前点附近对非线性方程做一阶近似，并求修正量：

{{< rawhtml >}}
$$
J_F(h^{(k)}) \delta h^{(k)}
=
-F(h^{(k)})
$$
{{< /rawhtml >}}

然后更新：

{{< rawhtml >}}
$$
h^{(k+1)}
=
h^{(k)} + \delta h^{(k)}
$$
{{< /rawhtml >}}

虽然原方程是非线性的，但每一轮 Newton 迭代只需要解一个线性系统。由于每个 $F_t$ 只依赖 $h_t$ 和 $h_{t-1}$，它的 Jacobian 具有块双对角结构：

{{< rawhtml >}}
$$
J_F
=
\begin{bmatrix}
I & 0 & 0 & \cdots \\
-J_2 & I & 0 & \cdots \\
0 & -J_3 & I & \cdots \\
\vdots & \vdots & \ddots & \ddots
\end{bmatrix}
$$
{{< /rawhtml >}}

这个结构对应一个线性递推，因此又可以使用 parallel scan 并行求解。

![ParaRNN：把非线性递推线性化，再用 parallel scan 求解（来源：Apple Machine Learning Research）](https://mlr.cdn-apple.com/media/figure5_55a0111ea2.png)

ParaRNN 的整体流程可以概括为：

1. 同时猜测整段序列的隐藏状态。
2. 计算每个时间步的残差和局部 Jacobian。
3. 用 parallel scan 并行解线性化后的修正量。
4. 更新整条状态轨迹，重复若干轮，直到足够接近顺序递推结果。

### 5.3 ParaRNN 和线性 SSM 的关系

线性 SSM 只需一次 scan 就能得到精确状态；ParaRNN 用多轮 Newton 迭代，每轮内部做一次线性 scan。因此它用额外计算换来了更自由的非线性递推。

两者的共同点是，都在利用序列 Jacobian 或递推算子的结构。区别在于：

| 对比项 | 线性 SSM / Mamba | ParaRNN |
|---|---|---|
| 状态更新 | 受结构约束的线性递推，参数可依赖输入 | 一般非线性递推 |
| 并行方法 | 一次 scan 或分块矩阵乘法 | Newton 迭代，每轮做 parallel scan |
| 训练结果 | 直接计算递推结果 | 迭代逼近顺序 RNN 结果 |
| 解码方式 | 逐 token 递推 | 逐 token 递推 |
| 主要优势 | 算法成熟、长序列效率高 | 可以重新使用 GRU、LSTM 等强非线性单元 |
| 主要代价 | 状态表达受线性结构限制 | 迭代次数、收敛性和 Jacobian 成本 |

### 5.4 ParaRNN 的实际限制

ParaRNN 并不意味着任意 RNN 都能无代价地并行化。

- **需要收敛**：Newton 方法依赖初值和系统性质，递推过于不稳定时可能需要更多迭代。
- **需要计算 Jacobian**：一般稠密 Jacobian 成本很高，因此论文和代码重点优化对角或块对角结构。
- **训练收益更明显**：自回归解码本来就一次只产生一个 token，没有整段未来状态可以并行求解。
- **数值误差会累积**：官方实现提醒，并行结果与顺序结果的误差大致会随机器精度和序列长度增长。

因此，ParaRNN 更像一个新的训练框架：研究者可以设计比线性 SSM 更自由的递归单元，再利用结构化 Jacobian 和 CUDA kernel 获得并行训练能力。

## 六、与 Transformer 放在一起比较

三类模型的差异可以先看最核心的“历史信息存在哪里”：

| 模型 | 历史信息载体 | 训练时序列并行 | 解码缓存 | 主要风险 |
|---|---|---|---|---|
| Transformer | 每个历史 token 的 KV | 强 | 随上下文增长 | 长序列注意力和 KV cache 成本 |
| Mamba / 线性 SSM | 固定大小递归状态 | 强 | 固定大小 | 历史被压缩后可能丢失细节 |
| ParaRNN | 固定大小非线性递归状态 | 通过迭代实现 | 固定大小 | 并行训练需要收敛和多轮求解 |

从复杂度看，设序列长度为 $L$：

- 标准全注意力训练计算量通常为 $O(L^2)$。
- SSM、Mamba 和 RNN 的顺序计算量通常为 $O(L)$。
- SSM 和 ParaRNN 的并行算法能降低顺序依赖深度，但不代表总工作量变成 $O(\log L)$。

最后一点很容易混淆。Parallel scan 的 $O(\log L)$ 描述的是并行深度；所有 token 对应的工作仍然要完成，总计算量通常仍随 $L$ 线性增长。

## 七、怎样形成整体认识

可以用两个问题来理解状态空间语言模型的演进。

第一个问题是：**怎样让递归模型可以并行训练？**

- S4、Mamba 等方法约束递推结构，使它能转化为卷积、scan 或分块矩阵乘法。
- ParaRNN 保留非线性递推，再把整条轨迹改写为方程组，通过 Newton 迭代反复求解结构化线性问题。

第二个问题是：**固定大小状态怎样保留足够多的信息？**

- Mamba-1 用输入相关的选择机制决定写入、读取和遗忘。
- Mamba-2 改进算法和状态空间对偶视角。
- Mamba-3 用指数梯形离散化、复数旋转和 MIMO 提升状态表达力。
- ParaRNN 则允许研究者使用更强的非线性门控状态更新。

所以，Mamba-3 和 ParaRNN 并不是简单的替代关系。Mamba-3 代表“继续把结构化线性状态空间做强”，ParaRNN 代表“设法解除线性约束，同时保留训练并行性”。它们共同说明：在 Transformer 之外，递归模型的算法空间仍然很大。

## 八、值得继续追踪的问题

- 固定大小状态能否可靠处理需要精确回忆原文的超长上下文任务？
- Mamba-3 的复数旋转与 Transformer 中的 RoPE，在真实语言任务里分别学到了什么？
- MIMO 增加算力利用率的收益，在不同 GPU 和推理 batch size 下是否稳定？
- ParaRNN 在更深网络、更长序列和更复杂递归单元上需要多少 Newton 迭代？
- 混合架构是否更实际：大部分层使用 SSM，少量层使用 attention 负责精确检索？

理解这些问题后，再阅读新模型时就不必只看它是否“线性复杂度”。更重要的是看：状态如何更新、训练怎样并行、解码保存什么，以及模型为了效率牺牲了哪些表达能力。
