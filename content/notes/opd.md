---
date: '2026-05-15T10:30:00+08:00'
draft: false
title: 'On-Policy Distillation：现象、机制与实践配方'
tags: ["LLM", "Distillation", "OPD", "Post-Training", "RL"]
summary: '整理 On-Policy Distillation 的基本形式、训练信号、成功条件、token 级机制、实践修复方法，以及长链路推理中的局限。'
---

`On-Policy Distillation`，简称 `OPD`，可以理解为一种介于传统知识蒸馏和强化学习之间的大模型后训练方法。它的核心不是让学生模型直接模仿老师已经写好的答案，而是让学生模型先按自己的当前策略生成回答，再让老师模型在这些学生实际访问到的状态上提供 token 级监督。

参考文章：

- [知乎：OPD 相关解读](https://zhuanlan.zhihu.com/p/2000612721868177979)
- [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/html/2604.13016v2)

<!--more-->

## 一、为什么需要 OPD

传统蒸馏最常见的方式是 `off-policy distillation`：先让 teacher 生成一批固定回答，再让 student 在这些 teacher 轨迹上做监督学习。这种方式实现简单，但有一个明显问题：训练时学生看到的是老师轨迹，推理时学生却会沿着自己的分布生成文本。

这会带来 `exposure bias`。也就是说，学生一旦在推理中走到了老师轨迹之外的前缀，传统蒸馏并没有教它在这些“自己造成的状态”上该怎么继续。

OPD 试图解决的正是这个问题：

- 轨迹由 student 当前策略采样，而不是 teacher 预先生成。
- teacher 不负责给出完整标准答案，而是在 student 已经生成的前缀上给出下一 token 分布。
- 训练目标在 student 实际访问的状态上对齐 teacher，因此监督信号更贴近推理时的真实分布。

从这个角度看，OPD 的优势很直接：它把蒸馏从“模仿老师走过的路”改成了“学生自己走路，老师沿途纠偏”。

但这也带来一个更深的问题：teacher 在 student 轨迹上的 token 级反馈，真的总是有用吗？论文的核心结论是：**不一定。OPD 成功需要满足特定条件，而且长链路推理里还会出现监督质量衰减。**

## 二、OPD 的基本形式

设有一个输入 prompt：

{{< rawhtml >}}
$$
x
$$
{{< /rawhtml >}}

student 模型为：

{{< rawhtml >}}
$$
\pi_\theta
$$
{{< /rawhtml >}}

teacher 模型为：

{{< rawhtml >}}
$$
\pi_T
$$
{{< /rawhtml >}}

student 按当前策略自回归采样出回答：

{{< rawhtml >}}
$$
y = (y_1, y_2, \ldots, y_L) \sim \pi_\theta(\cdot | x)
$$
{{< /rawhtml >}}

在第 `t` 步，已有前缀：

{{< rawhtml >}}
$$
y_{\lt t}
$$
{{< /rawhtml >}}

student 和 teacher 都会在同一个状态：

{{< rawhtml >}}
$$
(x, y_{\lt t})
$$
{{< /rawhtml >}}

上给出下一 token 分布：

{{< rawhtml >}}
$$
\pi_\theta(\cdot | x, y_{\lt t})
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
\pi_T(\cdot | x, y_{\lt t})
$$
{{< /rawhtml >}}

OPD 的目标就是在这些 student-generated states 上，让 student 的下一 token 分布靠近 teacher。

常见写法是最小化 student rollout 上的逐 token reverse KL：

{{< rawhtml >}}
$$
\mathcal{L}_{OPD}
= \mathbb{E}_{y \sim \pi_\theta}
\left[
\sum_{t=1}^{L}
D_{KL}
\left(
\pi_\theta(\cdot | x, y_{\lt t})
\Vert
\pi_T(\cdot | x, y_{\lt t})
\right)
\right]
$$
{{< /rawhtml >}}

这里使用 reverse KL 很关键。它有明显的 mode-seeking 特性：student 更倾向于把概率质量集中到 teacher 认为高概率的区域，而不是覆盖 teacher 的全部可能模式。

## 三、三种常见 OPD 目标

实际训练时，完整 vocabulary 的分布很大，teacher 查询也很贵。因此 OPD 通常有三种粒度。

### 3.1 Sampled-Token OPD

这是最轻量、也很常见的版本。

student 在第 `t` 步实际采样出了 token：

{{< rawhtml >}}
$$
y_t
$$
{{< /rawhtml >}}

那么只在这个 token 上读取 student 和 teacher 的 log probability，并构造单 token 的监督信号。直观上，就是问 teacher：

> 对于 student 刚刚选出来的这个 token，你觉得它应该有多大概率？

这种方法的优点是计算便宜，因为每一步只需要关注一个 token。缺点也很明显：监督信号非常窄。如果 student 采样出来的 token 本身质量不稳定，或者 teacher 在这个前缀上的判断不可靠，那么训练会很脆弱。

可以把 sampled-token OPD 看成 full-vocabulary KL 的单样本估计。它在计算上高效，但对采样噪声和长序列误差累积比较敏感。

### 3.2 Full-Vocabulary OPD

full-vocabulary OPD 会在每个 student-generated prefix 上计算完整词表上的 KL：

{{< rawhtml >}}
$$
D_{KL}
\left(
\pi_\theta(\cdot | x, y_{\lt t})
\Vert
\pi_T(\cdot | x, y_{\lt t})
\right)
$$
{{< /rawhtml >}}

它的好处是梯度最密集，信息最完整。teacher 不只是评价 student 已经采样出来的 token，还会告诉 student 整个 vocabulary 上哪些 token 更应该被提升或压低。

问题是成本非常高。对于 batch size、sequence length 和 vocabulary size 都很大的 LLM 后训练场景，完整 KL 会带来显著的显存和计算压力。

### 3.3 Top-k OPD

Top-k OPD 是折中方案。它不看完整词表，而是只选一个局部 token 集合，例如 student 当前分布里概率最高的 `k` 个 token：

{{< rawhtml >}}
$$
S_t = TopK(\pi_\theta(\cdot | x, y_{\lt t}))
$$
{{< /rawhtml >}}

然后把 student 和 teacher 在这个集合上的概率重新归一化，再计算子集 KL。

这种做法保留了多 token 监督，比 sampled-token 更稳定；同时又避免完整 vocabulary KL 的高成本。论文实验里使用的就是类似 `Student Top-k` 的设置，默认 `LogProb top-k = 16`。

Top-k OPD 背后的判断是：OPD 最重要的训练信号往往集中在 student 和 teacher 都认为比较可能的高概率 token 区域，而不是长尾 vocabulary。

## 四、诊断 OPD 的三个动态指标

论文不是只看最终 benchmark 分数，而是提出了几个训练过程中的动态指标，用来判断 OPD 是否真的在对齐。

### 4.1 Overlap Ratio

设 student 在第 `t` 步的 top-k token 集合为：

{{< rawhtml >}}
$$
S_t
$$
{{< /rawhtml >}}

teacher 的 top-k token 集合为：

{{< rawhtml >}}
$$
T_t
$$
{{< /rawhtml >}}

则 overlap ratio 衡量二者交集占比：

{{< rawhtml >}}
$$
OR_t = \frac{|S_t \cap T_t|}{k}
$$
{{< /rawhtml >}}

如果 overlap ratio 很低，说明 student 和 teacher 在同一个前缀上认为“最可能的下一 token”差异很大。此时 teacher 的 token 级信号不一定能有效推动 student，因为两者甚至不在同一片候选空间里。

如果 overlap ratio 持续上升，说明 student 正在逐步进入 teacher 的高概率区域。

### 4.2 Overlap-Token Advantage

只看 top-k 集合是否重合还不够，因为即使 token 集合相同，二者在这些 token 上分配的概率也可能差异很大。

Overlap-token advantage 用来衡量 student 和 teacher 在交集 token 上的相对偏好是否接近。直观理解即可：

- 数值接近 `0`，说明 student 在重合 token 上的概率分布和 teacher 更接近。
- 数值明显偏负，说明 student 对某些 token 过度自信，但 teacher 并不那么认可。

这个指标能看出 OPD 是否只是在“找到同一批候选 token”，还是进一步学会了“在这些 token 之间按 teacher 的偏好重新分配概率”。

### 4.3 Entropy Gap

entropy 衡量分布的不确定性。teacher 和 student 在同一个 prefix 上可能有不同的置信度。

熵差可以写成：

{{< rawhtml >}}
$$
\Delta H_t = H(\pi_\theta(\cdot | x, y_{\lt t})) - H(\pi_T(\cdot | x, y_{\lt t}))
$$
{{< /rawhtml >}}

如果 entropy gap 很大，说明二者不仅偏好的 token 不一样，连“该不该自信”都不一样。成功的 OPD 往往会伴随 entropy gap 缩小，表示 student 不只是模仿 teacher 的答案倾向，也在接近 teacher 的局部不确定性结构。

## 五、OPD 成功的两个条件

论文最重要的经验结论是：OPD 不是“teacher 越强越好”。成功 OPD 主要取决于两个条件。

### 5.1 条件一：思考模式一致

student 和 teacher 需要有相对兼容的 thinking pattern。

这里的 thinking pattern 不是一个抽象口号，而可以通过 token 分布体现出来：在 student 自己生成的前缀上，二者的高概率 token 集合是否有较高 overlap。

论文里有一个很有意思的现象：更强的 teacher 不一定带来更好的 OPD。如果 teacher 的推理风格和 student 差异过大，那么即使 teacher benchmark 分数更高，它给出的 token 级信号也可能无法被 student 有效利用。

这可以理解为局部优化问题。OPD 每一步只是在 student 当前访问到的状态附近做 token 级分布对齐。如果 teacher 的策略分布和 student 相距太远，那么 teacher 的高分答案虽然全局上更好，但局部梯度未必能把 student 推过去。

因此，OPD 成功时通常会看到：

- 初始 overlap ratio 较高。
- overlap ratio 在训练中稳定上升。
- entropy gap 逐渐缩小。
- benchmark 分数随这些动态同步改善。

失败时则常见：

- 初始 overlap ratio 低。
- overlap ratio 长时间停滞。
- teacher 和 student 的置信度结构不匹配。
- 分数没有明显提升，即使 teacher 本身很强。

### 5.2 条件二：teacher 需要提供新知识

只有 thinking pattern 一致还不够。teacher 还必须拥有 student 没见过、没学到的新能力。

如果 teacher 只是同一训练流程下更大规模的模型，它的 benchmark 分数可能更高，但从 student 的分布视角看，它未必提供了足够新的可迁移信号。论文通过 DeepSeek 和 Qwen 两个模型族的对比说明：同 pipeline 的大 teacher 带来的提升有限，而经过额外 RL post-training 的 teacher 能带来更强增益。

这说明 OPD 里的 teacher 不能只满足“更大”或“分数更高”。更重要的是：

- 它和 student 的策略空间不能相距过远。
- 它又必须包含 student 当前没有掌握的新能力。

这两个条件有张力。teacher 太像 student，可能没有新东西；teacher 太不像 student，局部 token 信号又无法利用。OPD 的有效区间就在两者之间。

## 六、OPD 的 token 级机制

论文对 OPD 机制的解释可以概括为一句话：

**成功的 OPD 是 student 在自己访问到的状态上，逐步对齐 teacher 的高概率 token 区域。**

这不是全词表平均意义上的对齐，而是集中发生在一个很小但非常关键的 overlap token 集合上。

### 6.1 高概率 token 的渐进对齐

在成功训练中，student 和 teacher 的 top-k token overlap 会持续上升。例如论文报告的动态里，成功 OPD 的 overlap ratio 会从较低水平逐渐增长到更高水平，同时 overlap-token advantage 接近 `0`，entropy gap 缩小。

这说明 student 学到的不是某个孤立答案，而是 teacher 在一系列 student-generated prefixes 上的局部决策结构。

换句话说，OPD 的学习对象不是完整 reasoning trace，而是：

> 在当前前缀下，哪些下一 token 是合理候选，以及这些候选之间应该怎样分配概率。

### 6.2 overlap token 承载主要概率质量

一个关键观察是，student 和 teacher 共同的 top-k token 虽然数量很少，却承载了绝大部分概率质量。论文报告中，这些 overlap tokens 大约覆盖 `97% - 99%` 的概率质量。

这解释了为什么 Top-k OPD 可以有效。因为真正重要的梯度不均匀分布在整个 vocabulary 上，而是集中在高概率候选区域。只要这个区域能对齐，student 的行为就会发生实质变化。

### 6.3 只优化 overlap token 也足够

论文进一步验证：只在 student 和 teacher 的 overlap token 上做优化，也能接近标准 OPD 的效果；而非 overlap token 的贡献相对有限。

这说明 OPD 的主要机制不是“从 teacher 那里发现一大批 student 完全没考虑过的 token”，而是更像：

- student 和 teacher 已经有一部分候选空间重合。
- teacher 在这片空间里提供更好的概率重分配。
- reverse KL 把 student 推向 teacher 更认可的高概率模式。

因此，初始 overlap 很重要。如果一开始几乎没有共同候选区域，OPD 的核心机制就缺少发力点。

## 七、失败 OPD 的修复配方

论文给出两个实用修复方向：一个从模型初始化入手，一个从数据 prompt 入手。

### 7.1 Off-policy cold start

如果 student 和 teacher 的 thinking pattern 差距太大，可以先做一段 off-policy distillation 作为冷启动。

流程可以写成两阶段：

1. 让 teacher 在一批 prompts 上生成完整回答。
2. 用这些 teacher rollouts 对 student 做 SFT。
3. 再切换到 OPD，让 student 自己 rollout，并接受 teacher token-level supervision。

这看起来像是把传统蒸馏和 OPD 串起来。它的作用不是替代 OPD，而是先把 student 拉到 teacher 附近，让后续 OPD 的局部 token 信号变得可利用。

论文实验中，SFT cold start 后的 student 初始 overlap ratio 更高，entropy gap 更小，随后 OPD 的训练曲线也更稳定。更重要的是，它不只是改善早期收敛，还会提高后续 OPD 的最终性能上限。

### 7.2 使用 teacher-aligned prompts

第二个修复方向是 prompt 选择。

teacher 的策略不是凭空形成的，它通常来自某个 post-training 数据分布。如果 OPD 使用的 prompts 和 teacher 后训练时的 prompt template 或内容分布更接近，teacher 在这些状态上的反馈通常更稳定、更可利用。

论文从两个层面验证：

- prompt template 对齐：同一个数学问题，用 teacher 后训练时更熟悉的指令格式，OPD 效果更好。
- prompt content 对齐：使用 teacher post-training 数据分布中的问题，比只使用泛泛的同领域数据更有效。

但这里也有副作用。过度使用 teacher-aligned prompts 可能让 student entropy 变得过低，导致分布变窄。因此实践中更稳妥的做法不是只用 teacher-aligned prompts，而是把它们和其他分布的 prompts 混合，避免 student 过早坍缩到太窄的模式。

## 八、OPD 和 RL 的关系

OPD 很容易被看成一种特殊形式的 RL。

在 outcome-reward RL 中，模型通常生成完整答案，然后根据最终对错、偏好模型或 verifier 得到一个稀疏 reward。这个信号可能很准，但太稀疏，credit assignment 很难。

OPD 则不同。teacher 在每个 token 位置都能提供 log probability，因此 reward 是稠密的。每一步都可以知道 student 当前 token 相对 teacher 是更好还是更差。

这也是 OPD 吸引人的地方：

- 比 SFT 更 on-policy，因为轨迹来自 student 当前策略。
- 比 outcome RL 更稠密，因为每个 token 都有 teacher 信号。
- 比完整 RL 更简单，因为不需要单独训练 reward model 或 verifier。

但这也解释了它的局限：OPD 的 reward 来自 teacher 在 student prefix 上的局部判断。如果 student 生成的前缀越来越偏离 teacher 熟悉的状态，teacher 的 token 级反馈就会变得不可靠。

## 九、长链路推理中的局限

论文最后指出了 OPD 的一个关键天花板：trajectory 越长，dense token-level reward 的可靠性越可能下降。

在短到中等长度的 reasoning trace 中，teacher 还能比较可靠地评估 student prefix 下的下一 token。但在很长的链路中，student 早期的小偏差会不断累积，后面的 prefix 可能变成 teacher 自己很少访问的状态。

这时会出现几个问题：

- teacher continuation 的优势随 prefix 深度下降。
- 后部 token 的 reward 更不稳定。
- 后部不稳定会逐渐向前传播，影响整体训练。
- 全局 reward 可能仍然区分正确和错误 rollout，但局部 token 梯度不一定可利用。

论文里一个重要结论是：失败 OPD 不一定是 teacher 的全局信号没信息。某些失败设置中，teacher 给正确 rollout 的平均 reward 仍然高于错误 rollout，AUROC 也不错。但局部 token 级梯度可能方向不一致，聚合后互相抵消，导致训练推不动。

这说明 OPD 的问题不只是 reward quality，而是 local optimization geometry。也就是说，teacher 的信号全局上知道什么是好答案，但在 student 当前策略附近，未必能给出稳定、方向一致的局部更新。

## 十、实践建议

如果要在大模型后训练里使用 OPD，可以按下面的顺序检查。

### 10.1 先看 teacher 是否合适

不要只看 teacher benchmark 分数。更重要的是同时检查：

- teacher 是否比 student 有新能力。
- teacher 和 student 是否有相近的 thinking pattern。
- teacher 是否来自过于不同的模型族、tokenizer 或 post-training recipe。
- teacher 在 student rollouts 上的 top-k overlap 是否足够高。

如果 teacher 很强但 overlap 很低，直接 OPD 可能失败。

### 10.2 训练前做小规模诊断

在正式训练前，可以抽一小批 prompts，让 student rollout，然后记录：

- top-k overlap ratio
- overlap token probability mass
- entropy gap
- teacher 对正确/错误 rollout 的 reward 区分度

如果 overlap ratio 很低，优先考虑 cold start 或 prompt 对齐，而不是直接扩大训练规模。

### 10.3 优先使用 Top-k OPD

sampled-token OPD 成本低，但信号窄；full-vocabulary OPD 信号完整，但成本高。多数工程场景可以从 Top-k OPD 起步。

一个合理配置是：

- student rollout 使用温度采样，保证探索。
- teacher 查询 student top-k token 的 log probability。
- 在 top-k 子集上计算 KL 或近似 reverse KL。
- 对特殊 token、格式 token 做必要 masking，避免训练被无意义 token 主导。

### 10.4 失败时先修 overlap

如果 OPD 不涨分，先不要急着调学习率或训练轮数。更应该先看动态指标：

- overlap ratio 是否上升？
- entropy gap 是否缩小？
- overlap-token advantage 是否接近 `0`？

如果这些指标都不动，说明 token 级机制没有启动。此时更有效的修复通常是：

- 用 teacher rollouts 做 SFT cold start。
- 换成 teacher 更熟悉的 prompt template。
- 混入 teacher post-training 数据分布的 prompts。
- 换一个 thinking pattern 更接近 student 的 teacher。

### 10.5 长链路任务要混合训练信号

对于特别长的 CoT、agentic multi-turn 或工具调用任务，不应该假设 OPD 的 dense reward 会一直可靠。

更稳妥的方向是混合：

- 短片段上的 token-level OPD。
- 长轨迹上的 outcome reward。
- verifier 或 process reward。
- curriculum，把监督长度逐步拉长。

OPD 很适合提供密集局部修正，但不一定适合独自承担所有长链路 credit assignment。

## 十一、总结

OPD 的价值在于：它让 student 在自己的分布上接受 teacher 的稠密 token 级监督，比传统 off-policy 蒸馏更贴近真实推理状态，也比稀疏 outcome RL 更容易获得训练信号。

但 OPD 不是“强 teacher 自动提升弱 student”的万能配方。它成功需要两个条件：

- student 和 teacher 的 thinking pattern 要足够兼容。
- teacher 要提供 student 尚未掌握的新知识。

从机制上看，OPD 的核心不是全词表平均对齐，而是 student 在自己访问的状态上，逐步对齐 teacher 的高概率 overlap tokens。这个 overlap 区域虽然小，却承载了绝大部分概率质量，也是主要梯度信号所在。

从实践上看，OPD 失败时应优先修复初始分布错位：用 off-policy cold start 拉近 student 和 teacher，用 teacher-aligned prompts 提高局部监督质量，再通过 Top-k OPD 进行稳定的 on-policy 对齐。

它最值得警惕的地方，是长链路任务中的 reward degradation。OPD 的 dense supervision 是优势，但 dense 不等于 reliable。当 student prefix 偏离 teacher 熟悉状态后，token 级信号可能变得局部不可利用。因此，在长 CoT 和 agentic 场景中，OPD 更适合作为混合后训练框架的一部分，而不是单独的终局方案。
