---
date: '2026-05-09T13:00:00+08:00'
draft: false
title: '大模型面试精讲（四）：强化学习与对齐'
tags: ["LLM", "面试", "RLHF", "PPO", "DPO", "GRPO", "Reward Model"]
summary: '从 RL 基础到 DPO/GRPO，系统整理大模型强化学习与对齐的面试考点。'
---

这部分建议按"RL 基础 -> actor/critic -> PPO -> DPO/GRPO -> KL 约束 -> 失败模式"来理解。真正的难点不在名词，而在这些方法分别解决什么问题、为什么适合大模型后训练。

参考教材：大模型面试题精讲教材

<!--more-->

## 一、强化学习的基本目标

在强化学习里，智能体和环境交互，在状态 $s$ 下采取动作 $a$，得到奖励 $r$，目标是最大化长期回报。

$$
J(\theta) = \mathbb{E}_{\pi_{\theta}}[G_t], \quad G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

找到参数 $\theta$，让策略产生的期望回报最大。

## 二、value-based、policy-based、actor-critic

### 2.1 value-based

不直接学策略，而是先学价值函数 $V(s)$ 或 $Q(s, a)$，然后根据价值函数导出动作选择规则。对离散动作空间直观有效，但在大模型这种超高维动作空间里不现实。

### 2.2 policy-based

直接学习策略 $\pi_{\theta}(a \mid s)$，不绕道先学 value 再间接推策略。能直接处理随机策略，更适合语言模型这种"动作是整个词表分布"的场景。但方差大，训练容易不稳定。

### 2.3 actor-critic

结合两者：actor 学策略，critic 学价值估计。纯 policy gradient 方差大，引入 critic 后可以用价值估计帮助 actor 降方差。这类结构是大模型 RLHF 中非常核心的思想背景。

## 三、DQN、REINFORCE、PPO、GSPO、GRPO、DPO、DAPO 的关系

这一组方法的面试重点：它们优化的是 value、policy 还是 preference objective；是在线 RL、离线 RL 还是"监督式偏好优化"；为什么适合或不适合 LLM 后训练。

### 3.1 DQN

典型的 value-based 方法，核心是学习动作价值函数 $Q(s,a)$：

$$
Q(s,a) \leftarrow r + \gamma \max_{a'} Q(s', a')
$$

离散动作空间常见，off-policy，不适合直接拿来做语言模型 RLHF——动作空间太大，输出是整个 token 分布。

### 3.2 REINFORCE

最基础的 policy gradient 方法：

$$
\nabla_{\theta} J(\theta) = \mathbb{E}[\nabla_{\theta} \log \pi_{\theta}(a \mid s) \cdot R]
$$

如果某个动作带来高回报，就提高它的概率；如果带来低回报，就降低它的概率。理论形式简单直观，但方差很大，训练容易震荡。更像理解 policy gradient 的起点，而不是大模型后训练的最终实用方案。

### 3.3 PPO

PPO 是 RLHF 中最高频的方法之一。核心不是"让模型更聪明"，而是"让策略更新更稳"。

**PPO 想解决什么问题**：原始 policy gradient 更新时可能一步走太远，新策略和旧策略差异太大，奖励估计噪声被放大，训练不稳定。

**关键公式**：

$$
L = \mathbb{E}[\min(r_t A_t, \operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t)]
$$

其中 $r_t = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta, old}(a_t \mid s_t)}$ 是新旧策略在该动作上的概率比，$A_t$ 是 advantage。

**为什么 PPO 稳**：若更新合理，正常优化；若更新太猛，clip 截断目标，避免继续鼓励走得太远。可以把 PPO 理解为：policy gradient 告诉你朝哪走，PPO 告诉你别一脚踩太猛。

**PPO 在 RLHF 中为什么常和 KL 一起出现**：clip 只限制"相对旧策略"的局部更新，而 KL 约束常用于限制"相对参考模型"的整体偏移。PPO clip 管单步稳定，KL 管整体别走偏。

### 3.4 GSPO

Alibaba Qwen 团队提出的 Group Sequence Policy Optimization。可以理解为对组式策略优化方法做 sequence-level 稳定化设计。

在一些组式 RL 训练里，如果重要性比率或优化信号定义得过于 token-level，容易出现梯度方差大、长回答训练不稳定的问题。GSPO 更强调 sequence-level 的优化视角，把序列作为整体做重要性比率或裁剪。

### 3.5 GRPO

基于组内相对比较信号进行优化的方案。不是为每个答案都依赖绝对分数，而是让模型看同一问题下多个候选答案之间谁更好。

更强调：相对排序、组内比较、利用比较信号代替一部分绝对价值估计。

在大模型后训练里，绝对奖励往往噪声大，value model 训练成本高，相对偏好信息有时更稳定，所以组内比较有工程吸引力。

### 3.6 DPO

DPO 最关键的点是：把偏好学习写成一个监督式优化问题，而不是完整在线 RL 闭环。

**输入**：一组三元组——prompt $x$、chosen 回答 $y^+$、rejected 回答 $y^-$。

**核心思想**：引入参考模型 $\pi_{ref}$，优化当前策略 $\pi_{\theta}$，让 chosen 相对 rejected 的对数概率比更有优势：

$$
\log \sigma \Big( \beta \big[ \log \frac{\pi_{\theta}(y^+ \mid x)}{\pi_{ref}(y^+ \mid x)} - \log \frac{\pi_{\theta}(y^- \mid x)}{\pi_{ref}(y^- \mid x)} \big] \Big)
$$

相对参考模型，让 chosen 的优势更大，让 rejected 的优势更小。

**为什么火**：省掉了 RLHF 中一些昂贵步骤——不一定需要在线 rollout，不一定需要显式 reward model + PPO 全流程，工程实现更简单。

**优势**：简洁、好实现、训练更像监督学习、工程成本低。

**限制**：强依赖偏好数据质量，偏好数据覆盖不到的地方泛化能力可能有限，loss 下降不等于真实能力必然提升。

### 3.7 DAPO

ByteDance Seed 提出的 Decoupled Clip and Dynamic Sampling Policy Optimization。

**提出背景**：直接用公开 GRPO / PPO 配方去复现高质量 reasoning RL 结果时，往往会遇到长 CoT 训练不稳定、长回答被长度偏置影响、clip 设计不够细等问题。

**两个关键词**：

- **Decoupled Clip**：不要把所有 token、所有位置、所有类型的更新都用同一套粗糙 clip 机制处理，而要更细致地控制更新范围。
- **Dynamic Sampling**：训练时样本分布、答案长度、难度结构都在动态变化，采样策略也应跟着调整。

**和 PPO / GRPO 的关系**：

- PPO：经典稳定化策略优化框架
- GRPO：更强调组内相对比较信号
- DAPO：面向大规模 LLM reasoning RL 的工程化稳定优化方案，重点在 clip 和 sampling 的改造

### 3.8 一张图理解

- **DQN**：value-based，偏传统离散动作 RL
- **REINFORCE**：最原始 policy gradient
- **PPO**：稳定化的在线策略优化
- **GSPO**：更强调 sequence-level 稳定性的组式策略优化
- **GRPO**：基于组内相对比较的策略优化
- **DPO**：监督式偏好优化，不显式跑完整 RL 闭环
- **DAPO**：面向大规模 reasoning RL 的工程化稳定优化方案

## 四、Reward Model 在 RLHF 里扮演什么角色

reward model 本质上是一个近似人类偏好的打分器。输入是 prompt + 模型回答，输出是一个标量分数，表示这个回答"看起来多好"。

因为人类不能在每一步训练中都实时打分，所以要先训练一个近似器，代替人类偏好做大规模优化。但 reward model 不是完美真理，只是近似器，会有偏差、噪声、被模型钻空子的风险。这也是 reward hacking 的根源之一。

## 五、KL 约束在对齐中的作用

### 5.1 为什么需要 KL 约束

无论是 PPO 还是某些变体，对奖励优化如果完全放开，模型可能会：过度迎合 reward model、语言分布偏移太远、变得机械单一过度自信、出现 reward hacking 或 mode collapse。

所以常常加入 KL 约束，把新策略拉回参考模型附近。

### 5.2 直觉解释

参考模型相当于一个"语言与能力锚点"。KL 惩罚本质上在说：可以变好，但别离原始模型太远。

### 5.3 在 RLHF 里为什么尤其重要

因为 reward model 的目标不是完整真实世界目标，只是近似的人类偏好。如果没有 KL，模型可能疯狂优化 reward 的漏洞。

一句话：KL 约束在对齐里本质上是一个 regularizer，用来平衡"提高奖励"和"保持模型不要偏离原始语言模型分布太远"。

## 六、online / offline RL，on-policy / off-policy

### 6.1 online vs offline

- **online RL**：边交互边收集新数据边训练
- **offline RL**：只在固定数据集上训练，不再实时探索

### 6.2 on-policy vs off-policy

- **on-policy**：主要依赖当前策略生成的数据
- **off-policy**：可以复用旧策略甚至别的策略生成的数据

PPO 通常是 on-policy，DQN 通常是 off-policy。

### 6.3 为什么不能混

online / offline 描述的是数据获取方式，on-policy / off-policy 描述的是数据是否必须来自当前策略。它们相关，但不是同一维度。

## 七、DPO 为什么会出现 loss 下降但性能没有提升

这是极高频追问，必须答得比"目标不一致"更具体。

### 7.1 第一层：训练目标不等于评测目标

loss 降低只能说明模型越来越符合 DPO 训练目标，不说明真实任务指标一定同步提升。

### 7.2 第二层：偏好数据有噪声

如果 chosen / rejected 本身质量不稳定，模型就可能学到错误偏好模式。

### 7.3 第三层：学到的是表面模式，不是能力

例如：更长的回答更容易被偏好、更自信的措辞更容易被偏好、更像模板答案的风格更容易被偏好。模型可能只学会这些外观特征，而不是真的提升了推理或事实能力。

### 7.4 第四层：分布外泛化差

DPO 在训练分布内可能越来越好，但测试分布稍微变化，收益就不明显。

### 7.5 第五层：超参数问题

例如 $\beta$ 太大或太小，都可能导致更新过强或过弱，与参考模型距离控制不当。

## 八、Reward Hacking

Reward hacking 指模型发现了奖励函数或奖励模型的漏洞，用投机方式拿高分，而不是真正完成目标。

**典型表现**：奖励模型喜欢长答案，模型就变得冗长；奖励模型喜欢自信语气，模型就更敢编；奖励模型偏好某些格式，模型就机械套格式。

**为什么本质上难避免**：因为 reward model 只是目标的代理，不是目标本身。只要是代理目标，就存在被"优化穿透"的风险。

**怎么缓解**：提升 reward model 质量、做多维评估不只看单一奖励、加 KL 约束、做人工审核、引入事实性安全性拒答能力等额外约束。

## 九、熵坍塌 / 模态坍塌

模型输出分布越来越集中，只剩少数高奖励模式，回答风格变单一，多样性下降。

如果奖励只强烈偏向某类答案，模型就会不断把概率质量堆到这些模式上。后果是回答越来越模板化、探索能力下降、创造性和多样性下降。

缓解方式：熵正则、KL 约束、更丰富的数据分布、更稳健的奖励设计、控制更新幅度。

## 十、面试作答框架

**如果问"PPO、DPO、GRPO 怎么选"**：

- PPO：在线 RL 风格更强，稳定、成熟，但工程链路更重
- DPO：更像监督式偏好优化，流程轻，但更依赖偏好数据质量
- GRPO：更强调相对比较信号，在一些大模型训练设定里能减少对显式 critic 的依赖

**如果问"KL 为什么重要"**：它不是装饰项，是防止模型为了追奖励而走偏的关键约束，本质上平衡了"变好"和"别偏离太远"。

**如果问"DPO 为什么 loss 降了效果不升"**：因为训练目标下降只说明更拟合偏好目标，不代表真实能力、泛化能力、事实性一定同步提升。
