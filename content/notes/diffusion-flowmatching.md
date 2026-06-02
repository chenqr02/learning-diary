---
date: '2026-06-02T10:30:00+08:00'
draft: false
title: 'Diffusion 与 Flow Matching 详细推导'
tags: ["Diffusion", "Flow Matching", "Generative Models", "DDPM", "CNF"]
summary: '从 DDPM 的前向加噪、后验、ELBO 和噪声预测目标开始，推到采样公式；再从连续归一化流和连续性方程出发，推导 Flow Matching、Conditional Flow Matching 与 Rectified Flow，并比较二者的关系。'
---

这篇笔记整理 `Diffusion Model` 和 `Flow Matching` 的核心推导。目标不是覆盖所有变体，而是把最常用的数学链条走通：为什么 DDPM 可以训练一个预测噪声的网络，为什么采样时能一步步去噪；为什么 Flow Matching 可以不模拟随机反向过程，直接学习一个把噪声搬到数据的速度场。

参考资料：

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)
- [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Cambridge MLG Blog: An introduction to Flow Matching](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)

<!--more-->

## 一、先统一生成模型的视角

生成模型的基本任务是：真实数据来自一个未知分布 $p_{\mathrm{data}}(x)$，我们只有样本，想学一个模型分布 $p_{\theta}(x)$，使得从 $p_{\theta}$ 里采样出来的样本像真实数据。

如果直接建模 $p_{\theta}(x)$ 很难，可以换一种思路：从一个简单分布出发，比如标准高斯：

{{< rawhtml >}}
$$
z \sim \mathcal{N}(0, I)
$$
{{< /rawhtml >}}

然后学习一个变换，把 $z$ 变成数据样本：

{{< rawhtml >}}
$$
x = G_{\theta}(z)
$$
{{< /rawhtml >}}

不同生成模型的差别，主要在于这个变换怎么定义、怎么训练：

- `GAN`：用判别器提供训练信号，直接学生成器。
- `VAE`：用隐变量模型和变分下界训练。
- `Normalizing Flow`：学一个可逆变换，精确计算密度。
- `Diffusion`：先把数据逐步加噪到高斯，再学反向去噪链。
- `Flow Matching`：学一个连续速度场，把噪声分布沿 ODE 搬运到数据分布。

Diffusion 和 Flow Matching 都可以理解为“从简单分布到数据分布的路径学习”。区别在于，Diffusion 常从随机过程和 score 出发，Flow Matching 常从确定性 ODE 和速度场出发。

下面先从离散 DDPM 推起。

## 二、DDPM 的前向过程：把数据逐步加噪

设干净数据为 $x_0$。DDPM 定义一个固定的马尔可夫前向过程，每一步往样本里加入一点高斯噪声：

{{< rawhtml >}}
$$
q(x_t | x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right)
$$
{{< /rawhtml >}}

这里通常写：

{{< rawhtml >}}
$$
\alpha_t = 1 - \beta_t
$$
{{< /rawhtml >}}

$\beta_t$ 是第 $t$ 步的噪声强度。它一般从很小的值开始，逐渐增大。

这个定义等价于下面的重参数形式：

{{< rawhtml >}}
$$
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{1-\alpha_t}\epsilon_t,
\quad
\epsilon_t \sim \mathcal{N}(0, I)
$$
{{< /rawhtml >}}

如果一步步递推，$x_t$ 可以直接由 $x_0$ 采样出来，而不必真的循环 $t$ 次。定义：

{{< rawhtml >}}
$$
\bar{\alpha}_t
=
\prod_{s=1}^{t}\alpha_s
$$
{{< /rawhtml >}}

可以得到：

{{< rawhtml >}}
$$
q(x_t | x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
$$
{{< /rawhtml >}}

也就是：

{{< rawhtml >}}
$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\quad
\epsilon \sim \mathcal{N}(0, I)
$$
{{< /rawhtml >}}

这个公式很关键。训练时我们可以随机采样一个时间步 $t$，一次性构造带噪样本 $x_t$，然后让模型学会从 $x_t$ 里恢复关于 $x_0$ 的信息。

直觉上：

- $t=0$ 时，$\bar{\alpha}_t$ 接近 $1$，样本基本是干净数据。
- $t$ 越大，$\bar{\alpha}_t$ 越小，噪声占比越高。
- $T$ 足够大时，$x_T$ 近似标准高斯。

配图可以参考 Lilian Weng 文章中对 DDPM 训练和采样过程的示意：

![DDPM training and sampling diagram](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)

图片来源：[Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

## 三、DDPM 的反向过程：从噪声一步步还原

如果前向过程最终把数据变成高斯噪声，那么生成时就从高斯噪声开始，沿反方向走回数据分布：

{{< rawhtml >}}
$$
p_{\theta}(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_{\theta}(x_{t-1}|x_t)
$$
{{< /rawhtml >}}

其中：

{{< rawhtml >}}
$$
p(x_T) = \mathcal{N}(0, I)
$$
{{< /rawhtml >}}

反向一步也用高斯建模：

{{< rawhtml >}}
$$
p_{\theta}(x_{t-1}|x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_{\theta}(x_t,t),
\Sigma_{\theta}(x_t,t)
\right)
$$
{{< /rawhtml >}}

问题是：真实反向条件分布 $q(x_{t-1}|x_t)$ 不好直接得到，因为它依赖整个数据分布。但如果额外给定原始数据 $x_0$，那么：

{{< rawhtml >}}
$$
q(x_{t-1}|x_t,x_0)
$$
{{< /rawhtml >}}

是可以解析算出来的高斯分布。这就是 DDPM 训练推导的核心。

## 四、关键后验：求出 q(x_{t-1}|x_t,x_0)

我们已经知道两个高斯条件：

{{< rawhtml >}}
$$
q(x_t|x_{t-1})
=
\mathcal{N}
\left(
\sqrt{\alpha_t}x_{t-1},
\beta_t I
\right)
$$
{{< /rawhtml >}}

以及：

{{< rawhtml >}}
$$
q(x_{t-1}|x_0)
=
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_{t-1}}x_0,
(1-\bar{\alpha}_{t-1})I
\right)
$$
{{< /rawhtml >}}

由贝叶斯公式：

{{< rawhtml >}}
$$
q(x_{t-1}|x_t,x_0)
\propto
q(x_t|x_{t-1})q(x_{t-1}|x_0)
$$
{{< /rawhtml >}}

两个高斯相乘仍然是高斯。整理指数中的二次项，可以得到：

{{< rawhtml >}}
$$
q(x_{t-1}|x_t,x_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde{\mu}_t(x_t,x_0),
\tilde{\beta}_t I
\right)
$$
{{< /rawhtml >}}

其中方差为：

{{< rawhtml >}}
$$
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}
\beta_t
$$
{{< /rawhtml >}}

均值为：

{{< rawhtml >}}
$$
\tilde{\mu}_t(x_t,x_0)
=
\frac{
\sqrt{\bar{\alpha}_{t-1}}\beta_t
}{
1-\bar{\alpha}_t
}
x_0
+
\frac{
\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})
}{
1-\bar{\alpha}_t
}
x_t
$$
{{< /rawhtml >}}

这个公式说明，如果我们知道 $x_0$，就能知道从 $x_t$ 回到 $x_{t-1}$ 的真实后验均值。训练模型的自然目标就是让：

{{< rawhtml >}}
$$
p_{\theta}(x_{t-1}|x_t)
\approx
q(x_{t-1}|x_t,x_0)
$$
{{< /rawhtml >}}

但采样时没有 $x_0$，所以需要神经网络从 $x_t$ 和 $t$ 中预测相关信息。

## 五、从 ELBO 推到 KL 训练项

DDPM 是一个隐变量模型。生成过程里 $x_T,x_{T-1},\dots,x_1$ 都是隐变量。我们希望最大化数据似然：

{{< rawhtml >}}
$$
\log p_{\theta}(x_0)
$$
{{< /rawhtml >}}

直接最大化很难，于是引入前向过程 $q(x_{1:T}|x_0)$ 作为变分后验，得到负 ELBO：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{vlb}}
=
\mathbb{E}_{q(x_{1:T}|x_0)}
\left[
-\log
\frac{
p_{\theta}(x_{0:T})
}{
q(x_{1:T}|x_0)
}
\right]
$$
{{< /rawhtml >}}

代入联合分布：

{{< rawhtml >}}
$$
\begin{aligned}
p_{\theta}(x_{0:T})
&= p(x_T)\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_t) \\
q(x_{1:T}|x_0)
&= \prod_{t=1}^{T}q(x_t|x_{t-1})
\end{aligned}
$$
{{< /rawhtml >}}

经过标准整理，负 ELBO 可以拆成：

{{< rawhtml >}}
$$
\begin{aligned}
\mathcal{L}_{\mathrm{vlb}}
&=
\mathbb{E}_{q}
\Big[
D_{\mathrm{KL}}
\left(
q(x_T|x_0)
\Vert
p(x_T)
\right)
\\
&\quad
+
\sum_{t=2}^{T}
D_{\mathrm{KL}}
\left(
q(x_{t-1}|x_t,x_0)
\Vert
p_{\theta}(x_{t-1}|x_t)
\right)
\\
&\quad
-
\log p_{\theta}(x_0|x_1)
\Big]
\end{aligned}
$$
{{< /rawhtml >}}

这里有三个部分：

- 第一项让最终前向噪声接近标准高斯；通常是常数或近似常数。
- 中间的 KL 项让模型反向一步接近真实后验。
- 最后一项是从 $x_1$ 重建 $x_0$。

核心训练项是中间的 KL：

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

因为两边都是高斯，如果方差固定，这个 KL 对模型参数的依赖主要来自均值误差：

{{< rawhtml >}}
$$
\left\|
\tilde{\mu}_t(x_t,x_0)
-
\mu_{\theta}(x_t,t)
\right\|^2
$$
{{< /rawhtml >}}

于是问题变成：网络应该预测后验均值，还是预测 $x_0$，还是预测噪声 $\epsilon$？

## 六、为什么 DDPM 最常训练预测噪声

由前向重参数公式：

{{< rawhtml >}}
$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$
{{< /rawhtml >}}

可以反解 $x_0$：

{{< rawhtml >}}
$$
x_0
=
\frac{
x_t - \sqrt{1-\bar{\alpha}_t}\epsilon
}{
\sqrt{\bar{\alpha}_t}
}
$$
{{< /rawhtml >}}

如果网络预测噪声：

{{< rawhtml >}}
$$
\epsilon_{\theta}(x_t,t)
\approx
\epsilon
$$
{{< /rawhtml >}}

那么可以构造对 $x_0$ 的预测：

{{< rawhtml >}}
$$
\hat{x}_0(x_t,t)
=
\frac{
x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_{\theta}(x_t,t)
}{
\sqrt{\bar{\alpha}_t}
}
$$
{{< /rawhtml >}}

再把这个 $\hat{x}_0$ 代入后验均值，就得到模型反向均值。整理后，常见写法是：

{{< rawhtml >}}
$$
\mu_{\theta}(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_{\theta}(x_t,t)
\right)
$$
{{< /rawhtml >}}

DDPM 原文发现，直接用一个简化目标训练噪声预测通常效果很好：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon
-
\epsilon_{\theta}
\left(
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
t
\right)
\right\|^2
\right]
$$
{{< /rawhtml >}}

这就是大家常说的“Diffusion 训练就是随机加噪，然后让模型预测加进去的噪声”。

它背后的严谨来源是 ELBO 的高斯 KL 项；简化目标省略了不同时间步上的权重，保留了最稳定、最容易优化的噪声回归形式。

## 七、DDPM 采样公式

训练好 $\epsilon_{\theta}$ 后，采样从高斯噪声开始：

{{< rawhtml >}}
$$
x_T \sim \mathcal{N}(0,I)
$$
{{< /rawhtml >}}

然后从 $t=T$ 到 $1$ 迭代：

{{< rawhtml >}}
$$
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_{\theta}(x_t,t)
\right)
+
\sigma_t z
$$
{{< /rawhtml >}}

其中：

{{< rawhtml >}}
$$
z \sim \mathcal{N}(0,I)
$$
{{< /rawhtml >}}

当 $t=1$ 时通常不再加随机噪声；$\sigma_t$ 可以取 $\sqrt{\beta_t}$ 或者 $\sqrt{\tilde{\beta}_t}$，不同实现略有差异。

这条公式可以拆成两件事：

- $\epsilon_{\theta}(x_t,t)$ 估计当前样本里有多少噪声。
- 均值项把 $x_t$ 往更干净的方向移动一步。

所以 DDPM 是一个随机、多步、逐渐去噪的生成过程。

## 八、Diffusion 和 Score 的关系

噪声预测不仅可以解释为估计 $\epsilon$，还可以解释为估计 score。

score 是对数密度关于样本的梯度：

{{< rawhtml >}}
$$
\nabla_x \log p_t(x)
$$
{{< /rawhtml >}}

对于条件高斯：

{{< rawhtml >}}
$$
q(x_t|x_0)
=
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
$$
{{< /rawhtml >}}

有：

{{< rawhtml >}}
$$
\nabla_{x_t}\log q(x_t|x_0)
=
-
\frac{
x_t-\sqrt{\bar{\alpha}_t}x_0
}{
1-\bar{\alpha}_t
}
$$
{{< /rawhtml >}}

又因为：

{{< rawhtml >}}
$$
x_t-\sqrt{\bar{\alpha}_t}x_0
=
\sqrt{1-\bar{\alpha}_t}\epsilon
$$
{{< /rawhtml >}}

所以：

{{< rawhtml >}}
$$
\nabla_{x_t}\log q(x_t|x_0)
=
-
\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$
{{< /rawhtml >}}

如果模型学到了 $\epsilon_{\theta}$，就等价于学到了一个 score 估计：

{{< rawhtml >}}
$$
s_{\theta}(x_t,t)
\approx
-
\frac{
\epsilon_{\theta}(x_t,t)
}{
\sqrt{1-\bar{\alpha}_t}
}
$$
{{< /rawhtml >}}

这也是 DDPM、score-based model 和 SDE diffusion 之间可以互相转化的原因。

## 九、从离散扩散到连续时间

把时间步从离散的 $1,\dots,T$ 改成连续的 $t \in [0,1]$，可以把加噪过程写成随机微分方程：

{{< rawhtml >}}
$$
dx
=
f(x,t)dt
+
g(t)dw
$$
{{< /rawhtml >}}

这里 $w$ 是布朗运动。反向生成过程也可以写成反向 SDE，其漂移项会用到 score：

{{< rawhtml >}}
$$
dx
=
\left[
f(x,t)
-
g(t)^2\nabla_x \log p_t(x)
\right]dt
+
g(t)d\bar{w}
$$
{{< /rawhtml >}}

也存在一个对应的 deterministic probability flow ODE：

{{< rawhtml >}}
$$
\frac{dx}{dt}
=
f(x,t)
-
\frac{1}{2}g(t)^2\nabla_x \log p_t(x)
$$
{{< /rawhtml >}}

这个 ODE 和 SDE 可以拥有相同的边缘分布 $p_t(x)$。这点非常重要：它告诉我们，生成不一定非要走随机去噪链，也可以学一个确定性速度场，让样本沿着 ODE 从噪声流到数据。

这就自然过渡到了 Flow Matching。

## 十、Flow Matching 的基本设定

Flow Matching 从连续归一化流 `CNF` 出发。设 $x_t$ 满足一个 ODE：

{{< rawhtml >}}
$$
\frac{dx_t}{dt}
=
v_{\theta}(x_t,t)
$$
{{< /rawhtml >}}

如果初始样本来自简单分布：

{{< rawhtml >}}
$$
x_0 \sim p_0
$$
{{< /rawhtml >}}

沿着这个 ODE 积分到 $t=1$，得到：

{{< rawhtml >}}
$$
x_1 = \phi_1(x_0)
$$
{{< /rawhtml >}}

我们希望 $x_1$ 的分布接近数据分布 $p_{\mathrm{data}}$。

CNF 的密度演化满足连续性方程：

{{< rawhtml >}}
$$
\frac{\partial p_t(x)}{\partial t}
+
\nabla \cdot
\left(
p_t(x)v_t(x)
\right)
=
0
$$
{{< /rawhtml >}}

这句话的直觉是：概率质量不会凭空产生或消失，只会被速度场 $v_t$ 搬运。

如果我们知道真实速度场 $v_t(x)$，就可以训练网络去拟合它：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{FM}}(\theta)
=
\mathbb{E}_{t \sim U(0,1), x \sim p_t}
\left[
\left\|
v_{\theta}(x,t)-v_t(x)
\right\|^2
\right]
$$
{{< /rawhtml >}}

问题是：真实的 $p_t$ 和 $v_t$ 通常都不知道。

Flow Matching 的关键技巧是构造条件路径。

## 十一、Conditional Flow Matching：先固定一对端点

采样一个噪声点 $x_0 \sim p_0$ 和一个数据点 $x_1 \sim p_{\mathrm{data}}$。先不要考虑整个边缘分布，只考虑这两个点之间的一条条件路径：

{{< rawhtml >}}
$$
x_t = \psi_t(x_0,x_1)
$$
{{< /rawhtml >}}

最简单、最常见的是线性插值：

{{< rawhtml >}}
$$
x_t
=
(1-t)x_0 + tx_1
$$
{{< /rawhtml >}}

对时间求导：

{{< rawhtml >}}
$$
u_t(x_t|x_0,x_1)
=
\frac{dx_t}{dt}
=
x_1-x_0
$$
{{< /rawhtml >}}

这就是条件速度。它非常容易得到，因为端点是已知的。

Conditional Flow Matching 的训练目标就是：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t,x_0,x_1}
\left[
\left\|
v_{\theta}(x_t,t)
-
(x_1-x_0)
\right\|^2
\right]
$$
{{< /rawhtml >}}

训练完成后，生成时只需要从 $x_0 \sim p_0$ 采样，然后解 ODE：

{{< rawhtml >}}
$$
\frac{dx_t}{dt}
=
v_{\theta}(x_t,t),
\quad
t:0 \rightarrow 1
$$
{{< /rawhtml >}}

这就是 Flow Matching 最直接的版本，也常被称为 `Rectified Flow` 风格的训练目标。

## 十二、为什么条件目标能学到边缘速度场

上面看起来有一个疑问：我们训练的是条件速度 $u_t(x|x_0,x_1)$，但采样时只有 $x_t$，不知道它来自哪一对端点。为什么这样可行？

设条件路径诱导的边缘分布为：

{{< rawhtml >}}
$$
p_t(x)
=
\int p_t(x|z)p(z)dz
$$
{{< /rawhtml >}}

这里 $z$ 可以表示条件变量，例如端点 $x_1$，或者端点对 $(x_0,x_1)$。每个条件路径有自己的速度 $u_t(x|z)$。对应的边缘速度场可以写成条件期望：

{{< rawhtml >}}
$$
v_t(x)
=
\mathbb{E}
\left[
u_t(x|z)
\mid
x_t=x
\right]
$$
{{< /rawhtml >}}

如果用平方误差训练：

{{< rawhtml >}}
$$
\mathbb{E}
\left[
\left\|
v_{\theta}(x_t,t)-u_t(x_t|z)
\right\|^2
\right]
$$
{{< /rawhtml >}}

那么在函数空间里的最优解就是条件均值：

{{< rawhtml >}}
$$
v_{\theta}^{*}(x,t)
=
\mathbb{E}
\left[
u_t(x_t|z)
\mid
x_t=x
\right]
=
v_t(x)
$$
{{< /rawhtml >}}

因此，虽然训练样本里用的是容易计算的条件速度，模型最终学到的是边缘分布所需要的平均速度场。

这就是 Flow Matching 训练可以“simulation-free”的原因：训练时不需要真的解 ODE 或模拟反向扩散链，只需要采样端点、采样时间、构造中间点、回归速度。

## 十三、Flow Matching 的高斯路径

线性插值不是唯一选择。Flow Matching 原文讨论了更一般的高斯条件路径：

{{< rawhtml >}}
$$
p_t(x|x_1)
=
\mathcal{N}
\left(
x;
\mu_t(x_1),
\sigma_t^2 I
\right)
$$
{{< /rawhtml >}}

可以用重参数写成：

{{< rawhtml >}}
$$
x_t
=
\mu_t(x_1)
+
\sigma_t \epsilon,
\quad
\epsilon \sim \mathcal{N}(0,I)
$$
{{< /rawhtml >}}

对时间求导：

{{< rawhtml >}}
$$
\frac{dx_t}{dt}
=
\dot{\mu}_t(x_1)
+
\dot{\sigma}_t \epsilon
$$
{{< /rawhtml >}}

由于：

{{< rawhtml >}}
$$
\epsilon
=
\frac{x_t-\mu_t(x_1)}{\sigma_t}
$$
{{< /rawhtml >}}

条件速度可以写成只依赖 $x_t,t,x_1$ 的形式：

{{< rawhtml >}}
$$
u_t(x_t|x_1)
=
\dot{\mu}_t(x_1)
+
\frac{\dot{\sigma}_t}{\sigma_t}
\left(
x_t-\mu_t(x_1)
\right)
$$
{{< /rawhtml >}}

选择不同的 $\mu_t$ 和 $\sigma_t$，就得到不同的概率路径。

例如一种接近最优传输直线插值的设定是：

{{< rawhtml >}}
$$
\mu_t(x_1) = t x_1,
\quad
\sigma_t = 1 - (1-\sigma_{\min})t
$$
{{< /rawhtml >}}

当 $t=0$ 时，$x_t$ 接近高斯噪声；当 $t=1$ 时，$x_t$ 集中到数据点 $x_1$ 附近。$\sigma_{\min}$ 是一个很小的正数，用来避免末端方差完全为零带来的数值问题。

## 十四、Rectified Flow：最直观的 Flow Matching

很多现代图像生成模型里会使用类似 Rectified Flow 的训练形式。它可以写得非常简单：

采样：

{{< rawhtml >}}
$$
x_0 \sim \mathcal{N}(0,I),
\quad
x_1 \sim p_{\mathrm{data}},
\quad
t \sim U(0,1)
$$
{{< /rawhtml >}}

构造中间点：

{{< rawhtml >}}
$$
x_t = (1-t)x_0 + tx_1
$$
{{< /rawhtml >}}

目标速度：

{{< rawhtml >}}
$$
u_t = x_1 - x_0
$$
{{< /rawhtml >}}

训练损失：

{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{RF}}
=
\mathbb{E}
\left[
\left\|
v_{\theta}(x_t,t)
-
(x_1-x_0)
\right\|^2
\right]
$$
{{< /rawhtml >}}

生成时：

{{< rawhtml >}}
$$
x_1
\approx
x_0
+
\int_0^1
v_{\theta}(x_t,t)dt
$$
{{< /rawhtml >}}

如果用 Euler 方法离散化，令步长 $\Delta t = 1/N$：

{{< rawhtml >}}
$$
x_{t+\Delta t}
=
x_t
+
\Delta t
v_{\theta}(x_t,t)
$$
{{< /rawhtml >}}

这和 DDPM 的“每一步预测噪声并去噪”很不同。Flow Matching 是“每一步预测当前应该朝哪个方向移动”。

## 十五、Diffusion 与 Flow Matching 的共同点

二者表面上训练目标不同，但共同点很深。

第一，它们都构造了从简单分布到数据分布的时间路径：

{{< rawhtml >}}
$$
p_0
\rightarrow
p_t
\rightarrow
p_1
$$
{{< /rawhtml >}}

Diffusion 通常从数据到噪声定义前向过程，生成时反过来走；Flow Matching 通常直接定义从噪声到数据的路径。

第二，它们都在学习局部信息：

- Diffusion 学的是 score 或噪声，也就是“如何从当前噪声状态往干净方向修正”。
- Flow Matching 学的是 velocity，也就是“当前点沿路径应该往哪里走”。

第三，二者都可以放到连续时间框架里。Diffusion 的 probability flow ODE 可以看成一个速度场：

{{< rawhtml >}}
$$
v_t(x)
=
f(x,t)
-
\frac{1}{2}g(t)^2\nabla_x \log p_t(x)
$$
{{< /rawhtml >}}

Flow Matching 直接训练一个 $v_{\theta}(x,t)$ 去拟合这种速度场或其他自定义路径的速度场。

因此可以说：Diffusion 是通过 score 学路径；Flow Matching 是直接通过速度学路径。

## 十六、二者训练和采样的关键差异

### 16.1 训练目标

DDPM 常用噪声预测：

{{< rawhtml >}}
$$
\mathbb{E}
\left[
\left\|
\epsilon-\epsilon_{\theta}(x_t,t)
\right\|^2
\right]
$$
{{< /rawhtml >}}

Flow Matching 常用速度预测：

{{< rawhtml >}}
$$
\mathbb{E}
\left[
\left\|
u_t-v_{\theta}(x_t,t)
\right\|^2
\right]
$$
{{< /rawhtml >}}

DDPM 的监督信号来自“加进去的噪声是多少”。Flow Matching 的监督信号来自“这条路径在当前时刻的导数是多少”。

### 16.2 采样过程

DDPM 原始采样是反向马尔可夫链：

{{< rawhtml >}}
$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
$$
{{< /rawhtml >}}

Flow Matching 采样是解 ODE：

{{< rawhtml >}}
$$
\frac{dx_t}{dt}
=
v_{\theta}(x_t,t)
$$
{{< /rawhtml >}}

DDPM 采样常带随机性，Flow Matching 的 ODE 采样通常是确定性的。当然，实际模型也可以引入随机采样或不同数值求解器。

### 16.3 路径形状

DDPM 的路径由噪声日程决定：

{{< rawhtml >}}
$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$
{{< /rawhtml >}}

Flow Matching 可以自由选择路径，例如直线插值：

{{< rawhtml >}}
$$
x_t
=
(1-t)x_0 + tx_1
$$
{{< /rawhtml >}}

或者更一般的高斯概率路径：

{{< rawhtml >}}
$$
x_t
=
\mu_t(x_1)+\sigma_t\epsilon
$$
{{< /rawhtml >}}

路径越直，ODE 求解通常越容易，采样步数也可能越少。这也是 Flow Matching 和 Rectified Flow 受关注的重要原因。

## 十七、从实现角度看训练循环

DDPM 的训练循环可以概括为：

1. 从数据集中采样 $x_0$。
2. 随机采样时间步 $t$。
3. 采样噪声 $\epsilon$。
4. 构造 $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$。
5. 最小化 $\|\epsilon-\epsilon_{\theta}(x_t,t)\|^2$。

Flow Matching 的训练循环可以概括为：

1. 从噪声分布采样 $x_0$。
2. 从数据集采样 $x_1$。
3. 随机采样时间 $t$。
4. 构造 $x_t=(1-t)x_0+tx_1$。
5. 令目标速度 $u=x_1-x_0$。
6. 最小化 $\|u-v_{\theta}(x_t,t)\|^2$。

它们都很像普通监督学习：构造一个中间状态，然后回归一个目标。但回归目标不同。

## 十八、一个统一记忆法

可以用下面这句话记住二者：

Diffusion 问的是：

> 当前这个样本里混进了多少噪声？把这些噪声去掉，就能往数据方向走。

Flow Matching 问的是：

> 当前这个样本处在从噪声到数据的路上，它此刻应该以什么速度往前走？

如果把生成过程想成开车：

- Diffusion 像每一步都估计“偏离干净图像的噪声残差”，然后修正一点。
- Flow Matching 像直接学习“当前位置的速度向量”，然后沿 ODE 积分。

数学上，二者都在定义并学习一条连接 $p_0$ 和 $p_{\mathrm{data}}$ 的路径。工程上，Flow Matching 的训练目标更直接，采样可用通用 ODE solver；Diffusion 的理论和实践积累更久，score、guidance、noise schedule、latent diffusion 等变体非常成熟。

## 十九、常见参数化：epsilon、x0、v

Diffusion 模型里除了预测噪声 $\epsilon$，也常见预测 $x_0$ 或预测 $v$。

由：

{{< rawhtml >}}
$$
x_t
=
\alpha_t x_0 + \sigma_t \epsilon
$$
{{< /rawhtml >}}

这里为了简化记号，把连续时间下的信号系数写成 $\alpha_t$，噪声系数写成 $\sigma_t$。常见的 velocity 参数化定义为：

{{< rawhtml >}}
$$
v
=
\alpha_t \epsilon - \sigma_t x_0
$$
{{< /rawhtml >}}

这和 Flow Matching 的 velocity 不是完全同一个概念，但都体现了“预测一个更稳定的方向量”。在一些噪声日程下，$v$ 参数化比直接预测 $\epsilon$ 或 $x_0$ 更平衡，因为它避免训练目标在高噪声或低噪声区域过度偏向某一端。

从 $x_t$ 和 $v$ 可以反推出 $x_0$ 与 $\epsilon$：

{{< rawhtml >}}
$$
\begin{aligned}
x_0
&=
\alpha_t x_t - \sigma_t v \\
\epsilon
&=
\sigma_t x_t + \alpha_t v
\end{aligned}
$$
{{< /rawhtml >}}

所以实际模型“预测什么”很多时候只是参数化选择；真正关键的是它对应怎样的路径、损失权重和采样公式。

## 二十、总结

DDPM 的主线是：

{{< rawhtml >}}
$$
\begin{aligned}
q(x_t|x_0)
&=
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
\\
q(x_{t-1}|x_t,x_0)
&=
\mathcal{N}
\left(
\tilde{\mu}_t,
\tilde{\beta}_t I
\right)
\\
\mathcal{L}_{\mathrm{simple}}
&=
\mathbb{E}
\left[
\|\epsilon-\epsilon_{\theta}(x_t,t)\|^2
\right]
\end{aligned}
$$
{{< /rawhtml >}}

也就是：定义可解析的前向加噪过程，用 ELBO 推出反向后验拟合，再简化成噪声预测。

Flow Matching 的主线是：

{{< rawhtml >}}
$$
\begin{aligned}
\frac{dx_t}{dt}
&=
v_{\theta}(x_t,t)
\\
x_t
&=
(1-t)x_0 + tx_1
\\
\mathcal{L}_{\mathrm{CFM}}
&=
\mathbb{E}
\left[
\|v_{\theta}(x_t,t)-(x_1-x_0)\|^2
\right]
\end{aligned}
$$
{{< /rawhtml >}}

也就是：先设计一条从噪声到数据的概率路径，再让网络回归这条路径的速度场。

如果只记一句话：

> Diffusion 学会“去噪”，Flow Matching 学会“搬运”。

前者从随机反向过程走向数据，后者用确定性速度场把概率质量推向数据。它们不是完全割裂的两类模型，而是同一个“连接噪声分布与数据分布”的连续生成思想下的两种训练方式。
