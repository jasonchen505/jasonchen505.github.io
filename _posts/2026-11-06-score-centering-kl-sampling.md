---
title: "SC 加 KL：正则用哪个 token，会改变什么？"
date: "2026-11-06 00:00:00 -0800"
description: "KL 正则用实际 token，还是复用 SC 的辅助样本？比较两种无偏估计如何共享随机性，以及为什么不存在统一的方差排序。"
toc: [{"title": "1. 先固定要估计的更新", "anchor": "target-update"}, {"title": "2. 同一个 KL，怎样与 SC 共用随机性", "anchor": "shared-randomness"}, {"title": "3. 为什么不存在统一的方差排序", "anchor": "variance-ordering"}, {"title": "4. 把区别落实到代码", "anchor": "implementation"}]
---
Score Centering（SC）可以用每个 prefix 下的一组辅助 token 估计 score 中心。再加 sampler 与 trainer 之间的 KL 正则时，可以用实际生成的 token，也可以复用这组辅助样本。相同的 KL 方向和系数，并不能唯一确定随机更新。

**固定参数和 KL 系数，若辅助 token 来自同一行为分布，且与实际动作及反馈条件独立，两种 k1 估计得到相同的期望梯度，但没有普遍更低方差的选择。** KL 与 SC 共用了哪些随机项，会改变噪声如何相加或抵消。

## 1. 先固定要估计的更新 {#target-update}

固定一个 prefix、sampler 版本和当前 trainer 参数。记行为分布为 $q$，trainer 为 $p_\theta$，给定动作后的反馈分布也固定。假设有限词表上两者概率为正，$p_\theta$ 可微，所需二阶矩有限。以下比较优化器处理前的条件上升梯度，不加入 IS、PPO clipping 或随机长度分母。

实际动作 $a\sim q$ 带来反馈 $A$；$A$ 可以依赖动作及后续回报，在反传时冻结。这里不预设 $A$ 在当前 prefix 下的条件均值为零。定义参数 score 及其中心：

$$
s_a=\nabla_\theta\log p_\theta(a),\qquad
\mu=\mathbb E_q[s_a].
$$

[SC](https://arxiv.org/html/2609.20807v1#S4) 的完整中心更新为 $A(s_a-\mu)$。采用 [NormalUhr §3.2 的辅助 token 估计](https://huggingface.co/blog/NormalUhr/sfc)时，固定 $M\ge1$，在同一 prefix 下独立抽取 $v_1,\ldots,v_M\sim q$，且整组与 $(a,A)$ 条件独立，令

$$
\widehat\mu=\frac1M\sum_{j=1}^M s_{v_j},\qquad
\widehat G_{\rm SC}=A(s_a-\widehat\mu).
$$

于是 $\mathbb E[\widehat G_{\rm SC}]=\operatorname{Cov}_q(A,s)$，期望包含动作、反馈和辅助采样的随机性。辅助 token 无需续写或奖励。这里的无偏针对 SC 的协方差目标，并非原来的 $\mathbb E_q[As]$。

现在给下降 loss 加上 $\beta D_{\rm KL}(q\Vert p_\theta)$，取固定 $\beta\ge0$。因为 $q$ 冻结，KL 对参数的梯度为 $-\mu$，完整的期望上升方向就是

$$
G=\operatorname{Cov}_q(A,s)+\beta\mu.
$$

下面保持这个目标，只改变 KL 的取样位置。

## 2. 同一个 KL，怎样与 SC 共用随机性 {#shared-randomness}

k1 用 $\log q(x)-\log p_\theta(x)$ 估计 KL。若取实际生成的 $x=a$，KL 提供上升项 $\beta s_a$，与 SC 合并为

$$
\widehat G_{\rm main}=(A+\beta)s_a-A\widehat\mu.
$$

若复用 SC 的同一组辅助 token，KL 提供 $\beta\widehat\mu$，合并为

$$
\widehat G_{\rm aux}=A s_a+(\beta-A)\widehat\mu.
$$

两者期望都是 $G$，但同一组样本上的差为

$$
\widehat G_{\rm main}-\widehat G_{\rm aux}
=\beta(s_a-\widehat\mu).
$$

右侧均值为零，逐样本通常不为零。实际 token 方案与 SC 的直接项共用 $s_a$；辅助方案与中心项共用 $\widehat\mu$。单独比较 KL 的估计方差，无法确定合并后的梯度噪声。

## 3. 为什么不存在统一的方差排序 {#variance-ordering}

取这个 prefix 下反馈恒为 $A=c$ 的情形。SC 的期望反馈项为零，两种更新的均值均为 $\beta\mu$，可以直接比较噪声。

记 $\Sigma_s=\operatorname{Cov}_q(s)$，则 $\operatorname{Cov}(\widehat\mu)=\Sigma_s/M$。由于 $s_a$ 与 $\widehat\mu$ 独立，线性组合的交叉协方差为零。一般的常数反馈 $c$ 下，有

$$
\begin{aligned}
\operatorname{Cov}(\widehat G_{\rm main})
&=\left[(c+\beta)^2+\frac{c^2}{M}\right]\Sigma_s,\\
\operatorname{Cov}(\widehat G_{\rm aux})
&=\left[c^2+\frac{(\beta-c)^2}{M}\right]\Sigma_s.
\end{aligned}
$$

取 $\beta>0$，表中系数乘以 $\beta^2\Sigma_s$，就是对应的参数梯度协方差：

| 固定反馈 | 实际 token 方案 | 辅助 token 方案 |
| --- | --- | --- |
| $c=\beta$ | $4+1/M$ | $1$ |
| $c=-\beta$ | $1/M$ | $1+4/M$ |

第一行的辅助方案中，$(\beta-A)\widehat\mu$ 恰好消失；第二行的实际 token 方案中，$(A+\beta)s_a$ 消失，只剩辅助均值。当 $\Sigma_s\ne0$ 时，协方差较小的方案随反馈改变。

这是按上述条件直接推导的反例，否定了统一的方差排序。也不能据此认为，按本次反馈切换后仍然无偏：若 $\lambda(A)\in\{0,1\}$ 决定是否选择实际 token 方案，相对辅助方案就多出 $\beta\lambda(A)(s_a-\widehat\mu)$，其期望为

$$
\beta\operatorname{Cov}_q(\lambda(A),s),
$$

这项不保证为零。固定规则各自无偏，依赖反馈的选择却可能改变期望更新。

## 4. 把区别落实到代码 {#implementation}

[Tinker 的 StableReinforce 文档](https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/#samplerlearner-kl)将默认 k1 正则放在实际生成的 token 上。[agentcore-rl-toolkit 的一份公开实现](https://github.com/Linbo-Liu/agentcore-rl-toolkit/blob/68885eda295f230cfe0976fcf9750e5534527cef/src/agentcore_rl_toolkit/backends/verl/score_centering.py#L239-L245)则使用辅助 token 的平均 log-prob；其年龄系数在固定 sampler 版本和当前 trainer step 后，作为 $\beta$ 分析。这些材料能核对 KL 的取样位置，尚不足以识别 StableReinforce 的完整 loss。

在下降 loss 中，两种正则的参数相关部分分别是 $-\beta\log p_\theta(a)$ 和 $-\beta M^{-1}\sum_j\log p_\theta(v_j)$。冻结 $A$、$\beta$ 和采样记录，当前模型的 log-prob 保留反传。比较还需统一有效位置与聚合尺度；依赖动作或整条回答的 mask、权重需另算。

上述等价针对固定参数点的新鲜联合抽样。模型若已反复适应同一组辅助 token，给定当前参数后，辅助 token 是否仍服从 $q$，以及原有的独立性是否仍成立，都需要重查；多个 token 的交叉协方差和 Adam 的非线性处理，也会影响参数步。比较 SC＋KL 配方时，需要说明 KL 与中心共用了哪些样本；相同的期望梯度，尚不足以确定训练轨迹或收益。正则的取样位置，因而也是随机更新规则的一部分。
