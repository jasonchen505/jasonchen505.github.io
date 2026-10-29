---
title: "Score Centering 之后，梯度还在估计什么？"
date: "2026-10-29 00:01:00 -0700"
description: "沿完整梯度链路检查奖励归一化、长度聚合与 baseline，理解哪些操作保留条件零均值，哪些改变最终更新。"
toc: [{"title": "1. 条件零均值保留了什么", "anchor": "conditional-centering"}, {"title": "2. 奖励归一化如何重加权任务", "anchor": "reward-normalization"}, {"title": "3. 长度聚合与随机分母", "anchor": "length-aggregation"}, {"title": "4. baseline 的数据来源也属于估计器", "anchor": "baseline-data"}, {"title": "5. 近似 baseline 与近似 SC 的联合误差", "anchor": "joint-approximation-error"}]
---
[上一篇](https://jasonchen505.github.io/blog/score-centering-and-importance-sampling/)讨论了 Score Centering（SC）与 Importance Sampling。实际训练还会减 baseline、除以奖励标准差、按长度聚合。这些操作会怎样改变 SC 留下的更新？

我的判断是：**SC 的条件零均值不足以确定最终的期望梯度，还要检查权重与 baseline。** 给定 prefix 后固定的系数保留零均值，却可能改变任务权重；依赖本次动作、续写或同批拟合的系数，则需要重新核对期望。

## 1. 条件零均值保留了什么 {#conditional-centering}

先固定 trainer 参数和行为策略，分析新样本的一次更新。用 $q$ 表示真实采样策略，$p_\theta$ 表示 trainer，$R$ 是沿 $q$ 续写得到的完整回报。假设轨迹长度有界、所需期望存在，且 $p_\theta$ 在 $q$ 的支持上为正。反馈、baseline 和缩放系数在反传时停止梯度；暂不加入 IS、clipping 或额外正则。

在 prefix $h$ 处，定义参数 score 及其精确中心：

$$
s=\nabla_\theta\log p_\theta(a\mid h),\qquad
\mu=\mathbb E_q[s\mid h],\qquad
\widetilde s=s-\mu.
$$

[SC](https://arxiv.org/html/2609.20807v1#S4) 保证 $\mathbb E_q[\widetilde s\mid h]=0$，因此减去固定的 $b(h)$ 不改变条件期望更新：

$$
\mathbb E_q[(R-b(h))\widetilde s\mid h]
=\operatorname{Cov}_q(R,s\mid h).
$$

若再乘一个随机标量 $W$，则

$$
\mathbb E_q[W\widetilde s\mid h]
=\operatorname{Cov}_q(W,s\mid h).
$$

给定 $h$ 后为常数的 $W$，是保留条件零均值的充分条件。随机 $W$ 是否保留它，由这项协方差决定；仅有统计依赖，还不能断言结果非零。

## 2. 奖励归一化如何重加权任务 {#reward-normalization}

对 prompt $x$，先假设总体奖励均值 $m_x$ 和标准差 $\sigma_x>0$ 已知。令 $T=\sum_{t=1}^{L}\widetilde s_t$，求和包含完整回答的所有生成动作及终止动作，记 $g_{\mathrm{SC}}(x)=\mathbb E_q[RT\mid x]$。这里的长度由生成过程的停止规则决定，例如 EOS 或预设长度上限；是否执行第 $t$ 次采样，在该动作产生前已经可知。逐 prefix 的零均值因而给出 $\mathbb E_q[T\mid x]=0$，于是

$$
g_{\mathrm{norm}}(x)
=\mathbb E_q\!\left[\frac{R-m_x}{\sigma_x}T\,\middle|\,x\right]
=\frac{g_{\mathrm{SC}}(x)}{\sigma_x}.
$$

同一 prompt 的期望更新只被正数缩放；不同 prompt 的 $\sigma_x$ 不同，混合方向就可能改变。这种任务重加权可以与精确 SC 并存。

Jiayan Fu 的 [GRPO 分析](https://ichthyosaur.github.io/blog/2026/grpo-variance-stabilization/)将二元奖励、on-policy 总体极限下的归一化解释为优化变换后的成功率。其方差稳定化结论针对该成功率估计的渐近方差，不证明策略梯度方差最小，也不能把这个目标解释直接搬到 $p\ne q$ 的 SC 上。实际 group 均值和标准差还依赖本批样本，不能当作上述总体常数。

## 3. 长度聚合与随机分母 {#length-aggregation}

长度聚合还要看分母由谁共享。沿用 Jiayan 的 [token-mean 分析](https://ichthyosaur.github.io/blog/2026/token-mean-policy-gradient/)，暂用外部固定的 baseline 和缩放，从固定分布独立同分布地抽取 prompt 与完整回答。每条回答的贡献只依赖自身样本：用 $A_{i,t}$ 表示 advantage，记 $S_i=\sum_t A_{i,t}\widetilde s_{i,t}$，有效长度为 $L_i\ge1$。两种聚合为

$$
\widehat g_{\mathrm{TM}}=\frac{\sum_i S_i}{\sum_i L_i},
\qquad
\widehat g_{\mathrm{SM}}=\frac1N\sum_i\frac{S_i}{L_i}.
$$

前者与同一 batch 的 $N^{-1}\sum_i S_i$ 只差一个正尺度；后者按各自长度重加权。若 $\mathbb E\|S\|<\infty$、$0<\mathbb E L<\infty$，大数定律给出几乎必然收敛：

$$
\widehat g_{\mathrm{TM}}\longrightarrow
\frac{\mathbb E S}{\mathbb E L},
\qquad
\widehat g_{\mathrm{SM}}\longrightarrow
\mathbb E\!\left[\frac SL\right].
$$

两个极限未必平行。公共分母仍可能带来有限 batch 的比值偏差；同一 batch 只差尺度，不等于期望也只差尺度。因此，有限 batch 下的公共 token 分母也可能使固定 baseline 留下非零期望，这一风险并不限于逐回答乘 $1/L_i$。若逐 prompt 做 token-mean 后再平均，组内大样本极限又会按 $1/\mathbb E[L\mid x]$ 重加权任务。这里保留的是所选分子 $\mathbb E S$ 的方向，不能自动解释成当前 trainer 的回报梯度，或“回报除以期望长度”的梯度。

逐回答的最终长度 $L$ 依赖当前动作和续写。把前面的 $W$ 换成 $1/L$，就有

$$
\mathbb E_q\!\left[\frac{\widetilde s_t}{L}\,\middle|\,h_t\right]
=\operatorname{Cov}_q\!\left(\frac1L,s_t\,\middle|\,h_t\right),
$$

右边未必为零，固定 baseline 因而可能不再抵消。这与 GRPO 的奖励平移不变性可以并存：baseline 若随奖励一起平移，advantage 仍可逐样本不变。

[SC 原论文附录 B.1](https://arxiv.org/html/2609.20807v1#A2.SS1)使用整批 token 总数作公共分母。核对实现时，要区分“所有有效 token 求和后除以总数”和“每条回答先平均、再对回答平均”；这两种 reduction 不能随意互换。

在[对应的公开实现](https://github.com/martin-marek/score-centering/blob/64842967077ea105470ca73fc5d714366d3f7bdb/losses.py#L138-L170)中，SC 校正先并入每个 token 的 score 项，再与主项一起应用 completion mask，最后共同除以整个 batch 的有效 token 总数。

## 4. baseline 的数据来源也属于估计器 {#baseline-data}

baseline 也需要检查数据来源。[Jiayan 对 SAO 更新顺序的讨论](https://ichthyosaur.github.io/blog/2026/sao-critic-updates/)指出：网络虽只输入 prefix，参数却可能已通过拟合本批轨迹吸收了本次回报。若再用它预测这些轨迹的 $\widehat b(h)$，对完整 Monte Carlo 回报减 baseline 的更新，即使 SC 精确，也有

$$
\mathbb E_q[(R-\widehat b)\widetilde s\mid h]
=\operatorname{Cov}_q(R,s\mid h)
-\operatorname{Cov}_q(\widehat b,s\mid h),
$$

期望还包括拟合数据的随机性。例如，在同一 prefix 下独立同分布地抽取 $N$ 个样本，用包含自身回报的批均值 $\bar R$ 作 baseline，就有 $\mathbb E[(R_i-\bar R)\widetilde s_i]=(1-1/N)\operatorname{Cov}(R,s)$。期望信号缩小，$N=1$ 时归零；一般 critic 的相关性项不必只是缩放。

`stop_gradient` 或 `detach()` 不消除统计依赖。[GAE 原论文 §6.1](https://arxiv.org/html/1506.02438v6#S6.SS1)也讨论了过拟合使 Bellman residual 归零的风险。不过，SAO 博客未核验实际更新顺序，[SAO 论文 §4.3](https://arxiv.org/html/2607.07508v1#S4.SS3)还报告了两次 critic 更新优于一次的对照；这项风险不能作为 SAO 实际有问题的证据。

## 5. 近似 baseline 与近似 SC 的联合误差 {#joint-approximation-error}

回到未额外加权的更新，固定 baseline 与 SC 仍可互补。令 $b(h),c(h)$ 在本次抽样前固定，分别近似奖励均值 $m=\mathbb E_q[R\mid h]$ 与 score 中心 $\mu$。展开得到

$$
\mathbb E_q[(R-b)(s-c)\mid h]
=\operatorname{Cov}_q(R,s\mid h)
+(m-b)(\mu-c).
$$

任一侧精确都能消掉最后一项，两侧误差则相乘。这只描述相对于协方差目标的期望偏差，不保证方差更小；$m$ 也必须对应行为策略的续写。若 advantage 还通过 bootstrap 使用后续状态的 value，固定 baseline 的抵消结论并不覆盖这些回报估计项。后续状态本身受当前动作影响，因此即使 value 参数已冻结，这部分也不等同于仅依赖当前 prefix 的 baseline。

检查完整 loss 时，我会追踪权重的归一化范围、baseline 的拟合数据，以及 SC 校正项是否与主项使用同一有效 token mask 和归一化。两项共用 mask 是实现一致性的要求；如果 mask 取决于当前 token 或完整回答，仍需把它当作随机权重 $W$ 检查。重设中心前，还要明确希望保留哪个目标：任务重加权可能正是训练者的选择。更准确的 critic 能否降低近似 SC 的需求，仍是需要结合梯度噪声、训练表现和成本验证的推测。

SC 给出的条件中心化，是理解完整更新的起点。最终梯度在估计什么，要由缩放、长度聚合、baseline 的数据来源及其与采样的关系共同确定。明确这些条件，才能区分有意的任务重加权与估计过程引入的额外偏差。
