---
title: "PPO + Score Centering：baseline 为什么仍会改变更新？"
date: "2026-10-31 00:00:00 -0700"
description: "两个分支各自零均值，为什么仍不能保证 baseline 不变性？从 PPO 裁剪分支、一个两动作反例与代码中的校正位置分析实际更新。"
toc: [{"title": "1. 固定权重下的 baseline 不变性", "anchor": "baseline-invariance"}, {"title": "2. PPO 按反馈选择分支", "anchor": "feedback-branch-selection"}, {"title": "3. 两个直接梯度都为零的反例", "anchor": "zero-direct-gradient-example"}, {"title": "4. 代码中裁剪与中心校正的位置", "anchor": "clipping-and-centering"}]
---
baseline 通常用于降低策略梯度噪声，同时保留期望更新。纯 Score Centering（SC）的条件零均值，使外部固定的 baseline 能在期望中抵消。与 PPO clipping 组合后，需要重新检查这项保证。

**即使 baseline 在抽样前固定、中心计算完全准确，PPO + SC 的期望更新仍可能随 baseline 改变。** 本文考察按正负分支分别中心化加权 score、再按本次 advantage 选分支的组合。两个分支各自零均值，不能保证选择后的向量仍零均值；其他组合方式需另行分析。

## 1. 固定权重下的 baseline 不变性 {#baseline-invariance}

固定一个 prefix、当前 trainer 参数、行为分布 $q$ 及给定动作后的回报分布。假设有限动作集上 $p_\theta(a),q(a)>0$，所需期望存在。$R$ 表示回报，参数 score 为 $s(a)=\nabla_\theta\log p_\theta(a)$。以下比较优化器处理前的条件期望上升梯度；反传时冻结反馈与加权系数。

先令权重 $w(a)$ 只依赖动作，不随回报或 baseline 改变。按照 [SC 论文附录 A.2](https://arxiv.org/html/2609.20807v1#A1.SS2)，中心化整个加权 score：

$$
F(a)=w(a)s(a)-\nu,
\qquad \nu=\mathbb E_q[w(a)s(a)].
$$

于是 $\mathbb E_q F=0$。对抽样前固定的 $b$，

$$
G(b)=\mathbb E_q[(R-b)F]=\mathbb E_q[RF].
$$

这项抵消同时用了 $F$ 零均值和 $F$ 不随 $b$ 改变。纯 SC 的 $w=1$，以及只按概率比率裁剪后再中心化的规则（权重不读取 advantage 的正负号），都满足这两个条件。所得条件更新不能直接当作 trainer 的完整轨迹回报梯度。

## 2. PPO 按反馈选择分支 {#feedback-branch-selection}

PPO 裁剪本身就不普遍保证 baseline 不变性；这里检查分支中心化能否恢复它。采用以行为分布 $q$ 为分母的 token 级比率 $r=p_\theta/q$，令 $A=R-b$、$0<\epsilon<1$。避开阈值处的不可微点，标准 [PPO clipped surrogate](https://arxiv.org/pdf/1707.06347#page=3) 的直接梯度为 $A w_\pm(a)s(a)$，其中

$$
w_+(a)=r(a)\mathbf1_{\{r(a)\le1+\epsilon\}},
\qquad
w_-(a)=r(a)\mathbf1_{\{r(a)\ge1-\epsilon\}}.
$$

正 advantage 使用 $w_+$，负 advantage 使用 $w_-$，$A=0$ 时贡献为零。暂不加入 dual clipping、额外 KL 或长度权重。

这里的 $w_\pm$ 是求导后写出的系数：用停止梯度的 $Aw_\pm$ 乘 $\log p_\theta(a)$ 可实现同一局部梯度；直接对原始 PPO ratio surrogate 求导时，$r$ 仍须参与求导。

分别中心化两个分支，令

$$
\nu_\pm=\mathbb E_q[w_\pm s],
\qquad F_\pm(a)=w_\pm(a)s(a)-\nu_\pm.
$$

这里的两个中心都在固定分支后对整个 $q$ 求期望；实际按回报进入正负分支的样本，则可能具有不同的动作分布。

两者各自满足 $\mathbb E_qF_\pm=0$。实际使用的分支由本次回报决定。记 $I_b=\mathbf1_{\{R>b\}}$，乘在 advantage 后面的向量为

$$
F_b=I_bF_++(1-I_b)F_-.
$$

将它取期望，得到

$$
\mathbb E[F_b]
=\mathbb E[I_b(F_+-F_-)].
$$

期望包含动作与续写的随机性。分支选择可能与 $F_+-F_-$ 相关，右侧因而没有恒为零的保证。若所有比率都在裁剪区间内，则 $w_+=w_-=r$；完整支持下两个中心均为零，baseline 仍会抵消。

[Fei Tian 的技术笔记 §8.6](https://fei-tian.ai/blog/what-score-function-centering-fixes/An_Error_Analysis_of_Off-Policy_LLM_RL_Updates.pdf?v=20261027#page=38)讨论了这一边界。进一步，若一段 $b$ 区间内所有可能回报的 advantage 符号均不变，$F_b$ 不随 $b$ 改变，直接得到

$$
\frac{\mathrm dG(b)}{\mathrm db}
=-\mathbb E[F_b],
\qquad G(b)=\mathbb E[(R-b)F_b].
$$

所以，baseline 依赖无需等到反馈改变正负号才出现。这里对 $b$ 求导是比较不同更新规则；参数反传时 baseline 仍然冻结。

## 3. 两个直接梯度都为零的反例 {#zero-direct-gradient-example}

取两个动作，确定回报为 $R=(1,0)$，令

$$
q=(0.2,0.8),\qquad p=(0.4,0.6),\qquad \epsilon=0.2.
$$

这个代数例子直接把 $p=\operatorname{softmax}(z)$ 的两个 logits 当作参数。对第一个 logit，两动作的 score 分量为 $s^{(1)}=(0.6,-0.4)$，比率为 $r=(2,0.75)$，因此

$$
w_+=(0,0.75),\qquad w_-=(2,0),
$$

$$
\nu_+^{(1)}=0.8\times0.75\times(-0.4)=-0.24,
\qquad
\nu_-^{(1)}=0.2\times2\times0.6=0.24.
$$

当 $0<b<1$ 时，动作 1 的 advantage 为正、动作 2 为负。前者比率高于上界，后者低于下界，实际使用的两个直接梯度都为零。减去分支中心后，分量为 $F_+^{(1)}(1)=0.24$ 和 $F_-^{(1)}(2)=-0.24$，平均更新完全来自校正项：

$$
G_1(b)
=0.2(1-b)\times0.24+0.8(-b)\times(-0.24)
=0.048+0.144b.
$$

此时选中分量的均值为 $\mathbb E[F_b^{(1)}]=0.2\times0.24+0.8\times(-0.24)=-0.144$，恰好对应第 2 节给出的斜率 $\mathrm dG_1/\mathrm db=0.144$。

$b$ 从 $0.25$ 改到 $0.5$，$G_1$ 从 $0.084$ 变为 $0.12$，其他条件和裁剪分支不变。这里改变的是更新幅度；两个更新仍偏向高回报动作。反例否定了 baseline 不变性，没有证明训练效果变差。

若奖励与 group baseline 同时加上常数，advantage 和分支选择不变，本文的更新也逐样本不变。这与固定奖励、单独调整 baseline 是不同的操作。

## 4. 代码中裁剪与中心校正的位置 {#clipping-and-centering}

对选中的分支 $\sigma$，即使 $w_\sigma(a)=0$，完整贡献仍为

$$
A\bigl(w_\sigma(a)s(a)-\nu_\sigma\bigr)=-A\nu_\sigma.
$$

当 $A$ 与中心都非零时，样本仍有贡献。[SC 公开实现](https://github.com/martin-marek/score-centering/blob/64842967077ea105470ca73fc5d714366d3f7bdb/losses.py#L124-L154)的 token 级 PPO 配置（[README 所列的 `outside=drop` 规则](https://github.com/martin-marek/score-centering/blob/64842967077ea105470ca73fc5d714366d3f7bdb/README.md#L106-L114)）按 advantage 符号计算直接权重和词表中心，再对完整项应用 completion mask。上例对应完整词表路径，并取 completion mask 为 1；代码的额外比率保护与 dual clipping 在例中不触发。

因此，统计 `w == 0` 的裁剪比例不能表示样本的完整贡献已关闭。将零 mask 乘在中心化后的完整项外，才会同时关掉直接项和中心项，这是另一条更新规则。这也不自动恢复 baseline 不变性。即便关闭这个样本，共享参数上的其他梯度仍可改变其动作概率。

baseline 依赖本身不构成实现错误。比较 PPO + SC 的 baseline 时，我会先检查选中分支后的期望更新是否相同，再讨论方差和训练收益。分支中心计算准确，保证的是两个固定分支各自对 $q$ 的零均值；baseline 不变性则要求完成反馈分支选择后，改变 $b$ 仍给出相同的期望更新。
