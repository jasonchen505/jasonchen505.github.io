---
title: "Score Centering & Importance Sampling：从零均值到策略更新"
date: "2026-10-27 00:00:00 -0700"
description: "从 score 的零均值出发，理解 Score Centering、Importance Sampling 与策略更新的关系，以及裁剪、筛选和中心近似的边界。"
toc: [{"title": "1. 从非零均值到 SC", "anchor": "from-score-to-sc"}, {"title": "2. JPC 给出的局部策略解释", "anchor": "jpc-policy-interpretation"}, {"title": "3. IS、裁剪与筛选", "anchor": "is-clipping-selection"}, {"title": "4. 中心的近似与实现", "anchor": "center-approximation"}, {"title": "5. 从局部性质到策略更新", "anchor": "policy-updates"}]
---
LLM RL 通常由 sampler 生成回答，再交给 trainer 计算梯度。检查点延迟、量化、温度或数值实现，都可能让两边给同一个 token 的概率不同。Score Centering（SC）在采样分布下中心化 score，Importance Sampling（IS）还改变动作权重。二者何时重合，取决于使用的分布与权重规则。

下文先比较固定 prefix、冻结动作反馈时的条件期望更新，再讨论筛选和实现近似。条件期望的性质、同一个 batch 的实际梯度和完整训练的收益，需要分开理解。

## 1. 从非零均值到 SC {#from-score-to-sc}

固定一个 prefix，也就是当前上下文。用 $q$ 表示行为分布，$p_\theta$ 表示 trainer，$R$ 表示动作及后续回答的回报。先不加 IS，反传时冻结回报。下文省略对 prefix $h$ 的条件，并用 $p$ 简写当前参数处的 $p_\theta$；$q$ 指实际产生 token 的分布，包括温度、采样截断等处理。

定义参数 score $s(a)=\nabla_\theta\log p_\theta(a)$，以及 $\mu=\mathbb E_q s$。概率归一化保证 $\mathbb E_p s=0$，换成 $q$ 后却通常不成立，于是原始更新可以分解为

$$
\mathbb E_q[Rs]
=\operatorname{Cov}_q(R,s)+\mathbb E_q[R]\,\mu.
$$

所有奖励一起加上常数 $d$，动作的相对优劣没变，更新却会多出 $d\mu$。[SC](https://arxiv.org/html/2609.20807) 减去 score 的条件均值，令 $t(a)=s(a)-\mu$：

$$
G_{\mathrm{SC}}
=\mathbb E_q[Rt]
=\operatorname{Cov}_q(R,s).
$$

这消除了奖励零点对条件期望更新的影响。准确减去每个 prefix 的条件 reward 均值，也能得到同一协方差。但实际 group baseline 通常按 prompt 计算，回答沿途的 prefix 均值并不相同。组内 advantage 和为零，不能自动使 token SC 冗余。反过来，group baseline 随 reward 平移时，advantage 本身就不变，所以单做奖励平移测试也不能证明 SC 在起作用。

## 2. JPC 给出的局部策略解释 {#jpc-policy-interpretation}

Jacobian Policy Correction（JPC）为这个 centered score 提供了一个策略解释。这里用 JPC 指代下面这个保留 trainer Jacobian 的局部策略构造。设 $p=\operatorname{softmax}(z_\theta)$，$J_z=\partial z_\theta/\partial\theta$，则

$$
s(a)=J_z^\top(e_a-p),\qquad
t(a)=J_z^\top(e_a-q).
$$

这里沿用 trainer 的 Jacobian，输出归一化的中心换成了 $q$。冻结锚点 $\theta_0$ 和行为分布，在其固定支持上定义

$$
r_\theta(a)=
\frac{q(a)\exp(z_\theta(a)-z_{\theta_0}(a))}
{\sum_v q(v)\exp(z_\theta(v)-z_{\theta_0}(v))}.
$$

它在锚点等于 $q$，此时的 score 恰好是 $t$，随后沿 trainer 的相对 log-odds 变化。令 $f(a)=\mathbb E[R\mid h,a]$，冻结给定动作后的回报条件分布，就有

$$
\left.\nabla_\theta\mathbb E_{r_\theta}[f(a)]\right|_{\theta_0}
=\mathbb E_q[Rt].
$$

SC 因而是这个归一化代理在锚点的回报梯度。JPC 选择了由 trainer 提供的可微学习响应。历史 sampler 即使已经冻结，也不妨碍这个目标成立。

这个构造保证的是锚点处的一阶等价：一般位置的代理 score 是 $J_z^\top(e_a-r_\theta)$，而固定 $q$ 的 SC 使用 $J_z^\top(e_a-q)$。二者离开锚点后通常不再相同。

## 3. IS、裁剪与筛选 {#is-clipping-selection}

IS 改变的是：在相同的固定动作反馈 $f$ 下，将当前动作的权重从 $q$ 换成 $p$。若 $q$ 覆盖 $p$ 的支持，采用精确、未裁剪的比率 $w=p/q$，则

$$
\mathbb E_q[ws]=\mathbb E_p[s]=0.
$$

精确 IS 已经恢复了加权 score 的零均值，还改变了动作之间的相对贡献。固定同一个 $f$ 时，两种更新分别为：

$$
G_{\mathrm{SC}}=\operatorname{Cov}_q(f,s),\qquad
G_{\mathrm{IS}}=\mathbb E_q[wRs]=\operatorname{Cov}_p(f,s).
$$

前者对应以 $q$ 为锚点的 JPC 响应，后者是 $\mathbb E_p f$ 的梯度。两者在 $p=q$ 时相同，失配时一般连方向都可能不同。

实际 IS 常因方差问题而裁剪比率，或屏蔽过大的偏离。这时 $\nu=\mathbb E_q[ws]$ 可能重新非零。这里先令 $w(a)$ 在给定 prefix 后只是动作的函数，并在反传时冻结。按 [SC 论文附录 A.2](https://arxiv.org/html/2609.20807)，应中心化的是整个加权 score：

$$
G_{w\mathrm{SC}}
=\mathbb E_q\!\left[R\bigl(w(a)s(a)-\nu\bigr)\right],
\qquad
\nu=\mathbb E_q[ws].
$$

不同于先算 $s-\mu$、再乘当前动作的 $w(a)$。当 $w=p/q$ 精确成立时，完整中心 $\nu$ 逐 prefix 恒为零，额外 SC 项对同一个 batch 也没有影响。这不需要等待采样量趋于无穷。

这里的中心指由分布计算的精确条件期望，而非当前 batch 的加权 score 样本均值。后者即使使用精确 IS，也通常不为零。若中心来自有限样本估计，或者当前动作与中心使用了不一致的数值保护规则，就不能直接套用上述逐 batch 结论。截断后，SC 可以去掉实际加权 score 的均值，却不会补回被截掉的奖励信号。

到这里，中心仍是在原始行为分布 $q$ 下定义的。若样本还经过筛选，还需要检查筛选如何改变实际参与更新的数据。

[Jian Hu 的分析](https://hijkzzz.notion.site/From-FlashREINFORCE-to-Score-Centering-What-Importance-Sampling-Already-Corrects-3a5d9a33ecc9835b9e9a812908a055d4) 提醒我们：原分布下的零中心，不等于筛选后数据的均值为零。例如 FlashREINFORCE 会筛选整条回答。令 $M$ 表示是否保留，$k(h,a)=\Pr_q(M=1\mid h,a)$；即使 token 比率精确，

$$
\mathbb E_q[Mws\mid h]
=\sum_a p(a\mid h)\,k(h,a)\,s(h,a)
$$

仍不必为零，因为保留概率可能依赖动作与续写。这与《增加 SC 不改变 batch》没有矛盾：增加的是原 $q$ 下的零中心，筛选后考察的是另一个期望。长度归一化也可能改变有效权重。token IS 此外只重加权当前动作，尚未转换 prefix 的访问分布和后续生成策略，因此不能把它直接当成完整轨迹纠偏。

## 4. 中心的近似与实现 {#center-approximation}

实现需要估计与所选权重匹配的中心：纯 SC 的 $\mu$，或加权 SC 的 $\nu$。纯 SC 的一种实现只记录词表 top-k 的 head $H$，将 tail $T$ 的剩余质量按 trainer 比例补齐。在 $p(T)>0$、数值保护未触发时，

$$
\rho=\frac{q(T)}{p(T)},\qquad
\widehat q_v=q_v\ (v\in H),\qquad
\widehat q_v=\rho p_v\ (v\in T).
$$

head 保留原始概率，不重新归一化。利用 $\mathbb E_p s=0$，近似中心可化成 head 上的求和：

$$
\widehat\mu=\sum_{v\in H}(q_v-\rho p_v)s(v).
$$

它是重建分布 $\widehat q$ 下的精确中心；相对于真实 $q$，仍有

$$
\mu-\widehat\mu
=\sum_{v\in T}(q_v-\rho p_v)s(v).
$$

因此，head 加 trainer tail 一般只能近似恢复真实 $q$ 下的零均值。奖励平移 $d$ 后，剩余的期望更新变化为 $d(\mu-\widehat\mu)$，由 tail 的分布近似误差决定。

令 $A$ 为冻结的 advantage，$\operatorname{sg}$ 表示停止梯度。可交给自动微分的上升代理是

$$
U=\operatorname{sg}(A)\left[
\log p_\theta(a)
-\sum_{v\in H}\operatorname{sg}(q_v-\rho p_v)\log p_\theta(v)
\right].
$$

它的梯度为 $A(s(a)-\widehat\mu)$，最小化 loss 时取 $-U$。系数必须冻结，梯度仍经过完整 forward 和 head（输出层），才能得到参数空间的校正。[公开实现](https://github.com/martin-marek/score-centering/blob/64842967077ea105470ca73fc5d714366d3f7bdb/losses.py#L143-L153) 采用了这条 head 求和思路。组合 IS 时，要在当前动作与中心中使用一致的权重规则，并冻结反传系数，不能继续套用未加权的 $\widehat\mu$。

更具体地，若权重只依赖当前 token 的比率，记 $w(v)=\psi(p_v/q_v)$。在 $\rho>0$ 时，重建 tail 上的比率恒为 $1/\rho$，于是 [附录 A.2](https://arxiv.org/html/2609.20807) 给出的对应近似中心为

$$
\alpha=\rho\,\psi(1/\rho),\qquad
\widehat\nu
=\sum_{v\in H}\bigl(q_vw(v)-\alpha p_v\bigr)s(v).
$$

纯 SC 时 $\psi\equiv1$，退回 $\widehat\mu$；精确 IS 时 $\psi(r)=r$，有 $\alpha=1$、$q_vw(v)=p_v$，head 系数逐项为零。这也从实现形式上解释了前面的冗余结论。计算当前动作的 IS 比率时，还应保存被采样 token 的真实行为 logprob，即使它不在 head 中。tail 截断或分母保护激活后，则须按实际系数核对近似。

这里的 top-k 是分布记录方式。如果 sampler 本身就在 top-k 或 top-p 支持集内采样，$q$ 可能没有覆盖完整 $p$，精确 IS 的零均值前提便不再满足。完整保存历史支持集及其中各 token 的真实采样概率，可以精确计算该 $q$ 的中心。把 trainer 也限制并归一化到这个集合，则改变了 IS 的目标策略；相应的 score 也应来自这个重新归一化后的策略。也可以从同一历史 $q$ 抽取辅助 token，在给定 prefix 后要求它们与本次动作及回报独立。这是在用采样噪声换取中心估计，区别于确定性的 head 加 trainer tail。

## 5. 从局部性质到策略更新 {#policy-updates}

SC 的局部目标不等于训练收益保证。固定 rollout 多次更新时，若历史 $q$ 与 advantage 保持固定、中心精确，纯 SC 可以继续优化冻结反馈下的经验 surrogate，却不能一直沿用锚点处与代理回报的等价关系。再加入显式 KL，则又增加了约束策略偏离的作用。它不属于必须消除的 score 漂移。

诊断指标也应与实际权重一致。采用裁剪 IS 时，若用 `drift_norm` 记录未加权中心的范数 $\|\mu\|$，它就不能直接代表实际加权中心的范数 $\|\nu\|$。

回到 SC 与 IS 的比较，纯 SC 在 $q$ 下去掉 score 的条件均值；精确 IS 把当前动作的权重换成 $p$，并同时恢复加权 score 的零均值。因此，精确 IS 下的加权 SC 可以冗余，裁剪后则仍可能留下可被中心化消除的项。筛选、长度聚合与中心近似，需要按最终进入梯度的规则另行检查。零均值本身并不唯一决定策略更新。JPC 为 SC 提供了明确的局部策略解释，IS 则通过目标分布决定当前动作之间的相对贡献。评价一种校正，既要验证它是否消除了所选分布下的均值项，也要说明校正后的梯度指向哪个策略目标。

