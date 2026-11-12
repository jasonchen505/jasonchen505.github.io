---
title: "SC 用在蒸馏里，究竟改变了什么？"
date: "2026-11-12 00:00:00 -0800"
description: "比较普通蒸馏更新与 Score Centering 的局部目标：平方残差、残差方差，以及为何同一个完成点不保证同样的训练效果。"
toc: [{"title": "1. 旧样本上的反馈，可以随学生变化", "anchor": "feedback-on-old-samples"}, {"title": "2. SC 把平方残差变成了方差残差", "anchor": "residual-variance"}, {"title": "3. 实现时，重算反馈与停止梯度要同时成立", "anchor": "recompute-and-detach"}, {"title": "4. 同一个完成点，为什么不保证同样的效果", "anchor": "same-optimum-different-updates"}]
---
Score Centering（SC）通过减去 sampler 下的 score 均值，去除训推失配引入的一项漂移。蒸馏也会使用异步或低精度 rollout，自然会遇到一个问题：把 SC 加进去，是否也能改善学习？

最近一篇 [OPD mismatch 研究](https://arxiv.org/html/2611.11450v1)给出了值得留意的对照：在其测试中，SC 阻止了部分 RL 崩溃，但未给 on-policy distillation（OPD）带来明确改善。理解这个结果，需要先看蒸馏的反馈怎样变化。

**在固定 prefix、固定行为分布下，使用当前学生与固定教师的 log-prob 差作为反馈时，普通更新对应平方残差，精确 SC 对应残差方差。二者都保留匹配教师的完成点，但优化路径与随机更新可能不同。** 这个结论不包含 IS、clipping 或额外正则，也不保证整条训练轨迹收敛。

## 1. 旧样本上的反馈，可以随学生变化 {#feedback-on-old-samples}

固定一个 prefix，记实际行为分布为 $q$、当前学生为 $p_\theta$、教师为 $p_T$。先假设有限词表上三者均为正，学生可微，$q$ 与教师不参与反传。定义

$$
A_\theta(a)=\log p_T(a)-\log p_\theta(a).
$$

$$
s_\theta(a)=\nabla_\theta\log p_\theta(a).
$$

教师认为某个动作比学生更值得分配概率时，反馈为正。学生提高其概率后，这个反馈会减弱。即使动作来自旧 rollout，重新计算当前学生的 log-prob，也能得到新的残差。

把 $A_\theta$ 作为本次反传的冻结系数，普通 OPD 的期望上升方向为

$$
g_{\rm OPD}=\mathbb E_q[A_\theta s_\theta]
=-\nabla_\theta L_{\rm sq},
\qquad L_{\rm sq}=\frac12\mathbb E_q[A_\theta^2].
$$

这是论文 [Appendix D.2](https://arxiv.org/html/2611.11450v1) 的结论，推导只需用到 $\nabla_\theta A_\theta=-s_\theta$。右边的平方目标正常求导，左边的策略梯度代理冻结当前反馈，二者方向相同。

当 $p_\theta=p_T$，每个动作的 $A_\theta$ 都为零，因此更新逐样本消失，即使 $q\ne p_\theta$。这个性质依赖**当前学生**的概率；若一直使用采样时保存的学生概率，反馈就不会随学习消退。

## 2. SC 把平方残差变成了方差残差 {#residual-variance}

令 $\mu_\theta=\mathbb E_q[s_\theta]$。[SC 的定义](https://arxiv.org/html/2609.20807v1#S4)给出

$$
g_{\rm SC}
=\mathbb E_q[A_\theta(s_\theta-\mu_\theta)]
=\operatorname{Cov}_q(A_\theta,s_\theta).
$$

沿着论文的平方目标继续推导，对残差方差求导，有

$$
\nabla_\theta\frac12\operatorname{Var}_q(A_\theta)
=\mathbb E_q[(A_\theta-\mathbb E_qA_\theta)(-s_\theta)]
=-g_{\rm SC}.
$$

这是本文的直接推论。记 $L_{\rm SC}=\tfrac12\operatorname{Var}_q(A_\theta)$，两个目标恰好相差

$$
L_{\rm sq}=L_{\rm SC}
+\frac12(\mathbb E_qA_\theta)^2.
$$

普通平方目标同时拟合残差的均值和动作间差异；SC 去掉前一项。因而它通常改变期望更新，不能只被解释为保持原目标的减方差操作。这里最小化的是**反馈残差的方差**，也不等于最小化随机梯度的方差。若在当前参数点 $q=p_\theta$，则 $\mu_\theta=0$，精确 SC 与普通 OPD 的更新相同。

去掉均值以后，会不会丢掉教师目标？若 $q$ 全支持，$L_{\rm SC}=0$ 意味着所有动作都有 $\log p_T(a)-\log p_\theta(a)=c$。两边概率都归一化，迫使 $c=0$，所以仍有 $p_\theta=p_T$。

这个目标在 logits 下尤其直观。令 $p_\theta=\operatorname{softmax}(z_\theta)$，归一化常数在方差中消失：

$$
L_{\rm SC}
=\frac12\operatorname{Var}_q(\log p_T-z_\theta).
$$

自由 logits 下，这是对教师相对 logits 的二次拟合。共享神经网络参数下，目标未必凸，梯度也可能因参数限制或相互抵消而停滞；不能把任意驻点称为“已经学会”。支持缺失时，匹配已覆盖动作的相对概率，也不足以确定完整教师分布。

## 3. 实现时，重算反馈与停止梯度要同时成立 {#recompute-and-detach}

若沿策略梯度代理实现，需要在每次更新时重算当前学生 log-prob、反馈和中心，然后把反馈作为冻结系数，经过学生的 forward 与 head 反传。`stop_gradient` 限制本次求导，不意味着跨更新缓存反馈。直接计算完整平方或方差目标时，则应正常对残差求导。

还要区分总体方差与当前 batch 方差。假设在同一 prefix 下抽到有限个动作，横线表示样本平均。使用精确总体中心的 SC 更新是

$$
\widehat g_{\rm SC}=\overline{A s}-\bar A\mu_\theta.
$$

直接对 batch 目标 $\tfrac12\overline{(A-\bar A)^2}$ 求导，负梯度却是 $\overline{A s}-\bar A\bar s$。两者相差 $\bar A(\bar s-\mu_\theta)$，一般不为零。因此，前面的总体恒等式不能直接用来把现有 loss 替换成 batch 方差；top-k 或有限辅助样本又会改变中心估计。在固定当前参数点，来自同一 $q$ 且与主动作独立的新鲜辅助 token，可以保持对 $g_{\rm SC}$ 的无偏估计；top-k 近似一般没有这一保证。

固定 $q$ 的分析也没有纠正 prefix 的访问分布。若每轮重新采样，或者同一批数据被反复使用后模型已适应其抽样误差，需要重新说明总体目标与经验更新的关系。旧数据可以重算反馈，不等于它能无限复用。

## 4. 同一个完成点，为什么不保证同样的效果 {#same-optimum-different-updates}

共享教师分布这个理想完成点，只规定了在哪里可以结束。去掉均值残差项会改变到达该处的方向；行为分布决定哪些动作获得权重，中心估计带来额外误差，优化器和共享参数还会改变实际步幅。

OPD 论文在两个权重噪声设置下报告，SC 在第 100 和 200 步的 held-out teacher KL 点估计相对同一 rollout engine 下未中心化 OPD 高约 2.3%–3.7%，其中一个末期差异的 95% 置信区间跨过零。这些是特定模型、预算和 top-128 中心的对照，所引 SC 配置各只有一次 200 步训练；它们没有证明精确 SC 普遍有害，也没有识别上述目标差异是否导致了结果。[实验与讨论：§6、Appendix D.1、G–H](https://arxiv.org/html/2611.11450v1#S6)

我更愿意据此保留一个判断：讨论 SC 的作用时，应把反馈随当前模型变化的规则写进目标。教师残差已经提供了逐动作归零的完成点；SC 在这里改变的是残差的拟合方式。

若进一步把蒸馏用于保留旧能力，可以复用旧 prefix 与动作，并在每次更新时重算当前学生的反馈：匹配时反馈归零，偏离时则可能重新出现残差信号。但持续恢复教师概率是否服务于当前任务，仍取决于教师目标是否适用。新的解法即使在任务上更好，也可能偏离教师的概率分配。因此，持续匹配教师概率和“允许更好的新解法，只防止任务表现退化”是不同要求。
