---
title: "用 SC 的 top-k 算“精确 KL”，究竟精确了什么？"
date: "2026-11-12 00:01:00 -0800"
description: "复用 Score Centering 的 top-k 概率，得到的是哪一种精确 KL？从合并 tail 的目标出发，区分估计噪声与正则化目标的变化。"
toc: [{"title": "1. 单 token 估计原本在估计什么", "anchor": "single-token-estimate"}, {"title": "2. top-k 中心对应一个更粗的 KL", "anchor": "coarse-kl"}, {"title": "3. 一个 KL 为零，另一个仍在推动模型", "anchor": "tail-counterexample"}, {"title": "4. 怎样解释新的训练对照", "anchor": "training-comparisons"}]
---
Score Centering（SC）训练往往已为每个 prefix 保存 sampler 的 top-k 概率。这些记录还能用来计算 KL 正则，避免只依赖实际生成的一个 token。[Drift and Distance §4.3](https://www.paperinstruments.com/blog/sc-drift-and-distance) 就比较了这两条路径，并把复用 top-k 的版本称为 exact KL。

这个替换值得先问清楚：去掉 KL 项中的单 token 抽样噪声以后，约束的还是同一个目标吗？

**在 head 与行为概率固定、tail 质量为正且数值保护未触发时，这条 top-k 路径给出的是“head 各动作＋一个合并 tail”之间 KL 的精确梯度。它一般不等于完整词表 KL 的梯度。** 因而，这个对照同时改变了估计噪声和正则化目标，训练收益不能只归因于“KL 算得更准”。

## 1. 单 token 估计原本在估计什么 {#single-token-estimate}

固定一个 prefix 和 sampler 版本，记真实行为分布为 $q$，当前 trainer 为 $p_\theta$，简写为 $p$。假设有限词表上概率均为正，$p_\theta$ 可微；本节固定参数点，$q$ 不参与反传。以下只比较 KL 正则项，不加入 IS、PPO clipping 或数据筛选。

给下降 loss 加上 $\beta K(\theta)$，其中 $\beta>0$ 固定，目标为

$$
K(\theta)=D_{\mathrm{KL}}(q\Vert p_\theta).
$$

实际动作 $a\sim q$ 上的 k1 估计是 $\log q_a-\log p_\theta(a)$，其期望等于 $K$。令参数 score 为 $s_v=\nabla_\theta\log p_\theta(v)$，则

$$
\nabla_\theta K=-\mathbb E_q[s]=-\mu.
$$

单个 token 的梯度 $-s_a$ 有抽样噪声，但在这些条件下无偏地估计完整 KL 的梯度。KL 对上升更新的贡献是 $+\beta\mu$，把 trainer 拉向行为分布。比较另一种实现时，首先要检查它是否仍给出这个期望。

## 2. top-k 中心对应一个更粗的 KL {#coarse-kl}

令 $H$ 为已记录的 sampler top-k 集合，$T$ 为其补集。这里的 top-k 是记录方式，不是把采样限制在 $H$ 内；head 概率保留原值，不重新归一化。记

$$
\tau=q(T),\qquad P_T=p_\theta(T),\qquad \rho=\frac{\tau}{P_T}.
$$

固定 $H$ 和 $q$，假设 $\tau,P_T>0$，且两侧 tail 质量的下限保护均未激活。[SC 的 head＋tail 近似](https://arxiv.org/html/2609.20807v1#A1.SS1) 将 tail 内的行为概率建模为 $\rho p_\theta(v)$，得到近似中心

$$
\widehat\mu=\sum_{v\in H}(q_v-\rho p_v)s_v.
$$

这与一个可直接求值的目标有关：

$$
K_H(\theta)
=\sum_{v\in H}q_v\log\frac{q_v}{p_v}
+\tau\log\frac{\tau}{P_T}.
$$

$K_H$ 比较 head 中每个动作的概率，以及整个 tail 的总质量。对 tail 项求导，并用 $\sum_v p_vs_v=0$ 将 tail 求和换成 head 求和，恰好得到

$$
-\nabla_\theta K_H
=\sum_{v\in H}(q_v-\rho p_v)s_v
=\widehat\mu.
$$

这是已有的 coarsened-KL 关系，见 [KLPO §5.2、附录 D.5](https://arxiv.org/pdf/2610.08963v1#page=12)。它解释了为什么复用 SC 系数能算出一个精确梯度，也指出了精确对应的目标。

[Drift and Distance 的 `exact_head` 实现](https://github.com/rl-cookbook/sc-drift-and-distance/blob/953185b9b2d7935ce3d451bd1e3fb2597cebc1fc/SkyRL/skyrl/backends/skyrl_train/utils/ppo_utils.py#L1034)正是这样复用中心项。忽略共同的聚合尺度，其反传代理可写成

$$
L_{\mathrm{back}}
=-\beta\sum_{v\in H}
\operatorname{sg}(q_v-\rho p_v)\log p_\theta(v).
$$

其中 $\operatorname{sg}$ 停止整个系数的梯度，当前 log-prob 仍通过模型反传，得到 $\beta\nabla K_H$。这些系数仍应在每次 forward 时由当前 $p_\theta$ 重新计算；停止梯度不意味着跨更新固定它们的数值。若直接计算上面的 $K_H$，则应正常求导，无需 detach。两条路径的梯度相同，前向数值通常不同，因此不能把反传代理的 loss 值直接当作 KL 日志。数值 floor 激活后，也需按实际系数重新核对。

## 3. 一个 KL 为零，另一个仍在推动模型 {#tail-counterexample}

合并 tail 究竟舍去了什么，可以用 KL 的分组链式法则精确表达：

$$
K-K_H
=\tau D_{\mathrm{KL}}\!\left(
q(\cdot\mid T)\Vert p_\theta(\cdot\mid T)
\right).
$$

差值就是 tail 内部的条件分布失配。在 $\tau>0$ 时，$K=K_H$ 当且仅当 tail 内的条件分布相同；这一条件也使当前参数点的两个梯度相同。固定 trainer 的 head 概率和 tail 总质量时，tail 动作之间怎样重新分配概率，都不会改变 $K_H$。

取一个三动作例子，只记录第一个动作：

$$
q=(0.6,\,0.3,\,0.1),\qquad
p=(0.6,\,0.1,\,0.3),\qquad H=\{1\}.
$$

两边的 head 都是 $0.6$，tail 总质量都是 $0.4$，所以 $K_H=0$，其梯度也为零。但 tail 中两个动作的相对偏好已经反转：

$$
K=0.3\log3+0.1\log(1/3)
=0.2\log3\approx0.220.
$$

若把三个 logits 当作独立参数，完整 KL 的梯度是 $p-q=(0,-0.2,0.2)$，仍会推动 tail 恢复原来的比例；合并后的正则则没有这个要求。这是说明信息损失的代数例子，不是训练效果实验。

因此，小的 $K_H$ 不能单独保证完整 KL 小。即使 tail 质量较小，还需知道其内部条件 KL 的大小。若某个训练保证需要完整 KL 的上界，也不能把 $K_H\le K$ 这个下界代进去。

## 4. 怎样解释新的训练对照 {#training-comparisons}

回到 SC＋KL。保持 SC 项、聚合和该次更新的系数 $\beta$ 不变，把 k1 换成上述 top-k 路径，就把 KL 的期望上升项从 $\beta\mu$ 换成了 $\beta\widehat\mu$。与“同一 KL 用实际 token 还是独立辅助 token 估计”的比较不同，这里连期望更新都可能改变。给定 prefix 和记录后，head 求和没有额外的 token 抽样噪声，但训练 batch 的随机性仍然存在。

Drift and Distance 报告，在其 0.8B、GSM8K、SGD 配方下，将衰减 KL 的单 token 估计换成 top-32 路径，末期训练 reward（最后三个 rollout 的均值）从 66.7% 升到 70.3%。[配套实验脚本](https://github.com/rl-cookbook/sc-drift-and-distance/blob/953185b9b2d7935ce3d451bd1e3fb2597cebc1fc/experiments/sec4_3_kl_penalty.sh)中，exact-head 配置只有一个种子。这组训练 reward 不能单独确立独立评测收益，也无法分离减少噪声与改变正则化目标各自的作用。

若要单独考察 KL 估计噪声，可以固定同一个 $K_H$：一条路径按 $q$ 在“head 各动作＋合并 tail”上的概率作新鲜抽样，使用对应类别的 k1 项（抽中 tail 时为 $\log\tau-\log P_T$），另一条直接求和。这样两条路径的期望梯度相同。

我的理解是，top-k 在这里还决定了约束的分辨率。放松 tail 内部的匹配可能允许有益变化，也可能漏掉需要控制的偏离；上面的恒等式和这个训练对照都没有判定哪一种占主导。

如果希望保留完整 sampler KL，来自同一真实 $q$ 的新鲜辅助抽样仍可在期望中估计原目标。若采用 head＋tail 聚合，则应明确接受它对 tail 内部重分配不敏感。比较两条路径时，应把目标差别写进方法说明，再用任务表现和成本判断取舍。
