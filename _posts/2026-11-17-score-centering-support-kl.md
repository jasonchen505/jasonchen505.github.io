---
title: "把 trainer 限制到采样支持集，SC 会变吗？"
date: "2026-11-17 00:00:00 -0800"
description: "把 trainer 归一化到历史采样支持集，为什么纯 Score Centering 的更新不变，KL 却少了一项？从概率质量约束到稀疏反传，拆解这个差别。"
toc: [{"title":"1. 归一化改变了 score，却被中心化抵消","anchor":"centering-invariance"},{"title":"2. 同样的归一化，为什么改变了 KL？","anchor":"support-mass"},{"title":"3. 稀疏反传保留了哪种更新","anchor":"sparse-backprop"},{"title":"4. 别把两种“少算一些 token”混在一起","anchor":"top-k-tail"}]
---
top-p 或 top-k 采样会把一部分 token 的生成概率置零。如果完整保存 sampler 当时保留的 token 及其概率，训练时就能在这个历史支持集上精确计算 Score Centering（SC），无需猜测未记录的 tail。[Prime-RL 的 SC + sampler KL 提案](https://github.com/PrimeIntellect-ai/prime-rl/pull/4233)采用了这条路径，并把 trainer 也归一化到同一个集合。

这个计算选择会不会改变训练目标？

**固定 prefix、历史支持集和行为分布，并使用相同的冻结反馈时，未加 IS 或 clipping 的精确 SC 更新不变；但额外的 KL 会少掉对支持集总概率质量的约束。** 因而，两种实现可以有完全相同的纯 SC 梯度，却在加入 KL 后推动模型去不同的方向。

## 1. 归一化改变了 score，却被中心化抵消 {#centering-invariance}

记行为分布为 $q$，其完整支持为 $S$；当前 trainer 为 $p_\theta$，简写为 $p$。假设有限词表上的 $p$ 为正，且可微。$q$ 和 $S$ 来自历史生成记录，不参与反传，也不随当前 trainer 重新选择。令

$$
P=p_\theta(S),\qquad
p^S_\theta(a)=\frac{p_\theta(a)}{P},\quad a\in S.
$$

这里的 $S$ 是 sampler 实际允许生成的全部动作，不是从一个全支持分布中只记录了概率最高的若干项。

对采到的动作 $a\in S$，[SC](https://arxiv.org/html/2609.20807v1#S4) 的下降 loss 可以写成

$$
L_{\rm SC}(a)
=-A_a\left[\log p_\theta(a)-\mathbb E_q\log p_\theta\right],
$$

其中反馈 $A_a$ 在本次反传中冻结。换成 $p^S$ 后，每个支持内动作的 log-prob 都少了同一个 $\log P$。由于 $\sum_{a\in S}q_a=1$，

$$
\log p^S(a)-\mathbb E_q\log p^S
=\log p(a)-\mathbb E_q\log p.
$$

两个标量 surrogate 本身就相同，因此通过 forward 和输出层传回去的完整参数梯度也相同。这个等式逐样本成立，无需等待有限 batch 收敛到期望；它并不是说单个 batch 的 SC 梯度为零。

## 2. 同样的归一化，为什么改变了 KL？ {#support-mass}

现在给 loss 加上系数 $\beta>0$ 的 sampler KL。需要区分完整 trainer 与支持内条件策略：

$$
K_{\rm full}=D_{\rm KL}(q\Vert p),\qquad
K_S=D_{\rm KL}(q\Vert p^S).
$$

将 $p(a)=Pp^S(a)$ 代入，直接得到

$$
K_{\rm full}=K_S-\log P.
$$

SC 中，$\log P$ 在实际动作与中心项之间抵消。KL 中只有对 log-prob 的期望，这一项保留了下来：$K_S$ 约束支持内各动作的相对概率，$-\log P$ 则惩罚 trainer 把概率质量移出历史支持集。

取三个动作，让历史 sampler 只生成前两个；对 $0<\varepsilon<1$，令

$$
q=(1/2,\,1/2,\,0),
$$

$$
p=(\varepsilon/2,\,\varepsilon/2,\,1-\varepsilon).
$$

无论 $\varepsilon$ 多小，前两个动作的相对比例都没有变，所以 $p^S=q$、$K_S=0$。但完整 KL 为 $-\log\varepsilon$，可以任意大。条件 KL 没有计算错误；它没有要求当前 trainer 继续给这个旧支持集分配足够的总质量。

条件 KL 对支持内 logits 的共同平移不敏感，但不会把支持外概率冻结：支持内相对概率的更新及网络参数共享，仍可能改变 $P$ 和其他 prefix 的分布。两种 KL 在这里都是软惩罚，固定 $\beta$ 不保证每一步满足某个 KL 上限。

## 3. 稀疏反传保留了哪种更新 {#sparse-backprop}

这个差别直接影响计算。设 $p=\operatorname{softmax}(z)$，$e_a$ 是动作 $a$ 的 one-hot 向量，并将 $q$、$p^S$ 在 $S$ 外补零。KL 按 $q$ 精确求和时，对同一套 logits，两种 SC + KL loss 的单样本上升方向分别为

$$
G_{\rm full}
=A_a(e_a-q)+\beta(q-p),
$$

$$
G_S
=A_a(e_a-q)+\beta(q-p^S).
$$

两者之差为 $G_{\rm full}-G_S=\beta(p^S-p)=\beta\nabla_z\log P$，正是完整 KL 中 $-\log P$ 额外提供的上升项。

纯 SC 的稠密 softmax 项已经抵消；差别全部来自 KL。完整 KL 通常重新引入稠密的 $p$，条件 KL 则只涉及 $S$ 内的概率。这解释了为什么支持内 SC + KL 可以采用稀疏输出层反传。

参数梯度仍是 $J_z^\top G$，其中 $J_z=\partial z/\partial\theta$ 包含完整网络。输出 logits 梯度稀疏，不代表只更新少数网络参数。

Prime-RL 提案中的[固定版本 loss](https://github.com/PrimeIntellect-ai/prime-rl/blob/0e2995a4c6139f9e2c5aaf312a1b4d632729e492/src/prime_rl/trainer/rl/loss.py)体现了这一选择：重放完整截断支持时，trainer 概率在支持内归一化；未截断时，其 loss 设计用辅助抽样估计完整 KL，后者仍有稠密梯度。[稀疏反传实现](https://github.com/PrimeIntellect-ai/prime-rl/pull/4445)因此也需要检查 loss 和支持条件。这里的等式是对这些目标的直接展开，并不说明哪种 KL 训练效果更好。

## 4. 别把两种“少算一些 token”混在一起 {#top-k-tail}

通常的 top-k head + modeled tail SC，记录的是完整行为分布的一部分。若据此计算合并 tail 的 KL，仍会约束 tail 的总质量，只忽略 tail 内部的概率分配。[KLPO §5.2](https://arxiv.org/pdf/2610.08963v1#page=12)讨论了这个目标。

本文的支持内归一化恰好不同：$q$ 在 $S$ 外本来就是零，所有非零行为概率都已记录。$q=0$ 的项本来就不贡献 $K_{\rm full}$ 的求和；产生目标差别的是将 trainer 在 $S$ 内重新归一化。它保留支持内的相对分配，额外 KL 却不再约束 trainer 留在 $S$ 内的总质量。两者都可能减少计算，但放松的是不同要求。

如果希望限制当前 trainer 对真实行为分布的完整偏离，就应保留 $-\log p_\theta(S)$ 这一项；若只希望约束旧支持内的相对概率，条件 KL 是一个明确的选择。历史支持由旧模型和采样设置共同决定，持续维持其总质量是否有利，还取决于当前学习目标。稀疏计算能说明怎样便宜地实现一个选择，不能代替这个判断。

*实现状态：截至 2026-11-13，Prime-RL 的完整截断支持采集 [#4232](https://github.com/PrimeIntellect-ai/prime-rl/pull/4232) 与稀疏反传 [#4445](https://github.com/PrimeIntellect-ai/prime-rl/pull/4445) 已合并，SC + KL loss 提案 #4233 仍开放。未截断辅助抽样还依赖尚未合并的 [renderers #202](https://github.com/PrimeIntellect-ai/renderers/pull/202)，以保留重复抽样记录。本文核对的是上述固定版本中的 loss 设计，没有运行端到端训练或作收益比较。*
