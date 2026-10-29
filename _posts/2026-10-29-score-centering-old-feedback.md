---
title: "旧反馈何时该停止推动模型？从 SC 到残差目标"
date: "2026-10-29 00:00:00 -0700"
description: "旧 advantage 应该推动模型多久？比较 SC、KLPO 与 BPO 的局部目标，讨论残差、目标完成度与旧能力保留。"
toc: [{"title": "1. 固定旧反馈后，纯 SC 何时停止", "anchor": "fixed-feedback"}, {"title": "2. SC 加 KL 与 KLPO 的局部目标", "anchor": "sc-kl-klpo"}, {"title": "3. 残差变大，也可能来自有益的超越", "anchor": "beyond-old-targets"}, {"title": "4. BPO 需要看实际采用的更新系数", "anchor": "bpo-update-coefficients"}, {"title": "5. 从拟合旧目标到保留旧能力", "anchor": "retaining-old-capabilities"}]
---
LLM RL 的 rollout 有生成成本，异步训练和数据复用都会让同一份反馈跨越多次参数更新。奖励记录保持不变，模型却已经变了：一份旧 advantage 还应该推动它多久？

我的判断是：**旧目标是否已经完成，与它是否仍值得追求，需要分别判断。** 历史概率目标被更好的策略超越后，继续恢复它未必有益；旧样本仍可保留，训练用途需要重新说明。这里讨论 sampler 只作为临时改进参考的训练；若长期匹配参考分布本身就是任务要求，持续约束仍然合理。

## 1. 固定旧反馈后，纯 SC 何时停止 {#fixed-feedback}

固定一个 prefix，以 $q$ 表示历史行为分布，$p_\theta=\operatorname{softmax}(z_\theta)$ 表示当前 trainer。令 $f_a$ 为采取动作 $a$ 后、沿旧策略续写的回报条件均值。以下固定 $q$ 和 $f$，假设有限词表上 $q_a,p_a>0$，使用精确条件均值，暂不加入 IS 或 clipping。定义

$$
A_a=f_a-\mathbb E_q f,\qquad
s_a=\nabla_\theta\log p_\theta(a).
$$

这里的 $A$ 是理想的条件优势，不等同于按整个 prompt 计算的 group advantage。先比较 logits 可独立调整时的总体方向 $g_z$；实际参数梯度为 $J_z^\top g_z$，其中 $J_z=\partial z_\theta/\partial\theta$。

[SC 的中心化公式](https://arxiv.org/html/2609.20807v1) 给出

$$
G_{\mathrm{SC}}
=\mathbb E_q\!\left[f_a(s_a-\mathbb E_q s)\right]
=\operatorname{Cov}_q(f,s).
$$

softmax score 中共同的归一化项被消掉后，得到 $g_{z,a}=q_aA_a$。固定 $q$ 和 $f$，这个方向不随当前 $p$ 减弱。因此，在这个设定下，纯 SC 没有为非零优势设置有限的“完成点”。

## 2. SC 加 KL 与 KLPO 的局部目标 {#sc-kl-klpo}

要让推动力随策略变化调整，可以给 SC 的上升目标 $\operatorname{Cov}_q(f,\log p)$ 减去 $\beta D_{\mathrm{KL}}(q\Vert p)$，取固定的 $\beta>0$。本文的 SC+KL 专指这一选择。

这个上升目标也可以写成

$$
U_{\mathrm{SC+KL}}
=\sum_a q_a(A_a+\beta)\log p_a+C,
$$

其中 $C$ 与当前 $p$ 无关。这个形式既能直接给出下面的梯度，也说明有限最优点是否存在取决于各项系数的符号。

[KLPO §3.1、3.3](https://arxiv.org/pdf/2610.08963v1) 则从 $\max_p\{\mathbb E_p f-\beta D_{\mathrm{KL}}(p\Vert q)\}$ 的最优性条件出发，拟合反馈与 log-ratio 的关系。注意这里 KL 的方向不同。记

$$
c_a=\log\frac{p_a}{q_a}
-\mathbb E_q\log\frac{p}{q},
$$

把对动作不变的截距作为自由变量最小化并消去后，目标为

$$
L_{\mathrm{KLPO}}
=\frac{1}{2\beta}
\operatorname{Var}_q\!\left(f-\beta\log\frac{p}{q}\right)
=\frac{1}{2\beta}\mathbb E_q[(A-\beta c)^2].
$$

其中 $c$ 记录已经发生的中心化 log-ratio 变化，$A-\beta c$ 衡量它与旧反馈要求的差距。对相应损失取负梯度，得到：

| 方法 | 总体上升方向 $g_{z,a}$ |
| --- | --- |
| 纯 SC | $q_aA_a$ |
| SC 加 $\beta D_{\mathrm{KL}}(q\Vert p)$ 惩罚 | $q_a[A_a-\beta(p_a/q_a-1)]$ |
| KLPO 的局部回归 | $q_a(A_a-\beta c_a)$ |

在 $p=q$ 处，三个方向相同。离开这个位置后，SC+KL 根据相对概率变化调整推动力，KLPO 则根据中心化 log-ratio 调整。

令后两行的方向为零，得到不同的理想目标：

$$
p_a^{\mathrm{SC+KL}}
=q_a\left(1+\frac{A_a}{\beta}\right),
\qquad
p_a^{\mathrm{KLPO}}
=\frac{q_a e^{A_a/\beta}}{\sum_b q_b e^{A_b/\beta}}.
$$

SC+KL 一式要求所有括号为正，才能得到内点解；若所有括号非负、但部分恰为零，理想最优点位于概率单纯形的边界，有限的 softmax logits 只能逼近它；若出现负数，上升目标甚至无上界。KLPO 给出 Gibbs 分布：回归与原始正则回报共享这个理想最优解，但梯度一般不同。这些结论针对自由分布；共享参数下，梯度抵消也能造成停滞，不能把驻点直接当作“已经学会”。

## 3. 残差变大，也可能来自有益的超越 {#beyond-old-targets}

超过目标后，反馈还可以反向。令 $q=(0.5,0.5)$、$f=(1,0)$、$\beta=1$。SC+KL 的好动作目标概率是 $0.75$，KLPO 约为 $0.731$。假如后续训练已把它提高到 $0.95$，这两个旧目标都会往回拉。但在这个单步任务中，未加正则的期望回报就是好动作的概率，当前表现已经更好。

这是代数例子，不是训练收益实验。**旧目标残差变大，也可能来自有益的超越。** 是否该往回拉，取决于历史约束是否仍属于当前任务；例子没有否定正则化本身。

## 4. BPO 需要看实际采用的更新系数 {#bpo-update-coefficients}

从残差目标推导出的算法，实际更新也可能不再使用完整的动态残差。这时需要检查最终采用的系数与门控，才能判断旧反馈何时停止推动模型。

[BPO §3.1、3.2](https://arxiv.org/html/2609.15987v1#S3) 也从轨迹平方残差出发，但实际算法在线性化时用旧回报优势替换动态残差因子，再采用 binary KL、平滑、mask 和权重上限。因此，不能把完整残差的完成度解释直接套到实际 BPO 上。

忽略平滑、mask 和上限时，BPO 使用停止梯度的权重 $\omega_a=(1-q_a)/(1-p_a)$。对已采样 token 自身的 logit，

$$
\omega_a\frac{\partial\log p_a}{\partial z_a}
=\frac{1-q_a}{1-p_a}(1-p_a)
=1-q_a.
$$

这个权重抵消了该分量随 $p_a$ 增大而出现的衰减。这里考察的是单个采样项对自身 logit 的贡献；对其他 logit $z_b$，$b\ne a$，相应分量为 $-(1-q_a)p_b/(1-p_a)$，一般仍随其余动作之间的相对概率变化。因此，这个恒等式不能直接推出整个总体梯度或参数梯度保持不变。实际 mask 按优势符号和 $\omega_a$ 阈值关闭该 token 的直接梯度项，平滑与上限则改变其大小。mask 既不能证明能力已掌握，也不是策略变化的硬约束：同一 prefix 上其他动作的梯度，以及经共享参数传来的其他梯度，仍能改变该概率。

实现中，动态残差和所需的中心应随当前 log-prob 重新计算；`stop_gradient` 只停止本次反传，不等于跨更新冻结数值。历史 $q$ 保持采样时的含义。表格比较的是总体期望：近似中心、有限样本会改变实际更新，期望等价也不保证固定 batch 上的经验目标相同。复用数据时，应按实际 loss 或更新规则判断“完成”。

## 5. 从拟合旧目标到保留旧能力 {#retaining-old-capabilities}

目标是否适用，还取决于反馈描述的条件。一条旧代码 rollout 失败，可能是后续调试没做好；当前模型已能修复时，同一早期动作的后续价值就可能改变。旧记录依然属实，却不足以单独决定当前该强化什么。旧优势仍可用于策略改进，但它与当前回报的关系涉及新策略的访问分布，不能直接沿用旧数据的平均值。[TRPO §2](https://arxiv.org/pdf/1502.05477) 给出了这一关系及局部近似。

前面的残差回答的是“离指定的历史目标还有多远”；是否继续拟合这个目标，还需要当前表现的证据。

因此，我把旧数据的“退出”理解为结束某项训练职责。局部目标完成或不再适用时，可以停止拟合它；若保留数据用于防退化，需要重新说明保护的是什么。

例如，对旧任务分布规定表现下限 $J_{\mathrm{old}}(p)\ge\tau$，其中 $J_{\mathrm{old}}$ 是当前策略的期望回报。这允许换解法和继续提高成绩。对应的单侧软惩罚可以写成

$$
L_{\mathrm{retain}}
=\lambda\,[\tau-J_{\mathrm{old}}(p)]_+,
\qquad \lambda\ge0,\quad [x]_+=\max(x,0).
$$

这是我的目标选择：表现不足时才要求恢复。软惩罚不保证约束一定满足，也不是经过验证的回放算法；$J_{\mathrm{old}}$ 不能靠旧 log-prob 直接算出，需要当前表现的证据。训练过的旧样本不能再充当独立泛化评测，旧答案概率下降也可能只是换了正确解法。

按当前表现决定旧数据何时退出、何时重新启用，是我想继续研究的方向。重新使用时还要核对目标与采样规则：只挑成功样本就改变了数据分布，原始 SC 在 $q$ 下的统计解释不能原样照搬。这种安排是否更有效，仍需独立评测与成本证据。

旧反馈何时停止推动模型，需要同时回答两个问题：所选更新规则是否还有未完成的局部目标，以及这个目标是否仍服务于当前任务。残差、mask 和旧答案概率各自提供有限的信号；旧数据继续承担什么训练职责，应由明确的目标与当前表现共同决定。
