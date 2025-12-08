# Assignment 5 — Reinforcement Learning (Theory)

Question 1

1. 所有可能的 1-word 和 2-word 序列（作为中间状态）：

- 1-word 序列（3 个）：
  - [I]
  - [like]
  - [pizza]

- 2-word 序列（3x3 = 9 个）：
  - [I, I], [I, like], [I, pizza]
  - [like, I], [like, like], [like, pizza]
  - [pizza, I], [pizza, like], [pizza, pizza]

2. 终止状态（3-word sentence）的数量：3^3 = 27 个。\
   其中只有 1 个会得到非零奖励（"I like pizza"），奖励值为 +10。

3. 在随机均匀策略下，价值函数 V(s) = 到达目标句子的概率 × 10。计算如下：

(a) s0 = [ ]（空状态）：需要连续生成 I, like, pizza；每步均为 1/3，因此概率 = (1/3)^3 = 1/27。\
    V(s0) = 10 * 1/27 = 10/27 ≈ 0.37037

(b) s1 = [I]：还需生成 [like, pizza]，概率 = (1/3)^2 = 1/9。\
    V([I]) = 10 * 1/9 = 10/9 ≈ 1.11111

(c) s2 = [I,like]：只需生成最后一个词 pizza，概率 = 1/3。\
    V([I,like]) = 10 * 1/3 = 10/3 ≈ 3.33333

(d) s3 = [I,pizza]：已经第二个词是 pizza，无法变成 I like pizza（长度受限为 3 且中间词已错误），概率 = 0。\
    V([I,pizza]) = 0

---

Question 2

1. Q-learning 的更新规则（写法与题目一致）：

   Q(s,a) ← Q(s,a) + α [ r + γ · max_{a'} Q(s',a') - Q(s,a) ]

2. 计算 Q([I], like) 的更新值：

   已知：Q([I], like) = 1.0，α = 0.5，γ = 1，r = 0（因为还未到句末产生奖励），\
   s' = [I, like] 时的 Q 值为 {1.0, 0.5, 2.0}，因此 max_{a'} Q(s',a') = 2.0。\

   更新量 = α [ r + γ·max Q(s',a') - Q(s,a) ] = 0.5 * (0 + 1*2.0 - 1.0) = 0.5 * 1.0 = 0.5。\
   因此 Q 新值 = 1.0 + 0.5 = 1.5。

   所以 Q([I], like) 更新后为 1.5。
