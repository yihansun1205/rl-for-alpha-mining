# Paper Reading Notes: AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets for Robust Exploration

> 任务来源：RL Alpha Mining Paper Reading Task Template  
> 论文：**AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets for Robust Exploration**  
> 版本：ICLR 2026 / arXiv 2509.25055v2  
> 代码：论文声称已开源于 `https://github.com/BerkinChen/AlphaSAGE`

---

## 1. One-Sentence Summary

```text
AlphaSAGE proposes a structure-aware GFlowNet-based formulaic alpha generator to solve reward sparsity, sequential representation weakness, and low diversity in RL-based alpha mining by constructing AST-form alphas with an RGCN encoder and a multi-faceted reward combining IC, structure-behavior alignment, novelty, and entropy regularization.
```

更直观地说：

**AlphaSAGE 的核心不是“训练一个策略去找最高 IC 的单个公式”，而是训练一个生成分布，让模型更容易采样到一批高质量、结构多样、相关性较低的公式 alpha。**

---

## 2. Core Problem

## Core Problem
- **Task type:** 公式化 alpha 挖掘 + alpha pool 构建 + 动态组合。
- **Input:** 历史市场数据，主要是 OHLCV 类基础特征，包括 `Open, Close, High, Low, Vwap, Volume`，再通过算子构造公式 alpha。
- **Output:** 一组 formulaic alphas / alpha pool，后续通过动态线性组合得到 time-varying Mega-Alpha / portfolio signal。
- **Main objective:** 学习一个生成策略 \(P_\theta(\alpha)\)，使得采样 alpha 的概率与 alpha 的综合 reward 成正比：
  \[
  P_\theta(\alpha)\propto R(\alpha).
  \]
  这里 \(R(\alpha)\) 不只是单因子 IC，还包括结构一致性和 novelty。
- **Financial target:** 预测未来股票收益 \(y_d\)，并在组合层面提升 IC、RankIC、ICIR、RankICIR、年化收益、Sharpe，并降低回撤。
- **Single-factor or pool-aware:** 单个 alpha 的生成 reward 中包含 novelty vs. known alpha pool，因此是 **standalone predictive reward + pool-aware novelty reward** 的混合；组合阶段显式维护 alpha pool。
- **Discovery-only or discovery-plus-deployment:** discovery-plus-deployment。论文不仅生成 alpha，还使用 AlphaForge 风格的动态 re-selection + linear regression 组合，形成 Mega-Alpha 并做 portfolio backtest。

### 与 AlphaGen 的核心差异

AlphaGen 的典型目标是：

\[
R(\alpha\mid F)=IC(c(X;F\cup\{\alpha\}))-IC(c(X;F)),
\]

也就是新 alpha 对现有 pool 的边际贡献。AlphaSAGE 认为这种做法有两个问题：

1. reward 随着 pool \(F_t\) 改变而变化，导致非平稳；
2. 策略容易沿着一个 greedy construction path 走向少数模式，缺少多样性。

AlphaSAGE 改成学习全局分布：

\[
P_\theta(\alpha)\propto R(\alpha), \quad \alpha\in \mathcal X,
\]

目的不是找唯一最优公式，而是覆盖多个高 reward 模式。

---

## 3. MDP Design

AlphaSAGE 虽然使用 GFlowNet，而不是传统 PPO / DQN，但仍然可以理解为一个序列决策过程。每一条 trajectory 对应从空表达式树逐步生成完整 alpha 的过程。

### State

论文定义状态为 **partially constructed AST**：

\[
s_t = T_t,
\]

其中 \(T_t\) 是当前已经构造到一半的抽象语法树。初始状态：

\[
s_0 = \varnothing,
\]

终止状态是完整且合法的公式树：

\[
s_n = \alpha \in \mathcal X.
\]

在代码伪代码中，每一步会：

1. 将当前 state \(s_t\) parse 成 AST；
2. 用 RGCN 得到 state embedding：
   \[
   e_t=f_{\mathrm{GNN}}(AST_t);
   \]
3. 基于 embedding 输出 action distribution：
   \[
   \pi_\theta(\cdot\mid AST_t).
   \]

因此，state 不是简单的 RPN prefix，而是结构化的 partial tree。

### Action

动作是往 partial AST 的 open leaf node 添加一个 token：

\[
a_t \in \mathcal A.
\]

token 可以是：

- 基础特征：`Open, Close, High, Low, Vwap, Volume`；
- 一元算子：`Abs, Slog1p, Inv, Sign, Log, Rank`；
- 二元算子：`Add, Sub, Mul, Div, Pow, Greater, Less`；
- 时序算子：`Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin, TsMed, TsMad, TsRank, TsDelta, TsDiv, TsPctChange, TsWMA, TsEMA, TsCov, TsCorr`；
- 终止 token：`SEP`。

动作形式可以写成：

\[
a_t = \text{AddToken}(\text{open leaf}, \text{operator/feature/SEP}).
\]

### Transition

状态转移是确定性的 grammar/tree update：

\[
s_{t+1}=T(s_t,a_t).
\]

如果 \(a_t\) 是普通 token，则将该 token 填入当前 open leaf，并根据 token 的 arity 创建新的 open child nodes。

例如：

- 选择 feature：该节点成为叶子，不再需要子节点；
- 选择 unary operator：新增一个 operand child；
- 选择 binary operator：新增两个 operand children；
- 选择 rolling operator：可能需要 feature operand + time window operand。

如果 \(a_t=SEP\)，则尝试 build expression：

\[
\alpha = \mathrm{BuildExpr}(s_t).
\]

若当前树合法，则进入 terminal evaluation；否则通常应被 mask 或视为非法终止。

### Reward

AlphaSAGE 的 reward 是 terminal alpha 的综合 reward：

\[
R(\alpha,T)=R_{IC}(\alpha)+\lambda(T)R_{SA}(\alpha)+\eta(T)R_{NOV}(\alpha).
\]

其中：

1. \(R_{IC}\)：预测能力；
2. \(R_{SA}\)：结构 embedding 与行为输出的一致性；
3. \(R_{NOV}\)：与已有高质量 alpha pool 的低相关 novelty。

虽然 reward 在公式完成后计算，但论文称其是 dense / multi-faceted，因为除了 terminal IC，还使用结构和 novelty 信号，为生成过程提供更丰富的学习信号。严格说，从 MDP 时间步角度看，它仍然主要是 **terminal reward**；从 reward component 角度看，它是 **multi-signal shaped reward**。

### Termination

episode 终止条件：

1. 采样到 `SEP`；
2. 当前 partial AST 已经是合法表达式并通过 early stop 机制终止；
3. 达到最大长度 `MaxLen=20`；
4. 构造出完整合法 AST。

Early stop 概率为：

\[
p_{es}(s_t)=\frac{\mathrm{Len}(s_t)}{\mathrm{MaxLen}}.
\]

论文文字中说：当当前 stack / tree 已经形成合法表达式时，以长度比例概率提前停止。直觉是：

- 越接近最大长度，越倾向于停止；
- 避免公式无限变长；
- 避免到最大长度时被强行截断成非法公式。

### Action Legality / Grammar Constraints

AlphaSAGE 使用 invalid action masking：

\[
\pi_\theta(a\mid s_t)=0,\quad a\notin \mathcal A_{\mathrm{valid}}(s_t).
\]

合法性约束包括：

1. **语法合法性**：operator arity 必须匹配；
2. **树结构合法性**：只能在 open leaf node 添加 token；
3. **终止合法性**：只有当前 AST 可以 build 成完整表达式时才允许 SEP；
4. **长度约束**：不能超过 MaxLen；
5. **关系类型约束**：RGCN 中显式区分 operand 类型，例如 unary operand、commutative operands、non-commutative left/right operands、rolling feature operand、rolling time operand。

紧凑 MDP 表达：

\[
\begin{aligned}
s_t &= T_t \quad \text{partial AST},\\
a_t &\in \mathcal A_{\mathrm{valid}}(T_t),\\
s_{t+1} &= \mathrm{Expand}(T_t,a_t),\\
s_T &= \alpha \in \mathcal X,\\
R_T &= R_{IC}(\alpha)+\lambda(T)R_{SA}(\alpha)+\eta(T)R_{NOV}(\alpha).
\end{aligned}
\]

---

## 4. Representation

## Representation
- **Representation type:** AST / expression tree + RGCN embedding。论文仍然会提到 RPN 作为已有方法的 flatten 表示，但 AlphaSAGE 的核心是结构化 AST。
- **Token/operator set:** 
  - Features: `Open, Close, High, Low, Vwap, Volume`;
  - Unary: `Abs, Slog1p, Inv, Sign, Log, Rank`;
  - Binary: `Add, Sub, Mul, Div, Pow, Greater, Less`;
  - Time-series: `Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin, TsMed, TsMad, TsRank, TsDelta, TsDiv, TsPctChange, TsWMA, TsEMA, TsCov, TsCorr`;
  - Special: `SEP`;
  - Time window token: rolling operator 的 lookback \(d\)，论文中最大长度为 20，具体窗口集合未在正文中完全展开。
- **Validity mechanism:** partial AST expansion + invalid action mask + early stop + MaxLen 约束。
- **Advantages:** 
  1. 保留公式层级结构；
  2. 对 commutative / non-commutative operator 的关系可区分；
  3. 比 LSTM on RPN 更能表达公式语义；
  4. 有助于定义结构-aware reward。
- **Limitations:** 
  1. AST 结构相似不一定代表经济含义相似；
  2. RGCN embedding 的质量依赖 operator relation type 设计；
  3. 图编码增加训练复杂度；
  4. 若不同公式数学等价但 AST 不同，仍可能无法完全识别等价性。

### RGCN 表示

每个 alpha 公式 \(\alpha\) 被 parse 成 AST：

\[
T_\alpha=(V_\alpha,E_\alpha),
\]

其中 \(V_\alpha\) 是 features/operators/time windows 节点，\(E_\alpha\) 是表达式计算依赖边。

RGCN 更新为：

\[
h_v^{(l)}
=
\mathrm{ReLU}
\left(
\sum_{r\in \mathcal R}
\sum_{u\in N_r(v)}
\frac{1}{c_{v,r}}W_r^{(l)}h_u^{(l-1)}
+
W_0^{(l)}h_v^{(l-1)}
\right).
\]

最终 graph-level alpha embedding：

\[
e_\alpha=\mathrm{MaxPooling}\left(\{h_v^{(L)}\}_{v\in V_\alpha}\right).
\]

### RGCN 的 relation types

论文附录列出 6 类边关系：

1. unary operator with operand；
2. commutative operator with operands；
3. non-commutative operator with left operand；
4. non-commutative operator with right operand；
5. rolling operator with feature operand；
6. rolling operator with time operand。

这很重要，因为：

- `Add(a,b)` 和 `Add(b,a)` 在数学上等价，属于 commutative；
- `Sub(a,b)` 和 `Sub(b,a)` 不等价，必须区分 left/right；
- `TsMean(close,10d)` 中 `close` 和 `10d` 的语义不同，必须区分 feature operand 和 time operand。

---

## 5. Algorithm Mechanism

## Algorithm Mechanism
- **RL/search method:** Generative Flow Networks, GFlowNets。
- **On-policy or off-policy:** 更接近 on-policy generative training；它从当前 forward policy 采样 trajectory，并用 Trajectory Balance loss 更新。不是传统 RL 的 policy gradient / actor-critic，也不是 off-policy Q-learning。
- **Learned components:**
  1. forward policy \(P_F(s_{t+1}\mid s_t;\theta)\)；
  2. backward policy \(P_B(s_t\mid s_{t+1};\theta)\)；
  3. RGCN state/alpha encoder；
  4. learnable partition scalar \(Z_\theta\)；
  5. action logits / masked action distribution。
- **Search component:** 没有显式 MCTS / beam search。它是通过 GFlowNet 学习一个覆盖多峰高 reward 解的生成分布。
- **Training loop:**
  1. 从空 AST 开始；
  2. RGCN 编码当前 partial AST；
  3. 输出 masked action distribution；
  4. 采样 token 并扩展 AST；
  5. 若触发 SEP，则构造 alpha；
  6. 计算 alpha 输出 \(z=\alpha(X)\)；
  7. 计算 \(R_{IC},R_{NOV},R_{SA}\)；
  8. 合成总 reward；
  9. 若通过阈值，则加入 alpha pool；
  10. 用 Trajectory Balance loss + entropy regularization 更新模型。
- **Key update objective:**
  \[
  L_{\mathrm{final}}
  =
  \mathbb E_{\tau\sim P_F(\tau;\theta)}
  [L_{\mathrm{TB}}(\tau)]
  +
  \beta L_{\mathrm{ENT}}.
  \]

### GFlowNet 目标

GFlowNet 的目标不是：

\[
\max_\theta \mathbb E_{\alpha\sim \pi_\theta}[R(\alpha)],
\]

而是学习：

\[
P_\theta(\alpha)=\frac{R(\alpha)}{Z},
\quad
Z=\sum_{\alpha'\in \mathcal X}R(\alpha').
\]

这意味着如果两个 alpha 都有较高 reward，模型应当都能采样到，而不是只把概率压到最高 reward 的那一个 alpha 上。

### Trajectory Balance Loss

一条完整 trajectory：

\[
\tau=(s_0\to s_1\to \cdots \to s_n=\alpha).
\]

TB loss：

\[
L_{\mathrm{TB}}(\tau)
=
\left(
\log Z_\theta
+
\sum_{t=1}^{n}\log P_F(s_t\mid s_{t-1};\theta)
-
\log R(s_n)
-
\sum_{t=1}^{n}\log P_B(s_{t-1}\mid s_t;\theta)
\right)^2.
\]

理解方式：

- forward policy 表示“怎么生成公式”；
- backward policy 表示“怎么从公式反推生成路径”；
- \(Z_\theta\) 是归一化常数；
- 当 TB loss 最小时，forward 生成完整 alpha 的概率与 reward 成正比。

### Entropy Regularization

论文还加入 action-level entropy：

\[
L_{\mathrm{ENT}}
=
-\mathbb E_{\tau\sim P_F(\tau;\theta)}
\left[
\sum_{t=0}^{n-1}H(\pi_\theta(\cdot\mid s_t))
\right].
\]

最终目标：

\[
L_{\mathrm{final}}=
\mathbb E_\tau[L_{\mathrm{TB}}(\tau)]
+\beta L_{\mathrm{ENT}}.
\]

因为 \(L_{\mathrm{ENT}}\) 前面是负 entropy，所以最小化时等价于鼓励更高 entropy，防止动作分布过早坍缩到少数 token。

---

## 6. Reward Design

## Reward Design
- **Reward signal:** 
  \[
  R(\alpha,T)=R_{IC}(\alpha)+\lambda(T)R_{SA}(\alpha)+\eta(T)R_{NOV}(\alpha).
  \]
- **Terminal or intermediate:** 主要是 terminal reward；但 reward components 比单一 IC 更丰富，属于 shaped / multi-faceted reward。
- **Pool-aware or standalone:** \(R_{IC}\) 是 standalone；\(R_{NOV}\) 对已有 known alpha library / alpha pool 做去相关，因此是 pool-aware。
- **Risk-aware or uncertainty-aware:** 不像 AlphaQCM 那样显式建模 return distribution / quantile uncertainty；主要是 diversity-aware 和 structure-aware，不是 distributional risk-aware。
- **Shaping mechanism:** structure-aware reward + novelty reward + entropy regularization + time-dependent annealing。
- **Possible reward bias:** reward scalarization 人为设定；IC 绝对值可能忽略方向稳定性；novelty 用 alpha-output IC 衡量，可能惩罚经济上相似但组合有效的 alpha，也可能放过非线性冗余。

### 6.1 Terminal Performance Reward: \(R_{IC}\)

论文使用 Information Coefficient 的绝对值：

\[
R_{IC}(\alpha)
=
IC(\alpha,y)
=
\left|
\mathbb E_d
\left[
\frac{
\mathrm{Cov}(\alpha(X_d),y_d)
}{
\sqrt{\mathrm{Var}(\alpha(X_d))\cdot \mathrm{Var}(y_d)}
}
\right]
\right|.
\]

其中：

- \(X_d\in \mathbb R^{N\times M}\)：第 \(d\) 天所有股票的特征；
- \(\alpha(X_d)\in \mathbb R^N\)：alpha 在横截面上的取值；
- \(y_d\in \mathbb R^N\)：未来收益；
- 每天计算横截面相关，再对时间求均值。

注意它取绝对值，因此正向 alpha 和反向 alpha 都可以被视为有效。实际组合时可以翻转符号。

### 6.2 Structure-Aware Reward: \(R_{SA}\)

目标：让结构 embedding 相近的 alpha，其行为输出也相近。

对 alpha \(\alpha_i\)，令：

\[
Z_i\in\mathbb R^{D\times N}
\]

表示它在 \(D\) 天、\(N\) 个股票上的横截面归一化输出。

两个 alpha 的行为距离：

\[
d_{\mathrm{behav}}(\alpha_i,\alpha_j)
=
\frac{1}{D}
\sum_{d=1}^{D}
(Z_i(d)-Z_j(d))^2.
\]

在 embedding 空间中，取 \(\alpha_i\) 的 KNN：

\[
N_K(\alpha_i).
\]

根据 embedding 距离定义权重：

\[
w_{ij}
=
\frac{
\exp(-\|e_{\alpha_i}-e_{\alpha_j}\|^2)
}{
\sum_{k\in N_K(\alpha_i)}
\exp(-\|e_{\alpha_i}-e_{\alpha_k}\|^2)
}.
\]

结构-aware reward：

\[
R_{SA}(\alpha_i)
=
\exp
\left(
-
\sum_{j\in N_K(\alpha_i)}
w_{ij}\cdot d_{\mathrm{behav}}(\alpha_i,\alpha_j)
\right).
\]

直观理解：

- 如果 embedding 空间认为两个公式接近；
- 它们实际输出行为也接近；
- 那么 \(R_{SA}\) 高；
- 这会鼓励 RGCN embedding 学到“结构-行为一致”的表示。

这部分本质上有一点 self-supervised / representation alignment 的味道。

### 6.3 Novelty Reward: \(R_{NOV}\)

\[
R_{NOV}(\alpha)
=
1-
\max_{\alpha'\in F_{\mathrm{known}}}
|IC(\alpha,\alpha')|.
\]

其中 \(IC(\alpha,\alpha')\) 是两个 alpha 输出之间的相关性。

含义：

- 如果新 alpha 和已有高质量 alpha 高度相关，则 novelty 低；
- 如果新 alpha 与已有 alpha 低相关，则 novelty 高；
- 这直接服务于 alpha pool 的多样性和组合稳定性。

从量化角度看，这和多因子组合中的 multicollinearity 控制一致：高相关 alpha 会导致线性组合权重不稳定，降低解释性和稳健性。

### 6.4 Time-dependent Reward Weight

\[
R(\alpha,T)
=
R_{IC}(\alpha)
+
\lambda(T)R_{SA}(\alpha)
+
\eta(T)R_{NOV}(\alpha).
\]

其中：

\[
\lambda(T)=\left(1-\frac{T}{T_{\mathrm{anneal}}}\right)\lambda_{\max},
\]

\[
\eta(T)=\left(1-\frac{T}{T_{\mathrm{anneal}}}\right)\eta_{\max}.
\]

这表示训练早期更强调 structure-aware / novelty，引导探索；训练后期逐渐回到 predictive performance。

### 6.5 Reward Design 的评价

AlphaSAGE 的 reward 设计解决的是三个问题：

1. **IC-only reward 太 sparse / 太单一**；
2. **RL policy 容易 mode collapse**；
3. **formula representation 和 alpha behavior 脱节**。

但它也有明显局限：

1. \(R_{IC},R_{SA},R_{NOV}\) 的加权是人工 scalarization；
2. \(R_{SA}\) 的合理性取决于 embedding 是否足够表达金融语义；
3. novelty 用线性相关衡量，无法完全识别非线性冗余；
4. reward 没有显式加入 turnover、transaction cost、capacity、risk exposure；
5. \(R_{IC}\) 是训练集 / 验证集 evaluator，仍可能发生 evaluator overfitting。

---

## 7. Factor Pool and Combination

## Factor Pool and Combination
- **Pool maintained:** 是。算法维护 alpha pool \(F\)、probability buffer \(B_{\mathrm{prob}}\)、embedding buffer \(B_{\mathrm{emb}}\)。
- **Pool update rule:** 当生成 alpha 的 total reward 通过阈值 `PassThreshold(R)` 后，加入 pool：
  \[
  F\leftarrow F\cup\{\alpha\}.
  \]
- **Factor selection/removal:** 论文正文说组合阶段使用 AlphaForge 风格的 dynamic re-selection，筛选 recently effective alphas，丢弃 stale / redundant signals。但具体 selection 阈值和 removal 细节主要依赖实现或 AlphaForge 方法。
- **Combination model:** simple linear regression。
- **Static or dynamic weights:** dynamic weights。每个 period 重新筛选并重估线性权重。
- **Deployment realism:** 比只报告单因子 IC 更接近实际组合，但仍缺少充分的 transaction cost、turnover、capacity、short-sale constraint 等细节。

组合形式：

\[
S_t
=
\sum_{k=1}^{K_t}
w_t^{(k)} f_t^{(k)}.
\]

其中：

- \(f_t^{(k)}\)：第 \(k\) 个被选中的 alpha 在 \(t\) 时刻的横截面信号；
- \(w_t^{(k)}\)：动态线性回归得到的权重；
- \(K_t\)：当前 period 被选中的 alpha 数量。

这与 AlphaGen 的不同是：

- AlphaGen 更强调生成阶段的 pool-aware marginal contribution；
- AlphaSAGE 更强调生成阶段的 diversity-aware distribution learning，组合阶段再用动态线性模型处理 regime shift。

---

## 8. Experiments and Evaluation

## Experiments and Evaluation
- **Dataset:** 中国市场 CSI300、CSI500；美国市场 S&P500。
- **Universe:** CSI300、CSI500、S&P500 成分股。
- **Time split:**
  - China:
    - train: 2010-01-01 至 2020-12-31；
    - validation: 2021-01-01 至 2021-12-31；
    - test: 2022-01-01 至 2024-12-31。
  - U.S.:
    - train: 2010-01-01 至 2016-12-31；
    - validation: 2017-01-01 至 2017-12-31；
    - test: 2018-01-01 至 2020-12-31。
- **Prediction target:** 未来收益 \(y_d\)。附录 backtest 设置显示中国市场和美国市场都使用持有 20 天的交易规则，因此更接近中频 / 日频 alpha。
- **Metrics:**
  - Correlation metrics: IC, ICIR, RankIC/RIC, RankICIR/RICIR；
  - Portfolio metrics: Annualized Return, Maximum Drawdown, Sharpe Ratio。
- **Baselines:**
  - MLP；
  - LightGBM；
  - XGBoost；
  - GP；
  - AlphaGen；
  - AlphaQCM；
  - AlphaForge。
- **Transaction costs:** 正文和附录指标定义中没有明确看到 transaction cost-adjusted return 的详细设置。需要在 review 中标注为“不充分明确”。
- **Turnover:** 没有系统报告 turnover。
- **Random seeds:** 论文表格没有清晰报告多 seed 均值 / 方差。
- **Robustness tests:** 有 ablation study、sensitivity analysis、training step comparison、candidate-pool size analysis；也有 CSI300/CSI500/S&P500 跨市场测试。
- **Code availability:** 论文声称代码可用。
- **Backtest protocol:**
  - CSI300/CSI500：每天买入 top 20% 股票，持有 20 天，long-only；
  - S&P500：每天买入 top 10%，卖空 bottom 10%，持有 20 天，long-short。

### 8.1 Main Results

论文报告 AlphaSAGE 在三个市场上总体最好：

- CSI300:
  - IC = 0.079；
  - ICIR = 0.496；
  - RIC = 0.094；
  - RICIR = 0.583；
  - AR = 7.62%；
  - MDD = -17.3%；
  - SR = 1.71。
- CSI500:
  - IC = 0.054；
  - ICIR = 0.379；
  - RIC = 0.084；
  - RICIR = 0.637；
  - AR = 5.53%；
  - MDD = -16.0%；
  - SR = 1.20。
- S&P500:
  - IC = 0.052；
  - ICIR = 0.493；
  - RIC = 0.038；
  - RICIR = 0.382；
  - AR = 19.47%；
  - MDD = -4.2%；
  - SR = 6.32。

### 8.2 Ablation Study

Ablation 在 CSI300 上做，结论：

1. base GFlowNet 已经有一定效果；
2. 只加 Early Stop 反而变差，说明 ES 需要强表示模型配合；
3. 加 GNN 后提升最大，说明结构表示是关键；
4. 加 SA reward 后 RankIC 稳定性和 drawdown 有改善；
5. 加 NOV 后信号质量和 tradability 继续提高；
6. 加 ENT 后整体最好，说明 entropy regularization 改善探索。

论文报告的 full model：

\[
IC=0.079,\quad ICIR=0.496,\quad RIC=0.094,\quad RICIR=0.583,\quad AR=7.62\%,\quad MDD=-17.3\%,\quad SR=1.71.
\]

### 8.3 Sensitivity Analysis

论文分析 novelty weight \(R_{NOV}\) 和 structure-aware weight \(R_{SA}\)：

- novelty weight 在小到中等范围提升明显，过大后收益趋缓或下降；
- \(R_{SA}\) 权重提升时，多个指标大体稳定改善；
- 没有出现 abrupt performance collapse，论文据此称模型对超参数较稳健。

### 8.4 Evaluation Critique

从 review 角度，应注意：

1. **缺少 turnover 和成本细节**：组合收益指标如果没有成本调整，在中频调仓下可能偏乐观。
2. **S&P500 时间截止到 2020**：论文解释是数据源限制，但这导致美国市场测试不覆盖 2021-2024 的疫情后和加息周期。
3. **没有多随机种子统计**：RL/GFlowNet 结果可能有较大随机性，缺少均值/方差会影响可信度。
4. **与 AlphaForge 的组合阶段关系需要拆清楚**：AlphaSAGE 的 portfolio gain 可能来自生成器，也可能部分来自 AlphaForge-style dynamic combiner。
5. **IC 取绝对值**：有利于评估 alpha 的信息强度，但实际交易方向和稳定性仍需验证。

---

## 9. Main Contributions

## Main Contributions

1. **The paper contributes structure-aware alpha representation by replacing flattened RPN/LSTM encoding with AST + RGCN, addressing the structural underrepresentation problem in formulaic alpha mining.**

   过去很多方法把公式写成 token 序列，例如 RPN，然后用 LSTM 编码。但公式本质上是树结构，operator 的层级、左右顺序、rolling window 语义都很重要。AlphaSAGE 用 AST + RGCN 显式表达这些关系。

2. **The paper contributes a diversity-seeking GFlowNet generator by learning \(P_\theta(\alpha)\propto R(\alpha)\), addressing mode collapse in standard expected-return-maximizing RL.**

   PPO/REINFORCE 类方法通常最大化期望 reward，容易把概率集中到少数高 reward 模式。GFlowNet 目标是按 reward 比例采样，因此更适合生成一组多样化 alpha。

3. **The paper contributes a multi-faceted reward design by combining predictive IC, structure-behavior alignment, novelty, and entropy regularization, addressing sparse and single-signal reward in alpha generation.**

   它不是只用 terminal IC，而是加入结构一致性和 pool novelty，从而给生成器更稳定的学习信号，并显式鼓励低相关 alpha。

4. **The paper contributes a discovery-to-combination pipeline by combining generated alphas through dynamic re-selection and linear regression, addressing alpha decay and regime shift at deployment stage.**

   这使论文不只是 formula discovery，而是更接近完整 alpha mining + alpha deployment workflow。

---

## 10. Limitations

## Limitations

### 10.1 Algorithmic limitation

1. **Reward 仍然主要在公式完成后计算，真正的 step-level dense reward 仍不足。**

   虽然 \(R_{SA}\)、\(R_{NOV}\) 增加了 reward 信息，但 reward 还是在 terminal alpha 构造完成后计算。对 partial AST 的中间状态没有直接 ground-truth reward。

2. **Reward scalarization 依赖人工超参数。**

   \[
   R=R_{IC}+\lambda R_{SA}+\eta R_{NOV}
   \]

   这种加权形式简单有效，但 \(\lambda,\eta,\beta\) 的选择具有经验性。不同市场、频率、数据源下可能需要重新调参。

3. **GFlowNet 训练复杂度较高。**

   相比 PPO/RNN 生成器，AlphaSAGE 多了：

   - AST parser；
   - RGCN encoder；
   - forward/backward policy；
   - trajectory balance；
   - KNN embedding；
   - novelty pool；
   - entropy regularization。

   实现和调试成本更高。

4. **RGCN 表示不等于金融语义理解。**

   AST 结构相近不一定代表 alpha 经济逻辑相近；AST 结构不同也可能计算行为高度相似。论文用 \(R_{SA}\) 缓解这一点，但不能完全解决公式等价性和经济语义问题。

5. **Novelty reward 可能与 performance reward 冲突。**

   过强 novelty 会鼓励“为了不同而不同”的公式，导致 IC 下降；论文 sensitivity 中也显示 novelty 过大后可能 taper。

### 10.2 Evaluation / finance limitation

1. **交易成本和 turnover 没有充分报告。**

   对公式 alpha 来说，尤其有 rolling / rank / short-term operators 时，turnover 可能较高。如果没有成本调整，portfolio metrics 可能偏乐观。

2. **没有容量 / liquidity / impact 分析。**

   CSI300/CSI500 和 S&P500 流动性较好，但策略容量仍取决于持仓集中度、换手率和交易冲击。

3. **美国市场测试截止 2020 年，时间覆盖不足。**

   没有覆盖 2021-2024 的重要 regime，例如高通胀、加息、AI bull market 等。

4. **缺少多随机种子报告。**

   GFlowNet/RL 方法受随机种子影响较大。单次表格结果可能无法体现训练稳定性。

5. **组合阶段借鉴 AlphaForge，使生成器贡献和组合器贡献难以完全分离。**

   如果 AlphaSAGE + dynamic combiner 优于 baselines，需要进一步确认收益主要来自 GFlowNet/RGCN，而不是动态组合机制。

6. **IC 使用绝对值可能掩盖方向稳定性。**

   真实交易中 alpha 方向是否稳定、是否需要频繁翻转，是部署时很重要的问题。

---

## 11. Relation to AlphaGen and Existing Literature

## Relation to Existing Literature

- **Relation to AlphaGen:**
  - AlphaGen：PPO / RL sequential expression generation，reward 是新 alpha 对当前 alpha pool 的边际贡献；
  - AlphaSAGE：GFlowNet generative distribution learning，reward 是 IC + structure-aware + novelty，目标是覆盖多个高 reward alpha 模式。
- **Main bottleneck addressed:**
  1. AlphaGen / PPO 的 reward sparsity；
  2. RPN/LSTM 表示无法保留树结构；
  3. expected reward maximization 导致生成分布 mode collapse；
  4. alpha pool 需要低相关、多样化候选。
- **Paradigm classification:**
  1. GFlowNet or diversity-first symbolic generation；
  2. Search-enhanced / tree-structured exploration；
  3. Reward shaping and variance reduction；
  4. Dynamic factor combination；
  5. Structure-aware representation learning。
- **Difference from prior work:**
  - vs. AlphaGen：从 PPO 目标换成 GFlowNet 目标；从 pool marginal reward 换成 global reward distribution；
  - vs. AlphaQCM：AlphaQCM 强调 distributional RL / uncertainty；AlphaSAGE 强调 diversity sampling + AST/RGCN；
  - vs. AlphaForge：AlphaForge 强调生成-预测模块与动态组合；AlphaSAGE 在生成阶段加入 GFlowNet 和结构-aware 表示；
  - vs. GP/AutoAlpha：AlphaSAGE 是神经生成分布，不是 mutation/crossover 主导；
  - vs. LLM/MCTS alpha mining：AlphaSAGE 更像 neural symbolic generator，而不是语言模型推理或 tree search。
- **Whether it is more RL-like or search-like:**
  AlphaSAGE 介于 RL 和 generative search 之间。它不是传统最大化期望 reward 的 RL，而是用 GFlowNet 学习 reward-proportional sampling distribution。更准确地说，它属于 **diversity-first generative policy learning for symbolic search**。

### 与 AlphaGen 的对比表

| Dimension | AlphaGen | AlphaSAGE |
|---|---|---|
| 生成对象 | RPN / formulaic alpha | AST / formulaic alpha |
| 表示模型 | 通常为 LSTM/sequence encoder | RGCN over AST |
| RL 机制 | PPO / policy gradient | GFlowNet / trajectory balance |
| 目标 | 最大化 pool marginal contribution | 学习 \(P(\alpha)\propto R(\alpha)\) |
| Reward | downstream combiner improvement | IC + SA + novelty |
| Diversity | 间接依赖 pool reward | 显式 novelty + GFlowNet mode coverage |
| Pool | 强 pool-aware | 生成阶段 novelty-aware，组合阶段 dynamic pool |
| 组合 | synergistic alpha pool | AlphaForge-style dynamic linear combiner |
| 主要优点 | 直接优化组合效果 | 多样性、结构表示、探索稳健性 |
| 主要问题 | reward 非平稳、mode collapse | reward scalarization、实现复杂、成本细节不足 |

---

## 12. How to Incorporate This Paper into My Review

### Suggested section
- **Section:** “Diversity-first and Structure-aware RL for Formulaic Alpha Mining” 或 “Beyond PPO: GFlowNet-based Alpha Generation”
- **Reason:** AlphaSAGE 是目前 RL alpha mining 里很有代表性的范式转移：从 expected-return-maximizing RL 转向 reward-proportional generative modeling，并将 formula representation 从序列扩展到 AST graph。

### Suggested one-line description

```text
AlphaSAGE replaces PPO-style single-mode formula generation with a GFlowNet over ASTs, using RGCN structure encoding and IC/novelty/structure-aware rewards to sample diverse high-quality alpha formulas.
```

### Suggested paragraph

AlphaSAGE is a representative diversity-first framework for formulaic alpha mining. Unlike AlphaGen, which formulates alpha generation as a PPO-style sequential decision process with pool-level marginal contribution rewards, AlphaSAGE learns a GFlowNet distribution over alpha formulas such that the sampling probability is proportional to a multi-faceted reward. Its key design is to represent formulas as ASTs and encode them with an RGCN, thereby preserving operator hierarchy and relation types that are lost in flattened RPN sequences. The reward combines predictive IC, structure-behavior alignment, novelty against known alphas, and entropy regularization, aiming to alleviate sparse rewards and mode collapse. Empirically, the paper reports improved IC/RankIC and portfolio metrics across CSI300, CSI500, and S&P500, although transaction costs, turnover, capacity, and multi-seed robustness remain under-specified.

### Suggested comparison table row

| Paper | Venue | Core formulation | Representation | RL/Search mechanism | Reward type | Pool-aware? | Dynamic allocation? | Key contribution | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| AlphaSAGE | ICLR 2026 / arXiv 2025 | Learn \(P_\theta(\alpha)\propto R(\alpha)\) over formulaic alphas | AST + RGCN | GFlowNet with Trajectory Balance | IC + structure-aware reward + novelty + entropy | Partly; novelty vs. known pool | Yes; AlphaForge-style dynamic linear combiner | Diversity-first structure-aware alpha generation | Reward scalarization, complex implementation, limited cost/turnover reporting |

---

## 13. My Overall Assessment

AlphaSAGE 主要价值在 **methodology 和 future research inspiration**，其次是 implementation。

### 13.1 Methodology Value

它提出了一个很清晰的新方向：

\[
\text{RL alpha mining} \quad \rightarrow \quad \text{GFlowNet alpha distribution learning}.
\]

这对你的 RL for alpha review 很重要，因为它说明“因子挖掘”不一定要用 PPO/REINFORCE 去找一个最高 reward policy，也可以用 GFlowNet 去覆盖多个高 reward 模式。这和真实量化研究的需求更一致：我们不是只需要一个最强单因子，而是需要一批低相关、可组合、可替换的 alpha。

### 13.2 Implementation Value

它的伪代码已经相对清晰，可以拆成几个模块：

```text
ASTEnv / FormulaBuilder
RGCNEncoder
GFlowNetForwardPolicy
BackwardPolicy
RewardEvaluator
NoveltyPool
TrajectoryBalanceLoss
DynamicLinearCombiner
```

如果你之后想在 AlphaGen 上改造，可以考虑：

1. 保留 AlphaGen 的 expression grammar；
2. 把 RPN prefix state 转成 AST state；
3. 用 GNN/RGCN 替代 LSTMSharedNet；
4. 将 PPO objective 替换或并行加入 GFlowNet TB loss；
5. 将 reward 从 single IC / pool marginal IC 扩展为：
   \[
   R=\lambda_1 IC+\lambda_2\text{novelty}+\lambda_3\text{turnover penalty}+\lambda_4\text{OOS stability}.
   \]

### 13.3 Evaluation Value

论文实验看起来覆盖了中国和美国市场，也做了 ablation 和 sensitivity。但从严谨量化研究角度，仍然需要补充：

1. transaction cost-adjusted return；
2. turnover；
3. capacity；
4. multiple seeds；
5. post-2020 U.S. market validation；
6. alpha pool 中每个 alpha 的相关矩阵和稳定性；
7. 不同 market regime 下的分段表现；
8. 与 AlphaForge dynamic combiner 的贡献拆分。

### 13.4 对你的 review 的定位建议

我建议把 AlphaSAGE 放在 review 中的一个单独小节：

```text
Diversity-first Alpha Generation: From Policy Optimization to Reward-Proportional Sampling
```

它可以作为 AlphaGen 之后的一个重要演化方向：

```text
AlphaGen: PPO + pool-aware marginal reward
AlphaQCM: distributional RL + uncertainty-aware exploration
QuantFactor REINFORCE: variance-bounded policy gradient
AlphaSAGE: GFlowNet + AST/RGCN + diversity-aware reward
```

如果你的 review 想形成主线，可以这样写：

> Existing RL alpha mining methods mainly optimize expected rewards through policy gradients, which may suffer from sparse terminal feedback and mode collapse. AlphaSAGE reframes formulaic alpha mining as reward-proportional generative modeling with GFlowNets, making diversity an intrinsic property of the learning objective rather than an auxiliary post-processing criterion.

---

# Appendix: Core Algorithm in My Own Words

AlphaSAGE 的训练可以写成如下流程：

```text
Initialize GFlowNet parameters θ
Initialize alpha pool F = ∅
Initialize embedding buffer B_emb = ∅

For each training episode:
    s0 = empty AST

    For t = 0,1,...,MaxLen:
        Encode partial AST st with RGCN:
            et = f_RGCN(st)

        Compute masked action distribution:
            πθ(a | st), a ∈ A_valid(st)

        If current AST is valid:
            with probability Len(st)/MaxLen choose SEP

        Otherwise sample next action:
            at ~ πθ(. | st)

        If at is not SEP:
            st+1 = Expand(st, at)

        If at is SEP:
            α = BuildExpr(st)
            z = ComputeAlpha(α, X)

            R_IC = IC(z, y)
            R_NOV = 1 - max correlation with known alpha pool
            R_SA = structure-behavior alignment reward
            R = R_IC + λ(T)R_SA + η(T)R_NOV

            If R passes threshold:
                F = F ∪ {α}
                store embedding eα

            Compute trajectory balance loss:
                L_TB = (logZ + log forward path - logR - log backward path)^2

            Add entropy regularization:
                L_final = L_TB + βL_ENT

            Update θ
            Reset state to empty AST
```

最关键的一句话：

**PPO 类方法学的是“下一步怎么选才能最大化期望回报”；AlphaSAGE/GFlowNet 学的是“怎样按 reward 比例采样整个公式空间中的多个高质量解”。**

---

# Appendix: Four Core Questions

## Q1. How does it define factor mining as a decision problem?

它将 alpha mining 定义为在 AST 状态空间中的逐步构造问题：

\[
s_0=\varnothing,\quad s_{t+1}=\mathrm{Expand}(s_t,a_t),\quad s_T=\alpha.
\]

动作是向 partial AST 的 open leaf 添加 feature/operator/SEP。终止后得到完整公式 alpha，并计算 reward。

## Q2. What exactly does the RL component learn?

严格说，AlphaSAGE 的 RL/generative component 学的是：

\[
P_F(s_{t+1}\mid s_t;\theta)
\]

和

\[
P_B(s_t\mid s_{t+1};\theta),
\]

使得终止 alpha 的采样概率满足：

\[
P_\theta(\alpha)\propto R(\alpha).
\]

所以它学的不是 value function，也不是 Q function，而是一个 reward-proportional formula generator。

## Q3. What weakness of AlphaGen or previous methods does it address?

它主要解决：

1. PPO / REINFORCE 的 sparse terminal reward；
2. RPN/LSTM 的结构表达不足；
3. expected reward maximization 的 mode collapse；
4. alpha pool 需要 low-correlation diversity，但传统 RL 目标不直接鼓励 diversity；
5. AlphaGen 的 pool marginal reward 非平稳问题。

## Q4. Does the experimental protocol support the paper's conclusion?

部分支持。

支持之处：

1. 覆盖 CSI300、CSI500、S&P500；
2. baselines 包括 ML、GP、AlphaGen、AlphaQCM、AlphaForge；
3. ablation 显示 GNN、SA、NOV、ENT 都有增益；
4. sensitivity 显示超参数不太脆弱；
5. portfolio metrics 和 correlation metrics 都有提升。

不足之处：

1. 缺少 transaction cost 和 turnover 的系统报告；
2. 缺少多 seed 统计；
3. S&P500 只到 2020；
4. dynamic combiner 借鉴 AlphaForge，生成器贡献需要更干净的拆分；
5. IC absolute value 和 long-only / long-short 差异可能影响结果解释。

因此，结论可以写成：

**AlphaSAGE 的实验结果支持其在结构表示、多样性探索和综合指标上的优势，但还不足以完全证明其在真实交易部署中的稳健性。**

---

## 14. 补充理解：为什么 AlphaSAGE 要从 RPN/LSTM 转向 AST/GNN？

论文中有一段话指出：已有方法往往依赖 sequential encoder，例如 LSTM，并且输入是 flattened representation，例如 Reverse Polish Notation, RPN。AlphaSAGE 认为这种做法有一个根本问题：公式本质上不是普通的一维文本序列，而是有层级结构的数学表达式。

### 14.1 这段话的核心含义

这段话主要是在解释：

> 以前很多方法把公式看成一串 token 序列，例如 RPN 序列，然后用 LSTM 读这个序列；但公式本质上是有树结构的数学表达式。只看序列会丢掉公式的层级结构，甚至会把数学上等价的公式看成不同东西。

例如：

\[
close + open
\]

和

\[
open + close
\]

从数学上看，因为加法满足交换律：

\[
close + open = open + close
\]

它们是等价的。

但是如果把它们展开成序列，可能分别是：

```text
close open +
```

和

```text
open close +
```

这两个序列是不一样的。LSTM 看到的 token 顺序不同，就可能认为它们是两个不同模式。因此，论文认为 sequence encoder 可能没有真正理解公式的数学结构。

### 14.2 AST 表示在做什么？

AlphaSAGE 的做法是把每个公式 alpha \(\alpha\) 解析成一棵 AST，即 Abstract Syntax Tree，抽象语法树：

\[
\alpha \rightarrow T_\alpha=(V_\alpha,E_\alpha)
\]

其中：

- \(V_\alpha\)：节点集合，包含操作符和变量，例如 `Add`, `Log`, `TsStd`, `close`, `open`, `low`, `10d`；
- \(E_\alpha\)：边集合，表示计算层级关系，即谁是谁的输入、哪个 operator 作用在哪个 feature 上。

例如公式：

\[
\log(\mathrm{TsStd}(low,10d))
\]

对应的树结构大致是：

```text
        Log
         |
      TsStd
      /    \
    low    10d
```

这里 `Log` 是最外层操作，`TsStd` 是它的输入，`low` 和 `10d` 是 `TsStd` 的两个输入。相比 RPN 序列：

```text
low 10d TsStd Log
```

AST 更明确地保留了“谁作用于谁”的计算层级。

### 14.3 为什么说 AST/GNN 更适合公式 alpha？

公式 alpha 的语义由计算结构决定，而不是简单由 token 的线性顺序决定。比如：

\[
\mathrm{TsStd}(low,10d)
\]

中的 `10d` 是 rolling window 参数，而不是普通变量；`low` 是被 rolling operator 作用的 market feature；`Log` 是对整个 `TsStd` 结果再做非线性变换。AST 可以把这些层级关系保留下来。

AlphaSAGE 进一步用 RGCN 编码 AST，是因为 AST 中不同边有不同语义。例如：

1. unary operator 与 operand 的关系；
2. commutative operator 与 operands 的关系；
3. non-commutative operator 的左操作数关系；
4. non-commutative operator 的右操作数关系；
5. rolling operator 与 feature operand 的关系；
6. rolling operator 与 time operand 的关系。

这些关系如果全部压平成 RPN 序列，就很难显式区分。RGCN 的优势是可以对不同关系类型使用不同参数，从而更细致地表达公式结构。

### 14.4 对最后一句话的理解

原文说：

> This representation is invariant to semantically inconsequential syntactic variations.

意思是：

> AST 表示可以减少一些“语法上不同但语义上没区别”的变化带来的干扰。

比如：

\[
close + open
\]

和

\[
open + close
\]

虽然写法顺序不同，但计算含义一样。论文希望通过树结构和结构编码，让模型更关注公式的真实计算结构，而不是被 token 顺序误导。

不过需要注意：普通 AST 本身不一定天然完全解决交换律等价问题。如果 AST 严格保留左右子节点，那么 `Add(close, open)` 和 `Add(open, close)` 仍然可能是两棵顺序不同的树。论文的说法更准确地理解为：相比 RPN + LSTM，AST + GNN/RGCN 更容易表达层级结构和关系类型，也更有机会学习到这类语义等价或近似等价。

### 14.5 对 AlphaSAGE 方法设计的启发

这部分可以总结为：

\[
\text{RPN + LSTM} \Rightarrow \text{把公式当作一维序列}
\]

\[
\text{AST + RGCN} \Rightarrow \text{把公式当作带关系类型的计算图}
\]

因此 AlphaSAGE 的结构创新不是单纯换了一个 encoder，而是改变了公式 alpha 的建模对象：

- AlphaGen 一类方法更像是在 token 序列空间中搜索合法表达式；
- AlphaSAGE 更像是在 AST 结构空间中生成和编码表达式；
- RGCN 提供了结构 inductive bias；
- GFlowNet 提供了 diversity-first 的采样机制；
- \(R_{SA}\) 进一步让结构 embedding 和真实行为输出对齐。

这也是 AlphaSAGE 与传统 RL alpha mining 方法的重要区别之一。

---

## 追加笔记：Entropy Regularization 在 AlphaSAGE 中的作用

### 1. 这段文字想表达什么？

论文这里想表达的是：**AlphaSAGE 在 GFlowNet 的 Trajectory Balance loss 之外，又加入了 policy entropy bonus，用来防止策略过早收敛，并鼓励 action-level 的细粒度探索。**

在 alpha 生成过程中，模型每一步都要从 action set 中选择下一个 token，例如：

\[
\text{close},\ \text{open},\ \text{Add},\ \text{TsMean},\ \text{Rank},\dots
\]

如果训练早期模型发现某些 token 或结构容易产生较高 reward，它可能很快把概率集中到少数动作上。例如：

\[
\pi_\theta(\text{Rank}\mid s_t)=0.9,
\]

其他动作概率都很小。这样会导致生成的公式越来越相似，探索空间快速缩小，这就是论文所说的 **premature convergence**。

因此，AlphaSAGE 在最终训练目标里加入 entropy regularization，让每一步的动作分布不要太快变得过于确定。

---

### 2. Entropy 是什么？

对于状态 \(s_t\) 下的动作分布：

\[
\pi_\theta(\cdot\mid s_t),
\]

entropy 衡量这个分布有多分散。

如果动作分布比较平均，例如：

\[
[0.25,0.25,0.25,0.25],
\]

entropy 较高，说明模型仍然在探索多个可能动作。

如果动作分布高度集中，例如：

\[
[0.97,0.01,0.01,0.01],
\]

entropy 较低，说明模型几乎只选择一个动作。

所以 entropy bonus 的核心作用是：

\[
\boxed{\text{让 policy 不要太快变得 deterministic。}}
\]

---

### 3. 公式 14 的含义

论文定义：

\[
L_{ENT}
=
-\mathbb{E}_{\tau\sim P_F(\tau;\theta)}
\left[
\sum_{t=0}^{n-1}
H(\pi_\theta(\cdot\mid s_t))
\right].
\]

其中：

\[
H(\pi_\theta(\cdot\mid s_t))
\]

表示在当前状态 \(s_t\) 下，动作选择分布的 entropy。

一条生成轨迹为：

\[
\tau=(s_0,s_1,\dots,s_n),
\]

因此论文把整条轨迹上每一步的 entropy 加起来：

\[
\sum_{t=0}^{n-1}H(\pi_\theta(\cdot\mid s_t)).
\]

外面加负号，是因为训练目标是 **minimize loss**。最小化负 entropy 等价于最大化 entropy：

\[
\min L_{ENT}
\Longleftrightarrow
\max \sum_t H(\pi_\theta(\cdot\mid s_t)).
\]

所以公式 14 的意思是：

\[
\boxed{\text{鼓励每一步动作分布保持随机性，防止动作概率过早塌缩。}}
\]

---

### 4. 公式 15 的含义

最终训练目标是：

\[
L_{final}
=
\mathbb{E}_{\tau\sim P_F(\tau;\theta)}[L_{TB}(\tau)]
+
\beta L_{ENT}.
\]

其中：

- \(L_{TB}\)：GFlowNet 的 Trajectory Balance loss；
- \(L_{ENT}\)：entropy regularization；
- \(\beta\)：控制 entropy bonus 强度的超参数。

\(L_{TB}\) 的目标是让 GFlowNet 学到：

\[
P_\theta(\alpha)\propto R(\alpha),
\]

即高 reward 的 alpha 被采样到的概率更高。

\(L_{ENT}\) 的目标是让 forward policy 不要过早变得过于确定，从而持续探索更多 token/action 组合。

因此，最终目标可以理解为：

\[
\boxed{
\text{学习一个 reward-proportional 的生成分布，同时保持足够探索。}
}
\]

---

### 5. 为什么叫 action-level fine-grained exploration？

AlphaSAGE 里面的 novelty reward \(R_{NOV}\) 和 GFlowNet 本身主要是在 **完整 alpha 层面** 鼓励多样性：

\[
\alpha_1,\alpha_2,\alpha_3,\dots
\]

但是 entropy bonus 是在每一个中间状态的动作分布上起作用：

\[
\pi_\theta(a_t\mid s_t).
\]

它不是等一个 alpha 完整生成之后才判断“这个 alpha 是否新颖”，而是在每一步 token 选择时都鼓励模型保留更多可能动作。

因此它比 \(R_{NOV}\) 更细粒度：

\[
R_{NOV}:\quad \text{alpha-level diversity}
\]

\[
L_{ENT}:\quad \text{action-level exploration}
\]

---

### 6. 和 PPO 里的 entropy bonus 的关系

这个思想和 PPO 中常见的 entropy bonus 很相似。

PPO 中常见形式是：

\[
L = L_{PPO} - c_{ent}H(\pi),
\]

目的是防止 policy 过早变成 deterministic policy。

AlphaSAGE 中对应的是：

\[
L_{final}=L_{TB}+\beta L_{ENT}.
\]

区别在于：

\[
\text{PPO: policy gradient objective + entropy bonus}
\]

\[
\text{AlphaSAGE: GFlowNet Trajectory Balance objective + entropy bonus}
\]

也就是说，entropy bonus 的探索思想类似，但主优化目标不同。

---

### 7. 对 AlphaSAGE 的理解

这部分说明 AlphaSAGE 的多样性设计有三个层次：

1. **GFlowNet 分布式生成**：目标不是只找一个最优 alpha，而是按 reward 比例采样一批高 reward alpha；
2. **Novelty reward**：在完整 alpha 层面惩罚与已有 alpha 高相关的公式；
3. **Entropy regularization**：在 action/token 层面防止生成策略过早塌缩。

因此，AlphaSAGE 的探索机制不是单独依赖某一个模块，而是：

\[
\boxed{
\text{GFlowNet mode coverage}
+
\text{alpha-level novelty}
+
\text{action-level entropy exploration}
}
\]

共同推动生成器产生更丰富、更不重复、更有预测能力的 formulaic alpha。

---

### 8. 一句话总结

这段话的核心意思是：

\[
\boxed{
\text{AlphaSAGE 在 GFlowNet 的 TB loss 上加入 entropy regularization，}
}
\]

\[
\boxed{
\text{让每一步动作选择保持更高随机性，防止早期模式塌缩，提升 alpha 多样性。}
}
\]

更直白地说：

> 不要让生成器太早只会生成一种公式结构，而是让它在 token/action 层面持续探索更多可能的公式。

