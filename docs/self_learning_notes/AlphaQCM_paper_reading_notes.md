# Paper Reading Notes: AlphaQCM: Alpha Discovery in Finance with Distributional Reinforcement Learning

> 说明：本文档按照 `RL Alpha Mining Paper Reading Task Template` 的结构整理。正文尽量使用中文，必要的英文专有名词、数学公式、变量符号保留英文。

---

## 1. One-Sentence Summary

This paper proposes **AlphaQCM**, a distributional reinforcement learning method, to solve **non-stationary and reward-sparse synergistic formulaic alpha discovery** by using **DQN to learn mean Q-values, IQN to learn return quantiles, and QCM-estimated variance as an exploration bonus**.

用中文概括：

**AlphaQCM 试图解决 AlphaGen 类公式因子生成任务中的非平稳 reward 与稀疏 reward 问题：它不再只用 PPO 直接学习生成策略，而是用 Q-network 学习动作价值，用 quantile network 学习累计 reward 分布，再通过 QCM 方法从可能有偏的 quantiles 中估计方差，并把方差作为 exploration bonus 来指导公式 token 的选择。**

---

## 2. Core Problem

- **Task type:** 不是单因子挖掘，而是 **synergistic formulaic alpha pool discovery**，目标是找到一组可以组合成线性 meta-alpha 的公式因子。
- **Input:** 股票历史市场数据，主要包括 `Open / High / Low / Close / Vwap / Volume`，以及由这些基础量经 operators 组合得到的公式表达式。
- **Output:** 一个 alpha pool：
  \[
  \mathcal F = \{f_1, f_2, \dots, f_P\},
  \]
  其中每个 \(f_p\) 是一个 formulaic alpha。最终用线性模型组合为 meta-alpha：
  \[
  \hat \alpha_s = \sum_{p=1}^{P} \alpha_{p,s}\hat\beta_p,
  \qquad
  \alpha_{p,s}=f_p(H_{s-1})\in\mathbb R^N.
  \]
- **Main objective:** 最大化由 alpha pool 构造出的线性 meta-alpha 对未来收益 \(y_s\) 的预测能力，主要用 out-of-sample IC 衡量。
- **Financial target:** 预测 20-day future stock returns。
- **Single-factor or pool-aware:** 明确是 **pool-aware**。新公式的 reward 不是单独 IC，而是它加入当前 pool 后对 meta-alpha IC 的边际提升。
- **Discovery-only or discovery-plus-deployment:** 主要是 discovery-plus-combination。论文没有真正进入完整交易部署，例如 portfolio construction、turnover、transaction cost、capacity 等环节；但它确实构造了 linear meta-alpha，而不只是输出孤立因子。

---

## 3. MDP Design

AlphaQCM 的 MDP 设计基本继承 AlphaGen：把公式生成过程看成一个 token-by-token 的 sequential decision-making problem。区别在于 AlphaQCM 强调该 MDP 具有两个困难：

1. **Non-stationarity:** 当前 alpha pool 会随 episode 更新，所以同一个公式在不同阶段的边际贡献不同，reward function 会变。
2. **Reward-sparsity:** 中间 token 没有 reward，大量生成公式无效或无边际贡献，reward 大量为 0。

### State

状态 \(x_t\) 是当前已经生成的 RPN token sequence，即 partial formula prefix：

\[
x_t = [\text{BEG}, a_1, a_2, \dots, a_{t-1}].
\]

初始状态是：

\[
x_0 = [\text{BEG}].
\]

为了保持公式可解释性，论文限制 token 序列长度小于 20。

### Action

动作 \(a_t\) 是下一个 token：

\[
a_t \in \mathcal A.
\]

token 可以是 feature、constant、time delta、unary operator、binary operator、time-series operator、cross-sectional operator、或者结束符 `SEP`。

### Transition

transition 是 deterministic grammar update：

\[
x_{t+1} = x_t \oplus a_t,
\]

即把动作 token append 到当前 token sequence 后面。

如果 \(a_t=\text{SEP}\)，或者 token 长度达到上限，则当前 episode 结束，环境重新回到 \(x_0\)。

### Reward

reward 是 terminal and pool-aware 的。对于未完成公式：

\[
r_t = 0.
\]

对于完成公式：

- 如果 token sequence 不能 parse 成合法公式，则
  \[
  r_t=-1.
  \]
- 如果 parse 成合法公式 \(f_{new}\)，则先加入当前 alpha pool，重新拟合线性 meta-alpha，再计算新 pool 相比旧 pool 的 IC 提升：
  \[
  r_t = IC(\mathcal F^*) - IC(\mathcal F).
  \]

其中 \(\mathcal F^*\) 是加入新公式并经过 pool update 后的 alpha pool。

### Termination

episode 结束条件：

1. agent 选择 `SEP` token；
2. token sequence 达到最大长度；
3. 公式生成完成后被 parse 和 evaluate。

### Action Legality / Grammar Constraints

不是所有 token 都能在任意 state 下选择。AlphaQCM 使用 action mask / invalid action masking：

\[
\hat Q(x,a)=-\infty, \qquad a\notin \mathcal A_{valid}(x).
\]

例如：

- time delta token 必须配合 time-series operator；
- RPN 序列必须保证栈结构合法；
- operator 的 arity 必须匹配已有 operands；
- 公式长度不能超过限制。

因此 MDP 可以紧凑写为：

\[
s_t=x_t=[\text{BEG},a_1,\dots,a_{t-1}],
\qquad
 a_t\in\mathcal A_{valid}(x_t),
\]

\[
x_{t+1}=x_t\oplus a_t,
\qquad
R_T = IC(\mathcal F^*)-IC(\mathcal F).
\]

---

## 4. Representation

- **Representation type:** Reverse Polish Notation, RPN。公式树被 flatten 成 token sequence。
- **Token/operator set:**
  - Features: `Open`, `High`, `Low`, `Close`, `Vwap`, `Volume`。
  - Constants: \(\{-30,-10,-5,-2,-1,-0.5,-0.01,0.01,0.5,1,2,5,10,30\}\)。
  - Time delta: \(\{10,20,30,40,50\}\)。
  - Time-series operators: `Ref`, `TsRank`, `Mean`, `Med`, `Sum`, `Std`, `Var`, `Max`, `Min`, `WMA`, `EMA`, `Cov`, `Corr`。
  - Cross-sectional operators: `Sign`, `Abs`, `Log`, `Rank`, `Add`, `Sub`, `Mul`, `Div`, `Greater`, `Less`。
- **Validity mechanism:** 通过 RPN grammar constraints 和 invalid action masking 保证公式可以被 parse；非法动作被 mask，非法完整公式 reward 为 \(-1\)。
- **Advantages:**
  1. RPN 把表达式树生成转化成序列决策问题，便于接入 RL。
  2. 公式结构可解释，比 black-box ML alpha 更容易被研究员理解和审查。
  3. 与 AlphaGen 表示方式一致，方便做公平对比。
- **Limitations:**
  1. RPN flatten 掉了显式 tree structure，LSTM 需要自己从序列中学习结构依赖。
  2. 表达能力强烈依赖 operator library。
  3. token 长度上限有助于可解释性，但会限制复杂公式。
  4. formula validity 主要是语法层面，不能保证经济意义、交易稳定性或低 turnover。

---

## 5. Algorithm Mechanism

### 总体机制

AlphaQCM 不是 policy-gradient 方法，而是一个 **value-based + distributional RL** 方法。它学习两个核心量：

1. **Q function:**
   \[
   Q^*(x,a)=\mathbb E[Z^*(x,a)],
   \]
   用 DQN-style Q-network 估计。
2. **Return distribution quantiles:**
   \[
   Z^*(x,a),
   \]
   用 IQN-style quantile network 估计其 quantiles。

然后，AlphaQCM 不直接把 quantiles 平均成 Q-value，而是用 QCM 方法估计 variance：

\[
\hat h(x,a) \approx \operatorname{Var}[Z^*(x,a)].
\]

动作选择使用：

\[
a_t = \arg\max_{a\in\mathcal A_{valid}(x_t)}
\left[
\hat Q(x_t,a) + \lambda\sqrt{\hat h(x_t,a)}
\right].
\]

其中：

- \(\hat Q(x_t,a)\)：利用当前估计的平均累计 reward，偏 exploitation；
- \(\sqrt{\hat h(x_t,a)}\)：不确定性或方差 bonus，偏 exploration；
- \(\lambda\)：risk-preference / exploration strength。

### Expected output format

- **RL/search method:** DQN + IQN + QCM variance bonus。没有显式 MCTS 或 beam search。
- **On-policy or off-policy:** Off-policy。使用 replay memory 和 prioritized experience replay。
- **Learned components:**
  1. Q-network：输出 \(\hat Q(x,a)\in\mathbb R^{|\mathcal A|}\)。
  2. Quantile network：输出 \(\hat\theta_k(x,a)\)，即每个 action 的 return quantiles。
  3. Target Q-network 和 target quantile network：用于 TD target。
- **Search component:** 没有传统搜索树；实际搜索通过 value-guided token selection 完成。
- **Training loop:**
  1. 从 \(x_0=[\text{BEG}]\) 开始生成公式。
  2. 对当前 state，计算 \(\hat Q(x,a)\) 与 \(\hat h(x,a)\)。
  3. 用 \(\hat Q+\lambda\sqrt{\hat h}\) 选择下一个 token。
  4. 得到 transition \((x_t,a_t,r_t,x_{t+1})\)，存入 replay memory。
  5. 从 replay memory 采样 batch，更新 Q-network 与 quantile network。
  6. 周期性同步 target networks。
  7. 每个 episode 结束时 parse formula、更新 pool、计算 terminal reward。
- **Key update objective:**

Q-network 使用 squared TD error：

\[
\ell(\omega)=\sum_{t\in batch}
\left[
 r_t+\gamma\max_{a'}\tilde Q(x_{t+1},a')-\hat Q(x_t,a_t)
\right]^2.
\]

Quantile network 使用 quantile TD Huber loss：

\[
\ell(\omega^*)=
\sum_{t\in batch}\sum_{k=1}^{K}\sum_{k'=1}^{K'}
\rho^{\kappa}_{\tau_k}(\delta_{k,k',t}),
\]

其中：

\[
\delta_{k,k',t}
=
 r_t+\gamma\tilde\theta_{k'}
\left(x_{t+1},\arg\max_{a'}\tilde Q(x_{t+1},a')\right)
-
\hat\theta_k(x_t,a_t).
\]

---

## 6. Reward Design

- **Reward signal:** pool-level marginal IC improvement：
  \[
  r_t = IC(\mathcal F^*) - IC(\mathcal F).
  \]
- **Terminal or intermediate:** terminal reward 为主。中间 incomplete token sequence reward 为 0。
- **Pool-aware or standalone:** pool-aware。reward 评估的是新公式对已有 alpha pool 的边际贡献，而不是 standalone IC。
- **Risk-aware or uncertainty-aware:** reward 本身不是 risk-aware；但 action selection 是 uncertainty-aware，因为使用 QCM variance bonus。
- **Shaping mechanism:** 没有传统 dense reward shaping；主要通过 QCM-estimated variance 改善 exploration，相当于 exploration shaping。
- **Possible reward bias:**
  1. Reward 来自训练集上的 IC 增量，容易产生 evaluator overfitting。
  2. 只优化 IC，不直接考虑 RankIC、turnover、transaction cost、capacity、industry/size exposure。
  3. Pool update 通过线性回归和剔除最小 \(|\hat\beta_p|\) 的因子完成，可能偏向训练期线性贡献。
  4. Reward function 随 alpha pool 更新而变化，因此同一个 formula 在不同 episode 中 reward 不一致。

### Reward 计算过程

给定当前 alpha pool：

\[
\mathcal F=\{f_1,\dots,f_{P^*}\},
\]

新生成公式：

\[
f_{new}=f_{P^*+1}.
\]

计算每个 alpha 的横截面值并标准化：

\[
\alpha_{p,s}=\frac{f_p(H_{s-1})-Mean(f_p(H_{s-1}))}{Std(f_p(H_{s-1}))}.
\]

构造设计矩阵：

\[
A_s=[\alpha_{1,s},\dots,\alpha_{P^*+1,s}].
\]

拟合线性模型：

\[
\hat\beta = \arg\min_\beta \sum_s \|y_s-A_s\beta\|^2.
\]

如果 pool 超过容量 \(P\)，剔除 \(|\hat\beta_p|\) 最小的 alpha：

\[
\bar p=\arg\min_p |\hat\beta_p|.
\]

得到更新后的 pool \(\mathcal F^*\)，并计算：

\[
IC(\mathcal F^*)=Mean(Corr(\hat\alpha_s,y_s)).
\]

最终 reward：

\[
r_t=IC(\mathcal F^*)-IC(\mathcal F).
\]

---

## 7. Factor Pool and Combination

- **Pool maintained:** 是。维护一个最多包含 \(P\) 个 formulaic alphas 的 alpha pool。
- **Pool update rule:** 每生成一个合法新 alpha，就将它临时加入 pool，然后重新拟合线性模型。如果 pool 数量超过 \(P\)，剔除线性系数绝对值最小的 alpha。
- **Factor selection/removal:** 使用 \(|\hat\beta_p|\) 作为 contribution proxy。最小 \(|\hat\beta_p|\) 的 alpha 被删除。
- **Combination model:** 线性 meta-alpha：
  \[
  \hat\alpha_s = \sum_{p=1}^{P}\alpha_{p,s}\hat\beta_p.
  \]
- **Static or dynamic weights:** 论文中 \(\hat\beta\) 是基于训练样本拟合得到的线性权重，不是随市场状态动态变化的 weights。
- **Deployment realism:** 中等。它比 standalone alpha discovery 更接近真实因子组合，但还没有完整 portfolio backtest、交易成本、换手、滑点、容量、风格暴露控制等部署要素。

---

## 8. Experiments and Evaluation

- **Dataset:** 中国 A 股市场数据；额外实验包含 U.S. S&P 500。
- **Universe:**
  1. CSI 300；
  2. CSI 500；
  3. Market：上海和深圳交易所全部股票；
  4. Robustness: S&P 500。
- **Time split:**
  - Train: 2010/01/01–2019/12/31；
  - Validation: 2020/01/01–2020/12/31；
  - Test: 2021/01/01–2022/12/31；
  - 额外 expanded test: 2021/01/01–2024/12/31。
- **Prediction target:** 20-day future stock returns。
- **Metrics:** 主要是 out-of-sample IC，即 meta-alpha 与未来收益的横截面相关系数的时间平均。
- **Baselines:**
  1. Human-designed formulaic alphas: Alpha101；
  2. Non-formulaic ML models: MLP, XGBoost, LightGBM；
  3. GP-based formulaic alphas: GP w/o filter, GP w/ filter；
  4. RL-based formulaic alphas: PPO w/ filter, AlphaGen。
- **Transaction costs:** 未纳入。
- **Turnover:** 未纳入。
- **Random seeds:** 对 indeterministic methods 使用 10 random seeds。
- **Robustness tests:**
  1. QCM variance vs no variance vs vanilla quantile variance；
  2. IQN backbone vs QRDQN backbone；
  3. with domain knowledge vs without domain knowledge；
  4. CSI 500 vs S&P 500；
  5. parameter size ablation；
  6. LSTM vs MAMBA；
  7. expanded test set to 2024。
- **Code availability:** 论文声称开源：`https://github.com/ZhuZhouFan/AlphaQCM`。

### 主要实验结果理解

在主实验中，AlphaQCM 在三个 A 股 universe 上都取得最高 IC：

| Universe | AlphaGen IC | AlphaQCM IC | 直观解释 |
|---|---:|---:|---|
| CSI 300 | 8.13% | 8.49% | 提升较小，说明较简单 universe 下 AlphaGen 已经较强 |
| CSI 500 | 8.08% | 9.55% | 提升明显，QCM exploration 更有效 |
| Market | 6.04% | 9.16% | 提升最大，说明复杂市场下 sparse/non-stationary 问题更严重 |

这个结果支持作者的核心论点：随着股票池变大，公式搜索空间和 reward 非平稳性更强，AlphaQCM 的 uncertainty-aware exploration 更有价值。

---

## 9. Main Contributions

1. **The paper contributes uncertainty-aware exploration by addressing sparse terminal reward in formulaic alpha generation.** 具体而言，它使用 QCM-estimated variance 作为 exploration bonus，使 agent 不只选择当前平均 Q-value 高的 token，也倾向探索高不确定性的 token。

2. **The paper contributes a distributional RL formulation for synergistic alpha pool discovery by addressing the limitation that AlphaGen only learns a policy/value expectation.** AlphaQCM 同时学习 \(Q(x,a)\) 与 \(Z(x,a)\) 的 quantiles，尝试利用 return distribution 中的高阶信息。

3. **The paper contributes a non-stationarity-aware improvement over AlphaGen by estimating variance from biased quantiles via QCM.** 作者认为在 non-stationary MDP 中 quantiles 可能有偏，但 QCM variance 仍可以保持较好的估计性质，因此适合不断更新 pool 的 alpha discovery 环境。

4. **The paper contributes stronger empirical validation than许多普通 alpha-mining 论文 by including multiple universes, 10 seeds, ablations, cross-market S&P 500 test, and expanded test period.** 这使得论文对“方法模块是否有效”的论证较完整。

---

## 10. Limitations

### Algorithmic limitation

1. **Reward sparsity 被缓解但没有被根本解决。** 中间 token 仍然没有真实 financial reward，QCM variance 只是让 exploration 更有效，并没有构造 dense semantic reward。

2. **Q-network 与 quantile network 分开训练，系统复杂度高。** 相比 AlphaGen 的 actor-critic，AlphaQCM 需要 online Q-network、online quantile network、target Q-network、target quantile network，以及 QCM regression，训练和调参复杂度更高。

3. **QCM variance 的有效性依赖 Cornish-Fisher expansion 与 quantile estimation 质量。** 如果 quantile network 学得很差，或者 return distribution 极其不规则，QCM variance 也可能不稳定。

4. **它没有真正建模 reward function 的动态变化机制。** Alpha pool 更新导致 MDP non-stationary，但 AlphaQCM 主要通过 exploration bonus 与 replay 学习适应，并没有显式学习 environment dynamics 或 reward drift。

5. **RPN + LSTM 仍然可能不是最适合表达式树的结构。** LSTM 处理 flattened sequence，可能无法充分利用 expression tree 的层级结构。虽然论文测试了 MAMBA，但没有测试 tree-based encoder、GNN 或 AST-specific architecture。

### Evaluation / finance limitation

1. **评价指标主要是 IC，没有完整交易回测。** 缺少 Sharpe、annualized return、max drawdown、turnover、transaction-cost-adjusted return 等部署指标。

2. **没有 transaction cost 和 turnover。** 对 formulaic alpha 来说，某些高 IC 因子可能换手极高，现实中未必可交易。

3. **缺少容量和流动性分析。** 特别是在全市场股票池上，因子是否可在实际资金规模下部署仍不清楚。

4. **线性 meta-alpha 可能有训练期过拟合。** Reward 直接来自训练样本上的 pool IC 增量，公式搜索过程会反复利用 evaluator，存在 data snooping / evaluator overfitting 风险。

5. **没有充分讨论风险暴露。** 例如行业、市值、风格、beta、流动性等 exposure 是否被控制并不清楚。

6. **虽然有 S&P 500 测试，但跨市场验证仍有限。** S&P 500 是很好的补充，但不同市场结构、频率、交易规则下的泛化仍需要更多证据。

---

## 11. Relation to AlphaGen and Existing Literature

- **Relation to AlphaGen:** AlphaQCM 是 AlphaGen 的直接扩展。它继承了 AlphaGen 的核心任务：用 RPN token sequence 生成 formulaic alphas，并用 alpha pool 的 linear meta-alpha IC 作为 reward。但它替换了 AlphaGen 的 PPO-based policy learning，改用 DQN + IQN + QCM variance bonus。

- **Main bottleneck addressed:**
  1. AlphaGen 中 reward 稀疏：只有公式结束后才有 reward，且大多数公式 reward 为 0。
  2. AlphaGen 中 reward 非平稳：pool 更新后，同一类公式的边际贡献会下降。
  3. AlphaGen 忽略 return distribution：只利用平均意义上的 reward/value，而不利用 uncertainty。

- **Paradigm classification:**
  1. Policy-based sequential expression generation：部分相关，因为仍是 token-by-token expression generation；
  2. Distributional or risk-aware RL：核心范式；
  3. Reward shaping and variance reduction：通过 variance bonus 改善 exploration；
  4. Expert-guided generation：只在 domain knowledge ablation 中涉及，不是最终推荐版本。

- **Difference from prior work:**
  - 相比 GP：AlphaQCM 用 RL value learning 引导搜索，而不是随机 evolution / mutation / crossover。
  - 相比 AlphaGen：AlphaQCM 不使用 PPO，而是用 value-based distributional RL。
  - 相比 vanilla DRL：AlphaQCM 不直接用 quantile variance，而是用 QCM 从 potentially biased quantiles 中估计 variance。
  - 相比 ML-based alpha：AlphaQCM 输出可解释公式，而不是 black-box prediction model。

- **Whether it is more RL-like or search-like:** 更 RL-like。它没有显式 search tree 或 MCTS，而是通过 learned Q-values 和 uncertainty bonus 进行 token-level sequential decision-making。

### 和 AlphaGen 的核心对比表

| 维度 | AlphaGen | AlphaQCM |
|---|---|---|
| 公式表示 | RPN token sequence | RPN token sequence |
| 目标 | synergistic alpha pool | synergistic alpha pool |
| RL 类型 | policy-gradient / actor-critic | value-based + distributional RL |
| 主要算法 | PPO | DQN + IQN + QCM |
| Reward | pool IC marginal improvement | pool IC marginal improvement |
| Non-stationarity 处理 | 基本没有显式处理 | 用 QCM variance bonus 提高适应效率 |
| Sparse reward 处理 | 依赖 PPO exploration | 用 variance-driven exploration |
| Action selection | policy sampling / PPO policy | \(\arg\max_a[Q+\lambda\sqrt h]\) |
| 训练数据 | on-policy rollouts | off-policy replay memory |
| 主要新增点 | — | distributional uncertainty-aware exploration |

---

## 12. How to Incorporate This Paper into My Review

### Suggested section

- **Section:** “Distributional and Uncertainty-Aware RL for Formulaic Alpha Discovery” 或 “Beyond PPO: Value-Based and Distributional RL Methods for Alpha Mining”。
- **Reason:** AlphaQCM 是目前 RL alpha mining 里对 AlphaGen 的一个自然升级：它没有改变公式表示和 pool-aware reward 的框架，而是专门攻击 AlphaGen 的训练不稳定、reward 稀疏、non-stationary reward 这几个 RL 层面的瓶颈。

### Suggested one-line description

AlphaQCM extends AlphaGen-style formulaic alpha generation by replacing PPO with a DQN–IQN–QCM framework that uses distributional variance as an exploration bonus for non-stationary and reward-sparse alpha pool discovery.

### Suggested paragraph

AlphaQCM can be positioned as a distributional-RL extension of AlphaGen. Like AlphaGen, it represents formulaic alphas as RPN token sequences and evaluates each newly generated alpha by its marginal contribution to a linear alpha pool. Its main methodological change is not the representation or reward definition, but the RL optimizer: AlphaQCM replaces PPO with a value-based distributional RL framework, where a Q-network estimates the expected return and an IQN-style quantile network estimates the distribution of cumulative rewards. To handle the non-stationary and reward-sparse nature of alpha pool discovery, it applies the quantiled conditional moment method to estimate return variance from potentially biased quantiles, and uses this variance as an exploration bonus in action selection. Empirically, AlphaQCM improves over AlphaGen, especially in larger and more complex stock universes, suggesting that uncertainty-aware exploration is useful when the formula search space is large and most generated expressions have zero or negligible marginal value.

### Suggested comparison table row

| Paper | Venue | Core formulation | Representation | RL/Search mechanism | Reward type | Pool-aware? | Dynamic allocation? | Key contribution | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| AlphaQCM | ICML 2025 | Non-stationary and reward-sparse MDP for synergistic formulaic alpha pool discovery | RPN token sequence | DQN + IQN + QCM variance bonus, off-policy replay | Terminal marginal IC improvement of linear meta-alpha | Yes | No, linear weights are fitted but not state-dynamic | Uses distributional variance to guide exploration under sparse/non-stationary reward | No full trading backtest; no transaction cost/turnover/capacity; still evaluator-overfitting risk |

---

## 13. My Overall Assessment

AlphaQCM 最有价值的地方主要是 **methodology and future research inspiration**，其次是 implementation reference。

从方法角度看，它明确指出 AlphaGen 类 alpha mining MDP 的两个核心 RL 难点：

\[
\boxed{\text{non-stationary reward} + \text{sparse terminal reward}}.
\]

这点非常重要。因为很多公式挖掘论文只说“搜索空间大”，但 AlphaQCM 更准确地指出：当 reward 是 pool-level marginal contribution 时，reward function 本身会随着 pool 变化而变化，因此这不是普通 stationary MDP。

从你的 review 文档角度，我建议把 AlphaQCM 放在 AlphaGen 之后，作为“AlphaGen 后续改进方向”的代表。它说明：在 formulaic alpha mining 中，未来改进不一定要重新设计公式表示，也不一定要重新设计 reward；也可以从 **RL optimization / exploration mechanism** 入手。也就是说，AlphaQCM 的贡献位置可以写成：

\[
\text{AlphaGen: 定义 pool-aware formula generation MDP}
\]

\[
\text{AlphaQCM: 针对该 MDP 的 sparse/non-stationary training bottleneck 设计 distributional RL 解法}
\]

不过，从量化金融实用性角度看，AlphaQCM 仍然停留在 alpha discovery + IC evaluation 层面。它没有回答：这些因子是否低换手、是否抗交易成本、是否有容量、是否中性化后仍有效、是否能形成真实可交易组合。因此在 review 中应避免把它写成“完整交易系统”，更准确的定位是：

**一个面向 formulaic alpha pool discovery 的 uncertainty-aware RL optimizer。**

---

## 14. Four Core Questions Revisited

### 1. How does it define factor mining as a decision problem?

它把公式因子的 RPN token generation 定义为 MDP：state 是当前 partial token sequence，action 是下一个 token，transition 是 append token，terminal reward 是新公式加入 alpha pool 后带来的 IC 边际提升。

### 2. What exactly does the RL component learn?

RL 组件学习两个对象：

\[
Q^*(x,a)=\mathbb E[Z^*(x,a)]
\]

以及

\[
Z^*(x,a) \text{ 的 quantiles } \theta_k(x,a).
\]

然后用 QCM 从 quantiles 中估计 variance \(h(x,a)\)，用于 action selection。

### 3. What weakness of AlphaGen or previous methods does it address?

它主要解决 AlphaGen 没有显式处理的两个问题：

1. alpha pool 更新导致 reward non-stationary；
2. 公式生成任务 terminal reward sparse，大多数公式 reward 为 0。

### 4. Does the experimental protocol support the paper's conclusion?

基本支持“AlphaQCM 比 AlphaGen 更适合大股票池、复杂市场下的 formulaic alpha discovery”这一结论。因为论文提供了多 universe、10 seeds、ablation、S&P 500、expanded test period 等证据。  
但它还不足以支持“AlphaQCM 因子能直接实盘部署”这一更强结论，因为缺少 transaction cost、turnover、capacity、portfolio-level backtest 和风险暴露控制。
