# Paper Reading Notes: RiskMiner: Discovering Formulaic Alphas via Risk Seeking Monte Carlo Tree Search

> 论文：Tao Ren, Ruihan Zhou, Jinyang Jiang, Jiafeng Liang, Qinghao Wang, Yijie Peng. **RiskMiner: Discovering Formulaic Alphas via Risk Seeking Monte Carlo Tree Search**. arXiv:2402.07080v2, 2024.

---

## 1. One-Sentence Summary

This paper proposes **RiskMiner**, a risk-seeking MCTS-based formulaic alpha mining framework, to solve the sparse-reward and inefficient-exploration problems of RL alpha generation by formulating alpha mining as a reward-dense MDP and alternating Monte Carlo Tree Search with risk-seeking policy optimization.

用中文概括：

> RiskMiner 用 **reward-dense MDP + MCTS + risk-seeking policy** 来挖掘一组“高 IC 且低相关”的公式化 alpha，目标是相对于 AlphaGen 的 PPO 序列生成方式，更好地利用表达式搜索空间结构，并把优化目标从平均表现转向“最好情况表现”。

---

## 2. Core Problem

- **Task type:** 公式化 alpha 挖掘，目标不是单个 factor，而是一个可组合的 alpha pool。
- **Input:** A 股日频 OHLCV 类数据，包括 `open, high, low, close, volume, vwap`。
- **Output:** 一组 formulaic alphas，以及线性组合后的 composite alpha / mega-alpha。
- **Main objective:** 找到既有较高预测能力，又与现有 alpha pool 相关性较低的公式 alpha。
- **Financial target:** 未来 5 日收益率和未来 10 日收益率。论文使用 close price 计算未来收益。
- **Single-factor or pool-aware:** pool-aware。单个 alpha 的评价不仅看自身 IC，也看它与 pool 内已有 alpha 的 mutIC。
- **Discovery-only or discovery-plus-deployment:** discovery-plus-deployment。论文不仅报告 IC、RankIC、ICIR，也做了周频调仓的多头回测。

核心问题可以写成：

$$
\max_{\mathcal F = \{f_1,\ldots,f_K\}} \operatorname{IC}\left(\sum_{k=1}^K \omega_k f_k(X_t), r_{t+1}\right),
$$

同时希望：

$$
\operatorname{IC}(f_i(X_t), r_{t+1}) \text{ 高},
\qquad
\operatorname{corr}(f_i(X_t), f_j(X_t)) \text{ 低}.
$$

---

## 3. MDP Design

### State

状态是当前已经生成的 token 序列，即一个 RPN 表达式前缀：

$$
s_t = (\text{BEG}, a_1, a_2, \ldots, a_t).
$$

这里的 state 不直接包含市场状态，也不包含一个显式的 portfolio 状态；它主要是 **partial formula state**。不过 reward 计算依赖当前 alpha pool，因此 reward 是 pool-aware 的。

### Action

动作是在当前 token 序列后面选择下一个 token：

$$
a_t \in \mathcal A,
$$

其中 action space 包括：价格量特征、时间窗口、常数、算子以及 `BEG` / `END` 等特殊 token。

### Transition

状态转移是确定性的 token append：

$$
s_{t+1} = s_t \Vert a_t.
$$

如果当前 token 序列已经可以解析成合法 RPN 表达式，则可以立刻计算该 partial expression 的 alpha 值，并给 intermediate reward。

### Reward

RiskMiner 的关键改动是把 AlphaGen 类方法中偏 sparse 的终端 reward 改成 **reward-dense MDP**。

#### Intermediate reward

当当前 token 序列形成一个合法但尚未终止的 RPN 表达式时，论文计算：

$$
R_{\text{inter}}
= \operatorname{IC}(f)
- \lambda \frac{1}{k}\sum_{i=1}^{k}\operatorname{mutIC}(f, f_i),
$$

其中 $f$ 是当前合法 partial expression 对应的 alpha，$f_i$ 是 alpha pool 中已有 alpha，$\lambda=0.1$。

这个 reward 的含义是：当前表达式越有预测能力越好，但如果它和已有 alpha 太相似，就要扣分。

#### Terminal reward

当 episode 选择 `END` 结束时，当前 alpha 被加入 alpha pool，然后重新优化 pool 的线性组合权重，最终 composite alpha 的 IC 作为终端 reward：

$$
R_{\text{end}}
= \operatorname{IC}\left(c(X_t \mid \mathcal F, \omega), r_{t+1}\right).
$$

其中组合模型是线性 alpha pool：

$$
z_t = c(X_t \mid \mathcal F, \omega)
= \sum_{i=1}^{k}\omega_i f_i(X_t).
$$

### Termination

一个 episode 在以下情况结束：选择 `END` token，或者达到最大 episode length。论文设最大长度为 30。

### Action Legality / Grammar Constraints

论文明确使用 RPN 表达式表示 alpha。RPN 天然有 grammar legality 约束：一元算子需要栈中已有一个合法子表达式，二元算子需要两个合法子表达式，时间序列算子需要表达式和时间窗口参数，`END` 应该只在当前序列能形成完整合法表达式时使用。

论文正文没有像 AlphaGen / MaskablePPO 那样详细展开 invalid action mask 的具体实现，但从 MCTS expansion 与合法表达式 reward 的描述看，它至少需要在表达式解析阶段做合法性检查，否则无法判断当前 token 序列是否能计算 IC。也就是说，RiskMiner 的合法性机制更接近 RPN parser / stack validity check、operator arity check、time-window operand compatibility check 和 expression evaluability check。

紧凑 MDP 表达：

$$
\begin{aligned}
s_t &= (a_0, a_1, \ldots, a_t), \quad a_0=\text{BEG},\\
a_t &\in \mathcal A_{\text{token}},\\
s_{t+1} &= s_t \Vert a_t,\\
R_t &=
\begin{cases}
\operatorname{IC}(f_t)-\lambda \frac{1}{k}\sum_i\operatorname{mutIC}(f_t,f_i), & s_t \text{ forms a valid non-terminal expression},\\
\operatorname{IC}(c(X_t\mid \mathcal F\cup\{f_t\},\omega)), & a_t=\text{END}.
\end{cases}
\end{aligned}
$$

---

## 4. Representation

- **Representation type:** Reverse Polish Notation, RPN。公式 alpha 被展开成 token sequence。
- **Tree relation:** RPN 来自表达式树的后序遍历，因此它把 tree structure flatten 成 sequence，但仍隐含表达式树结构。
- **Token/operator set:**
  - price/volume operands: `open, high, close, low, volume, vwap`；
  - time deltas: `1, 5, 10, 20, 30, 40, 50`；
  - constants: `-30, -10, -5, -2, -1, -0.5, -0.01, 0.5, 1, 2, 5, 10, 30`；
  - cross-sectional unary: `Sign, Abs, Log, CSRank`；
  - arithmetic binary: `Add, Sub, Mul, Div`；
  - comparison binary: `Greater, Less`；
  - time-series unary with window: `Ref, Rank, Skew, Kurt, Mean, Med, Sum, Std, Var, Max, Min, WMA, EMA`；
  - time-series binary with window: `Cov, Corr`。
- **Validity mechanism:** 通过 RPN 解析、算子 arity、时间窗口参数和表达式可计算性检查保证合法性。
- **Advantages:** RPN 把表达式生成转化为离散序列决策，适合 MDP 和 MCTS；比直接生成字符串更容易做 parser legality check；相比纯神经网络预测模型，公式 alpha 仍保留一定可解释性。
- **Limitations:** 表达能力强烈依赖人工设计的 operator library；flatten 成序列后，树结构信息没有被显式建模；公式合法不等于经济含义合理，仍可能生成噪声型表达式。

一个例子：

$$
\operatorname{Add}(\operatorname{Std}(\$close,10), \$open)
$$

可以表示成 RPN：

```text
BEG $close 10 Std $open Add END
```

---

## 5. Algorithm Mechanism

- **RL/search method:** Risk-seeking Monte Carlo Tree Search + risk-seeking policy optimization。
- **On-policy or off-policy:** 不属于标准 PPO 式 on-policy，也不是典型 Q-learning off-policy。更准确地说，它是 **MCTS sampler + replay buffer trajectories + policy-gradient style risk-seeking update**。由于 trajectories 来自当前 MCTS 与当前 policy 交互，具有 on-policy 采样色彩；但又用 replay buffer 暂存 MCTS 轨迹，因此不能简单归类为标准 on-policy / off-policy。
- **Learned components:** risk-seeking policy network $\pi_{\text{risk}}(a\mid s;\theta)$。网络结构是 GRU feature extractor + MLP policy head。没有显式 value network 或 Q-network；MCTS edge 里维护搜索统计量 $N,P,Q,R$。
- **Search component:** MCTS。每个搜索循环包含 Selection, Expansion, Rollout, Backpropagation。
- **Training loop:** 初始化 alpha pool、权重、risk policy 和 replay buffer；重置 MCTS 根节点；用 PUCT 选择到 leaf；调用 risk policy 扩展 leaf 的 prior；用 risk policy rollout 到 `END`；计算 intermediate reward 和 terminal reward；反向更新 MCTS edge value；将完整 trajectory 加入 replay buffer；buffer 满后用 quantile-based risk-seeking objective 更新 policy network。

### MCTS selection objective

论文使用 PUCT：

$$
a_t = \arg\max_a
\left[
Q(s,a)
+ P(s,a)\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
\right].
$$

其中 $Q(s,a)$ 是当前 edge 的平均回报估计，$P(s,a)$ 是 risk policy 给出的 prior，$N(s,a)$ 是访问次数，第二项鼓励探索 prior 高但访问次数少的动作。

### Backpropagation

rollout 完成后，论文把 rollout 中的 intermediate rewards 和 terminal reward 加总为 leaf value $v_l$，然后对路径上的 edge 做 bootstrap 累积回报：

$$
G_k = \sum_{i=0}^{l-1-k}\gamma^i r_{k+1+i} + v_l,
$$

并更新：

$$
Q(s_{k-1},a_k)
= \frac{N(s_{k-1},a_k)Q(s_{k-1},a_k)+G_k}{N(s_{k-1},a_k)+1},
$$

$$
N(s_{k-1},a_k) \leftarrow N(s_{k-1},a_k)+1.
$$

论文设置 $\gamma=1$，原因是它不想惩罚长表达式探索。

### Risk-seeking policy objective

传统 policy gradient 最大化：

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R(\tau)].
$$

RiskMiner 改为最大化 reward distribution 的上分位数：

$$
J_{\text{risk}}(\theta;\alpha)=q(\theta;1-\alpha),
$$

其中 $q$ 是 cumulative reward 的分位点。直观上，它不是让 policy 平均生成还不错的 alpha，而是让 policy 更偏向生成 top-tail 高质量 alpha。

分位数用递推估计：

$$
q_{i+1} = q_i + \beta\left(1-\alpha - \mathbf{1}\{R(\tau_i)\le q_i\}\right).
$$

policy 参数更新方向为：

$$
D(\tau;\theta,q)
= -\mathbf{1}\{R(\tau_i)\le q\}
\sum_{t=1}^T \nabla_\theta \log \pi(a_t\mid s_t;\theta),
$$

$$
\theta_{i+1}=\theta_i+\gamma D(\tau_i;\theta_i,q_i).
$$

这里最容易困惑的一点是：这个更新看起来使用了 $R(\tau)\le q$ 的 indicator，但它来自 quantile objective 的 CDF gradient 推导。论文的思想是通过调整 reward distribution 的分位点，而不是直接优化均值。

---

## 6. Reward Design

- **Reward signal:** intermediate reward 是单 alpha IC 减去与 pool 中已有 alpha 的平均 mutIC 惩罚；terminal reward 是加入 alpha pool 并重新优化权重后的 composite alpha IC。
- **Terminal or intermediate:** dense reward。既有 intermediate reward，也有 terminal reward。
- **Pool-aware or standalone:** pool-aware。intermediate reward 已经包含 mutIC penalty，terminal reward 是组合 alpha 的 pool-level IC。
- **Risk-aware or uncertainty-aware:** risk-seeking。它不是建模收益分布风险，也不是金融意义的 downside risk，而是优化搜索过程中 cumulative reward 的上分位数。
- **Shaping mechanism:** reward shaping 来自合法 partial expression 的 IC-based reward，以及 mutIC penalty 鼓励 alpha diversity。
- **Possible reward bias:** IC 和 mutIC 的线性标量化依赖 $\lambda$；intermediate reward 可能鼓励局部合法但最终组合价值有限的表达式；terminal reward 使用 pool IC 而不是显式边际贡献；没有在 reward 中直接纳入 turnover、transaction cost、行业/市值暴露或容量约束。

### 与 AlphaGen 的 reward 差异

AlphaGen 的主要特点是用 alpha pool downstream performance 作为 RL reward，但 reward 更偏 terminal 和 sparse。RiskMiner 的主要改变是：

$$
\text{Sparse terminal pool reward}
\quad\rightarrow\quad
\text{Dense IC/mutIC reward + terminal pool IC reward}.
$$

这使得搜索过程在还没有完整结束时就能获得反馈，从而降低长表达式生成中的 credit assignment 难度。

---

## 7. Factor Pool and Combination

- **Pool maintained:** 是。论文维护 alpha pool $\mathcal F=\{f_1,\ldots,f_k\}$。
- **Pool update rule:** 每发现一个新 alpha，就将其加入 pool，并重新优化线性组合权重。
- **Factor selection/removal:** 当 alpha 数量超过预设容量 $K$ 时，移除绝对权重 $|\omega_i|$ 最小的 alpha。
- **Combination model:** 线性组合模型：

$$
z_t = \omega\cdot f(X_t)=\sum_{i=1}^k \omega_i f_i(X_t).
$$

- **Optimization objective:** 用 MSE 拟合未来收益：

$$
\mathcal L(\omega)
= \frac{1}{nT}\sum_{t=1}^T \|z_t-r_{t+1}\|^2.
$$

- **Static or dynamic weights:** 静态权重。权重在训练 / pool update 时优化，但不是随市场状态动态变化的 allocation policy。
- **Deployment realism:** 论文做了一个简单部署测试：使用 5 日 alpha 信号，每 5 天调仓一次，选择预测分数最高的 top-k 股票等权持有。验证集搜索得到 $k=40$。但这仍然是较简化的 long-only deployment，没有完整讨论交易成本、滑点、冲击成本和容量。

### 与你当前 AlphaGen 理解的对应关系

RiskMiner 仍然保留了 AlphaGen 里的 “B 模块” 思想：不是只看单因子 IC，而是维护一个 alpha pool；pool 里的 alphas 通过线性权重合成 composite alpha；新 alpha 的价值要看它是否提升组合表现、是否与已有 alpha 重复。不同点在于 RiskMiner 的 factor generator 不再主要是 PPO policy，而是 **MCTS + risk-seeking policy**。

---

## 8. Experiments and Evaluation

- **Dataset:** 中国 A 股日频数据。
- **Universe:** CSI300 和 CSI500 成分股。
- **Input features:** `open, high, low, close, volume, vwap`。
- **Time split:** train 为 2010-01-01 至 2019-12-31；validation 为 2020-01-01 至 2020-12-31；test 为 2021-01-01 至 2022-12-31。
- **Prediction target:** 5-day return 和 10-day return。
- **Metrics:** IC、ICIR、RankIC、backtest cumulative return。
- **Baselines:** Alpha101、gplearn GP、AlphaGen、MLP、GRU、XGBoost、LightGBM、CatBoost。
- **Transaction costs:** 论文说使用 realistic trading setting，并排除涨跌停和停牌股票，但正文中没有清楚说明具体 transaction cost 数值。这是一个需要注意的 evaluation limitation。
- **Turnover:** 有周频调仓，但没有系统报告 turnover 指标。
- **Random seeds:** 主表显示所有实验重复 10 次，并报告 mean 和 standard deviation。
- **Robustness tests:** 不同 quantile level 的 risk-seeking 强度分析；MCTS 与 risk-seeking policy 的 ablation study。
- **Code availability:** 论文正文没有看到明确代码仓库链接。

### Main result interpretation

主表显示 RiskMiner 在 CSI300 / CSI500、5 日 / 10 日目标下，在 IC、ICIR、RankIC 上整体领先所有 baselines。尤其 ICIR 提升明显，说明作者认为 dense reward + MCTS search 使 alpha 表现更稳定。

论文还发现 quantile level 并不是越高越好：从 0.6 到约 0.85，IC 随 risk-seeking 程度上升而提高；超过 0.85 后，表现下降。作者解释为过度 risk-seeking 会让搜索陷入局部最优，把过多 budget 花在当前最优附近，而不是探索多个潜在局部最优。

这个结论对 alpha mining 很重要：因子挖掘不是只需要找到一个全局最优公式，而是需要找到一组互补的局部优质公式。因此过强的 exploitation 反而可能降低 pool diversity。

---

## 9. Main Contributions

1. **The paper contributes reward-dense MDP design by addressing the sparse-reward bottleneck of RL formula generation.**  
   它让合法 partial expression 也能产生 IC/mutIC reward，从而给长序列表达式生成提供更频繁的学习信号。

2. **The paper contributes search-enhanced exploration by addressing the limitation of pure neural policy generation.**  
   相比 AlphaGen 主要依靠 PPO policy，RiskMiner 用 MCTS 记录已探索的离散表达式空间，并用 PUCT 在 exploitation 和 exploration 之间平衡。

3. **The paper contributes risk-seeking policy optimization by addressing the mismatch between average-return RL objective and alpha mining's best-case search objective.**  
   因子挖掘中生成坏公式的成本较低，但发现一个好公式的收益很高，因此作者把目标从 expected reward 改成 upper quantile reward。

4. **The paper contributes a pool-aware symbolic alpha mining pipeline by combining individual IC, diversity penalty, and composite alpha IC.**  
   它不是只挖单因子，而是围绕 alpha pool 的协同组合效果设计 reward 和 pool update。

---

## 10. Limitations

### Algorithmic limitation

1. **MCTS 的计算成本可能较高。** 表达式空间巨大，MCTS 每轮需要 selection、expansion、rollout、backpropagation，并频繁调用 alpha evaluator 计算 IC / mutIC。论文没有充分报告计算成本、运行时间、硬件资源和扩展性。

2. **risk-seeking objective 可能牺牲 diversity。** 论文自己也观察到 quantile level 太高会性能下降，因为搜索可能集中在某些局部最优区域。对于 alpha pool 来说，过度追求 top-tail reward 可能减少候选公式多样性。

3. **reward credit assignment 仍不完全精确。** terminal reward 使用 composite alpha IC，而不是显式 marginal contribution：

$$
\Delta R = \operatorname{Perf}(\mathcal F\cup\{f\})-\operatorname{Perf}(\mathcal F).
$$

因此新 alpha 对 pool 的真实边际贡献可能没有被完全隔离。

4. **表达式空间依赖人工 operator library。** 模型能发现什么 alpha，很大程度取决于给定算子和窗口集合。若 operator set 不适合某市场，搜索算法再强也会受限。

### Evaluation / finance limitation

1. **交易成本与换手率披露不足。** 论文做了回测，并提到排除涨跌停和停牌股票，但没有系统报告 transaction cost 数值、turnover、slippage、market impact。这对判断真实可交易性不够充分。

2. **只在 A 股 CSI300 / CSI500 做验证。** 论文没有展示跨市场、跨资产类别、跨频率验证。例如美股、期货、加密货币、分钟级数据是否适用仍不清楚。

3. **回测策略较简单。** 只使用 top-k long-only weekly rebalance。没有 long-short、行业中性、市值中性、风险模型约束、容量约束等更贴近机构使用的设置。

4. **可能存在 evaluator overfitting 风险。** alpha mining 反复搜索并使用 IC 作为 evaluator，容易对训练期统计特征过拟合。虽然论文有 validation/test split，但没有充分讨论多重检验、data snooping、因子衰减等问题。

5. **代码可复现性不明确。** 如果没有公开代码，MCTS 细节、合法 action 过滤、alpha evaluator 和 pool update 的实现差异都可能影响复现。

---

## 11. Relation to AlphaGen and Existing Literature

- **Relation to AlphaGen:** RiskMiner 可以看作对 AlphaGen 的三点改造：reward 从偏 terminal / sparse 改为 dense；generator 从纯 PPO neural policy 改为 MCTS-guided search；objective 从 expected reward 改为 risk-seeking upper-quantile objective。
- **Main bottleneck addressed:** AlphaGen 的 sparse reward 让长表达式生成难学；PPO 不显式记录和利用表达式搜索树结构；平均 reward objective 不符合 alpha mining 中“发现少数特别好公式”的目标。
- **Paradigm classification:** Policy-based sequential expression generation；Search-enhanced or tree-based exploration；Reward shaping and variance reduction；Distributional or risk-aware RL；Pool-aware symbolic alpha generation。
- **Difference from prior work:** 相比 GP，RiskMiner 不是简单 mutation / crossover，而是把表达式生成建模为 MDP，并用 policy-guided MCTS 搜索；相比 Alpha101，它是自动发现公式；相比 AlphaGen，它更像 neural-guided symbolic search；相比 end-to-end forecasting models，它生成可解释公式 alpha，而不是直接预测收益分数。
- **Whether it is more RL-like or search-like:** RiskMiner 是 **search-like 更强的 RL alpha mining 方法**。它有 policy gradient 更新，但核心优势主要来自 MCTS 对离散表达式空间的结构化搜索。

一句话定位：

> AlphaGen 是 “PPO-based symbolic alpha generator”；RiskMiner 是 “MCTS-guided risk-seeking symbolic alpha searcher”。

---

## 12. How to Incorporate This Paper into My Review

### Suggested section

- **Section:** RL-based formulaic alpha mining / Search-enhanced alpha generation / Risk-aware RL for alpha mining。
- **Reason:** RiskMiner 是 AlphaGen 之后非常典型的改进路线：不是改变 alpha 表达本身，而是改进搜索机制、reward density 和 policy optimization objective。

### Suggested one-line description

RiskMiner formulates formulaic alpha pool construction as a reward-dense MDP and combines MCTS with risk-seeking policy optimization to search for high-IC and low-correlation symbolic alphas.

### Suggested paragraph

RiskMiner extends the RL-based alpha mining paradigm by replacing purely neural policy generation with a risk-seeking Monte Carlo Tree Search framework. Similar to AlphaGen, it represents formulaic alphas as RPN token sequences and maintains a pool of synergistic alphas combined through a linear model. Its key methodological improvement is the design of a reward-dense MDP: valid intermediate expressions receive IC-based rewards penalized by mutual correlation with existing pool members, while terminal rewards are based on the IC of the updated composite alpha. The MCTS module exploits the tree structure of the discrete formula space, and the risk-seeking policy optimizes upper-quantile performance rather than expected reward. This makes the method particularly relevant to alpha discovery, where discovering a small number of exceptional formulas matters more than improving the average quality of generated formulas.

### Suggested comparison table row

| Paper | Venue | Core formulation | Representation | RL/Search mechanism | Reward type | Pool-aware? | Dynamic allocation? | Key contribution | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| RiskMiner | arXiv 2024 | Reward-dense MDP for formulaic alpha pool construction | RPN token sequence / implicit expression tree | MCTS guided by risk-seeking policy network | Intermediate IC - mutIC penalty + terminal composite-alpha IC | Yes | No, static optimized linear weights | Uses MCTS and upper-quantile policy optimization to improve exploration and best-case alpha discovery | Computational cost, incomplete trading-cost/turnover analysis, dependence on operator set |

---

## 13. My Overall Assessment

这篇论文主要有 **methodology value** 和 **future research inspiration value**，其次才是 implementation 和 evaluation value。

### Methodology value

RiskMiner 对你现在的 RL 挖因子 review 很重要，因为它指出了 AlphaGen 类方法的三个核心弱点：sparse reward、pure neural policy 不利用 symbolic expression search space 的树结构、expected reward objective 不完全符合 alpha discovery 的 best-case search 属性。这三个点都可以作为你 review 中组织文献脉络的主线。

### Implementation value

从实现角度，它提示你可以把 AlphaGen 的 PPO generator 替换或增强为：MCTS sampler、policy prior network、RPN legality checker、dense alpha evaluator、quantile-based risk-seeking policy update。

如果你未来想在 AlphaGen repo 上做改进，RiskMiner 的可迁移模块主要是：

```text
AlphaEnv / ExpressionBuilder
        ↓
legal token expansion + evaluator
        ↓
MCTS tree search
        ↓
risk-seeking policy prior
        ↓
alpha pool update
```

### Evaluation value

实验结果比较完整，包括 signal metrics、backtest、quantile sensitivity 和 ablation。但是从量化实盘角度，它还缺少 transaction cost 数值、turnover、capacity、neutralization、cross-market validation、factor decay / data snooping analysis。因此在 review 里可以认可它的方法贡献，但对实盘有效性保持谨慎。

### Final judgment

> RiskMiner is best viewed as a search-enhanced and risk-seeking extension of AlphaGen. Its strongest contribution is not a new alpha representation, but a better exploration and reward-learning mechanism for symbolic alpha pool construction.

---

## 14. Four Core Questions Answered

### 1. How does it define factor mining as a decision problem?

它把 alpha 公式生成定义成 token-by-token RPN sequence construction。状态是当前 token prefix，动作是选择下一个 token，转移是 deterministic append，reward 来自 intermediate formula IC/mutIC 和 terminal pool IC。

### 2. What exactly does the RL component learn?

RL component 学习的是 risk-seeking policy：

$$
\pi_{\text{risk}}(a\mid s;\theta),
$$

它在 MCTS 中同时作为 tree policy prior 和 rollout policy，用于指导哪些 token 更值得扩展和采样。

### 3. What weakness of AlphaGen or previous methods does it address?

它主要解决 AlphaGen 的 sparse reward、PPO 对离散表达式空间结构利用不足、expected reward objective 与 alpha discovery 目标不匹配这三个问题。

### 4. Does the experimental protocol support the paper's conclusion?

基本支持其 “signal-level 表现更好” 的结论，因为它在 CSI300/CSI500、5 日/10 日目标上报告了 10 次重复实验，并与 AlphaGen、GP、Alpha101 和多种机器学习模型比较。但对 “realistic trading profitability” 的支持还不够充分，因为 transaction cost、turnover、capacity 和中性化细节不够完整。

---

## 14. Supplementary Note: Understanding RiskMiner's MCTS Statistics \(N, P, Q, R\)

这一节用于补充理解论文第 4.3 节 **Risk-based Monte Carlo Tree Search**。如果之前只学习过最基础的 MCTS，通常每个节点只记录两个量：

$$
N_i = \text{访问次数}, \qquad V_i = \text{节点价值}.
$$

而 RiskMiner 中，每条边 \((s,a)\) 记录的是：

$$
\{N(s,a), P(s,a), Q(s,a), R(s,a)\}.
$$

这不是完全换了一套 MCTS 逻辑，而是把普通 MCTS 中的“节点价值”拆得更细，并额外加入了神经网络给出的先验概率 \(P\)。

### 14.1 普通 MCTS 的 \(N,V\) 与 RiskMiner 的 \(N,P,Q,R\) 对应关系

普通 MCTS 中，常见选择公式是：

$$
UCB_i = \bar V_i + c\sqrt{\frac{\log N}{N_i}},
$$

其中：

$$
\bar V_i = \frac{T_i}{N_i}
$$

表示第 \(i\) 个节点的平均价值。

RiskMiner 更接近 AlphaZero / MuZero 风格的 PUCT 写法。它不是把统计量主要放在节点上，而是放在边 \((s,a)\) 上。这里：

- \(s\)：当前已经生成的 RPN token 前缀；
- \(a\)：下一步选择的 token；
- \((s,a)\)：在当前公式前缀下选择某个下一 token 的动作边。

对应关系如下：

| 普通 MCTS | RiskMiner | 含义 |
|---|---|---|
| \(N_i\) | \(N(s,a)\) | 这条边被访问过多少次 |
| \(V_i\) 或 \(\bar V_i\) | \(Q(s,a)\) | 从这条边走下去的平均累计价值 |
| 通常没有 | \(P(s,a)\) | risk-seeking policy network 给出的动作先验概率 |
| 通常不单独存 | \(R(s,a)\) | 经过这条边后，如果形成合法表达式，立即得到的中间奖励 |

因此可以粗略记为：

$$
Q(s,a) \approx \text{普通 MCTS 中的平均价值 } \bar V_i.
$$

而 RiskMiner 额外加入：

$$
P(s,a), \qquad R(s,a).
$$

其中 \(P\) 用于引导搜索方向，\(R\) 用于记录 dense reward 中的即时反馈。

---

### 14.2 为什么 RiskMiner 把统计量存在边 \((s,a)\) 上？

公式生成问题本质上是 token-by-token 的序列决策问题。状态是当前已经生成的 token 序列：

$$
s_t = [\mathrm{BEG}, token_1, \ldots, token_t],
$$

动作是选择下一个 token：

$$
a_t = token_{t+1}.
$$

因此一条边 \((s,a)\) 表示：

> 在当前公式前缀 \(s\) 下，选择下一个 token \(a\)。

例如当前状态为：

$$
s = [\mathrm{BEG},\ \$close,\ 10],
$$

候选动作可能是：

$$
a_1 = Std, \qquad a_2 = Mean, \qquad a_3 = Max.
$$

于是三条边分别是：

$$
(s,Std), \qquad (s,Mean), \qquad (s,Max).
$$

RiskMiner 要判断的是：

> 当前公式前缀下，下一步选哪个 token 更可能生成高质量 alpha？

因此把统计量记录在边上，比只记录在节点上更贴合“动作选择”的问题。

---

### 14.3 四个量分别是什么？

#### 1. \(N(s,a)\)：访问次数

$$
N(s,a)
$$

表示 MCTS 过去选择过这条边多少次。

例如：

$$
s = [\mathrm{BEG},\ \$close,\ 10].
$$

如果过去 MCTS 有 20 次从这个状态选择 `Std`，那么：

$$
N(s,Std)=20.
$$

如果只选择过 3 次 `Mean`，那么：

$$
N(s,Mean)=3.
$$

它和普通 MCTS 中的访问次数 \(N_i\) 作用一致，用于平衡 exploitation 和 exploration。

---

#### 2. \(P(s,a)\)：先验概率

$$
P(s,a)
$$

是 risk-seeking policy network 给出的动作概率：

$$
P(s,a)=\pi_{risk}(a|s).
$$

例如当前状态：

$$
s=[\mathrm{BEG},\ \$close,\ 10].
$$

policy network 可能输出：

| action | \(P(s,a)\) |
|---|---:|
| `Std` | 0.40 |
| `Mean` | 0.25 |
| `Max` | 0.10 |
| `Log` | 0.02 |

这表示网络认为，在当前公式前缀下，`Std` 比 `Log` 更可能是有前途的下一步 token。

---

#### 3. \(Q(s,a)\)：平均累计价值

$$
Q(s,a)
$$

表示从状态 \(s\) 选择动作 \(a\) 之后，继续 rollout 生成完整公式所能得到的平均累计价值。

粗略地说：

$$
Q(s,a) \approx \mathbb E[\text{从 }(s,a)\text{ 继续生成完整公式后的总 reward}].
$$

它不是当前这一步的 IC，而是包含后续 rollout、intermediate rewards 和 terminal reward 的长期平均价值估计。

因此 \(Q(s,a)\) 更接近普通 MCTS 中的平均节点价值 \(\bar V_i\)。区别是 RiskMiner 用边价值 \(Q(s,a)\) 表示某个动作分支的长期前景。

---

#### 4. \(R(s,a)\)：即时奖励

$$
R(s,a)
$$

表示选择动作 \(a\) 后，当前这一步立即产生的 reward。

在 RiskMiner 中，如果选择 \(a\) 后，新 token 序列刚好构成一个合法 RPN 表达式，就可以把当前表达式当作一个临时 alpha，计算中间奖励：

$$
R(s,a)
=
\operatorname{IC}(f)
-
\lambda\frac{1}{k}\sum_{i=1}^k \operatorname{mutIC}(f,f_i).
$$

其中：

- \(f\)：当前 token prefix 解析得到的合法临时 alpha；
- \(\operatorname{IC}(f)\)：当前 alpha 对未来收益的预测能力；
- \(\operatorname{mutIC}(f,f_i)\)：当前 alpha 与 alpha pool 中第 \(i\) 个 alpha 的相关性；
- \(\lambda\)：相关性惩罚强度，论文中设为 0.1。

所以：

$$
R(s,a) \neq Q(s,a).
$$

\(R(s,a)\) 是这一步的局部即时奖励；\(Q(s,a)\) 是从这一步继续生成完整公式后的长期平均价值。

---

### 14.4 为什么需要先验概率 \(P\)？

普通 MCTS 通常只靠：

$$
\bar V_i + c\sqrt{\frac{\log N}{N_i}}
$$

来选择节点。这个方法在搜索空间较小时可行，但公式 alpha 的 token 空间非常大。当前状态下可能有很多候选 token，例如：

$$
\$open,\ \$close,\ \$volume,\ Add,\ Sub,\ Mul,\ Div,\ Std,\ Mean,\ Corr,\ Cov,\ldots
$$

如果完全随机扩展或只靠访问次数探索，效率会很低。因此 RiskMiner 引入 policy network 的输出 \(P(s,a)\)，作为一个“先验方向”。

它的作用是：

$$
\boxed{\text{告诉 MCTS：哪些 action 在当前状态下更值得优先尝试。}}
$$

例如：

$$
P(s,Std)=0.40, \qquad P(s,Log)=0.02.
$$

那么 MCTS 更倾向于优先探索：

$$
[\mathrm{BEG},\ \$close,\ 10,\ Std]
$$

而不是：

$$
[\mathrm{BEG},\ \$close,\ 10,\ Log].
$$

从直觉上看，`Std(close, 10)` 更像一个有意义的时序技术因子，而某些 token 路径可能语法上不合适或经济含义较弱。\(P\) 就是把 risk-seeking policy network 学到的历史搜索经验注入 MCTS。

---

### 14.5 RiskMiner 的 PUCT 选择公式

RiskMiner 使用类似 PUCT 的选择公式：

$$
a_t
=
\arg\max_a
\left[
Q(s,a)
+
P(s,a)\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
\right].
$$

这个公式由两部分组成：

$$
\text{选择分数}
=
\text{历史价值项}
+
\text{先验引导的探索项}.
$$

其中：

$$
Q(s,a)
$$

负责 exploitation，即利用历史上表现好的路径；

$$
P(s,a)\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
$$

负责 exploration，但这个 exploration 是被 policy prior 引导的。

与普通 UCB 对比：

| 项 | 普通 UCB | RiskMiner PUCT |
|---|---|---|
| 历史价值 | \(\bar V_i\) | \(Q(s,a)\) |
| 探索项 | \(c\sqrt{\frac{\log N}{n_i}}\) | \(P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}\) |
| 是否有神经网络先验 | 没有 | 有 |
| 探索方向 | 主要看访问次数少不少 | 访问少 + policy 认为有前途 |

因此，\(P\) 的意义是：

$$
\boxed{\text{不是所有访问少的 action 都同等值得探索，而是优先探索 policy 认为更可能有前途的 action。}}
$$

---

### 14.6 \(Q\) 和用 IC 计算的 \(R\) 的关系

这部分最容易混淆。

可以简单区分为：

$$
R(s,a)=\text{这一步的即时奖励},
$$

$$
Q(s,a)=\text{这条边长期累计奖励的平均估计}.
$$

例如一条 RPN 公式路径是：

$$
\mathrm{BEG}
\rightarrow \$close
\rightarrow 10
\rightarrow Std
\rightarrow \$open
\rightarrow Add
\rightarrow END.
$$

其中有些中间状态可以形成合法 alpha，于是产生 intermediate reward：

| 当前 token 序列 | 是否合法公式 | 奖励 |
|---|---:|---|
| \(\$close\) | 是 | \(r_1=\operatorname{IC}(close)-\lambda\operatorname{mutIC}\) |
| \(\$close,10\) | 否 | \(r_2=0\) |
| \(\$close,10,Std\) | 是 | \(r_3=\operatorname{IC}(Std(close,10))-\lambda\operatorname{mutIC}\) |
| \(\$close,10,Std,\$open\) | 否，栈中有两个表达式 | \(r_4=0\) |
| \(\$close,10,Std,\$open,Add\) | 是 | \(r_5=\operatorname{IC}(Add(Std(close,10),open))-\lambda\operatorname{mutIC}\) |
| \(END\) | 终止 | \(r_T=\operatorname{IC}(\text{composite alpha pool})\) |

这些单步奖励 \(r_t\) 对应边上的：

$$
R(s_{t-1},a_t).
$$

而 \(Q(s,a)\) 是 backpropagation 后得到的平均累计值。

例如对于边：

$$
(s_2,Std),
$$

它的即时奖励可能是：

$$
R(s_2,Std)=0.035.
$$

但是从这条边继续 rollout，后面还可能得到：

$$
r_5=0.020, \qquad r_T=0.060.
$$

于是这一次搜索从这条边得到的累计回报可以近似写成：

$$
G = 0.035 + 0.020 + 0.060 = 0.115.
$$

然后用这个 \(G\) 去更新 \(Q(s_2,Std)\)。如果这是第一次访问，则：

$$
Q(s_2,Std)=0.115.
$$

如果之前已经访问过很多次，则做平均更新：

$$
Q_{new}(s,a)
=
\frac{N(s,a)Q_{old}(s,a)+G}{N(s,a)+1}.
$$

所以：

$$
\boxed{R \text{ 是单步局部奖励，} Q \text{ 是多次搜索后对长期总收益的平均估计。}}
$$

---

### 14.7 一个具体数值例子

假设当前状态为：

$$
s=[\mathrm{BEG},\ \$close,\ 10],
$$

候选动作是：

$$
Std, \qquad Mean.
$$

policy network 给出：

$$
P(s,Std)=0.4, \qquad P(s,Mean)=0.2.
$$

当前树统计量为：

| action | \(N(s,a)\) | \(P(s,a)\) | \(Q(s,a)\) | \(R(s,a)\) |
|---|---:|---:|---:|---:|
| `Std` | 10 | 0.4 | 0.080 | 0.035 |
| `Mean` | 2 | 0.2 | 0.060 | 0.025 |

总访问次数为：

$$
\sum_b N(s,b)=12.
$$

RiskMiner 的选择分数为：

$$
Score(Std)
=0.080 + 0.4\frac{\sqrt{12}}{1+10}
=0.080+0.4\frac{3.464}{11}
=0.206.
$$

$$
Score(Mean)
=0.060 + 0.2\frac{\sqrt{12}}{1+2}
=0.060+0.2\frac{3.464}{3}
=0.291.
$$

虽然 `Std` 的 \(Q\) 更高，但 `Mean` 访问次数更少，因此探索项更大，这一次可能选择 `Mean`。这就是 MCTS 的核心思想：

$$
\text{既利用高价值路径，也探索访问较少的潜在路径。}
$$

但 RiskMiner 的探索不是均匀探索。因为有 \(P(s,a)\)，如果 policy network 认为某个 action 很没前途，它的探索项也会比较小。

---

### 14.8 为什么不直接用 \(R\) 选动作，而还要 \(Q\)？

因为一个 token 当前看起来 reward 高，不代表最终公式好。

例如：

$$
s=[\mathrm{BEG},\ \$close].
$$

单独 `close` 可能 IC 不错：

$$
R=0.03.
$$

但是继续生成下去以后，可能变成：

$$
Log(Log(close))
$$

或者：

$$
close / close,
$$

最终公式没有意义。

反过来，有些中间步骤暂时没有 reward，例如：

$$
[\$close,10].
$$

因为它还不是合法公式，所以：

$$
R=0.
$$

但后面接一个 `Std` 后：

$$
[\$close,10,Std]
$$

可能形成一个有效技术因子：

$$
Std(close,10).
$$

所以如果只看 \(R\)，agent 会短视；如果看 \(Q\)，MCTS 会考虑这条路径未来能不能长成好公式。

因此：

$$
\boxed{R \text{ 评价当前一步，} Q \text{ 评价当前选择的长期前景。}}
$$

---

### 14.9 为什么不直接用 \(Q\)，还要存 \(R\)？

因为 RiskMiner 是 dense-reward MDP。

它不只是最后 `END` 时给一次 reward，而是中间合法表达式也会给 reward。为了在 backpropagation 时计算累计回报，需要知道路径上每一步的即时 reward：

$$
G_k
=
\sum_{i=0}^{l-1-k}\gamma^i r_{k+1+i} + v_l.
$$

其中这些 \(r\) 就来自边上的 \(R(s,a)\)。

所以：

$$
R(s,a)
$$

是计算累计回报 \(G_k\) 的原材料；

$$
Q(s,a)
$$

是累计回报经过多次 MCTS 访问后得到的平均估计。

---

### 14.10 公式生成场景中的完整例子

假设当前公式前缀为：

$$
s=[\mathrm{BEG},\ \$close,\ 10],
$$

候选动作：

$$
a=Std.
$$

选择后得到：

$$
s'=[\mathrm{BEG},\ \$close,\ 10,\ Std].
$$

这个序列是合法公式：

$$
f=Std(close,10).
$$

于是可以计算即时奖励：

$$
R(s,Std)
=
\operatorname{IC}(Std(close,10))
-
\lambda\cdot avgMutIC(Std(close,10),\mathcal P).
$$

例如：

$$
\operatorname{IC}=0.045, \qquad avgMutIC=0.10, \qquad \lambda=0.1.
$$

那么：

$$
R(s,Std)=0.045-0.1\times0.10=0.035.
$$

这只是当前边的即时 reward。

之后 rollout 继续生成：

$$
[\mathrm{BEG},\ \$close,\ 10,\ Std,\ \$open,\ Add,\ END].
$$

完整公式为：

$$
Add(Std(close,10),open).
$$

终止时将这个公式加入 alpha pool，并计算组合 alpha 的 IC：

$$
Reward_{end}=0.060.
$$

那么这次从 \((s,Std)\) 出发的累计价值可以近似写成：

$$
G=0.035+0.060=0.095.
$$

用 \(G\) 更新 \(Q(s,Std)\)。如果原来：

$$
N(s,Std)=4, \qquad Q(s,Std)=0.080,
$$

则更新后：

$$
Q_{new}(s,Std)
=
\frac{4\times0.080+0.095}{5}=0.083,
$$

并且：

$$
N_{new}(s,Std)=5.
$$

---

### 14.11 最终总结

普通 MCTS 中，每个节点通常只存：

$$
N, \qquad V.
$$

RiskMiner 将它扩展为每条边存：

$$
N(s,a), \qquad P(s,a), \qquad Q(s,a), \qquad R(s,a).
$$

四者含义为：

$$
\boxed{N(s,a)=\text{访问次数}}
$$

$$
\boxed{P(s,a)=\text{policy network 认为这个 action 有多值得优先尝试}}
$$

$$
\boxed{R(s,a)=\text{当前这一步生成合法表达式时，用 IC-mutIC 算出的即时奖励}}
$$

$$
\boxed{Q(s,a)=\text{从这条边继续生成完整公式后，累计 reward 的平均价值}}
$$

三者关系可以压缩为：

$$
\boxed{
P \text{ 决定优先往哪里探索，}
\quad
R \text{ 提供每一步的局部反馈，}
\quad
Q \text{ 记录这条路径长期来看是否真的好。}
}
$$

这也是 RiskMiner 相比普通 MCTS 的关键差异：它不是盲目搜索公式树，而是用 risk-seeking policy 的先验概率 \(P\) 引导探索，并用 dense reward 的即时反馈 \(R\) 与累计价值 \(Q\) 共同更新搜索树。

---

# Appendix B: Risk-Policy-Network 的输入、输出、架构、损失与训练迭代

本节补充理解 RiskMiner 中的 **risk-policy-network**。它本质上是一个 token-level policy network：

$$
\pi_{\text{risk}}(a_t\mid s_t;\theta)
$$

它输入当前已经生成的 RPN token 前缀，输出下一个 token 的概率分布。它和 PPO/AlphaGen 中的 policy network 都是“公式生成策略网络”，但训练目标不同：PPO 通常优化平均累计 reward，而 RiskMiner 的 risk-policy-network 优化的是 reward 分布的高分位数，即 best-case performance。

---

## B.1 这个 network 到底学什么？

它学习的是：

$$
\boxed{\text{在当前公式前缀 }s_t\text{ 下，下一个 token }a_t\text{ 应该选什么。}}
$$

例如当前状态是：

$$
s_t=[BEG,\ \$close,\ 10]
$$

risk-policy-network 输出一个 categorical distribution：

$$
\pi_{\text{risk}}(\cdot\mid s_t)
$$

例如：

| Candidate token | Probability |
|---|---:|
| `Std` | 0.40 |
| `Mean` | 0.22 |
| `Rank` | 0.15 |
| `Add` | 0.01 |
| `END` | 0.00 |

这表示网络认为：在当前前缀下，选 `Std` 更可能通向高 reward alpha。

---

## B.2 Input 是什么？

输入是当前状态 $s_t$，即已经生成的 token 序列：

$$
s_t=[BEG,a_1,a_2,\ldots,a_{t-1}]
$$

在论文场景中，token 包括：

- price/volume features：`$open`, `$high`, `$low`, `$close`, `$volume`, `$vwap`；
- constants：例如 `-30`, `-10`, `-5`, `-2`, `-1`, `-0.5`, `0.5`, `1`, `2`, `5`, `10`, `30`；
- time windows / deltas：例如 `1`, `5`, `10`, `20`, `30`, `40`, `50`；
- operators：例如 `Add`, `Sub`, `Mul`, `Div`, `Std`, `Mean`, `Corr`, `Cov`, `Rank`, `Ref` 等；
- special tokens：`BEG`, `END`。

例如：

$$
[BEG,\ \$close,\ 10,\ Std]
$$

这个前缀对应已经生成了一个部分公式：

$$
Std(close,10)
$$

注意：policy network 的输入不是股票原始 OHLCV 张量，而是 symbolic token sequence。股票数据只在 evaluator 里用于计算 IC、mutIC 和 pool IC，不直接作为 policy network 的输入。

---

## B.3 Output 是什么？

输出是所有 token 的概率分布：

$$
p_t=\pi_{\text{risk}}(\cdot\mid s_t;\theta)\in\mathbb R^{|\mathcal A|}
$$

其中 $|\mathcal A|$ 是 token vocabulary 的大小。

如果 token vocabulary 是：

$$
\mathcal A=\{\$open,\$close,\$volume,1,5,10,Std,Mean,Add,Sub,END,\ldots\}
$$

那么 network 输出：

$$
[p(\$open),p(\$close),p(10),p(Std),p(Add),p(END),\ldots]
$$

这个输出有两个用途：

1. **在 MCTS expansion / selection 中作为 prior probability：**

$$
P(s,a)=\pi_{\text{risk}}(a\mid s;\theta)
$$

2. **在 rollout 阶段采样动作：**

$$
a_t\sim \pi_{\text{risk}}(\cdot\mid s_t;\theta)
$$

因此 risk-policy-network 不是单独替代 MCTS，而是给 MCTS 提供搜索方向。

---

## B.4 网络架构是什么？

论文中的结构可以概括为：

$$
\boxed{\text{Token sequence}\rightarrow \text{GRU feature extractor}\rightarrow \text{MLP policy head}\rightarrow \text{Categorical distribution over tokens}}
$$

具体流程是：

```text
RPN token ids
    ↓
Embedding layer
    ↓
4-layer GRU
    ↓
last hidden state h_t
    ↓
MLP policy head
    ↓
logits over token vocabulary
    ↓
softmax
    ↓
π_risk(a|s)
```

论文实现细节给出：

- GRU feature extractor：4 层；
- GRU hidden dimension：64；
- policy head：MLP，两层 hidden layer；
- MLP hidden dimension：32。

数学形式可以写成：

$$
e_i=Emb(token_i)
$$

$$
h_t=GRU(e_1,e_2,\ldots,e_t)
$$

$$
\ell_t=MLP(h_t)
$$

$$
\pi_{\text{risk}}(a\mid s_t;\theta)=\frac{\exp(\ell_{t,a})}{\sum_{a'}\exp(\ell_{t,a'})}
$$

其中 $\ell_t$ 是 action logits。

---

## B.5 训练数据从哪里来？

risk-policy-network 不是用监督学习标签训练的，而是用 MCTS 采样得到的 trajectories 训练。

一条 trajectory 是：

$$
\tau=\{s_0,a_1,r_1,s_1,\ldots,s_{T-1},a_T,r_T,s_T\}
$$

例如：

```text
BEG → $close → 10 → Std → $open → Add → END
```

MCTS 搜索过程中会把这些 trajectories 放进 replay buffer：

$$
\mathcal B=\{\tau_1,\tau_2,\ldots,\tau_m\}
$$

然后用每条 trajectory 的 cumulative reward 来更新 risk-policy-network。

完整交替关系是：

$$
\boxed{\text{MCTS 采样 trajectories}\rightarrow \text{训练 risk-policy-network}\rightarrow \text{更新后的 policy 再指导 MCTS}}
$$

---

## B.6 每条 trajectory 的 reward 怎么来？

一条 trajectory 的累计 reward 是：

$$
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t
$$

在 RiskMiner 中，$r_t$ 主要来自两类。

第一类是中间合法表达式的 intermediate reward：

$$
r_t=IC(f_t)-\lambda\frac{1}{k}\sum_{i=1}^k mutIC(f_t,f_i)
$$

第二类是终止时加入 alpha pool 后的 terminal reward：

$$
r_T=IC(\text{composite alpha pool})
$$

因此一条 trajectory 的 cumulative reward $R(\tau)$ 大致反映：

$$
\text{这条公式生成路径有没有产生高 IC、低冗余、能提升 pool 的 alpha。}
$$

---

## B.7 Objective / optimal 是什么？

普通 policy gradient 优化的是平均累计收益：

$$
J_{\text{mean}}(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R(\tau)]
$$

RiskMiner 认为 alpha mining 不应该只看平均公式质量，而应该关注 best-case performance，也就是生成结果中的高分位 reward。

因此它定义 risk-seeking objective：

$$
J_{\text{risk}}(\theta;\alpha)=q(\theta;1-\alpha)
$$

其中 $q(\theta;1-\alpha)$ 是 cumulative reward 分布的高分位数。

直观地说：

$$
\boxed{\theta^*=\arg\max_\theta \text{Upper-quantile of }R(\tau)}
$$

而不是：

$$
\theta^*=\arg\max_\theta \text{Mean of }R(\tau)
$$

这就是 risk-seeking 的含义：它更关注 reward 分布右尾中的优秀 alpha，而不是所有生成公式的平均表现。

---

## B.8 Quantile 怎么估计？

论文用递推式估计当前 reward 分布的分位点：

$$
q_{i+1}=q_i+\beta\left(1-\alpha-\mathbf 1\{R(\tau_i)\le q_i\}\right)
$$

其中：

- $q_i$：当前估计的分位点；
- $\beta$：quantile estimation 的学习率；
- $R(\tau_i)$：第 $i$ 条 trajectory 的累计 reward；
- $\mathbf 1\{R(\tau_i)\le q_i\}$：判断当前样本是否低于分位点。

这个递推的目标是让：

$$
P(R(\tau)\le q)\approx 1-\alpha
$$

例如如果 $\alpha=0.1$，那么关注的是：

$$
q_{0.9}
$$

也就是 top 10% reward 的阈值。

---

## B.9 Loss 是什么？

论文写的是 gradient ascent 形式，而不是传统监督学习里的 loss。

它给出的更新方向是：

$$
D(\tau;\theta,q)=-\mathbf 1\{R(\tau)\le q\}\sum_{t=1}^{T}\nabla_\theta\log\pi(a_t\mid s_t;\theta)
$$

然后更新：

$$
\theta_{i+1}=\theta_i+\eta D(\tau_i;\theta_i,q_i)
$$

其中 $\eta$ 是 policy network 的学习率。论文实现细节中，quantile regression 的学习率为 $\beta=0.01$，网络参数更新学习率为 $0.001$。

如果改写成 PyTorch 中的 loss minimization，可以写成：

$$
\mathcal L_{\text{risk}}(\theta)=\mathbf 1\{R(\tau)\le q\}\sum_{t=1}^{T}\log\pi_\theta(a_t\mid s_t)
$$

然后用 gradient descent 最小化这个 loss。

注意这个 loss 和普通 REINFORCE 不同。普通 REINFORCE 通常类似：

$$
\mathcal L_{\text{REINFORCE}}(\theta)=-R(\tau)\sum_t\log\pi_\theta(a_t\mid s_t)
$$

RiskMiner 的 risk-policy loss 更像是：

$$
\mathcal L_{\text{risk}}(\theta)=\mathbf 1\{R(\tau)\le q\}\sum_t\log\pi_\theta(a_t\mid s_t)
$$

---

## B.10 这个 loss 的直觉

这个更新方向一开始比较反直觉，因为它不是显式强化高 reward trajectory，而是惩罚低于分位点阈值的 trajectory。

原因是它在优化 reward 分布的分位数。设：

$$
F_R(q;\theta)=P(R(\tau)\le q)
$$

如果想提高高分位点 $q$，本质上要让低于当前 threshold 的轨迹概率变小。因此更新方向可以理解为：

$$
\boxed{\text{降低低 reward trajectory 的生成概率。}}
$$

如果一条 trajectory 的 reward 低于当前 quantile threshold $q$，那么训练会压低这条轨迹中 action 的 log-probability。

如果一条 trajectory 的 reward 高于 $q$，则：

$$
\mathbf 1\{R(\tau)\le q\}=0
$$

这条轨迹不被惩罚。相对而言，高 reward 轨迹的概率质量会上升。

因此它的逻辑不是“显式奖励好轨迹”，而是：

$$
\boxed{\text{惩罚低于高分位阈值的轨迹，让策略分布向右尾移动。}}
$$

---

## B.11 一个 batch 例子

假设 MCTS 采样得到 5 条 trajectory：

| Trajectory | Formula | Cumulative reward $R(\tau)$ |
|---|---|---:|
| $\tau_1$ | `close` | 0.010 |
| $\tau_2$ | `Std(close,10)` | 0.035 |
| $\tau_3$ | `Corr(close,volume,20)` | 0.060 |
| $\tau_4$ | `Add(Std(close,10),open)` | 0.080 |
| $\tau_5$ | `Log(volume)` | -0.010 |

假设当前 high-quantile threshold 是：

$$
q=0.050
$$

那么低于 $q$ 的轨迹是：

$$
\tau_1,\tau_2,\tau_5
$$

这些轨迹会被惩罚：

$$
\mathcal L=\sum_{\tau_i:R_i\le q}\sum_t\log\pi_\theta(a_t\mid s_t)
$$

优化后，policy 会降低这些轨迹中的 token 选择概率。例如 `Log(volume)` 表现差，那么以后在类似状态下选 `Log` 的概率会下降。

高于 $q$ 的轨迹：

$$
\tau_3,\tau_4
$$

不会被这个 loss 惩罚。相对而言，策略会更容易保留或偏向这些高 reward 路径。

---

## B.12 完整训练迭代流程

RiskMiner 的训练 pipeline 可以写成：

```text
Initialize alpha pool F
Initialize risk-policy-network π_risk(a|s; θ)
Initialize quantile estimate q

For each mining iteration:
    1. Reset MCTS tree root = BEG
    2. Empty replay buffer B

    3. Run MCTS search cycles:
        a. Selection:
           choose action using PUCT:
           Q(s,a) + P(s,a)*sqrt(sum_b N(s,b))/(1+N(s,a))

        b. Expansion:
           expand leaf node;
           set P(s,a)=π_risk(a|s;θ)

        c. Rollout:
           sample remaining tokens using π_risk

        d. Reward:
           compute intermediate reward and terminal pool reward

        e. Backpropagation:
           update N(s,a), Q(s,a), R(s,a)

        f. Store trajectory τ into replay buffer B

    4. For each trajectory τ in B:
        a. Compute cumulative reward R(τ)
        b. Update quantile estimate q
        c. Compute risk-policy gradient
        d. Update θ

    5. Use updated π_risk in next MCTS iteration
```

---

## B.13 PyTorch 风格伪代码

下面是一个近似实现逻辑，方便和 PPO 的训练做对比：

```python
# token_seq: [B, T]   当前轨迹中的 token 序列
# actions:   [B, T]   每一步实际选择的 token
# returns:   [B]      每条 trajectory 的累计 reward R(tau)
# q: scalar           当前 quantile estimate

logits = policy_net(token_seq)          # [B, T, vocab_size]
log_probs = torch.log_softmax(logits, dim=-1)

chosen_log_probs = log_probs.gather(
    dim=-1,
    index=actions.unsqueeze(-1)
).squeeze(-1)                           # [B, T]

trajectory_log_prob = chosen_log_probs.sum(dim=1)  # [B]

# indicator: 低于 quantile threshold 的轨迹被惩罚
bad_mask = (returns <= q).float()        # [B]

loss = (bad_mask * trajectory_log_prob).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()

# update quantile estimate
q = q + beta * ((1 - alpha) - (returns <= q).float().mean())
```

普通 REINFORCE 可能是：

```python
loss = -(returns * trajectory_log_prob).mean()
```

RiskMiner 风格更像是：

```python
loss = indicator_low_return * trajectory_log_prob
```

它的目标是减少低于分位阈值轨迹的概率。

---

## B.14 和 PPO / AlphaGen policy network 的区别

| 维度 | PPO / AlphaGen 风格 | RiskMiner risk-policy-network |
|---|---|---|
| 输入 | 当前 RPN token prefix | 当前 RPN token prefix |
| 输出 | 下一个 token 概率 | 下一个 token 概率 |
| 网络 | 常见为 LSTM/GRU + policy/value head | GRU + MLP policy head |
| 是否有 value head | PPO 通常有 critic/value head | 论文主要强调 policy network |
| 训练数据 | policy 自己 rollout 的 trajectories | MCTS 采样的 trajectories |
| 目标 | 最大化平均 reward | 最大化 reward 高分位数 |
| loss | PPO clipped surrogate | quantile policy gradient |
| 在搜索中的作用 | 直接采样生成公式 | 同时作为 MCTS tree policy 和 rollout policy |
| 关注点 | 平均表现 | best-case alpha discovery |

---

## B.15 一句话总结

RiskMiner 的 risk-policy-network 是一个 **GRU + MLP 的 token 生成策略网络**：输入当前 RPN token 前缀 $s_t$，输出下一个 token 的概率分布 $\pi_{\text{risk}}(a_t\mid s_t)$。它不通过 PPO 的平均 reward 目标训练，而是用 MCTS 采样的 trajectories 估计 cumulative reward 的高分位阈值 $q$，再通过 quantile policy gradient 惩罚低于 $q$ 的轨迹，从而让策略分布向高 reward 右尾移动；训练好的 policy 又作为 MCTS 的 prior $P(s,a)$ 和 rollout policy，指导下一轮公式搜索。

