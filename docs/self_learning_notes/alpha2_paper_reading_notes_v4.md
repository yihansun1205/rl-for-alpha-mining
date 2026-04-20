# Paper Reading Notes: Alpha2: Discovering Logical Formulaic Alphas using Deep Reinforcement Learning

## 1. One-Sentence Summary

This paper proposes **Alpha2**, an RL-guided MCTS framework for discovering logical formulaic alphas, to solve the problems of huge symbolic search space, high alpha correlation, and invalid financial formulas by representing alpha generation as an incremental **program construction** process with dimension-aware pruning and diversity-aware evaluation.

---

## 2. Core Problem

- **Task type:** Formulaic alpha discovery; the method mines a collection of formulaic alphas, but the core search process evaluates one candidate alpha program at a time.
- **Input:** Daily Chinese A-share OHLCV-style raw features: `open`, `close`, `high`, `low`, `volume`, `vwap`.
- **Output:** A set of generated formulaic alphas represented as alpha programs, which can be converted into expression trees and then used by a downstream combination model.
- **Main objective:** Search for formulaic alphas with high predictive IC, low redundancy with already discovered alphas, and logical dimensional consistency.
- **Financial target:** 20-day future stock return.
- **Single-factor or pool-aware:** The candidate alpha is generated individually, but its evaluation is partially pool-aware because the reward penalizes high correlation with the already mined alpha set.
- **Discovery-only or discovery-plus-deployment:** Mainly discovery. The paper explicitly states that Alpha2 generates alphas, while the downstream combination model can be customized. In experiments, XGBoost is used as the combination model.

核心理解：

Alpha2 不是像 AlphaGen 那样直接围绕「alpha pool 的组合性能」来定义 PPO 生成器，而是更接近 **AlphaZero / AlphaDev 风格的 neural-guided symbolic search**。它的主要改进点不是组合模型，而是：

1. 把公式 alpha 表示成 program instructions；
2. 用 MCTS 在 program space 中搜索；
3. 用 DRL 网络提供 policy prior 和 value estimate；
4. 在 MCTS expansion 时做 dimension check，提前剪枝；
5. 在 evaluator 中加入 diversity penalty，降低 alpha 之间的相关性。

---

## 3. MDP Design

### State

Alpha2 的状态 \(s_t\) 是当前已经构造到第 \(t\) 步的 alpha program。论文中说 state space \(S\) 包含所有可能的 alpha programs，每个 state 对应一个唯一的 alpha function \(\zeta\)，并且 state 是该 alpha function 的 vectorized representation。

可以写成：

\[
s_t = \mathrm{Vectorize}(\zeta_t)
\]

其中 \(\zeta_t\) 是当前部分构造出的 alpha program 对应的函数。

更具体地说，state 至少需要隐含包含：

- 当前 instruction sequence；
- 当前 register 中存储的中间表达式；
- 每个 register 中表达式的维度信息；
- 当前 program 是否可以结束；
- 当前 program 是否满足 operator/operand 的合法性约束。

### Action

动作 \(a_t\) 是选择下一条 program instruction。每个 instruction 是四元组：

\[
a_t = (\mathrm{Operator}, \mathrm{Operand}_1, \mathrm{Operand}_2, \mathrm{Operand}_3)
\]

Operator 包括 unary、binary、ternary、indicator；Operand 包括 scalar、matrix feature、register、Null。

例如：

```text
Start Null Null Null
Sub   close open Null
Sub   high  low  Null
Div   Reg0  Reg1 Null
End   Null  Null Null
```

对应公式：

\[
\frac{\mathrm{close}-\mathrm{open}}{\mathrm{high}-\mathrm{low}}
\]

### Transition

Transition 是确定性的。执行一条 instruction 后，当前 alpha program 被扩展成新的 alpha program：

\[
p(s_{t+1}\mid s_t,a_t)=1
\]

其中：

\[
s_{t+1} = \mathrm{AppendInstruction}(s_t,a_t)
\]

同时，环境会更新：

- instruction sequence；
- expression tree；
- register 内容；
- register dimension；
- action legality mask / possible child nodes。

### Reward

论文的 reward 是 evaluation metric 的增量：

\[
r(s_t,a_t,s_{t+1}) = \mathrm{Perf}(\zeta_{t+1})-\mathrm{Perf}(\zeta_t)
\]

这里 \(\mathrm{Perf}\) 不是简单 IC，而是经过 diversity discount 的 IC：

\[
\mathrm{Perf}(\zeta_t)
=
(1-\mathrm{MaxCorr}(z_t,G))\cdot \mathrm{IC}(z_t,\mu)
\]

其中：

\[
\mathrm{MaxCorr}(z_t,G)=\max_i \mathrm{IC}(z_t,z^i)
\]

\(G=\{z^1,z^2,\ldots,z^n\}\) 是已经发现的 alpha values 集合，\(\mu\) 是未来收益。

因此 Alpha2 的 reward 可以理解为：

\[
r_t =
\left[(1-\mathrm{MaxCorr}(z_{t+1},G))\mathrm{IC}(z_{t+1},\mu)\right]
-
\left[(1-\mathrm{MaxCorr}(z_t,G))\mathrm{IC}(z_t,\mu)\right]
\]

这比纯 terminal IC reward 更 dense，因为每扩展一步就可以根据当前 program 的 \(\mathrm{Perf}\) 变化给增量 reward。

### Termination

episode / program 结束条件主要是：

- agent 选择 `End` instruction；
- 达到最大 program length；
- 无合法动作可选；
- 搜索过程在 MCTS 中终止并进行 value backup。

`Start` 和 `End` 是 indicator operators，用于标记 alpha program 的起点和终点。

### Action Legality / Grammar Constraints

Alpha2 的核心 action constraint 是 **dimension-aware pruning**。例如：

- `close - open` 的 dimension 是 currency；
- `high` 的 dimension 也是 currency；
- `volume` 的 dimension 是 unit。

因此：

\[
\mathrm{Add}(\mathrm{Reg0}, \mathrm{high})
\]

合法，因为二者都是 currency；但：

\[
\mathrm{Add}(\mathrm{Reg0}, \mathrm{volume})
\]

非法，因为 currency 和 unit 不能直接相加。

这和 AlphaGen 的区别很关键：

- AlphaGen 用 RPN token sequence，只有完整表达式生成后才容易做 dimension check；
- Alpha2 在 program construction / MCTS node expansion 阶段就知道当前 register 的 dimension，因此可以提前剪枝；
- 这相当于把「金融语义合法性」前移到了搜索过程内部。

---

## 4. Representation

- **Representation type:** Alpha program / register-machine-like instruction sequence，可转换为 expression tree。
- **Token/operator set:**
  - Unary: `Abs`, `Ln`, `Sign`, ...
  - Binary: `Add`, `Sub`, `Mul`, `TS-Mean`, ...
  - Ternary: `Correlation`, `Covariance`, ...
  - Indicator: `Start`, `End`
  - Operands: scalar constants, matrix features, registers, `Null`
- **Validity mechanism:**
  - instruction format 固定为四元组；
  - register 管理中间结果；
  - operator arity 决定需要几个 operands；
  - `Null` 用于占位；
  - dimension information 跟随 register 一起维护；
  - MCTS expansion 时根据 dimension consistency 剪枝非法节点。
- **Advantages:**
  1. 比 RPN sequence 更容易知道每一步对应表达式树中的位置；
  2. 可以在生成过程中做 semantic pruning；
  3. 适合和 MCTS 结合，因为 partial program naturally defines a search-tree node；
  4. 能表达更深、更复杂的公式结构；
  5. register mechanism 让中间表达式可复用。
- **Limitations:**
  1. 搜索空间仍然很大；
  2. 表示能力强依赖 operator set 和 dimension rules；
  3. dimension consistency 只保证形式逻辑，不保证经济逻辑；
  4. register assignment 是隐式规则，复现时需要仔细读代码；
  5. 对 cross-sectional normalization、delay、winsorization、缺失值处理等金融工程细节，论文描述不够充分。

与 AlphaGen 的表示差异：

| 维度 | AlphaGen | Alpha2 |
|---|---|---|
| 表示方式 | RPN token sequence | Program instruction sequence |
| 每步动作 | 选择下一个 token | 选择下一条 instruction |
| 结构信息 | 扁平序列，树结构隐式 | program 可直接转 expression tree |
| 合法性检查 | grammar mask，dimension check 较难提前做 | expansion 阶段可以做 dimension pruning |
| 搜索方式 | PPO sampling | DRL-guided MCTS |
| 核心 inductive bias | 顺序生成公式 | 程序合成 + register + dimension system |

---

## 5. Algorithm Mechanism

- **RL/search method:** DRL-guided MCTS，类似 AlphaZero / AlphaDev。
- **On-policy or off-policy:** 不是标准 PPO/A2C 这种纯 on-policy 算法；更接近 AlphaZero-style search + neural network training。MCTS 产生 improved policy / value targets，网络再学习这些搜索结果。
- **Learned components:**
  1. Policy head：输出每个 action / instruction 的 prior probability；
  2. Value head：估计当前 partial alpha program 的 expected cumulative reward。
- **Search component:** MCTS 在 alpha program tree 上展开、模拟、评估和 backup。
- **Training loop:**
  1. 从空 program `Start` 开始；
  2. 当前 state \(s_t\) 输入 neural network；
  3. 网络输出 action prior \(P(a\mid s_t)\) 和 value \(V(s_t)\)；
  4. MCTS 根据 prior、value、当前 Q 估计选择要展开的 instruction；
  5. expansion 时使用 dimension check 剪掉非法动作；
  6. 对展开后的 partial / complete program 计算 \(\mathrm{Perf}\)；
  7. 进行 value backup；
  8. MCTS 产生更强的 action distribution；
  9. 用搜索产生的数据训练 policy/value network；
  10. 输出 top alphas，并用 XGBoost 等组合模型形成最终 trading signal。

### Key update objective

论文没有像 PPO 那样给出显式 clipped surrogate loss，而是沿用 AlphaZero / AlphaDev 风格。可以把学习目标概括为：

\[
\min_\theta
\left[
\left(V_\theta(s)-\hat{V}^{\mathrm{MCTS}}(s)\right)^2
-
\pi^{\mathrm{MCTS}}(\cdot\mid s)^\top \log P_\theta(\cdot\mid s)
\right]
\]

其中：

- \(P_\theta(\cdot\mid s)\)：policy head 输出的 action prior；
- \(V_\theta(s)\)：value head 输出的 state value；
- \(\pi^{\mathrm{MCTS}}\)：MCTS 搜索后得到的 improved policy；
- \(\hat V^{\mathrm{MCTS}}\)：由搜索/rollout/evaluator backup 得到的 value target。

### MCTS value backup modification

Alpha2 的关键不是普通 MCTS mean backup，而是 mean-max 混合 backup：

\[
Q(s,a)
=
r(s,a)
+
\beta \cdot \mathrm{mean}(V_s)
+
(1-\beta)\cdot \max(V_s)
\]

其中：

- \(V_s=\{v_1,\ldots,v_k\}\) 是该 node 记录的 top-k value backup；
- \(\beta\in[0,1]\) 控制 mean 和 max 的权衡；
- \(\beta=1\) 更保守，接近 average performance；
- \(\beta=0\) 更激进，接近 best-case search；
- top-k values 用 min-heap 维护，单次更新复杂度为 \(O(\log k)\)。

这可以理解为 Alpha2 的 robustness design：

- 只用 mean：早期大量无效 alpha 使 value 估计偏低，学习慢；
- 只用 max：容易追逐偶然高 IC、参数敏感的 alpha；
- mean-max mixture：在探索高潜力表达式和避免过拟合之间折中。

---

## 6. Reward Design

- **Reward signal:** diversity-adjusted IC improvement。
- **Terminal or intermediate:** 论文定义为每一步 evaluation metric 的增量，因此比纯 terminal reward 更 dense；但实际是否每个 partial program 都能稳定评估，取决于 implementation。
- **Pool-aware or standalone:** 半 pool-aware。它不是 AlphaGen 那种 downstream pool-combination reward，而是在单 alpha IC 上乘以与已发现 alpha 集合的低相关性因子。
- **Risk-aware or uncertainty-aware:** 不是传统意义上的 risk-aware。没有显式 VaR、CVaR、drawdown、turnover、transaction cost、risk exposure 约束。robustness 主要来自 MCTS value backup 的 mean-max mixture。
- **Shaping mechanism:** reward shaping 来自：
  1. incremental reward：\(\mathrm{Perf}(\zeta_{t+1})-\mathrm{Perf}(\zeta_t)\)；
  2. diversity discount：\((1-\mathrm{MaxCorr})\)；
  3. dimension pruning：虽然不是 reward，但通过 action feasibility 改变搜索分布；
  4. mean-max backup：不是 reward 本身，但影响搜索偏好。
- **Possible reward bias:**
  1. 仍然高度依赖 training-period IC；
  2. \((1-\mathrm{MaxCorr})\cdot \mathrm{IC}\) 是手工 scalarization；
  3. 若 IC 为负，相关性折扣的解释会变复杂；
  4. MaxCorr 只惩罚与已有 alphas 的最大相关，可能忽略整体多重共线性；
  5. 没有直接优化交易成本、换手、容量、行业/市值暴露；
  6. diversity 以 Pearson correlation 衡量，未必等价于组合模型中的边际贡献。

### 和 AlphaGen reward 的关键区别

| 维度 | AlphaGen | Alpha2 |
|---|---|---|
| Reward 核心 | alpha 加入 pool 后的组合性能提升 | diversity-adjusted standalone IC |
| 是否直接优化组合模型 | 是 | 否，组合模型在生成后使用 |
| 是否惩罚相关性 | 通过 pool / synergy 间接处理 | 显式 MaxCorr penalty |
| reward 稀疏性 | 主要 terminal reward | 增量 Perf reward，更 dense |
| 核心瓶颈 | credit assignment / evaluator dependency | search cost / IC overfitting / scalarization |

---

## 7. Factor Pool and Combination

- **Pool maintained:** Alpha2 维护一个已发现 alpha set \(G\)，用于计算 MaxCorr diversity penalty；但它并不像 AlphaGen 那样维护一个可训练的 alpha pool 并在 RL 中直接优化组合模型。
- **Pool update rule:** 论文没有特别详细说明插入/删除规则。实验中使用 top 20 generated alphas ranked by IC 进入组合模型。
- **Factor selection/removal:** 主要按 IC 排序选择 top 20；相关性通过生成时 reward penalty 被提前控制。
- **Combination model:** XGBoost。论文也说 combination model 可以自定义为 linear regression、deep neural network、gradient boosting trees 等。
- **Static or dynamic weights:** XGBoost 在训练集上拟合，训练后固定用于 test set；不是动态 factor allocation。
- **Deployment realism:** 有 top-k/drop-n backtest，但没有充分讨论真实交易成本、冲击成本、停牌涨跌停、行业/风格中性、容量等细节。

可以写成：

\[
\hat r_{d,i}
=
\mathrm{XGBoost}\left(
f^{(1)}_{d,i},
f^{(2)}_{d,i},
\ldots,
f^{(20)}_{d,i}
\right)
\]

这里不是线性组合：

\[
F_t=\sum_k w_k f_t^{(k)}
\]

而是非线性 tree-based combination。权重不是 RL 学出来的，也不是按 regime 动态变化的。

---

## 8. Experiments and Evaluation

- **Dataset:** Chinese A-share daily data from baostock.
- **Universe:** CSI300 and CSI500 constituents.
- **Time split:**
  - Train: 2009/01/01–2018/12/31
  - Validation: 2019/01/01–2020/12/31
  - Test: 2021/01/01–2023/12/31
- **Prediction target:** 20-day future return.
- **Raw features:** `open`, `close`, `high`, `low`, `volume`, `vwap`.
- **Metrics:** IC and Rank IC. The paper also mentions MDD, turnover, and Sharpe as important metrics, but the main reported predictive metrics are IC and Rank IC.
- **Baselines:**
  - MLP
  - XGBoost
  - LightGBM
  - gplearn
  - AlphaGen
- **Transaction costs:** Not clearly specified in the main experiment table. Backtest uses top-k/drop-n with \(k=50,n=5\), but cost assumptions are not sufficiently detailed.
- **Turnover:** Mentioned as a relevant metric, but not central in reported results.
- **Random seeds:** Tables report mean ± std, implying multiple runs, but details of seeds are limited.
- **Robustness tests:**
  - CSI300 and CSI500;
  - train/validation/test temporal split;
  - IC/correlation comparison of generated alphas;
  - top-k/drop-n backtest on CSI300.
- **Code availability:** The paper states that code is available at GitHub `x35f/alpha2`.

### Main reported results

#### Alpha quality and correlation on CSI300

| Method | IC | Correlation |
|---|---:|---:|
| gplearn | 0.0164 ± 0.0167 | 0.7029 ± 0.1824 |
| AlphaGen | 0.0257 ± 0.0153 | 0.3762 ± 0.6755 |
| Alpha2 | 0.0407 ± 0.0219 | 0.1376 ± 0.3660 |

Interpretation:

Alpha2 improves both standalone IC and diversity. Its mean correlation is much lower than gplearn and AlphaGen, consistent with the MaxCorr penalty.

#### Test IC / RankIC

| Method | CSI300 IC | CSI300 RankIC | CSI500 IC | CSI500 RankIC |
|---|---:|---:|---:|---:|
| MLP | 0.0123 | 0.0178 | 0.0158 | 0.0211 |
| XGBoost | 0.0192 | 0.0241 | 0.0173 | 0.0217 |
| LightGBM | 0.0158 | 0.0235 | 0.0112 | 0.0212 |
| gplearn | 0.0445 | 0.0673 | 0.0557 | 0.0665 |
| AlphaGen | 0.0500 | 0.0540 | 0.0544 | 0.0722 |
| Alpha2 | 0.0576 | 0.0681 | 0.0612 | 0.0731 |

Interpretation:

Alpha2 在两个股票池上都有最高 IC；RankIC 上也略优于 AlphaGen/gplearn。但提升幅度不是数量级提升，更像是 representation + search + diversity + validity 带来的稳健增益。

---

## 9. Main Contributions

1. **The paper contributes a program-construction representation for formulaic alpha mining by addressing the weakness of RPN-style token generation in preserving expression-tree structure and enabling early validity checks.**

2. **The paper contributes DRL-guided MCTS for alpha search by addressing the large, sparse, and irregular symbolic search space where pure policy sampling or GP mutation can be inefficient.**

3. **The paper contributes dimension-aware generation-time pruning by addressing the practical problem that many statistically high-IC formulas may be semantically invalid, such as adding price and volume.**

4. **The paper contributes a diversity-aware evaluation function by addressing the tendency of previous methods to generate highly correlated alphas.**

5. **The paper contributes a mean-max mixed value backup mechanism by addressing the instability of mean-only MCTS value estimates in sparse formulaic alpha spaces and the over-aggressiveness of max-only search.**

---

## 10. Limitations

### Algorithmic limitations

1. **The algorithm is computationally heavier than PPO-only generation.**  
   DRL-guided MCTS requires repeated tree search, expansion, evaluation, and backup. This may be expensive when the operator set grows or when factor evaluation is slow.

2. **The paper does not fully specify the neural network architecture and training loss.**  
   It says the network outputs action distributions and value predictions, but the exact architecture, target construction, and optimization details are not as explicit as a PPO-style paper.

3. **Dimension consistency is necessary but not sufficient.**  
   It prevents obviously invalid operations such as adding price and volume, but it cannot guarantee economic interpretability, robustness, or tradability.

4. **Reward still depends heavily on historical IC.**  
   Even after diversity discount, the primary predictive signal is IC on historical data, so evaluator overfitting remains possible.

5. **The mean-max backup introduces hyperparameter sensitivity.**  
   \(\beta\) and top-k backup size control the exploration/robustness tradeoff. Poor tuning may lead to either conservative search or overfitting to lucky expressions.

### Evaluation / finance limitations

1. **Transaction cost and market frictions are not sufficiently modeled.**  
   The top-k/drop-n backtest restricts daily changes, but the paper does not provide a detailed cost-adjusted backtest protocol.

2. **No explicit risk exposure control.**  
   There is no clear industry, size, liquidity, beta, or style neutralization analysis.

3. **Combination model may introduce confounding.**  
   Since Alpha2 uses XGBoost as the downstream combiner, part of the test performance may come from nonlinear combination capacity rather than purely from alpha quality.

4. **Limited cross-market validation.**  
   Experiments are on China A-share CSI300/CSI500. There is no US, futures, crypto, or cross-country validation.

5. **Code availability is positive, but reproducibility still depends on implementation details.**  
   In particular, dimension rules, register mechanics, data preprocessing, and feature delay alignment must be checked carefully in code.

---

## 11. Relation to AlphaGen and Existing Literature

- **Relation to AlphaGen:**  
  Alpha2 can be viewed as addressing AlphaGen's structural and search-efficiency bottlenecks. AlphaGen formulates formula generation as RPN token-level policy generation with PPO, while Alpha2 formulates it as program synthesis and uses DRL-guided MCTS with dimension-aware pruning.

- **Main bottleneck addressed:**  
  Alpha2 targets three bottlenecks:
  1. sparse and huge formula search space;
  2. invalid or semantically illogical expressions;
  3. high correlation among generated alphas.

- **Paradigm classification:**
  1. Search-enhanced or tree-based exploration;
  2. Policy/value-guided symbolic program synthesis;
  3. Reward shaping and diversity-aware evaluation;
  4. Grammar / semantics constrained alpha generation.

- **Difference from prior work:**

| Prior work | Limitation | Alpha2 response |
|---|---|---|
| GP / gplearn | local optima, slow mutation, high correlation | MCTS + neural policy/value guidance |
| AlphaGen | RPN sequence, volatile value estimates, similar expressions | program representation + MCTS + diversity-aware reward |
| AlphaDev | program discovery but not financial alpha mining | adapted to formulaic alpha generation with IC/diversity/dimension |
| Symbolic regression MCTS | generic expression fitting | financial target + alpha correlation + dimension constraints |

- **Whether it is more RL-like or search-like:**  
  Alpha2 is more **search-like** than AlphaGen. The RL network does not directly output final formulas by pure sampling. Instead, it guides MCTS by providing action priors and value predictions. The final behavior is a neural-guided search process over program nodes.

### Compact comparison: AlphaGen vs RiskMiner vs Alpha2

| Paper | Core idea | Search/RL mechanism | Representation | Reward / value signal | Main improvement |
|---|---|---|---|---|---|
| AlphaGen | Generate synergistic alpha pool | PPO | RPN sequence | downstream pool-combination performance | pool-aware reward |
| RiskMiner | Risk-seeking MCTS for formulaic alphas | MCTS + risk-seeking policy | expression-tree search | quantile/risk-seeking objective | focuses search on high-reward tail |
| Alpha2 | Logical diverse alpha programs | DRL-guided MCTS | program instructions + registers | diversity-adjusted IC + mean-max backup | dimension pruning + program synthesis |

---

## 12. How to Incorporate This Paper into My Review

### Suggested section

- **Section:** Search-enhanced / tree-based RL alpha mining.
- **Reason:** Alpha2 is a representative method showing that RL alpha mining should not only be viewed as policy-gradient token generation. It reframes alpha discovery as program synthesis and uses MCTS plus semantic constraints to improve search quality.

### Suggested one-line description

Alpha2 formulates formulaic alpha discovery as a DRL-guided program synthesis problem, using MCTS, diversity-adjusted IC, and dimension-aware pruning to generate logical and low-correlation alphas.

### Suggested paragraph

Alpha2 extends RL-based formulaic alpha mining from token-level sequence generation to neural-guided program search. Instead of representing formulas as RPN sequences, it constructs alpha programs from instruction tuples containing operators, operands, and registers, which can be translated into expression trees. This representation makes it possible to maintain dimension information during generation and prune semantically invalid actions, such as adding price and volume, before the formula is fully generated. The search process is guided by a policy/value network in an AlphaZero/AlphaDev-style MCTS framework. Its reward is based on a diversity-adjusted IC metric, where the IC of a candidate alpha is discounted by its maximum correlation with already discovered alphas. Compared with AlphaGen, Alpha2 is less focused on directly optimizing downstream pool-combination performance and more focused on improving the structure, validity, and exploration efficiency of symbolic alpha search.

### Suggested comparison table row

| Paper | Venue | Core formulation | Representation | RL/Search mechanism | Reward type | Pool-aware? | Dynamic allocation? | Key contribution | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| Alpha2 | arXiv 2024 | Alpha discovery as program construction | Instruction tuples + registers + expression tree | DRL-guided MCTS, AlphaZero/AlphaDev-style policy-value search | Diversity-adjusted IC and mean-max value backup | Partially, via MaxCorr with discovered alpha set | No | Enables dimension-aware pruning during generation and improves diversity/logical validity | Still IC-driven; limited transaction-cost/risk modeling; computationally heavy |

---

## 13. My Overall Assessment

Alpha2 is mainly valuable for **methodology and future research inspiration**.

Its strongest contribution is not simply that it reports higher IC than AlphaGen, but that it changes the way we think about RL alpha mining:

\[
\text{formula generation}
\quad \rightarrow \quad
\text{program synthesis with semantic pruning}
\]

For your review document, Alpha2 should be used as the representative paper for the argument:

> After AlphaGen established PPO-based formula generation, later work began to treat alpha mining as a structured symbolic search problem. Alpha2 is an important example: it adds MCTS, program representation, dimension consistency, and diversity-aware evaluation.

My evaluation:

- **Methodology value:** High. Program construction + dimension pruning is a meaningful design.
- **Implementation value:** Medium to high. Code is available, but MCTS + JAX + dimension rules require careful reading.
- **Evaluation value:** Medium. Results are promising, but transaction cost/risk/capacity realism is not fully established.
- **Future research value:** High. The framework can be extended with:
  1. transaction-cost-aware reward;
  2. risk-neutralized IC;
  3. industry/size exposure constraints;
  4. dynamic alpha allocation;
  5. surrogate evaluator / model-based reward prediction;
  6. GFlowNet-style diverse formula generation;
  7. integration with AlphaGen-style pool marginal contribution.

---

## 14. Four Core Questions from the Task Template

### 1. How does it define factor mining as a decision problem?

Alpha2 defines factor mining as an MDP over alpha programs. State is the current partial program, action is the next instruction, transition appends the instruction deterministically, and reward is the improvement in a diversity-adjusted IC-based evaluation function.

### 2. What exactly does the RL component learn?

The neural network learns:

\[
P_\theta(a\mid s)
\]

as an action prior over possible next instructions, and:

\[
V_\theta(s)
\]

as the value estimate of a partial alpha program. These outputs guide MCTS rather than directly replacing search.

### 3. What weakness of AlphaGen or previous methods does it address?

It addresses:

1. AlphaGen's RPN representation makes generation-time dimension pruning difficult;
2. AlphaGen's MDP can produce volatile value estimates in sparse alpha space;
3. AlphaGen and GP methods can generate similar / correlated alphas;
4. GP methods are sensitive to initialization and can converge to local optima;
5. many generated alphas are statistically strong but semantically illogical.

### 4. Does the experimental protocol support the paper's conclusion?

Partially yes. The IC/RankIC and correlation results support the claim that Alpha2 finds stronger and more diverse alphas than gplearn and AlphaGen on CSI300/CSI500. However, the evidence is less complete for real deployment because transaction costs, capacity, turnover, risk exposure, and cross-market robustness are not deeply analyzed.


---

# Appendix: Alpha2 vs. RiskMiner — Why They Are Not the Same Although Both Use MCTS

## 1. Core Intuition

Alpha2 and RiskMiner are indeed similar at the highest level because both belong to **neural-guided MCTS-based alpha discovery**. Both methods use a search tree, policy/value guidance, and formula evaluation to explore a large symbolic alpha space.

However, their essential difference is not whether they use MCTS, but rather:

1. what object MCTS is searching over;
2. how the state and action are represented;
3. what weakness of previous methods they mainly address;
4. how reward / value is defined;
5. whether the main contribution is representation, validity, diversity, or risk-seeking search.

A compact distinction is:

\[
\boxed{\text{Alpha2 is program-construction MCTS for logical and diverse formulaic alphas.}}
\]

\[
\boxed{\text{RiskMiner is risk-seeking / high-reward MCTS for expression mining.}}
\]

In other words, Alpha2 is closer to **AlphaDev-style program synthesis**, while RiskMiner is closer to **AlphaZero-style expression search**.

---

## 2. Shared High-Level Framework

Both Alpha2 and RiskMiner can be written as:

\[
\text{Policy/Value Network} + \text{MCTS Search} + \text{Formula Evaluation}.
\]

The broad workflow is:

\[
s_t \rightarrow \text{MCTS expansion} \rightarrow \text{child-node evaluation} \rightarrow \text{backup} \rightarrow \text{select promising formula paths}.
\]

Therefore, both papers should be classified as:

\[
\boxed{\text{search-enhanced RL-based alpha discovery}.}
\]

But this shared MCTS shell hides substantial design differences.

---

## 3. Difference 1: Search Object and Representation

### Alpha2

Alpha2 reformulates formulaic alpha discovery as **alpha program generation**.

A candidate alpha is not directly represented as an RPN token sequence. Instead, it is represented as a sequence of program instructions:

\[
(\text{Operator},\ \text{Operand1},\ \text{Operand2},\ \text{Operand3}).
\]

Intermediate results are stored in registers. For example:

\[
\frac{close-open}{high-low}
\]

can be represented as:

```text
Sub close open Null -> Reg0
Sub high low Null -> Reg1
Div Reg0 Reg1 Null -> Reg0
```

Thus, Alpha2's state is closer to:

\[
s_t = \text{current partial alpha program + register state + dimension information}.
\]

### RiskMiner

RiskMiner is closer to expression-tree search. Its MCTS node usually corresponds to a partial expression, a formula construction state, or a node in the symbolic search tree.

The state is closer to:

\[
s_t = \text{current partial expression / current formula node}.
\]

### Key distinction

| Dimension | Alpha2 | RiskMiner |
|---|---|---|
| Search object | Alpha program | Formula / expression tree |
| Basic unit | Instruction tuple | Operator / operand / expression expansion |
| State | Partial program + register state | Partial expression / tree state |
| Main representation idea | Program synthesis | Expression search |

The key point is:

\[
\boxed{\text{Alpha2 first redesigns the representation of formulaic alpha.}}
\]

RiskMiner mainly focuses on how to search effectively within the expression space.

---

## 4. Difference 2: Search-Space Pruning and Logical Validity

Alpha2's most important methodological contribution is **dimension-aware pruning**.

For example:

\[
close + open
\]

is dimensionally valid because both operands are price-like variables. But:

\[
close + volume
\]

is usually dimensionally invalid because price and volume have different dimensions.

Alpha2 maintains the dimension of intermediate expressions in registers. During MCTS expansion, it can check whether an action is legal before expanding the node:

\[
\mathcal A_{legal}(s_t)
=
\{a_t: \text{dimension-compatible}(s_t,a_t)\}.
\]

This allows Alpha2 to prune invalid branches before fully generating and evaluating the formula.

RiskMiner may also use grammar or validity constraints, but its core emphasis is not dimension-aware program construction. Its main emphasis is using MCTS statistics and neural guidance to search high-reward formulas more effectively.

| Dimension | Alpha2 | RiskMiner |
|---|---|---|
| Main pruning logic | Dimension consistency and program legality | MCTS selection statistics and reward/value guidance |
| Validity focus | Strong; central contribution | Usually not the main contribution |
| Search-space reduction | Pre-evaluation pruning | Search-guided exploration |
| Main bottleneck addressed | Illogical formulas and wasted search | Inefficient exploration of high-reward formulas |

Thus:

\[
\boxed{\text{Alpha2 mainly improves the definition of the searchable space.}}
\]

\[
\boxed{\text{RiskMiner mainly improves the exploration mechanism inside the search space.}}
\]

---

## 5. Difference 3: Reward Design — Diversity vs. Risk-Seeking Search

### Alpha2 reward

Alpha2 does not evaluate an alpha only by standalone IC. It discounts the alpha's IC according to its maximum correlation with already discovered alphas:

\[
Perf(\zeta_t)
=
(1-MaxCorr(z_t,G))\cdot IC(z_t,\mu),
\]

where:

\[
G = \{z_1,z_2,\dots,z_n\}
\]

is the set of already discovered alpha values, and

\[
MaxCorr(z_t,G)=\max_i IC(z_t,z_i).
\]

This means that an alpha with high IC but high similarity to existing alphas will be penalized.

Example:

| Alpha | IC | Max correlation with existing alphas | Alpha2 score |
|---|---:|---:|---:|
| A | 0.06 | 0.10 | \(0.9\times0.06=0.054\) |
| B | 0.07 | 0.80 | \(0.2\times0.07=0.014\) |

Although alpha B has higher standalone IC, Alpha2 prefers A because A contributes more diversity.

### RiskMiner reward / value emphasis

RiskMiner is more focused on searching for high-reward / high-IC expressions through MCTS. In the MCTS node statistics, the common quantities can be interpreted as:

| Symbol | Meaning |
|---|---|
| \(N\) | Visit count |
| \(P\) | Prior probability from policy network |
| \(Q\) | Backed-up value estimate |
| \(R\) | Realized reward, often based on IC or another evaluation score |

RiskMiner's key question is how to guide search toward promising formulas with high reward, especially when the expression space is huge and sparse.

### Key distinction

| Dimension | Alpha2 | RiskMiner |
|---|---|---|
| Reward focus | IC adjusted by diversity penalty | High reward / risk-seeking formula search |
| Correlation penalty | Explicit and central | Not usually the central design |
| Pool awareness | Uses existing alpha set \(G\) for diversity discount | More focused on search statistics and high-value paths |
| Main learning signal | Logical + diverse + effective alpha | High-potential / high-reward alpha |

---

## 6. Difference 4: MCTS Value Backup

Alpha2 modifies the value backup in MCTS. Instead of using only the mean value of child rollouts or only the maximum value, it uses a mean-max mixture:

\[
Q(s,a)
=
r(s,a)
+
\beta\cdot mean(V_s)
+
(1-\beta)\cdot max(V_s),
\]

where \(\beta\in[0,1]\) controls the balance between robust average performance and optimistic maximum performance.

The motivation is that alpha search is sparse. Most generated formulas are weak or non-informative. If MCTS uses only the mean, promising sparse signals may be underestimated. If it uses only the maximum, the search may overfit to accidental high scores. The mean-max mixture is a compromise.

RiskMiner is closer to a standard AlphaZero-style MCTS selection process using node statistics such as:

\[
N(s,a),\quad P(s,a),\quad Q(s,a),\quad R(s,a).
\]

A typical selection rule has the form:

\[
a^* = \arg\max_a \left[Q(s,a)+U(s,a)\right],
\]

where the exploration bonus \(U(s,a)\) depends on policy prior and visit counts, for example:

\[
U(s,a)\propto P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}.
\]

Therefore:

| Dimension | Alpha2 | RiskMiner |
|---|---|---|
| Value backup specialty | Mean-max mixture | Node statistics \(N,P,Q,R\) and PUCT-like search |
| Search bias | Balance mean robustness and max optimism | Prior-guided and reward/value-driven exploration |
| Main concern | Sparse alpha rewards and parameter robustness | Efficient high-reward expression exploration |

---

## 7. Difference 5: Relation to AlphaGen

Both Alpha2 and RiskMiner can be interpreted as responses to limitations of AlphaGen, but they respond to different limitations.

### Alpha2's critique of AlphaGen

Alpha2 mainly argues that AlphaGen's RPN-based sequential generation has two problems:

1. it is hard to perform dimension checking before a formula is fully generated;
2. generated alphas may be statistically strong but logically questionable or highly correlated.

Therefore, Alpha2 replaces RPN-style formula generation with alpha-program generation and dimension-aware pruning.

### RiskMiner's critique of AlphaGen

RiskMiner is more concerned with exploration inefficiency. A pure policy-based generator such as PPO may fail to explore the vast symbolic space effectively. MCTS helps reuse search statistics and guide exploration toward promising formula regions.

### Key distinction

| Question | Alpha2 | RiskMiner |
|---|---|---|
| What AlphaGen weakness is addressed? | RPN representation, logical validity, alpha diversity | Inefficient exploration and missing high-reward formulas |
| What is changed? | Representation and validity constraints | Search algorithm and exploration mechanism |
| Main methodological move | Program synthesis + dimension pruning | MCTS-enhanced expression mining |

---

## 8. One-Sentence Comparison

A concise review-level comparison is:

> Alpha2 and RiskMiner both use MCTS for symbolic alpha discovery, but Alpha2 mainly contributes a program-based representation that enables dimension-aware pruning and diversity-aware evaluation, whereas RiskMiner mainly contributes a neural-guided / risk-seeking MCTS mechanism for more efficient high-reward expression search.

More compactly:

\[
\boxed{\text{Alpha2: better search space.}}
\]

\[
\boxed{\text{RiskMiner: better search strategy.}}
\]

---

## 9. Suggested Review Paragraph

Alpha2 and RiskMiner can both be categorized as search-enhanced reinforcement learning methods for symbolic alpha discovery because they use MCTS to explore large formula spaces under neural guidance. However, their methodological emphases differ. Alpha2 reformulates formulaic alpha generation as a program construction problem, where instruction tuples and registers enable dimensional consistency checks and early pruning of invalid branches. It further introduces a diversity-adjusted evaluation metric that penalizes formulas highly correlated with previously discovered alphas. In contrast, RiskMiner focuses more on improving the exploration mechanism itself: MCTS node statistics such as visit count, policy prior, value estimate, and realized reward are used to guide the search toward high-reward expressions. Therefore, Alpha2 mainly improves the representation and validity of the search space, while RiskMiner mainly improves the search dynamics within the expression space.

---

## 10. Suggested Comparison Table Row

| Paper | Core formulation | Representation | RL/Search mechanism | Reward type | Main contribution | Main limitation |
|---|---|---|---|---|---|---|
| Alpha2 | Formulaic alpha discovery as alpha program generation | Instruction sequence + registers + expression tree | DRL-guided MCTS with mean-max value backup | IC discounted by max correlation with existing alpha set | Defines a logically constrained and diversity-aware alpha search space | Deployment evaluation still lacks detailed transaction-cost, turnover, and capacity analysis |
| RiskMiner | Formulaic alpha discovery as MCTS-guided expression search | Expression tree / symbolic search space | Neural-guided MCTS with node statistics such as \(N,P,Q,R\) | High-reward / IC-oriented search signal | Improves exploration efficiency and searches for high-potential formulas | The quality of discovered formulas still depends strongly on reward/evaluator design and may overfit to high in-sample scores |

---

# Appendix B: Understanding Alpha2 MDP with Code-Level Mapping

This appendix explains Section 4.1.3 of the Alpha2 paper, namely how the paper formulates alpha generation as a Markov decision process (MDP), and maps each MDP component to the public Alpha2 code structure.

The public `JunyueLiu/alpha2` repository is a fork of the original Alpha2 repository and explicitly states that it contains pseudocode, algorithm design, and code structure rather than a fully runnable implementation. Therefore, the code should be read as an implementation blueprint rather than as a complete production system.

## 1. What the MDP Section Is Saying

The paper defines alpha discovery as:

\[
\mathcal M = (\mathcal S, \mathcal A, p, r, \gamma, \rho_0),
\]

where the agent starts from an empty alpha program and repeatedly chooses the next instruction until a complete formulaic alpha is constructed.

The key point is:

\[
\boxed{\text{Alpha2's RL state is not a market state; it is a formula-construction state.}}
\]

That is, the state does not mean today's OHLCV market observation. Instead, it means the current partially generated alpha program.

A compact engineering view is:

```text
state      = current alpha program / expression tree / register state
action     = next legal instruction
transition = append instruction and update expression tree/registers
reward     = Perf(new alpha) - Perf(old alpha)
done       = Finish token or maximum program length
```

## 2. State \(s_t\): Current Alpha Program, Not Market State

The paper says that each state corresponds to a unique alpha function \(\zeta\). This can be confusing, because in trading RL we often think of state as market information. In Alpha2, however:

\[
s_t = \text{the alpha program already constructed up to step }t.
\]

For example, the initial state is an empty or start-only program:

```text
Start Null Null Null
```

After choosing the instruction:

```text
Sub close open Null -> Reg0
```

we get a new state corresponding to:

\[
Reg0 = close - open.
\]

After choosing:

```text
Sub high low Null -> Reg1
```

we get:

\[
Reg0 = close - open, \qquad Reg1 = high - low.
\]

After choosing:

```text
Div Reg0 Reg1 Null -> Reg0
```

we get the complete formula:

\[
Reg0 = \frac{close-open}{high-low}.
\]

So the Alpha2 state can be understood as:

\[
s_t = \{\text{program prefix},\ \text{register contents},\ \text{expression tree},\ \text{dimension information}\}.
\]

### Code-level mapping

In the public repository, the state is mainly reflected in `expression/tree.py`. The `ExpressionTree` class contains fields such as:

```python
self.root_node = None
self.reg_nodes = [None for _ in range(num_registers)]
self.reg_strs = ["" for _ in range(num_registers)]
self.action_history = []
```

This matches the paper-level idea that the state stores the current program history, expression tree, and register status.

The `program` property returns `action_history`, so the alpha program is effectively represented as a sequence of actions/instructions.

## 3. Action \(a_t\): The Next Program Instruction

Alpha2's action is not a buy/sell/hold trading action. It is the next alpha-program instruction:

\[
a_t = (\text{Operator}, \text{Operand1}, \text{Operand2}, \text{Operand3}).
\]

Examples include:

```text
Sub close open Null
Add Reg0 high Null
Div Reg0 Reg1 Null
Corr close volume 20
Finish Null Null Null
```

The action space is therefore:

\[
\mathcal A = \{\text{all possible operator-operand instruction tuples}\}.
\]

### Code-level mapping

The code-level action components are distributed across:

```text
expression/operators.py
expression/operands.py
expression/tokens.py
expression/structure.py
```

In `tokens.py`, operators are wrapped as token classes, for example:

```python
class AddToken(BinaryOpToken):
    ...

class SubToken(BinaryOpToken):
    ...

class CorrelationToken(TernaryOpToken):
    ...
```

The operand tokens include constants, market features, registers, and `NullToken` for padding. The code constructs:

```python
operand_tokens = scalar_operands + vector_operands + matrix_operands + register_tokens + [null_token]
operator_tokens = UNARY_OP_TOKENS + BINARY_OP_TOKENS + TERNARY_OP_TOKENS + START_FINISH_TOKENS
```

This corresponds directly to the paper's statement that the action space is the set of all possible instructions.

## 4. Transition \(p(s_{t+1}\mid s_t,a_t)\): Deterministic Program Update

The paper says the transition is deterministic:

\[
p(s_{t+1}\mid s_t,a_t)=1
\]

if \(s_{t+1}\) is the result of appending instruction \(a_t\) to program \(s_t\); otherwise the probability is zero.

This is very different from ordinary trading RL. In trading RL, the next market state is random because the market evolves stochastically. In Alpha2, the environment is not the market. The environment is the formula-construction process. If the current program and next instruction are fixed, the next program is uniquely determined.

For example:

\[
s_t: Reg0 = close - open
\]

Choose:

\[
a_t = Sub(high, low, Null).
\]

Then the next state must be:

\[
s_{t+1}: Reg0 = close-open,\quad Reg1=high-low.
\]

There is no stochastic market transition in this step.

### Code-level mapping

In `expression/tree.py`, this idea is represented by:

```python
def add_action(self, action):
    self.action_history.append(action)
    operator_tk = action[0]
    ...
```

Although many implementation details are left as pseudocode/pass statements in this public version, the intended structure is clear: applying an action mutates the current expression tree and register state in a deterministic way.

## 5. Reward \(r\): Incremental Improvement in Evaluation Metric

The paper defines reward as:

\[
r(s_t,a_t,s_{t+1}) = Perf(\zeta_{t+1}) - Perf(\zeta_t).
\]

This means the reward is the improvement in alpha quality caused by adding the new instruction.

Example:

\[
Perf(close-open)=0.02,
\]

and after adding more instructions:

\[
Perf\left(\frac{close-open}{high-low}\right)=0.05.
\]

Then the reward for that transition is:

\[
r_t = 0.05 - 0.02 = 0.03.
\]

So Alpha2 does not only score the final expression. In the paper formulation, each step can receive a shaped reward based on the incremental change in the current program's performance.

### What is \(Perf\)?

The paper first says the evaluation metric is primarily IC, but then refines it to encourage diversity:

\[
Perf(\zeta_t) = (1 - MaxCorr(z_t,G)) \cdot IC(z_t,\mu),
\]

where:

- \(z_t\) is the current alpha value over stocks and dates;
- \(\mu\) is the future return target;
- \(G\) is the set of already discovered alpha values;
- \(MaxCorr(z_t,G)\) is the maximum correlation between the current alpha and the discovered alpha set.

Therefore, Alpha2's reward encourages:

\[
\boxed{\text{high predictive IC} + \text{low correlation with existing alphas}.}
\]

### Code-level mapping

In the public code structure, `expression/evaluate.py` defines the evaluation function:

```python
def fast_evaluate(alpha: Value):
    return compute_metric(alpha.value)

@jax.jit
def compute_metric(alpha: jnp.ndarray):
    # definition of evaluation metric here, e.g., ic, sharpe
    return 0
```

This shows that the repository leaves the exact metric as a placeholder, but the intended design is that the expression tree produces an alpha value, and the evaluator computes an IC-, Sharpe-, or other performance-based metric.

## 6. Legal Actions and Dimension Constraints

A key Alpha2 innovation is that the action space is not blindly expanded. At each state, the system should compute the legal action set:

\[
\mathcal A_{legal}(s_t) \subseteq \mathcal A.
\]

For example:

\[
close + open
\]

is dimensionally valid because both are price-like quantities. But:

\[
close + volume
\]

is dimensionally invalid because price and volume have different units.

Thus Alpha2 should forbid the invalid action before evaluating the formula.

### Code-level mapping

In the repository README, `expression/legal_actions.py` is described as calculating legal actions when expanding an MCTS node, while `expression/tokens.py` is described as wrapping operators and implementing a `validity_check` function for legal action checking.

In `tokens.py`, each operator token contains a `validity_check` method, for example:

```python
class AddToken(BinaryOpToken):
    @staticmethod
    def validity_check(*values):
        # customized validity check here
        return True
```

The public version uses placeholders, but the design is that `validity_check` should encode dimension and semantic constraints. The corresponding dimension system is defined in `structure.py`:

```python
class DimensionType(Enum):
    price = 1
    trade = 2
    volume = 3
    condition = 4

class Dimension(NamedTuple):
    numerator: List[DimensionType]
    denominator: List[DimensionType]
```

And the market feature dimensions are initialized in `operands.py`:

```python
matrix_operands = {
    "open":  Dimension(numerator=[DimensionType.price], denominator=[]),
    "close": Dimension(numerator=[DimensionType.price], denominator=[]),
    "high":  Dimension(numerator=[DimensionType.price], denominator=[]),
    "low":   Dimension(numerator=[DimensionType.price], denominator=[]),
    "vwap":  Dimension(numerator=[DimensionType.price], denominator=[]),
    "volume": Dimension(numerator=[DimensionType.volume], denominator=[]),
}
```

This is exactly the code-level basis for the paper's dimension-aware pruning.

## 7. Termination: Finish Token or Maximum Length

An episode ends when the model outputs a finish instruction or reaches the maximum allowed program length.

### Code-level mapping

In `legal_actions.py`, the logic is:

```python
if isinstance(tree.action_history[-1][0], FinishToken):
    legal_action_list = []
elif len(tree.action_history) >= tree.max_length:
    legal_action_list = [finish_action_idx]
else:
    legal_action_list = get_legal_action_idxs(tree, computation_data)
```

This corresponds to:

\[
done = \mathbf 1\{a_t = Finish\ \text{or}\ t \geq T_{max}\}.
\]

So if the program is already finished, no more legal actions are available. If the program is too long, only the finish action remains legal.

## 8. MCTS on Top of This MDP

Once the formula-construction process is defined as an MDP, MCTS can search over possible program continuations.

A search tree node is a state:

\[
s_t = \text{current partial alpha program}.
\]

An edge is an action:

\[
a_t = \text{next instruction}.
\]

A path from root to leaf is a complete or partial alpha program:

```text
Start
  -> Sub(close, open)
  -> Sub(high, low)
  -> Div(Reg0, Reg1)
  -> Finish
```

The neural network provides policy and value guidance for this search. In the repository, `trainer.py` shows the high-level training structure:

```python
class AlphaSearcher:
    def run_alpha_search(...):
        game = play_game(config, network, rng, computation_data)
        metric = game.environment.evaluate()
        return game, info
```

and:

```python
class NetworkTrainer:
    def train_network(self, replay_buffer):
        data_batches = replay_buffer.sample_batches(...)
        ...
        _update_weights(...)
```

So the practical loop is:

1. MCTS/searchers generate alpha programs using `play_game`;
2. the environment evaluates the generated alpha;
3. the searched game is stored in a replay buffer;
4. the network trainer samples games and updates the policy/value network;
5. updated network parameters guide later MCTS searches.

This is why Alpha2 is closer to AlphaZero/AlphaDev-style neural-guided search than to plain PPO-based token generation.

## 9. Complete Example: Generating \((close-open)/(high-low)\)

### Step 0: Initial state

```text
s0 = [Start]
```

### Step 1: Choose first instruction

```text
a0 = Sub(close, open, Null)
```

Update:

\[
Reg0 = close-open.
\]

Reward:

\[
r_0 = Perf(close-open) - Perf(empty).
\]

### Step 2: Choose second instruction

```text
a1 = Sub(high, low, Null)
```

Update:

\[
Reg0 = close-open, \qquad Reg1 = high-low.
\]

Reward:

\[
r_1 = Perf(s_2)-Perf(s_1).
\]

### Step 3: Combine registers

```text
a2 = Div(Reg0, Reg1, Null)
```

Update:

\[
Reg0 = \frac{close-open}{high-low}.
\]

Reward:

\[
r_2 = Perf\left(\frac{close-open}{high-low}\right)-Perf(s_2).
\]

### Step 4: Finish

```text
a3 = Finish
```

Episode terminates, and the final alpha is:

\[
\zeta = \frac{close-open}{high-low}.
\]

## 10. Key Takeaway

The MDP in Alpha2 should be read as a formula-construction MDP:

\[
\boxed{
\begin{aligned}
&s_t = \text{current alpha program / expression tree / register state},\\
&a_t = \text{next legal instruction},\\
&s_{t+1} = \text{program after applying the instruction},\\
&r_t = Perf(\zeta_{t+1}) - Perf(\zeta_t),\\
&done = \text{Finish token or max length}.
\end{aligned}
}
\]

The important distinction from standard trading RL is that Alpha2 is not learning a portfolio policy directly. It is learning/searching how to construct symbolic alpha formulas. The generated formulas are then evaluated and later passed into a separate combination or trading model.


---

# Appendix C. Alpha2 Policy/Value Network: Input, Output, Loss, Optimization, and Iteration

This appendix explains how the policy/value network in Alpha2 is trained. The key point is that Alpha2 is closer to an AlphaZero/AlphaDev-style self-search framework than to a PPO-style policy-gradient framework.

In Alpha2, the neural network does not directly act as a standalone actor that samples the next instruction and is then updated by policy gradient. Instead, the network provides a policy prior and a value estimate to MCTS. MCTS performs look-ahead search, produces an improved action distribution, and this improved distribution is then used as a supervised learning target for the network.

The core closed loop is:

\[
s_t
\xrightarrow{f_\theta}
\bigl(p_\theta(\cdot\mid s_t), v_\theta(s_t)\bigr)
\xrightarrow{\mathrm{MCTS}}
\bigl(\pi^{\mathrm{MCTS}}(\cdot\mid s_t), z_t\bigr)
\xrightarrow{\mathrm{loss}}
\theta \leftarrow \theta - \eta \nabla_\theta L(\theta).
\]

Therefore, the network learns to predict two things:

\[
f_\theta(s_t)=\left(p_\theta(\cdot\mid s_t),v_\theta(s_t)\right),
\]

where:

- \(p_\theta(a\mid s_t)\) is the prior probability of choosing each possible next instruction;
- \(v_\theta(s_t)\) is the estimated future value of the current partial alpha program.

The network is trained by MCTS-generated supervision, not by PPO clipped policy-gradient updates.

---

## C.1 Input: What Does the Network Receive?

The input to Alpha2's network is not the raw stock data tensor directly. Instead, it is the current alpha-construction state:

\[
s_t = \text{current alpha program state}.
\]

More explicitly, the state contains information such as:

\[
s_t = \{\text{program prefix},\ \text{program length},\ \text{register state},\ \text{memory/location state}\}.
\]

A program prefix is the sequence of instructions already generated. For example, for the alpha:

\[
\frac{close-open}{high-low},
\]

an intermediate state may be:

```text
Start
Sub close open Null -> Reg0
```

At this point:

\[
Reg0 = close-open.
\]

Another later state may be:

```text
Start
Sub close open Null -> Reg0
Sub high low Null -> Reg1
```

At this point:

\[
Reg0 = close-open, \qquad Reg1 = high-low.
\]

The network needs to encode this partial program so that it can predict which instruction is worth expanding next.

In the public Alpha2 code structure, this corresponds to the observation passed into the network, which contains fields such as program, program length, registers, and memory. The program instructions are encoded as discrete tokens, usually including the operator and operands. Conceptually:

\[
x_t = \operatorname{Encode}\left([a_1,a_2,\ldots,a_{t-1}],\text{registers}_t,\text{memory}_t\right).
\]

Here \(a_k\) is a previously generated instruction.

---

## C.2 Output: What Does the Network Predict?

The network outputs two main objects:

\[
f_\theta(s_t)=\left(p_\theta(\cdot\mid s_t),v_\theta(s_t)\right).
\]

### C.2.1 Policy prior

The policy head outputs logits over possible actions/instructions:

\[
\ell_\theta(s_t)\in\mathbb R^{|\mathcal A|}.
\]

After applying softmax over legal actions, this becomes:

\[
p_\theta(a\mid s_t)
=
\frac{\exp(\ell_\theta(s_t,a))}
{\sum_{b\in\mathcal A_{\mathrm{legal}}(s_t)}\exp(\ell_\theta(s_t,b))},
\qquad a\in\mathcal A_{\mathrm{legal}}(s_t).
\]

This is not necessarily the final action distribution used to execute the next step. It is the prior distribution used by MCTS to guide search.

### C.2.2 Value estimate

The value head estimates the quality of the current partial program:

\[
v_\theta(s_t) \approx
\mathbb E\left[\sum_{k\ge 0}\gamma^k r_{t+k}\mid s_t\right].
\]

In the Alpha2 paper's context, the reward is tied to the improvement of the alpha evaluation metric:

\[
r(s_t,a_t,s_{t+1})
=
Perf(\zeta_{t+1})-Perf(\zeta_t).
\]

The evaluation function is primarily based on IC and is further adjusted by diversity:

\[
Perf(\zeta_t)
=
\left(1-\operatorname{MaxCorr}(z_t,G)\right)\cdot IC(z_t,\mu).
\]

Thus, the value network is trying to answer:

> Given the current partial alpha program, how promising is this state if we continue expanding it?

---

## C.3 Network Architecture

The Alpha2 implementation follows an AlphaDev-style design with two conceptual modules:

```text
Current alpha program state
        |
        v
Representation Network
        |
        v
state embedding h_t
        |
        v
Prediction Network
        |----------------------|
        v                      v
policy logits             value estimate
```

### Representation network

The representation network encodes the current alpha program state into a latent embedding:

\[
h_t = g_\theta(s_t).
\]

It usually needs to represent:

- the generated instruction sequence;
- the program length;
- register contents or register status;
- memory/location information;
- possibly legality-related structure.

Since the program is discrete, instructions are typically one-hot or embedded token representations. A simplified instruction representation is:

\[
\text{instruction}_t = (\text{operator},\text{operand}_1,\text{operand}_2,\text{operand}_3).
\]

The program sequence is then embedded into a vector representation.

### Prediction network

The prediction network maps the state embedding into policy and value outputs:

\[
\ell_t = W_p h_t + b_p,
\]

\[
v_t = v_\theta(h_t).
\]

In code, the prediction module may contain multiple heads, for example:

- a policy head for action logits;
- a metric/value head for predicting alpha performance;
- an auxiliary heuristic reward head inherited from AlphaDev-style code structure.

For understanding Alpha2, the most important outputs are policy logits and value estimates.

---

## C.4 How MCTS Uses the Network

At each state \(s_t\), Alpha2 does not simply sample an action from \(p_\theta(a\mid s_t)\). Instead, it runs MCTS.

The network first gives:

\[
(P_\theta(s_t,a), V_\theta(s_t)).
\]

MCTS then uses the prior and value to select and expand child nodes. A typical PUCT-style selection rule is:

\[
a^* = \arg\max_a \left[Q(s,a)+U(s,a)\right],
\]

where:

\[
U(s,a)
\propto
P_\theta(a\mid s)
\frac{\sqrt{N(s)}}{1+N(s,a)}.
\]

Here:

- \(P_\theta(a\mid s)\) is the network prior;
- \(N(s,a)\) is the visit count of action \(a\);
- \(Q(s,a)\) is the backed-up value estimate;
- \(U(s,a)\) encourages exploration of promising but less-visited actions.

Alpha2 also modifies the value backup using a mean-max mixture:

\[
Q(s,a)
=
r(s,a)
+
\beta\cdot \operatorname{mean}(V_s)
+
(1-\beta)\cdot \max(V_s).
\]

This is designed for sparse alpha search spaces. A pure mean backup may underestimate nodes that contain rare high-quality descendants, while a pure max backup may overestimate noisy or parameter-sensitive formulas. The mixture is a compromise between robustness and optimism.

---

## C.5 Where Do the Training Labels Come From?

The training data are generated by MCTS self-search.

For each visited state \(s_t\), MCTS produces:

1. an improved policy target \(\pi^{\mathrm{MCTS}}(\cdot\mid s_t)\);
2. a value target \(z_t\), based on the searched/evaluated future reward.

The improved policy target is usually derived from MCTS visit counts:

\[
\pi^{\mathrm{MCTS}}(a\mid s_t)
=
\frac{N(s_t,a)^{1/\tau}}
{\sum_b N(s_t,b)^{1/\tau}}.
\]

This target is usually stronger than the raw network policy, because it incorporates look-ahead search, legality constraints, reward evaluation, and value backup.

Therefore, the training sample is approximately:

\[
(s_t,\ \pi^{\mathrm{MCTS}}(\cdot\mid s_t),\ z_t).
\]

The replay buffer stores these search trajectories. The network is then trained on batches sampled from the replay buffer.

---

## C.6 Loss Function

The total loss can be written as:

\[
L(\theta)
=
L_{\mathrm{policy}}(\theta)
+
L_{\mathrm{value}}(\theta)
+
L_{\mathrm{aux}}(\theta).
\]

### Policy loss

The policy loss is a cross-entropy loss between the MCTS-improved policy and the network policy:

\[
L_{\mathrm{policy}}(\theta)
=
-
\sum_{a\in\mathcal A}
\pi^{\mathrm{MCTS}}(a\mid s)
\log p_\theta(a\mid s).
\]

This means:

\[
p_\theta(\cdot\mid s)
\leftarrow
\pi^{\mathrm{MCTS}}(\cdot\mid s).
\]

So the network is trained to imitate the improved search policy, not the raw action that was sampled once.

### Value loss

A simplified value loss can be written as:

\[
L_{\mathrm{value}}(\theta)
=
\left(z_t-v_\theta(s_t)\right)^2.
\]

However, the AlphaDev-style code often represents scalar values using categorical support / two-hot encoding. In that case, the value loss is closer to:

\[
L_{\mathrm{value}}(\theta)
=
\operatorname{CE}
\left(
\operatorname{TwoHot}(z_t),
q_\theta(\cdot\mid s_t)
\right).
\]

Here \(q_\theta\) is a predicted categorical distribution over value bins, and the scalar value is recovered from its expected value.

### Auxiliary loss

Some AlphaDev-style implementations also include an auxiliary heuristic reward head. This can be written generically as:

\[
L_{\mathrm{aux}}(\theta)
=
\operatorname{CE}
\left(
\operatorname{TwoHot}(h_t),
q_\theta^{\mathrm{aux}}(\cdot\mid s_t)
\right).
\]

For conceptual understanding of Alpha2, this auxiliary head is less important than the policy and value heads.

---

## C.7 Optimization Objective

From the network training perspective, the objective is:

\[
\theta^*
=
\arg\min_\theta
\mathbb E_{s\sim\mathcal D}
\left[
L(\theta;s)\right],
\]

where \(\mathcal D\) is the replay buffer of MCTS-generated search trajectories.

From the full algorithm perspective, the goal is to make the network approximate the search-improved policy and value:

\[
p_\theta(\cdot\mid s)
\approx
\pi^{\mathrm{MCTS}}(\cdot\mid s),
\]

\[
v_\theta(s)
\approx
\mathbb E\left[\sum_{k\ge 0}\gamma^k r_{t+k}\mid s_t=s\right].
\]

The network therefore learns:

> Under the current alpha program state, which instruction should MCTS explore first, and how valuable is this partial program likely to become?

Parameter updates are standard gradient-based optimization:

\[
g_t=\nabla_\theta L(\theta_t),
\]

\[
\theta_{t+1}=\theta_t-\eta g_t.
\]

If momentum is used:

\[
m_{t+1}=\mu m_t+g_t,
\]

\[
\theta_{t+1}=\theta_t-\eta m_{t+1}.
\]

A target network may also be maintained for more stable bootstrapped value targets, similar to AlphaZero/MuZero-style implementations.

---

## C.8 Full Iteration Loop

The Alpha2 training loop can be summarized as follows.

### Step 1. Initialize network

Initialize \(f_\theta\). At the very beginning, the network may be untrained, so MCTS may use nearly uniform priors.

### Step 2. Run MCTS search

For the current alpha program state \(s_t\), compute:

\[
f_\theta(s_t)=\left(p_\theta(\cdot\mid s_t),v_\theta(s_t)\right).
\]

MCTS then uses these outputs to explore the legal action space:

\[
\mathcal A_{\mathrm{legal}}(s_t).
\]

### Step 3. Select and execute an instruction

After MCTS simulations, choose an action according to the MCTS visit distribution:

\[
a_t\sim \pi^{\mathrm{MCTS}}(\cdot\mid s_t).
\]

The environment deterministically updates the program:

\[
s_{t+1}=T(s_t,a_t).
\]

### Step 4. Continue until termination

Repeat until the End instruction is chosen or the maximum program length is reached. The final program corresponds to an alpha formula \(\zeta\).

### Step 5. Evaluate the generated alpha

Compute the alpha values \(z=\zeta(X)\), then evaluate:

\[
Perf(\zeta)
=
(1-\operatorname{MaxCorr}(z,G))\cdot IC(z,\mu).
\]

### Step 6. Store search data

Store MCTS-generated training data in the replay buffer:

\[
(s_t,\pi_t^{\mathrm{MCTS}},z_t).
\]

### Step 7. Train the network

Sample batches from the replay buffer and minimize:

\[
L(\theta)
=
L_{\mathrm{policy}}+L_{\mathrm{value}}+L_{\mathrm{aux}}.
\]

### Step 8. Repeat

The updated network guides the next round of MCTS. This creates a self-improving loop:

\[
\boxed{
\text{Network} \rightarrow \text{MCTS search} \rightarrow \text{improved targets} \rightarrow \text{network update} \rightarrow \text{better network}
}
\]

---

## C.9 Difference from PPO / AlphaGen-style Policy Training

It is important not to confuse Alpha2's network with a PPO actor.

| Dimension | PPO / AlphaGen-style generation | Alpha2-style MCTS-guided training |
|---|---|---|
| Policy role | Directly samples the next token/action | Provides prior for MCTS |
| Action choice | Sampled from \(\pi_\theta(a\mid s)\) | Chosen after MCTS search using visit counts |
| Policy loss | Policy gradient / PPO clipped objective | Cross entropy against MCTS visit distribution |
| Value role | Critic estimates \(V(s)\) for advantage | Prediction head estimates leaf/state value for search |
| Data source | On-policy rollouts | Self-search replay buffer |
| Optimization target | Maximize expected return directly | Imitate MCTS-improved policy and fit value target |

PPO uses an objective such as:

\[
L^{\mathrm{PPO}}(\theta)
=
\mathbb E\left[
\min\left(
 r_t(\theta)\hat A_t,
 \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

Alpha2 is closer to AlphaZero:

\[
L(\theta)
=
-
\pi^{\mathrm{MCTS}}(\cdot\mid s)^\top
\log p_\theta(\cdot\mid s)
+
L_{\mathrm{value}}.
\]

Thus, Alpha2's policy network is not an actor trained by policy gradient. It is a prior network trained by MCTS-generated supervision.

---

## C.10 Minimal Example

Suppose the current state is:

```text
Start
Sub close open Null -> Reg0
```

So:

\[
Reg0=close-open.
\]

The network outputs prior probabilities over legal actions:

| Candidate action | Network prior |
|---|---:|
| Sub high low Null -> Reg1 | 0.30 |
| Div Reg0 high Null -> Reg0 | 0.20 |
| Abs Reg0 Null Null -> Reg0 | 0.10 |
| End Null Null Null | 0.05 |

MCTS uses these priors but then performs look-ahead search. Suppose it discovers that the path:

```text
Sub high low Null -> Reg1
Div Reg0 Reg1 Null -> Reg0
End
```

leads to a strong alpha:

\[
\frac{close-open}{high-low}.
\]

After many simulations, the visit counts may be:

| Candidate action | Visit count | MCTS target |
|---|---:|---:|
| Sub high low Null -> Reg1 | 80 | 0.80 |
| Div Reg0 high Null -> Reg0 | 10 | 0.10 |
| Abs Reg0 Null Null -> Reg0 | 8 | 0.08 |
| End Null Null Null | 2 | 0.02 |

Then the policy head is trained to move from its original prior toward the MCTS-improved target:

\[
p_\theta(\cdot\mid s)
\approx
[0.80,0.10,0.08,0.02].
\]

Next time the network encounters a similar state, it will give a higher prior to `Sub high low`, making MCTS more efficient.

---

## C.11 One-Sentence Summary

Alpha2's policy/value network can be summarized as follows:

\[
\boxed{
\text{Input: current alpha program state}
\rightarrow
\text{Output: instruction prior + state value}
\rightarrow
\text{MCTS improves them}
\rightarrow
\text{network learns from MCTS targets.}
}
\]

In words: Alpha2's policy network is not a standalone PPO actor. It is a search prior network. It receives the current partial alpha program, predicts which instruction is promising and how valuable the state is, MCTS uses these predictions to search, and the MCTS search results are used as supervised targets to update the network.
