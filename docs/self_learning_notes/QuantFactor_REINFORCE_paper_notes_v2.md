# Paper Reading Notes: QuantFactor REINFORCE

> Paper: **QuantFactor REINFORCE: Mining Steady Formulaic Alpha Factors with Variance-bounded REINFORCE**  
> Version: arXiv:2409.05144v3, 2025-06-17  
> Notes language: 中文为主，保留专有名词、数学符号和算法名。

---

## 1. One-Sentence Summary

This paper proposes **QuantFactor REINFORCE (QFR)** to solve **PPO/actor-critic 在公式化 alpha 因子挖掘中 critic 难以从 terminal reward 学到有效 value、训练有偏且较慢的问题** by **用无 critic 的 REINFORCE 训练 RPN 公式生成策略，并加入 greedy reward baseline 与 IR-based reward shaping 来降低方差、鼓励稳健因子**。

一句中文概括：

> QFR 本质上是在 AlphaGen 的 RPN 序列生成框架上，把 PPO 替换为带 greedy baseline 的 REINFORCE，并用 IR 约束奖励，从而更适合“只有完整公式生成后才有 reward”的 terminal-feedback alpha mining 任务。

---

## 2. Core Problem

- **Task type:** 公式化 alpha 因子挖掘；更准确地说，是学习一个策略网络生成一组可组合的 formulaic alpha factors。
- **Input:** 历史资产价量数据。实验中只使用公开可复现的 6 个基础价量特征：`open`, `close`, `high`, `low`, `volume`, `vwap`。形式上，每个交易日 \(l\)，资产 \(i\) 的输入为
  \[
  X_{li}\in\mathbb{R}^{m\times d},
  \]
  其中 \(m\) 是原始市场特征数量，\(d\) 是回看天数。
- **Output:** 一个 formulaic alpha factor generator \(\pi_\theta\)，可不断生成 RPN token sequence，对应公式化 alpha；同时维护 alpha factor pool 和线性组合权重。
- **Main objective:** 学习一个生成策略，使其生成的因子组合与未来收益具有更高相关性，并在不同波动环境下更稳定。
- **Financial target:** 预测 ground-truth **5-day asset returns**，主要使用 IC / RankIC 评价。
- **Single-factor or pool-aware:** 不是纯单因子挖掘。单条 trajectory 生成一个新公式，但奖励评估会经过 combination model，所以更接近 **pool-aware formula alpha generation**。
- **Discovery-only or discovery-plus-deployment:** 主要是因子发现方法，但论文也做了 top-50 选股回测，用于证明 deployment-stage utility。

### 和 AlphaGen 的任务关系

QFR 没有重新定义公式化 alpha 挖掘任务，它基本沿用 AlphaGen 的核心问题设定：把公式生成建模为 MDP，用 RPN token 序列表示公式，策略网络逐 token 生成表达式。它真正改动的是 **policy optimization mechanism** 与 **reward shaping**。

---

## 3. MDP Design

论文沿用 AlphaGen 的 MDP 建模方式。该 MDP 不是金融市场本身的 MDP，而是 **公式生成过程的 MDP**。

### State

状态是当前已经生成的 RPN token prefix：

\[
s_t = a_{1:t-1} = [a_1,a_2,\ldots,a_{t-1}]^\top.
\]

例如：

```text
BEG -1 open Mul
```

表示当前已经生成了部分 RPN 表达式，策略需要决定下一个 token 是 `Log`、`SEP` 还是其他合法 token。

### Action

动作是下一个 token：

\[
a_t \sim \pi_\theta(\cdot \mid a_{1:t-1}).
\]

动作空间 \(\mathcal A\) 是有限 token set，包括：

- sequence indicator: `BEG`, `SEP`；
- feature token: `open`, `close`, `high`, `low`, `volume`, `vwap` 等；
- constant token: 如 \(-10,-5,-0.01,0.01,5,10\)；
- time delta token: 如 `10d`, `20d`, `50d`；
- operator token: cross-sectional operators 与 time-series operators。

### Transition

转移是确定性的 append 操作：

\[
s_{t+1}=a_{1:t}=[a_1,\ldots,a_{t-1},a_t]^\top.
\]

论文强调这一点非常关键：给定当前状态和动作后，下一个状态唯一确定，因此 transition function 满足 Dirac distribution：

\[
P(s_{t+1}\mid a_{1:t})=
\begin{cases}
1, & s_{t+1}=a_{1:t},\\
0, & \text{otherwise}.
\end{cases}
\]

这也是作者认为 REINFORCE 在该任务中可行的主要理由之一：环境转移没有额外随机性，高方差主要来自 policy sampling 和 reward，而不是环境 dynamics。

### Reward

奖励是 terminal reward。中间 token 没有 reward，只有完整公式生成结束后才评价：

\[
r(s_t,a_t)=
\begin{cases}
0, & t\neq T,\\
r(a_{1:T}), & t=T.
\end{cases}
\]

AlphaGen 使用 IC 作为 reward；QFR 在 IC 基础上加入 IR-based shaping：

\[
r(a_{1:T})=IC-\lambda\mathbf{1}\{IR\leq \operatorname{clip}[(t-\alpha)\eta,0,\delta]\}.
\]

其中：

- \(IC\)：组合因子值与真实未来收益的相关性；
- \(IR=\mathbb{E}[IC]/\sqrt{\operatorname{Var}(IC)}\)：IC 的稳定性指标；
- \(\lambda\)：惩罚强度；
- \(\alpha\)：延迟启动 IR 检验的时间；
- \(\eta\)：IR 门槛随训练时间上升的斜率；
- \(\delta\)：最大 IR test threshold。

### Termination

一个 episode 在以下情况结束：

1. 生成 `SEP` token；
2. 达到最大公式长度。

完整 trajectory \(a_{1:T}\) 可以转换为一个 RPN 表达式，再转换为公式化 alpha factor。

### Action Legality / Grammar Constraints

论文说明并非所有 token 序列都是合法 RPN 公式，因此必须在特定状态下只允许选择特定动作。它没有重新设计 grammar constraint，而是沿用 AlphaGen 的做法：通过合法动作约束保证 RPN 格式正确。

可以理解为：

\[
\pi_\theta(a_t\mid s_t)=0,\quad a_t\notin\mathcal{A}_{\text{legal}}(s_t).
\]

这里 \(\mathcal{A}_{\text{legal}}(s_t)\) 由 RPN 栈深度、操作符 arity、time operator 对 time delta 的需求、是否可结束等规则决定。

### Compact MDP Form

\[
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r),
\]

其中：

\[
s_t=a_{1:t-1},\qquad a_t\sim\pi_\theta(\cdot\mid s_t),\qquad s_{t+1}=a_{1:t},
\]

\[
P(s_{t+1}\mid s_t,a_t)=\mathbf{1}\{s_{t+1}=s_t\oplus a_t\},
\]

\[
J(\theta)=\mathbb{E}_{a_{1:T}\sim\pi_\theta}[r(a_{1:T})].
\]

---

## 4. Representation

- **Representation type:** Reverse Polish Notation (RPN) sequence。每个公式先表示为 token sequence，也可以唯一对应到 expression tree。
- **Token/operator set:**
  - **Features:** `open`, `close`, `high`, `low`, `volume`, `vwap`；
  - **Constants:** \(-10,-5,-0.01,0.01,5,10\) 等；
  - **Time deltas:** `10d`, `20d`, `50d` 等；
  - **Unary cross-sectional operators:** `Abs(x)`, `Log(x)`；
  - **Binary cross-sectional operators:** \(x+y\), \(x-y\), \(x\times y\), \(x/y\), `Larger(x,y)`, `Smaller(x,y)`；
  - **Time-series operators:** `Ref(x,l)`, `Mean(x,l)`, `Medium(x,l)`, `Sum(x,l)`, `Std(x,l)`, `Var(x,l)`, `Max(x,l)`, `Min(x,l)`, `Mad(x,l)`, `Delta(x,l)`, `WMA(x,l)`, `EMA(x,l)`, `Cov(x,y,l)`, `Corr(x,y,l)`；
  - **Sequence indicators:** `BEG`, `SEP`。
- **Validity mechanism:** 使用 RPN grammar constraint / action mask 思路，保证 token sequence 可以解析成合法表达式。
- **Advantages:**
  1. RPN 将树结构生成问题转化为序列生成问题，方便使用 policy network 逐 token 采样；
  2. 表达式具有可解释性，比 end-to-end deep alpha 更容易做人工审查；
  3. transition 是确定性 append，理论上有利于 REINFORCE 的方差控制。
- **Limitations:**
  1. flatten 成 sequence 后，长公式的 credit assignment 仍然困难；
  2. 表达能力强依赖 operator library；
  3. grammar 保证形式合法，不保证经济意义、稳定性、容量或交易可实现性；
  4. 搜索空间仍随 token 长度指数级膨胀。

### 一个例子

论文给出的示例：

\[
\operatorname{Mul}(-1,\operatorname{Corr}(open,volume,10d))
\]

对应 RPN：

```text
BEG -1 open volume 10d Corr Mul SEP
```

这说明 policy 实际上学的是“如何生成 token 序列”，而不是直接在连续参数空间中拟合一个黑箱预测模型。

---

## 5. Algorithm Mechanism

- **RL/search method:** REINFORCE with greedy baseline，论文命名为 QuantFactor REINFORCE (QFR)。
- **On-policy or off-policy:** On-policy。每次用当前 \(\pi_\theta\) 采样 trajectory，并用这些 trajectory 的 terminal reward 更新策略。
- **Learned components:**
  1. policy model \(\pi_\theta\)：公式生成器；
  2. combination model weights \(\omega\)：因子池线性组合权重。
- **Search component:** 没有 MCTS / beam search。它不是 search-enhanced 方法，而是 policy-based sequential expression generation。
- **Training loop:**
  1. 用当前 policy random sampling 生成一个 normal factor \(f_n\)；
  2. 用当前 policy greedy decoding 生成一个 greedy factor \(f_g\)；
  3. 分别计算 normal factor value \(z_n\) 和 greedy factor value \(z_g\)；
  4. 经过 combination model 得到组合因子值 \(z'_n,z'_g\)；
  5. 用 shaped reward 计算 \(r(a_{1:T})\) 与 \(r(\bar a_{1:T})\)；
  6. 用二者差值作为 REINFORCE baseline 后的 learning signal 更新 \(\theta\)；
  7. 用 MSE loss 更新组合权重 \(\omega\)。

### Key Objective

原始目标：

\[
J(\theta)=\mathbb{E}_{a_{1:T}\sim\pi_\theta}[r(a_{1:T})].
\]

REINFORCE policy gradient：

\[
\hat g(\theta)=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(a_t^i\mid a_{1:t-1}^i)r(a_{1:T}^i).
\]

QFR 的 greedy baseline 版本：

\[
\tilde g(\theta)=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(a_t^i\mid a_{1:t-1}^i)
\left[r(a_{1:T}^i)-r(\bar a_{1:T}^i)\right].
\]

其中 \(\bar a_t\in\arg\max_a\pi_\theta(a\mid \bar a_{1:t-1})\)，即 greedy decoding 生成的 trajectory。

### 为什么作者认为 REINFORCE 比 PPO 更适合这里？

PPO/actor-critic 的优势通常来自 critic 估计 \(V(s_t)\) 或 advantage \(A(s_t,a_t)\)，从而降低 policy gradient 方差。但在这个任务中：

\[
r(s_t,a_t)=0,\quad t<T.
\]

也就是说，绝大多数中间状态没有即时反馈。critic 需要从 token prefix 预测“未来完整公式的好坏”，但公式最终的 IC 只有生成完才能知道。因此：

1. **critic target 信号稀疏**：中间状态 value 很难准确估计；
2. **critic bias 会传给 actor**：advantage 估计偏差可能影响 policy update；
3. **计算成本更高**：PPO 需要 actor + critic，两套网络或共享 backbone + value head，rollout 中还要计算 value；
4. **该 MDP transition deterministic**：没有市场环境随机转移，REINFORCE 的环境方差来源较小。

所以 QFR 的核心判断是：在“deterministic transition + terminal reward + formula generation”的 MDP 中，少一个 critic 反而可能更稳、更快。

---

## 6. Reward Design

- **Reward signal:** 基于 IC 的终止奖励，并引入 IR-based shaping。
- **Terminal or intermediate:** Terminal reward。完整公式生成前 reward 为 0。
- **Pool-aware or standalone:** 更接近 pool-aware。论文在 combination model 后计算组合因子值 \(z'_t\)，再计算 IC / IR。
- **Risk-aware or uncertainty-aware:** 是 risk-aware / steadiness-aware。IR 衡量 IC 的均值相对波动，即信号稳定性。
- **Shaping mechanism:** 随训练时间逐步提高 IR threshold，对 IR 不达标的因子进行惩罚。
- **Possible reward bias:** 奖励仍高度依赖 IC 与 IR 的 scalarization；可能偏好低波动但弱收益的因子，也可能因阈值超参数选择导致过强惩罚。

### IC Reward

对组合因子值 \(z'_l\) 与真实收益 \(y_l\)，IC 定义为：

\[
IC(z'_l,y_l)=\frac{\operatorname{Cov}(z'_l,y_l)}{\sigma_{z'_l}\sigma_{y_l}}.
\]

平均 IC：

\[
\overline{IC}=\frac{1}{L}\sum_{l=1}^{L}IC(z'_l,y_l).
\]

### IR Reward Shaping

单因子或组合因子的 IR：

\[
IR=\frac{\mathbb{E}_t[IC(z'_t,y_t)]}{\sqrt{\operatorname{Var}(IC(z'_t,y_t))}}.
\]

QFR shaped reward：

\[
r(a_{1:T})=IC-\lambda\mathbf{1}\{IR\leq \operatorname{clip}[(t-\alpha)\eta,0,\delta]\}.
\]

直观解释：

- 训练早期：\(\operatorname{clip}[(t-\alpha)\eta,0,\delta]\) 较低，允许低 IR 因子存在，鼓励探索；
- 训练后期：IR threshold 上升，如果因子 IC 不稳定，则 reward 被扣除 \(\lambda\)；
- 因此，policy 不只追求短期高 IC，也会偏好跨时间更稳定的 IC。

### Reward 的关键含义

QFR 不是直接优化 Sharpe 或回测收益，而是在公式挖掘阶段用 **IC + IR penalty** 作为 proxy。它假设：

\[
\text{高 IC 且 IC 序列稳定}\Rightarrow\text{更稳健的选股信号}\Rightarrow\text{更好的回测表现}.
\]

这在量化上是合理的，但仍然不是完整交易目标，因为没有直接把交易成本、换手、容量、行业/风格暴露纳入训练 reward。

---

## 7. Factor Pool and Combination

- **Pool maintained:** 是。论文沿用 AlphaGen 式 factor pool \(\mathcal F=\{f_1,\ldots,f_K\}\)。
- **Pool update rule:** 新生成的 alpha 会加入 pool；如果超过阈值，则丢弃权重最小的 alpha。
- **Factor selection/removal:** 根据组合模型中的 factor weight 大小做删除，weight 最小的被移除。
- **Combination model:** 线性组合。
- **Static or dynamic weights:** 论文中组合权重 \(\omega\) 通过 gradient descent 优化；不是显式时变权重 \(w_t\)，更接近 static weight after training / within training updated weights。
- **Deployment realism:** 做了 top-50 选股回测，但训练 reward 本身未直接包含交易成本、换手、滑点、容量。

### Combination Form

给定因子池：

\[
\mathcal F=\{f_1,f_2,\ldots,f_K\},
\]

每个因子有权重 \(w_k\)，组合因子值为：

\[
z'_l=\sum_{k=1}^{K}w_k f_k(X_l).
\]

权重通过 MSE 目标学习：

\[
\mathcal L(\omega)=\frac{1}{L}\sum_{l=1}^{L}\|z'_l-y_l\|^2.
\]

这说明 QFR 的 generator 和 combination model 是耦合的：policy 生成公式，combination model 评估并维护已有因子池。reward 不是纯粹的单因子 standalone IC，而是会受到当前 pool 与权重的影响。

---

## 8. Experiments and Evaluation

- **Dataset:** 中国 A 股与美国股票市场公开价量数据。
- **Universe:**
  - CSI300；
  - CSI500；
  - CSI1000；
  - S&P 500 / SPX；
  - Dow Jones Industrial Average / DJI；
  - NASDAQ 100 / NDX。
- **Time split:**
  - Train: 2016-01-01 至 2020-01-01；
  - Validation: 2020-01-01 至 2021-01-01；
  - Test: 2021-01-01 至 2024-01-01。
- **Prediction target:** 5-day asset returns。
- **Metrics:**
  - IC；
  - RankIC；
  - cumulative return；
  - Sharpe ratio；
  - maximum drawdown；
  - turnover；
  - volatility-regime backtest performance。
- **Baselines:**
  - Tree model: XGBoost, LightGBM；
  - Heuristic: GP；
  - Deep model: MLP；
  - RL: AlphaGen, PPO, TRPO, A3C；
  - QFR variants for ablation。
- **Transaction costs:** 论文回测部分没有清晰纳入显式 transaction cost。它假设 top-50 调仓时 expected execution price 与 actual execution price difference 为 0，并认为 TWAP/VWAP 可中和 slippage。这个假设偏乐观。
- **Turnover:** 报告了 turnover，尤其在按季度回测表中比较 QFR 与 AlphaGen 的换手。
- **Random seeds:** 每个实验使用 5 个 random seeds，学习曲线显示半个标准差阴影。
- **Robustness tests:**
  1. 六个不同指数成分股；
  2. 中国与美国市场；
  3. 不同市场压力/波动阶段；
  4. reward shaping 超参数敏感性；
  5. greedy baseline 与 reward shaping 的 ablation。
- **Code availability:** 论文引用 AlphaGen、gplearn、Stable-Baselines3、Qlib 的开源实现作为 baseline；该论文自身 QFR 代码是否完整公开，在论文文本中没有明确说明。

### Main Reported Results

1. **RL learning curve:** 除 CSI500 外，QFR 在 5 个指数上相对 AlphaGen/PPO 等方法有更快收敛和更高 RankIC，论文称相对最佳 PPO 提升 3.83%。
2. **Factor evaluation:** 在 CSI300 上，QFR 的 IC 为 0.0588，高于 AlphaGen 的 0.0500；在 CSI500 上，QFR 的 IC 为 0.0708，高于 AlphaGen 的 0.0544。
3. **Backtesting:** CSI300 top-50 strategy 中，QFR 在 weekly、quarterly、yearly cumulative returns 上优于 baseline；monthly 略低于 AlphaGen。
4. **Volatility regime:** 在高波动、快速波动变化、低波动阶段，QFR 都表现较好，尤其高波动阶段优势明显。
5. **Ablation:** 去掉 greedy baseline 或去掉 IR reward shaping 都会降低最终表现，说明二者互补。

### 对实验协议的评价

优点：

- 跨中美市场、多个指数，覆盖不同市值与行业结构；
- 有 train/validation/test 明确时间划分；
- 有多 seed；
- 同时看因子 IC 和投资回测；
- 有 ablation 与 sensitivity analysis。

主要问题：

- 回测没有显式交易成本和滑点，且 top-50 每日换手可能很高；
- 只使用基础 OHLCV，不一定代表真实机构因子挖掘的完整信息集；
- reward 与 deployment objective 仍有 gap，训练阶段没有直接优化 after-cost portfolio utility；
- QFR 与 AlphaGen 的对比高度依赖实现细节、operator set、pool size、训练预算是否完全公平。

---

## 9. Main Contributions

1. **The paper contributes a REINFORCE-based policy optimization design by addressing PPO/critic bias in terminal-reward formula generation.**  
   它指出 AlphaGen 使用 PPO 时，critic 需要从 token prefix 预测完整公式的最终 IC，而中间状态没有 reward，这会导致 value estimation bias 和额外计算成本。QFR 直接用 Monte Carlo return 更新 policy，避免 critic。

2. **The paper contributes a greedy reward baseline by addressing the high-variance weakness of vanilla REINFORCE.**  
   QFR 同时生成 sampled trajectory 和 greedy trajectory，用 \(r(a_{1:T})-r(\bar a_{1:T})\) 作为 centered learning signal。作者证明该 estimator 无偏，并给出方差上界与相对 vanilla REINFORCE 的方差降低条件。

3. **The paper contributes IR-based reward shaping by addressing the instability of high-IC but volatile alpha factors.**  
   它把 Information Ratio 引入公式化 alpha 挖掘 reward，使策略不仅追求 IC 均值，也关注 IC across time 的稳定性。

4. **The paper contributes a clearer theoretical argument for why deterministic formula-generation MDPs differ from standard stochastic RL environments.**  
   作者强调该环境的 transition 是 RPN append，不是金融市场随机演化，因此 REINFORCE 的环境方差较低。这一点对理解 RL alpha mining 很有启发。

---

## 10. Limitations

### Algorithmic limitation

1. **Sparse terminal reward 仍然存在。**  
   QFR 去掉 critic 并加入 baseline，但没有从根本上解决 token-level credit assignment。一个公式最终好坏仍然要到 `SEP` 后才知道，中间 token 的具体贡献难以区分。

2. **Greedy baseline 不是 learned value function，也不一定接近最优 baseline。**  
   Greedy trajectory reward 可降低方差，但它只是当前策略的 deterministic decoding 结果。若 greedy factor 本身质量波动大，baseline 可能引入不稳定的相对比较信号。

3. **理论分析有简化条件。**  
   方差降低证明包括 2-armed bandit 特例、positive rewards、softmax parameterization 等条件；真实 RPN token action space 大得多，reward 也可能为负。

4. **搜索空间仍然巨大。**  
   QFR 没有引入 MCTS、beam search、GFlowNet、grammar-aware exploration bonus 等机制，因此长公式、大 operator set 下仍可能陷入局部最优。

5. **强依赖 AlphaGen 式 representation 和 pool design。**  
   它主要是优化器层面的改进，不是重新设计 alpha 表达空间或更强的经济约束。

### Evaluation / finance limitation

1. **交易成本与滑点假设偏乐观。**  
   论文认为 TWAP/VWAP 可使 expected/actual execution price difference 为 0，但真实 top-50 策略在 A 股和美股都有冲击成本、流动性约束、停牌/涨跌停等问题。

2. **训练 reward 没有直接包含 turnover、capacity、risk exposure。**  
   IR 惩罚关注 IC 稳定性，但没有显式惩罚换手、行业暴露、市值暴露、风格暴露或交易容量。

3. **只使用基础价量特征，外推到更复杂特征集时未验证。**  
   论文为了可复现只用 OHLCV/vwap，但机构真实 alpha mining 往往包含 fundamental、alternative data、Level-2、news、order flow 等。

4. **回测策略较简单。**  
   top-50 等权或排序持仓只能证明信号方向性，不能充分代表真实组合优化、风险控制和执行系统中的表现。

5. **代码可复现性不完全明确。**  
   若 QFR 完整代码未公开，复现实验尤其是 pool update、reward shaping、训练预算公平性会有困难。

---

## 11. Relation to AlphaGen and Existing Literature

- **Relation to AlphaGen:**  
  QFR 直接建立在 AlphaGen 的问题设定上：RPN token generation、MDP formulation、formulaic alpha pool、linear combination model。它不是替代 AlphaGen 的 representation，而是替代 AlphaGen 的 policy optimization：用 REINFORCE + greedy baseline 替代 PPO/actor-critic。

- **Main bottleneck addressed:**  
  AlphaGen 中 PPO critic 在 terminal-feedback 公式生成任务中难以学习有效 value function，导致 bias、收敛慢和计算开销大。QFR 针对这个瓶颈提出 critic-free optimization。

- **Paradigm classification:**
  1. Policy-based sequential expression generation；
  2. Reward shaping and variance reduction；
  3. Risk-aware / steadiness-aware RL；
  4. Pool-aware formula alpha generation。

- **Difference from prior work:**
  - 相比 AlphaGen：主要改 policy gradient estimator 和 reward shaping；
  - 相比 GP：QFR 学习一个生成策略，不是靠进化变异和选择；
  - 相比 tree models：QFR 生成显式公式，而不是直接用树模型预测收益；
  - 相比 end-to-end deep alpha：QFR 输出可解释公式，而非黑箱 embedding/prediction model；
  - 相比 MCTS/RiskMiner 类方法：QFR 没有显式树搜索，而是纯 policy sampling。

- **Whether it is more RL-like or search-like:**  
  更 RL-like。它的核心是 on-policy policy gradient，而不是 tree search / genetic search。虽然公式生成本身可以看成搜索，但论文贡献集中在 REINFORCE estimator、baseline 和 reward shaping。

### 和 AlphaGen 的关键对比表

| Dimension | AlphaGen | QuantFactor REINFORCE |
|---|---|---|
| Formula representation | RPN sequence | RPN sequence |
| State | token prefix | token prefix |
| Action | next token | next token |
| Transition | deterministic append | deterministic append |
| RL algorithm | PPO / actor-critic | REINFORCE with greedy baseline |
| Critic / value network | 有 | 无 |
| Reward | IC / pool performance | IC with IR shaping |
| Variance reduction | critic / advantage | greedy baseline |
| Main bottleneck addressed | formulaic alpha pool generation | PPO critic bias and REINFORCE variance |
| Main risk | critic bias, local optima | high variance, terminal reward, reward scalarization |

---

## 12. How to Incorporate This Paper into My Review

### Suggested section

- **Section:** `Reward Shaping and Variance Reduction in RL-based Formulaic Alpha Mining`
- **Reason:** 这篇论文最重要的不是新的公式表示，而是指出 AlphaGen/PPO 在 terminal-feedback 因子生成任务中的 critic bias，并提出 REINFORCE + greedy baseline + IR shaping。它适合放在 AlphaGen 之后，作为“对 AlphaGen 优化器和 reward 的改进”。

### Suggested one-line description

> QuantFactor REINFORCE modifies the AlphaGen-style RPN formula generation framework by replacing PPO with a critic-free REINFORCE estimator equipped with a greedy reward baseline and IR-based reward shaping for steadier formulaic alpha discovery.

### Suggested paragraph

QuantFactor REINFORCE is best understood as an optimizer-level and reward-level refinement of AlphaGen. It preserves the AlphaGen formulation in which a formulaic alpha is generated as an RPN token sequence and the current token prefix is treated as the MDP state. Its main argument is that PPO is not necessarily suitable for this setting, because rewards are only available after a complete formula is generated, making the critic network difficult to train and potentially biased. QFR therefore removes the critic and trains the policy using REINFORCE. To control the high variance of Monte Carlo policy gradients, it subtracts the reward of a greedily decoded formula as a baseline. In addition, it introduces an Information Ratio based shaping term to penalize factors whose IC is unstable across time. This paper is valuable for a review because it explicitly analyzes the mismatch between actor-critic RL and terminal-feedback symbolic alpha generation, and it shows that variance reduction and risk-aware reward shaping can be central design dimensions in RL-based alpha mining.

### Suggested comparison table row

| Paper | Venue | Core formulation | Representation | RL/Search mechanism | Reward type | Pool-aware? | Dynamic allocation? | Key contribution | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| QuantFactor REINFORCE | arXiv 2025 | Formulaic alpha generation as terminal-reward MDP | RPN token sequence / expression tree | On-policy REINFORCE with greedy reward baseline | IC with IR-based shaping | Yes, through linear factor pool combination | No explicit dynamic allocation | Replaces PPO critic with variance-bounded REINFORCE and adds IR shaping for steady factors | Still sparse terminal reward; no explicit cost/capacity/risk-exposure-aware training objective |

---

## 13. My Overall Assessment

这篇论文主要价值在 **methodology** 和 **future research inspiration**，其次才是 implementation/evaluation。

### 为什么有方法论价值？

它提出了一个很重要的判断：

> 公式化 alpha generation 的 MDP 和一般 RL 控制任务不同，因为 transition 是确定性的 token append，reward 是完整公式后的 terminal evaluation。因此，PPO/actor-critic 未必天然优于 REINFORCE。

这个观点对 RL alpha mining review 很有价值。很多论文会默认“PPO 是更现代的 policy gradient，所以更好”，但 QFR 提醒我们：算法是否合适取决于 MDP 的 reward structure 和 transition structure。

### 为什么有研究启发？

QFR 指出未来可以沿着几个方向继续改进：

1. **更好的 baseline:** greedy baseline 是简单有效的，但可以研究 learned baseline、control variate、trajectory-level normalization、group relative baseline 等。
2. **更强 reward shaping:** IR 是稳定性指标，但可以继续加入 turnover、cost、capacity、sector/style neutralization、drawdown penalty。
3. **credit assignment:** terminal reward 仍然稀疏，可以研究 partial expression evaluator、surrogate reward model、grammar-aware shaping。
4. **exploration:** QFR 没有解决搜索空间巨大问题，可以和 MCTS、GFlowNet、diversity-driven search、mutation refinement 结合。

### 对你写 review 时的一句话评价

> QFR 是 AlphaGen 路线中一个典型的“优化器替换 + 方差控制 + 风险稳健 reward shaping”工作，其核心贡献不在新的 alpha 表达形式，而在重新审视 terminal-feedback symbolic generation 中 actor-critic 与 Monte Carlo policy gradient 的适配性。

---

## 14. 补充说明：policy random sampling 与 policy greedy decoding

这一节解释训练循环中的两个关键概念：

1. 用当前 policy random sampling 生成一个 normal factor \(f_n\)；
2. 用当前 policy greedy decoding 生成一个 greedy factor \(f_g\)。

它们本质上是：**同一个策略网络 \(\pi_\theta\) 用两种不同方式生成公式因子**。QFR 让它们分别生成一个 normal factor 和一个 greedy factor，然后用二者的 reward 差值来更新策略。

### 14.1 policy random sampling 是什么？

在第 \(t\) 步，policy network 输入当前已经生成的 RPN token 序列：

\[
s_t = a_{1:t-1}
\]

然后输出下一个 token 的概率分布：

\[
\pi_\theta(a_t \mid a_{1:t-1}).
\]

例如当前状态是：

\[
\text{BEG } -1\ \text{open}\ \text{Mul}
\]

policy 输出：

\[
\pi_\theta(\cdot \mid s_t)
=
\begin{cases}
P(\text{Log})=0.3,\\
P(\text{open})=0.1,\\
P(\text{SEP})=0.6.
\end{cases}
\]

**random sampling** 的意思是：不是永远选择概率最大的 token，而是按照这个概率分布随机抽样：

\[
a_t \sim \pi_\theta(\cdot \mid s_t).
\]

所以即使 \(\text{SEP}\) 概率最高，仍然可能抽到 \(\text{Log}\)。

代码形式可以写成：

```python
probs = policy(state)                  # 例如 [0.3, 0.1, 0.6]
dist = Categorical(probs)
action = dist.sample()                 # 按概率随机抽样
```

这种方式生成出来的因子在论文中叫：

\[
f_n,
\]

也就是 **normal factor**。

它的作用是：**保持探索性**。因子挖掘的搜索空间很大，如果永远只选当前概率最大的 token，就很容易卡在早期局部最优公式里。random sampling 允许策略偶尔尝试概率较低但可能更有价值的公式路径。

### 14.2 policy greedy decoding 是什么？

**greedy decoding** 也是用同一个 policy network，但是每一步不随机采样，而是直接选择当前概率最大的 token：

\[
\bar a_t \in \arg\max_a \pi_\theta(a \mid \bar a_{1:t-1}).
\]

仍然使用上面的例子：

\[
\pi_\theta(\cdot \mid s_t)
=
\begin{cases}
P(\text{Log})=0.3,\\
P(\text{open})=0.1,\\
P(\text{SEP})=0.6.
\end{cases}
\]

greedy decoding 一定选择：

\[
\bar a_t = \text{SEP}.
\]

代码形式可以写成：

```python
probs = policy(state)                  # 例如 [0.3, 0.1, 0.6]
action = torch.argmax(probs)           # 选择最大概率 token
```

这种方式生成出来的因子在论文中叫：

\[
f_g,
\]

也就是 **greedy factor**。

它的作用是：**给当前 policy 一个确定性基准表现**。也就是说，当前策略如果不探索、只按最大概率走，它大概能生成什么质量的因子。

### 14.3 为什么 QFR 要同时生成 normal factor 和 greedy factor？

QFR 的核心不是直接用：

\[
r(a_{1:T})
\]

去做 REINFORCE，而是用：

\[
r(a_{1:T}) - r(\bar a_{1:T})
\]

作为学习信号。

其中：

- \(a_{1:T}\)：random sampling 生成的 normal trajectory；
- \(\bar a_{1:T}\)：greedy decoding 生成的 greedy trajectory；
- \(r(a_{1:T})\)：normal factor 的 reward；
- \(r(\bar a_{1:T})\)：greedy factor 的 reward。

所以 QFR 的 policy gradient 近似是：

\[
\tilde g(\theta)
=
\frac{1}{N}
\sum_{i=1}^N
\sum_{t=1}^T
\nabla_\theta \log \pi_\theta(a_t^i \mid a_{1:t-1}^i)
\left[
r(a_{1:T}^i) - r(\bar a_{1:T}^i)
\right].
\]

这句话非常关键：**它不是只看 normal factor 的绝对好坏，而是看它相对于当前 policy 的 greedy 表现有没有更好。**

### 14.4 这样设计的直觉

可以把 greedy factor 理解为当前 policy 的“默认答案”。

random sampling 生成一个探索因子：

\[
f_n,
\]

greedy decoding 生成当前策略最自信的因子：

\[
f_g.
\]

然后比较二者：

\[
r(f_n) - r(f_g).
\]

#### 情况一：random factor 比 greedy factor 好

如果：

\[
r(f_n) > r(f_g),
\]

说明这次随机探索找到了比当前默认策略更好的公式。此时：

\[
r(f_n)-r(f_g)>0.
\]

对应的 sampled actions 会被强化，policy 会增加这些 token 路径的概率。

也就是说：

> 这条随机探索路径比你当前最自信的路径还好，以后多生成类似公式。

#### 情况二：random factor 比 greedy factor 差

如果：

\[
r(f_n) < r(f_g),
\]

说明这次随机探索不如当前策略默认生成的公式。此时：

\[
r(f_n)-r(f_g)<0.
\]

对应 sampled actions 的概率会被压低。

也就是说：

> 这条探索路径比当前默认路径还差，以后少生成类似公式。

### 14.5 为什么不用普通 REINFORCE 的 \(r(f_n)\)，而要减去 greedy reward？

普通 REINFORCE 的梯度是：

\[
\hat g(\theta)
=
\sum_{t=1}^T
\nabla_\theta \log \pi_\theta(a_t \mid a_{1:t-1})
r(a_{1:T}).
\]

它的问题是：**reward 的绝对值波动很大，梯度方差很高。**

例如两个因子 reward 分别是：

\[
0.050,\quad 0.052.
\]

单看绝对 reward 都是正的，REINFORCE 都会鼓励它们。但如果当前 policy 的 greedy factor 已经有：

\[
r(f_g)=0.055,
\]

那么这两个 sampled factor 都不如当前默认生成结果，理论上不应该被强化。

QFR 减去 greedy baseline 后：

\[
0.050 - 0.055 = -0.005,
\]

\[
0.052 - 0.055 = -0.003.
\]

这时学习信号会变成负的，策略会减少这些探索路径的概率。

所以它的好处是：

\[
\text{绝对 reward 学习}
\quad \rightarrow \quad
\text{相对当前策略基准的 advantage 学习}.
\]

这和 actor-critic 中 advantage 的思想很像：

\[
A(s,a)=Q(s,a)-V(s).
\]

但是 QFR 不训练 critic，而是用 greedy factor 的 reward 作为 baseline。

### 14.6 为什么这个设计特别适合公式因子挖掘？

公式生成这个 MDP 有一个特殊点：**状态转移是确定性的**。

当前序列是：

\[
s_t = a_{1:t-1}.
\]

选择 token \(a_t\) 之后，下一个状态一定是：

\[
s_{t+1}=a_{1:t}.
\]

没有环境随机性。论文把这个写成 Dirac distribution：

\[
P(s_{t+1}\mid a_{1:t}) =
\begin{cases}
1, & s_{t+1}=a_{1:t},\\
0, & \text{otherwise}.
\end{cases}
\]

所以这里的随机性主要来自 policy sampling，而不是环境本身。这样一来，REINFORCE 的高方差问题相对可控；再加上 greedy baseline，可以进一步降低 policy gradient 的方差。

这也是论文认为 QFR 比 PPO 更适合 AlphaGen 式公式生成任务的核心理由：

- PPO 需要 critic 去估计中间状态价值；
- 但公式生成任务只有完整公式生成后才有 reward；
- 中间 token 的价值很难估计，critic 容易有 bias；
- QFR 干脆不要 critic，直接用完整 trajectory 的真实 reward；
- 再用 greedy reward 作为 baseline 来降方差。

### 14.7 一个完整小例子

假设当前 policy 生成一条公式序列。

#### random sampling 路径

\[
\text{BEG}, -1, \text{open}, \text{volume}, 10d, \text{Corr}, \text{Mul}, \text{SEP}
\]

得到 normal factor：

\[
f_n = \operatorname{Mul}(-1, \operatorname{Corr}(\text{open}, \text{volume}, 10d)).
\]

它的 reward 是：

\[
r(f_n)=0.061.
\]

#### greedy decoding 路径

每一步都选最大概率 token，得到 greedy factor：

\[
f_g = \operatorname{Rank}(\text{close}).
\]

它的 reward 是：

\[
r(f_g)=0.055.
\]

那么 QFR 的学习信号是：

\[
r(f_n)-r(f_g)=0.006.
\]

这个值是正的，所以 random sampling 这条路径上的 token 会被强化。

如果反过来：

\[
r(f_n)=0.048,\quad r(f_g)=0.055,
\]

则：

\[
r(f_n)-r(f_g)=-0.007.
\]

这条随机路径会被惩罚。

### 14.8 一句话总结

**policy random sampling** 用来探索新的公式路径，**policy greedy decoding** 用来生成当前策略的确定性基准公式；QFR 用二者 reward 的差值作为 REINFORCE 的 baseline-adjusted learning signal，从而在不训练 critic 的情况下保留 unbiased Monte Carlo 更新，并降低梯度方差。
