# Paper Reading Notes: Learning from Expert Factors: Trajectory-level Reward Shaping for Formulaic Alpha Mining

> 论文：**Learning from Expert Factors: Trajectory-level Reward Shaping for Formulaic Alpha Mining**  
> 主题：RL 公式化 Alpha 挖掘；expert factors；trajectory-level reward shaping；reward centering  
> 笔记目的：按照 RL alpha mining paper reading task template，抽取该论文的 MDP 设计、reward 设计、算法机制、实验协议，以及它相对 AlphaGen / QFR 的位置。

---

## 1. One-Sentence Summary

This paper proposes **Trajectory-level Reward Shaping, TLRS** to solve the **sparse and delayed reward problem in RL-based formulaic alpha mining** by using **expert-designed alpha formulas as demonstrations and giving subsequence-level dense shaping rewards during RPN expression generation**.

中文概括：

> 这篇论文提出 TLRS，用专家因子公式中的 token 子序列匹配来给 PPO 公式生成过程提供中间奖励，从而缓解 AlphaGen 类方法中“只有完整公式生成后才有 IC reward”的稀疏奖励问题，并通过 reward centering 降低训练方差。

---

## 2. Core Problem

- **Task type:** 公式化 alpha factor mining。论文主要是在 RL 框架下挖掘 formulaic alpha，并维护一个 alpha factor pool。
- **Input:** OHLCV 类基础价量数据，包括 open、close、high、low、volume、vwap；另有专家设计公式库作为 reward shaping 的 demonstration source。
- **Output:** 一组可解释的 formulaic alpha factors，以及由 factor pool 线性组合得到的预测信号。
- **Generated object:** RPN token sequence，对应 expression tree / formulaic alpha。
- **Main objective:** 学习一个 policy \(\pi_\theta\)，逐 token 生成更高预测能力的 alpha 公式，并借助专家公式子序列提升探索效率。
- **Financial target:** 预测未来 5-day asset returns；评价指标主要是 IC 与 RankIC。
- **Single-factor or pool-aware:** 生成过程是单条公式序列；评价和部署层面使用 factor pool 及线性组合模型，因此具有 pool-aware 的组合背景，但 TLRS 的核心创新是 generator reward shaping，而不是 pool construction 本身。
- **Discovery-only or discovery-plus-deployment:** 更偏 factor discovery；deployment realism 较弱，因为实验主要报告 IC/RankIC，没有完整交易回测、换手、交易成本和容量分析。

---

## 3. MDP Design

论文沿用 AlphaGen 的基本 MDP，把公式生成看作一个有限 horizon 的确定性序列决策问题：

\[
\mathcal M = (\mathcal S, \mathcal A, P, r, \gamma).
\]

### State

状态是当前已经生成的 RPN token prefix：

\[
s_t = a_{1:t-1} = [a_1,a_2,\ldots,a_{t-1}]^\top.
\]

也就是说，policy 在第 \(t\) 步看到的是当前尚未完成的公式前缀，而不是市场状态本身。这里的 environment 是“公式生成环境”，不是金融市场随机过程。

### Action

动作是从 token vocabulary 中选择下一个 token：

\[
a_t \sim \pi_\theta(\cdot \mid a_{1:t-1}).
\]

动作空间包括：价量特征 token、常数 token、时间窗口 token、横截面 / 单日算子、时间序列算子，以及 BEG / SEP 特殊 token。

### Transition

状态转移是确定性的 prefix append：

\[
s_{t+1}=a_{1:t}.
\]

转移概率为：

\[
P(s_{t+1}\mid s_t,a_t)=
\begin{cases}
1, & s_{t+1}=a_{1:t},\\
0, & \text{otherwise}.
\end{cases}
\]

### Reward

原始 reward 是完整公式生成后计算的 IC：

\[
r_t=
\begin{cases}
r(a_{1:T}), & t=T,\\
0, & t<T.
\end{cases}
\]

其中 \(r(a_{1:T})\) 通常是生成公式对应 factor value 与真实 future return 的 IC / combination IC。无 reward shaping 时，该 reward 是典型的 terminal sparse reward。

TLRS 后的 shaped reward 为：

\[
r'_t=
\begin{cases}
f_t, & t\ne T,\\
f_t + r(a_{1:T}), & t=T.
\end{cases}
\]

其中 \(f_t\) 来自当前生成轨迹与专家公式子序列的精确匹配增量。

### Termination

episode 在以下情况结束：

1. policy 生成 SEP token；
2. 达到最大序列长度；
3. 生成非法或无法计算的表达式时，最终 reward 被设为 \(-1\)，即 IC 的下界，用来惩罚 invalid expression。

### Action Legality / Grammar Constraints

论文沿用 AlphaGen 的 RPN grammar constraints：某些状态下只允许选择特定 token；通过 action mask / grammar restriction 保证 RPN 表达式尽量语法合法。但语法合法不等于数值可计算，例如 Log 作用在负值上可能导致 invalid expression；这类表达式用 \(-1\) reward 惩罚。

### Compact Form

\[
\boxed{
\begin{aligned}
s_t &= a_{1:t-1},\\
a_t &\sim \pi_\theta(\cdot\mid a_{1:t-1}),\\
s_{t+1} &= a_{1:t},\\
R_T &= \mathrm{IC}(f_{a_{1:T}}(X),Y),\\
r'_t &= r_t + f_t.
\end{aligned}
}
\]

---

## 4. Representation

- **Representation type:** RPN sequence。每个 alpha 公式既可以表示为数学表达式，也可以表示为 expression tree；RPN 是 expression tree 的 post-order traversal。
- **Token/operator set:** 包含 unary operators、binary operators、time-series operators、price-volume features、constants、time windows、sequence indicators。
- **Validity mechanism:** RPN grammar + action legality constraints；非法或不可计算表达式给最低 reward。
- **Advantages:** RPN 序列天然适合 sequential policy generation；表达式可解释；operator library 可以注入金融先验。
- **Limitations:** RPN flatten 后丢失显式树结构；token index 的数值距离没有金融语义；语义等价但语法不同的表达式很难识别。

---

## 5. Algorithm Mechanism

- **RL/search method:** PPO-based policy optimization，基本框架来自 AlphaGen；TLRS 不是替换 PPO，而是在 PPO 训练中加入 trajectory-level reward shaping 与 reward centering。
- **On-policy or off-policy:** PPO 是 on-policy。
- **Learned components:** policy network \(\pi_\theta(a_t\mid s_t)\)、value function / critic、factor pool 线性组合权重 \(\omega\)。
- **Search component:** 没有显式 MCTS、beam search 或 genetic search；核心仍是 neural policy sequential sampling。
- **Training loop:**
  1. 用 policy \(\pi_\theta\) 逐 token 采样，构造一个 RPN 公式 \(f_n\)；
  2. 在市场数据 \(X_l\) 上计算 factor values \(z_{n,l}=f_n(X_l)\)；
  3. 把新 factor 放入组合模型，计算组合预测 \(z'_{n,l}\)；
  4. 计算原始 terminal reward，即 IC / RankIC 类预测质量；
  5. 用 TLRS 根据专家公式子序列匹配计算每一步 shaping reward \(f_t\)；
  6. 用 reward centering 估计平均 reward \(\bar r_t\)，更新 differential value；
  7. 用 PPO clipped objective 更新 \(\theta\)；
  8. 用 MSE loss 更新组合权重 \(\omega\)。

### Key PPO Objective

\[
L^{\mathrm{surr}}(\theta)=
\mathbb E
\left[
\sum_{t=1}^{T}
\hat A(a_{1:t})
\min\left(
\psi(a_{1:t}),
\operatorname{clip}(\psi(a_{1:t}),1-\delta,1+\delta)
\right)
\right],
\]

其中

\[
\psi(a_{1:t})=\frac{\pi_\theta(a_t\mid a_{1:t-1})}{\pi_{\theta_{old}}(a_t\mid a_{1:t-1})}.
\]

---

## 6. Reward Design

### Reward signal

原始 reward 是生成公式对应 factor 与 future returns 的 IC 类指标。论文同时使用 RankIC 作为训练过程和测试过程的重要评价指标。

\[
\mathrm{IC}(z'_l,y_l)=\frac{\operatorname{Cov}(z'_l,y_l)}{\sigma_{z'_l}\sigma_{y_l}},
\qquad
\overline{\mathrm{IC}}=\frac{1}{L}\sum_{l=1}^{L}\mathrm{IC}(z'_l,y_l).
\]

### Terminal or intermediate

原始 AlphaGen 式 reward 是 terminal reward；TLRS 把它变成 terminal reward + dense intermediate shaping reward。

### Pool-aware or standalone

reward 的最终性能计算与 factor pool 组合有关，但 TLRS 的核心 shaping 信号是当前 token prefix 与 expert formula subsequences 的匹配程度，因此 shaping 本身不是 pool-aware。

### Risk-aware or uncertainty-aware

不是 risk-aware，也不是 distributional RL。没有显式建模收益分布、downside risk、transaction cost、turnover、capacity、risk exposure。

### Shaping mechanism: TLRS

TLRS 定义当前生成 prefix \(s_t\) 与专家公式子序列的匹配比例：

\[
\Phi(s_t)=\frac{n_{1,t}}{N_t},
\]

其中 \(n_{1,t}\) 是专家 demonstration subsequences 中与当前生成序列 \(s_t\) 完全一致的子序列数量，\(N_t\) 是所有长度为 \(t\) 的专家子序列总数。

shaping reward 是相邻两步 matching ratio 的差分：

\[
f_t=\delta(s_t,s_{t+1})=rac{n_{1,t+1}}{N_{t+1}}-\frac{n_{1,t}}{N_t}.
\]

如果把 \(\Phi(s_t)=n_{1,t}/N_t\) 看作 potential function，并且在 factor-mining MDP 中设置 \(\gamma=1\)，则：

\[
f_t=\gamma\Phi(s_{t+1})-\Phi(s_t)=\Phi(s_{t+1})-\Phi(s_t),
\]

这与 PBRS 形式一致，因此论文声称保持 optimal policy invariance。

### 为什么必须 \(\gamma=1\)

论文强调 factor-mining MDP 必须设置 \(\gamma=1\)。原因是 terminal reward 只在公式结束时出现，若 \(\gamma<1\)，目标为：

\[
J(\theta)=\mathbb E_{a_{1:T}\sim\pi_\theta}[\gamma^T r(a_{1:T})].
\]

此时 \(T\) 由 policy 决定，policy 可以通过更早输出 SEP 来减少折扣损失。因此即使长公式 reward 更高，只要

\[
r_T > \gamma r_{T+1},
\]

模型也可能偏好较短公式。这会诱导 premature termination，导致表达式过短、表达能力不足、探索浪费。

### Reward centering

TLRS 还引入 reward centering。平均 reward 在线估计：

\[
\bar r_{t+1}=\bar r_t+\beta(r_{t+1}-\bar r_t).
\]

随后使用 centered reward 更新 differential value：

\[
\widetilde V_{\pi_\theta}(a_{1:t-1}) \leftarrow
\widetilde V_{\pi_\theta}(a_{1:t-1})+
\alpha\left[
(r_{t+1}-\bar r_t)+
\widetilde V_{\pi_\theta}(a_{1:t})-
\widetilde V_{\pi_\theta}(a_{1:t-1})
\right].
\]

直观上，它让 critic 不必拟合一个很大的 state-independent reward offset，而更专注于 state relative value，从而降低方差、稳定训练。

### Possible reward bias

1. **专家公式偏置:** TLRS 会鼓励生成与专家公式局部结构相似的表达式，因此可能限制探索到完全不同但有效的新结构。
2. **exact-match bias:** 只奖励 token-level exact subsequence match，无法识别语义等价但语法不同的公式。
3. **feature-set ceiling:** 如果输入只有 6 个基础价量特征，最终 IC 的提升可能受信息量上限限制。
4. **shaping reward 与真实收益目标不完全一致:** 子序列像专家公式不一定代表最终 factor 更有预测力。

---

## 7. Factor Pool and Combination

- **Pool maintained:** 是。论文沿用 AlphaGen 风格，维护一个 factor pool \(\mathcal F=\{f_1,\ldots,f_K\}\)。
- **Pool update rule:** 新生成且验证通过的 factor 加入 pool；当 pool 超过容量限制时，删除组合权重最小的 factor。
- **Factor selection/removal:** 权重较小的因子被认为边际贡献低，因此被移除。
- **Combination model:** 线性组合：

\[
z'_l=\sum_{k=1}^{K}w_k f_k(X_l).
\]

- **Weight optimization:** 通过最小化预测误差优化 \(\omega\)：

\[
\mathcal L(\omega)=\frac{1}{L}\sum_{l=1}^{L}\|z'_l-y_l\|^2.
\]

- **Static or dynamic weights:** 静态或训练期更新的组合权重，不是市场状态依赖的 dynamic allocation。
- **Deployment realism:** 有组合模型，但没有完整 portfolio construction / backtest / transaction cost / turnover / capacity 设置，因此更偏研究型 factor mining，而非真实交易部署。

---

## 8. Experiments and Evaluation

- **Dataset:** 中国 A 股与美国股票市场。
- **Universe:** CSI300、CSI500、CSI1000、SPX、DJI、NDX 六个指数成分股。
- **Input features:** open、close、high、low、volume、vwap 六个基础价量特征。
- **Time split:** train 为 2016-01-01 到 2020-01-01；validation 为 2020-01-01 到 2021-01-01；test 为 2021-01-01 到 2024-01-01。
- **Prediction target:** 未来 5-day asset returns。
- **Metrics:** IC、RankIC，训练曲线中重点比较 RankIC 收敛速度和稳定性。
- **Baselines:** MLP、XGBoost、LightGBM、GP/gplearn、AlphaGen/No Shaping PPO、QFR、PBRS、DPBA。
- **Transaction costs:** 未纳入。
- **Turnover:** 未纳入。
- **Random seeds:** 每个实验使用 5 个 random seeds，报告均值与标准差。
- **Robustness tests:** 六个市场 universe；discount factor sensitivity；expert demonstration 数量 \(N\) sensitivity；reward centering learning rate \(\beta\) sensitivity；ablation：去掉 reward shaping / 去掉 reward centering。
- **Code availability:** 论文中提到依赖 AlphaGen、gplearn、Stable-Baselines3、Qlib 等开源实现；未明显说明 TLRS 自身代码是否公开。

### Key experimental findings

1. TLRS 在训练阶段比 NS、PBRS、DPBA 更快、更稳定，除 CSI500 上提升不明显外，多数 universe 上表现更好。
2. TLRS 相比 potential-based shaping baselines 的 RankIC 提升约 9.29%。
3. \(\gamma=1\) 的表现优于 \(\gamma<1\)，支持论文关于短公式偏置的理论分析。
4. expert demonstrations 数量增加通常提升效果，在论文实验中 \(N=130\) 最优。
5. reward centering learning rate \(\beta\) 呈非单调影响，论文选择 \(\beta=2\times 10^{-3}\)。
6. 测试集上 TLRS 与 QFR 接近，但未显著全面超过 QFR；说明 TLRS 的主要价值更偏训练效率和稳定性，而非绝对 out-of-sample IC 碾压。

---

## 9. Main Contributions

1. **The paper contributes expert-guided reward shaping by addressing sparse terminal reward in RL alpha formula generation.**  
   它把 Alpha101 风格专家公式拆成子序列，用 partial RPN sequence 的 exact subsequence match ratio 作为 potential function，从而给每一步生成动作提供中间信号。

2. **The paper contributes a trajectory-level symbolic similarity mechanism by addressing the failure of distance-based RSfD in tokenized alpha formulas.**  
   它指出 token index 的欧氏距离 / 曼哈顿距离不具有金融语义，因此用 exact matching 代替 vector distance，避免 open 与 close index 接近这类伪相似问题。

3. **The paper contributes reward centering by addressing high-variance PPO training under shaped rewards.**  
   它用在线平均 reward 估计 \(\bar r_t\) 将 reward 中心化，使 value function 学习相对值而不是全局 offset。

4. **The paper contributes a clear theoretical argument for \(\gamma=1\) in formula-mining MDP.**  
   它证明 \(\gamma<1\) 会使 policy 偏好提前 SEP，从而生成过短公式；这对所有 RPN/AST/program synthesis 型 alpha mining 都很有参考价值。

---

## 10. Limitations

### Algorithmic limitations

1. **Exact-match shaping 不能识别语义等价表达式。** 例如 \(x+y\) 和 \(y+x\)，或者某些代数等价变换，token 序列不一致但语义接近。
2. **对专家公式库质量依赖较强。** 如果 expert formulas 质量一般、风格单一，TLRS 可能把 policy 引导到狭窄区域。
3. **仍然依赖 AlphaGen 式 token grammar 与 operator set。** 方法没有从根本上扩大表达空间。
4. **没有替代 PPO，也没有引入更强搜索机制。** TLRS 是 reward-level 改造，不是 MCTS、GFlowNet、model-based RL 或 surrogate evaluator。

### Evaluation / finance limitations

1. **没有完整交易回测。** 论文主要报告 IC / RankIC，缺少年化收益、Sharpe、最大回撤、换手率、交易成本后收益等部署指标。
2. **没有 turnover / transaction cost / capacity 分析。** 对真实量化部署而言，公式因子的预测 IC 不等于可交易 alpha。
3. **特征集较窄。** 只用六个基础价量特征，可能导致多个算法在测试集上接近同一 performance ceiling。
4. **TLRS 测试集并未显著全面超过 QFR。** 论文的强项是训练收敛与 reward shaping 机制，但 out-of-sample predictive power 相对 QFR 的优势并不绝对。
5. **代码公开不明确。** 如果没有 TLRS 官方实现，复现 exact subsequence matching、reward centering 和实验设置会有困难。

---

## 11. Relation to AlphaGen and Existing Literature

- **Relation to AlphaGen:** TLRS 基本继承 AlphaGen 的公式生成 MDP、RPN representation、PPO generator、factor pool 线性组合框架。它主要解决 AlphaGen 的 sparse terminal reward 问题，而不是重新设计 expression representation 或 pool optimizer。
- **Main bottleneck addressed:** AlphaGen 在完整公式生成后才得到 IC reward，导致 credit assignment 长、sample efficiency 低、训练不稳定。TLRS 用专家公式子序列对 partial trajectory 给 dense shaping reward。
- **Paradigm classification:** Policy-based sequential expression generation；Reward shaping and variance reduction；Expert-guided or imitation-assisted generation。
- **Difference from PBRS / DPBA:** PBRS / DPBA 使用距离型 potential function，需要把 token 序列数值化，再计算欧氏距离等；但 token index 不具备连续语义。TLRS 直接做 exact subsequence matching，避免 token distance 的语义错配，并降低复杂度。
- **Difference from QFR:** QFR 主要改进 REINFORCE 的稳定性和方差控制；TLRS 仍基于 PPO/AlphaGen，但通过 expert demonstration shaping 改善 sparse reward。
- **Whether it is more RL-like or search-like:** 更 RL-like。它没有显式树搜索或种群搜索，而是在 PPO 序列生成器中嵌入专家引导的 dense reward。

### Comparison with AlphaGen / QFR / PBRS

| Method | Generator | Reward signal | Main improvement | Weakness |
|---|---|---|---|---|
| AlphaGen | PPO RPN generator | terminal IC / pool performance | 首次系统建模公式 alpha pool generation | sparse reward，训练效率低 |
| PBRS / DPBA | PPO + potential shaping | distance-based shaping | 提供中间 reward | token distance 语义不可靠，计算成本较高 |
| QFR | improved REINFORCE | factor reward + variance control | policy gradient 方差控制，更稳定 | 不主要利用专家公式结构 |
| TLRS | PPO + expert subsequence shaping | terminal IC + exact-match dense shaping + reward centering | 用专家公式局部结构缓解 sparse reward | 依赖专家库，exact match 忽略语义等价 |

---

## 12. How to Incorporate This Paper into My Review

### Suggested section

- **Section:** Reward shaping and expert-guided RL for formulaic alpha mining
- **Reason:** 这篇论文不是主要改 representation，也不是主要改 search，而是针对 AlphaGen/QFR 之后的核心痛点：公式生成的 terminal reward 太稀疏、credit assignment 太长。它可以作为 “expert demonstration + reward shaping” 路线的代表。

### Suggested one-line description

TLRS introduces expert-formula-guided trajectory-level reward shaping for RL alpha mining, using exact RPN subsequence matching and reward centering to improve the sample efficiency and stability of PPO-based formula generation.

### Suggested paragraph

TLRS can be viewed as an expert-guided extension of AlphaGen. Instead of changing the RPN representation or replacing PPO, it modifies the reward signal: partial generated formulas receive dense shaping rewards according to their exact subsequence-level similarity to expert-designed alpha formulas. This directly targets the sparse terminal IC reward problem in formulaic alpha mining. The method also argues that the discount factor should be set to \(\gamma=1\), because \(\gamma<1\) creates an artificial preference for shorter expressions through the \(\gamma^T\) term. Compared with distance-based reward shaping methods such as PBRS and DPBA, TLRS avoids treating token indices as continuous semantic vectors and reduces the feature-dimension-dependent computation. Its main value lies in training efficiency and stability, although its out-of-sample predictive performance is comparable rather than decisively superior to strong baselines such as QFR.

### Suggested comparison table row

| Paper | Venue | Core formulation | Representation | RL/Search mechanism | Reward type | Pool-aware? | Dynamic allocation? | Key contribution | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| TLRS / Learning from Expert Factors | arXiv 2025 preprint | Formulaic alpha mining as RPN token MDP | RPN sequence / expression tree | PPO + expert-guided reward shaping + reward centering | Terminal IC plus subsequence-level dense shaping reward | Partly; uses factor pool and linear combination | No | Uses expert formula subsequence matching to reduce sparse-reward difficulty in AlphaGen-style generation | Depends on expert formula library; exact match ignores semantic equivalence; lacks full trading-cost backtest |

---

## 13. My Overall Assessment

这篇论文最有价值的地方是 **methodology and future research inspiration**，其次是 implementation inspiration。

从 RL 挖因子 review 角度，它的重要性不在于“又提出了一个比所有 baseline 都强很多的 alpha miner”，而在于它非常清楚地指出了 AlphaGen 类方法的一个核心结构性问题：

\[
\text{公式生成是长序列决策，但金融评价 reward 只在完整公式生成后出现。}
\]

这导致 policy 训练初期几乎没有有效 credit assignment。TLRS 的回答是：不用等完整公式生成后才知道好坏，而是把专家公式拆成局部结构，在生成过程中逐步奖励“向专家局部结构靠近”的行为。

对你自己的研究启发主要有三点：

1. **Reward shaping 可以从金融专家知识中来。** 这比单纯用 IC/RankIC 更接近“把人工因子设计经验注入 RL generator”。
2. **在公式生成任务中，\(\gamma=1\) 不是随便设的。** 如果 \(\gamma<1\)，模型会因为折扣项偏好短公式，这一点对所有 RPN/AST/program synthesis 型 alpha mining 都很重要。
3. **token-level similarity 不等于 semantic similarity。** TLRS 用 exact match 避免了距离度量的伪语义问题，但仍未真正解决表达式语义等价问题。未来可以考虑 symbolic simplification、expression embedding、LLM-assisted formula canonicalization 或 algebra-aware reward shaping。

如果放到你的 review 中，我建议把它归入：

> **Reward shaping / expert-guided RL / variance reduction for symbolic alpha generation**

而不是归入 search-enhanced、risk-aware、distributional RL 或 dynamic allocation。

---

## 14. 最后用四个核心问题复盘

### 1. How does it define factor mining as a decision problem?

它把公式化 alpha 挖掘定义为 RPN token sequence generation MDP：状态是当前 token prefix，动作是下一个 token，转移是 deterministic append，终止是 SEP / 最大长度，terminal reward 是公式预测能力。

### 2. What exactly does the RL component learn?

RL component 学习一个 policy \(\pi_\theta(a_t\mid a_{1:t-1})\)，即在任意 partial formula prefix 下选择下一个 token 的概率分布。它学的是 formula generator，不是直接学 portfolio weights，也不是学市场 dynamics。

### 3. What weakness of AlphaGen or previous methods does it address?

它主要解决 AlphaGen 的 sparse and delayed reward 问题，以及 PBRS/DPBA 在 tokenized formula 上使用 distance-based similarity 时的语义错配问题。

### 4. Does the experimental protocol support the paper's conclusion?

部分支持。训练曲线、ablation、discount factor sensitivity 支持 TLRS 能提升收敛速度和稳定性；但测试集 IC/RankIC 相对 QFR 并没有绝对统治优势，且缺少交易成本、换手和真实组合回测，因此更能支持“训练效率和稳定性提升”，不能充分支持“真实交易部署表现显著更强”。
