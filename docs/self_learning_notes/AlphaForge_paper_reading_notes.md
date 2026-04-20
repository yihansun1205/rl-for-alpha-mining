# Paper Reading Notes: AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors

> 说明：本文按照 `RL Alpha Mining Paper Reading Task Template` 的结构整理。由于 AlphaForge 并不是典型的 PPO / MDP 式强化学习方法，下面会在 “MDP Design” 和 “Algorithm Mechanism” 中明确区分：论文自身没有显式 MDP，而是采用 generative-predictive symbolic generation + dynamic factor combination 的两阶段框架。

---

## 1. One-Sentence Summary

This paper proposes **AlphaForge**, a two-stage framework for **formulaic alpha factor mining and dynamic factor combination**, to solve the limitations of sparse alpha search and fixed-weight Mega-Alpha by using a **generative-predictive neural network** to mine low-correlation strong factors and a **dynamic linear combination model** to select and reweight factors over time.

中文概括：

AlphaForge 提出一个两阶段公式 Alpha 框架：第一阶段用 Generator + Predictor 在公式空间中挖掘高 IC、低相关的 factor zoo；第二阶段在每个交易日根据近期因子表现动态筛选因子并用线性回归组合成当天的 Mega-Alpha，从而解决传统公式因子挖掘中搜索空间稀疏、固定因子权重难以适应市场变化的问题。

---

## 2. Core Problem

- **Task type:** 公式 Alpha 挖掘 + 动态因子组合。不是只挖一个 factor，也不是只做静态 alpha pool，而是先挖一个 factor zoo，再在每个时间截面动态组合。
- **Input:**
  - 股票历史特征矩阵：open, high, low, close, volume, vwap；
  - 过去 \(\tau\) 天 rolling window 数据；
  - future return label，即文中使用的 `Ref(VWAP, -21)/Ref(VWAP, -1) - 1`；
  - 已经挖出的 factor zoo \(Z\)，用于约束新因子与已有因子的相关性。
- **Output:**
  - 第一阶段输出：一组低相关、高质量的 formulaic alpha factors，即 factor zoo \(Z=\{f_1,\dots,f_k\}\)；
  - 第二阶段输出：每个交易日的动态 Mega-Alpha 信号 \(\hat y_t\)，以及当天被选中的因子集合和对应线性权重。
- **Main objective:**
  - 挖掘阶段：生成 valid、high IC、low correlation 的公式因子；
  - 组合阶段：在每个时间点根据最近因子表现动态选择并加权，提升 Mega-Alpha 的预测效果。
- **Financial target:** 预测未来收益并构造 stock selection signal。论文评估 IC、RankIC，并进一步用 Qlib 做模拟交易。
- **Single-factor or pool-aware:**
  - 挖掘阶段是 standalone factor IC + factor zoo correlation constraint；
  - 组合阶段是 pool-aware dynamic selection and weighting。
- **Discovery-only or discovery-plus-deployment:** discovery-plus-deployment。它不只是生成公式，还明确讨论动态组合、模拟交易和 real money investment。

---

## 3. MDP Design

### 3.1 论文是否显式定义 MDP？

AlphaForge **没有显式定义 MDP**，也没有像 AlphaGen 那样把公式生成过程表述为：

\[
s_t = \text{partial expression},\qquad a_t=\text{next token},\qquad R_T=\text{pool performance}.
\]

它的核心不是 policy gradient，而是：

\[
z\sim \mathcal N(0,I) \xrightarrow{G_{\theta_G}} \text{logit matrix}
\xrightarrow{M(\cdot)} x\in\{0,1\}^{D\times S}
\xrightarrow{parse} f
\xrightarrow{fitness} \pi(f,Z,X,Y).
\]

因此，严格说 AlphaForge 更接近 **neural symbolic generation / surrogate-model-guided optimization**，而不是标准强化学习。

### 3.2 如果从 RL alpha mining 的阅读框架强行对应

#### State

论文中没有显式 \(s_t\)。如果按照生成公式的序列过程做等价解释，可以把状态理解为：

\[
s_j = (x_{1:j-1}, Z, X, Y),
\]

其中 \(x_{1:j-1}\) 是已经生成的 RPN token prefix，\(Z\) 是已有 factor zoo，\(X,Y\) 用于评价公式表现。但这只是分析者的等价解释，不是论文正式定义。

#### Action

如果做 MDP 化解释，动作可以对应为：

\[
a_j \in \mathcal A = \{\text{features},\text{constants},\text{unary operators},\text{binary operators},\text{time-series operators}\}.
\]

但论文实际不是逐 token 采样 policy，而是 Generator 一次性输出一个 \(D\times S\) logit matrix，再经 sequence rule mask 和 gumbel-softmax 得到 one-hot formula matrix。

#### Transition

等价解释下：

\[
s_{j+1}=T(s_j,a_j),
\]

由 RPN grammar / sequence rule mask 决定。实际论文中 transition 不是 environment transition，而是从 latent noise \(z\) 到 formula one-hot matrix 的神经网络映射。

#### Reward

论文不使用 RL reward，而使用 factor fitness score：

\[
\pi(x,Z,X,Y)=
\begin{cases}
|IC(f,X,Y)|, & f\text{ is valid and }\psi(f,Z,X,Y)<CORR'\\
0, & \text{otherwise}
\end{cases}
\]

其中：

- \(f=parse(x)\)；
- \(Z\) 是已有 factor zoo；
- \(\psi(f,Z,X,Y)\) 是新因子和已有因子之间的最大绝对相关性；
- \(CORR'\) 是人工设置的相关性阈值；
- 使用 \(|IC|\) 是因为负 IC 因子可以通过取反变成正向因子。

如果把它看成 episodic reward，则可以写成：

\[
R_T = \pi(x,Z,X,Y),
\]

但更准确地说，它是 surrogate predictor 要拟合的 supervised target，而不是 policy rollout reward。

#### Termination

没有显式 episode termination。公式长度由最大长度 \(S\) 约束。生成器输出固定维度 \(D\times S\) 的公式 one-hot matrix；公式是否有效由 mask、parse 和 qualification rule 决定。

#### Action Legality / Grammar Constraints

AlphaForge 使用 \(M(\cdot)\) 模块，将 Generator 输出的 logit matrix 转为 one-hot formula matrix。该模块包含：

1. **sequence rule mask**：保证生成序列满足公式语法规则；
2. **gumbel-softmax**：让离散 one-hot 近似过程保持可微，使 Predictor 的梯度可以传回 Generator。

这和 AlphaGen / MaskablePPO 的 invalid action masking 有相似目标：防止非法表达式进入搜索空间；但机制不同。AlphaGen 是逐步 mask action logits，AlphaForge 是一次性对生成矩阵进行 rule mask + differentiable relaxation。

---

## 4. Representation

- **Representation type:** RPN sequence + one-hot matrix。论文沿用 AlphaGen / Yu et al. 2023 的思路，用 formula tree 的 post-order traversal 得到 Reverse Polish Notation，再把公式表示成 \(x\in\{0,1\}^{D\times S}\)。
- **Token/operator set:**
  - 原始特征：open, high, close, low, volume, vwap；
  - 常数：optional constants；
  - unary operators：abs, log / log1p, Inv 等；
  - binary operators：+, -, *, /；
  - time-series operators：Sum, ts cov, ts corr, ts min, ts std, ts mad, ts var, Ref 等；
  - rolling window：例如 5d、10d、20d、30d、40d、50d。
- **Validity mechanism:**
  - sequence rule mask 保证 RPN 结构合法；
  - parse 后检查公式是否 valid；
  - qualification rule 检查 \(|IC|\) 和与已有因子的相关性；
  - 不合格公式 fitness 直接置 0。
- **Advantages:**
  1. RPN 把树结构 flatten 成序列/矩阵，方便神经网络生成；
  2. one-hot matrix 适合 Predictor 学习 formula fitness；
  3. gumbel-softmax 让从 logit 到离散 token 的过程近似可微；
  4. 公式仍然可解释，适合量化投资中的因子归因和人工审查。
- **Limitations:**
  1. RPN flatten 会弱化显式 tree structure inductive bias；
  2. 公式质量高度依赖 operator set、window set 和 grammar rule；
  3. Gumbel-softmax relaxation 可能导致训练阶段的连续近似和最终离散公式之间存在 gap；
  4. 生成的公式虽然 symbolic，但复杂公式仍然不一定容易解释。

---

## 5. Algorithm Mechanism

### 5.1 RL/search method

AlphaForge 不使用 PPO、DQN、REINFORCE 或 MCTS。它使用的是：

\[
\text{Generative model }G + \text{Predictive surrogate model }P + \text{gradient-based optimization}.
\]

这更接近 deep symbolic regression / neural-guided symbolic search，而不是典型强化学习。

### 5.2 On-policy or off-policy

不适用。因为它不是 policy rollout 框架。若从搜索角度看，它是基于已评价样本库 \(R\) 的 surrogate learning，然后通过 Generator 优化 Predictor 估计分数。

### 5.3 Learned components

1. **Predictor \(P_{\theta_P}\):**
   - 输入：公式 one-hot matrix \(x\in\{0,1\}^{D\times S}\)；
   - 输出：预测的 fitness score；
   - 训练目标：拟合真实 evaluator 给出的 \(fitness(x)\)。

   \[
   L_P = \sqrt{\frac{1}{n}\sum_{i=1}^n\left(P(x_i)-fitness(x_i)\right)^2}.
   \]

2. **Generator \(G_{\theta_G}\):**
   - 输入：latent noise \(z\in\mathbb R^Q\)，其中 \(z\sim\mathcal N(0,I)\)；
   - 输出：\(D\times S\) logit matrix；
   - 经过 \(M(\cdot)\) 得到 formula one-hot matrix；
   - 训练目标：让生成公式在 Predictor 下得分更高，同时保持多样性。

   \[
   L_{Fitness}=-P(M(G(z))).
   \]

   最终 Generator loss：

   \[
   L_G=L_{Fitness}+L_{Diversity}
   \]

   展开为：

   \[
   L_G = -P(x_1)
   +\lambda_{onehot}\,Similarity_{onehot}(f(z_1),f(z_2))
   +\lambda_{hidden}\,Similarity_{hidden}(f(z_1),f(z_2)).
   \]

### 5.4 Search component

没有显式 tree search / MCTS / beam search。搜索是通过 Generator 在 latent space 中采样并优化完成的：

\[
z\rightarrow G(z)\rightarrow M(G(z))\rightarrow parse(x)\rightarrow f.
\]

搜索效率来自两个方面：

1. Predictor 学到公式空间中高 fitness 区域的 surrogate landscape；
2. Generator 可以用 Predictor 的梯度往高分区域移动，而不是像随机搜索或 GP 一样完全依赖离散变异。

### 5.5 Training loop

AlphaForge factor mining pipeline 可以概括为：

1. 初始化 factor zoo：\(Z=\emptyset\)。
2. 随机采样一批公式 one-hot matrix：\(R=\{x_1,\dots,x_r\}\)。
3. 对每个 \(x_i\) 计算真实 fitness：

   \[
   R_{fitness}=\{\pi(x_1,Z,X,Y),\dots,\pi(x_r,Z,X,Y)\}.
   \]

4. 用 \((R,R_{fitness})\) 训练 Predictor \(P\)。
5. 冻结 Predictor，训练 Generator：

   \[
   \min_{\theta_G}L_G.
   \]

6. 从 Generator 生成新公式 \(x_1,x_2\)，parse 成公式集合 \(Z_{new}\)。
7. 如果新公式 valid、qualified 且不在 \(Z\) 中，则加入 factor zoo。
8. 把生成的新样本加入 \(R\)，继续迭代直到 \(|Z|\) 达到目标数量。

### 5.6 Key update objective

整体可以写成：

\[
\min_{\theta_P}\;\mathbb E_{x\in R}\left(P_{\theta_P}(x)-\pi(x,Z,X,Y)\right)^2,
\]

\[
\min_{\theta_G}\; -P_{\theta_P}(M(G_{\theta_G}(z))) + L_{Diversity}.
\]

与 AlphaGen 的区别是：AlphaGen 学的是 sequential policy \(\pi_\theta(a_t|s_t)\)，而 AlphaForge 学的是从 latent noise 到完整公式矩阵的 Generator。

---

## 6. Reward Design

严格说 AlphaForge 中没有 RL reward，而是 **fitness score / evaluator signal**。但从 alpha mining 的角度，它承担了 reward 的角色。

- **Reward signal:**

  \[
  \pi(x,Z,X,Y)=
  \begin{cases}
  |IC(f,X,Y)|, & f\text{ is valid and }\psi(f,Z,X,Y)<CORR'\\
  0, & \text{otherwise}
  \end{cases}
  \]

- **Terminal or intermediate:** terminal-like。一个完整公式生成后才计算 fitness。没有逐 token dense reward。
- **Pool-aware or standalone:** 半 pool-aware。
  - 主要评价项是 standalone \(|IC(f)|\)；
  - 但通过 \(\psi(f,Z,X,Y)<CORR'\) 引入了与已有 factor zoo 的低相关约束。
  - 它不是 AlphaGen 那种直接用整个 pool 的 combined performance 作为 reward。
- **Risk-aware or uncertainty-aware:** 不是显式 risk-aware，也没有 distributional uncertainty modeling。它主要关注 IC、ICIR、correlation constraint。
- **Shaping mechanism:**
  1. invalid formula fitness = 0；
  2. high correlation with existing factors fitness = 0；
  3. diversity loss 约束 Generator 不要 collapse 到单一公式模式。
- **Possible reward bias:**
  1. 使用 \(|IC|\) 可能忽略交易方向稳定性和经济含义，只是假设负 IC 可以简单取反；
  2. IC-based fitness 不直接等价于 after-cost portfolio return；
  3. correlation threshold \(CORR'\) 是人工设定，可能影响 factor zoo 多样性；
  4. Predictor 学的是 historical fitness，存在 evaluator overfitting 风险；
  5. diversity loss 是表示层和 hidden 层相似度，不一定等价于真实收益来源的多样性。

---

## 7. Factor Pool and Combination

AlphaForge 的最重要贡献之一是把 factor mining 和 dynamic factor timing / combination 分开。

- **Pool maintained:** 是。第一阶段生成固定的 factor zoo：

  \[
  Z=\{f_1,\dots,f_k\}.
  \]

- **Pool update rule:**
  - 生成阶段不断向 \(Z\) 中加入 valid、qualified、low-correlation 的新公式；
  - 达到 TargetFactorNum 后，factor zoo 固定；
  - 推理和交易阶段不再修改 factor zoo，而是动态选择其中一部分因子。

- **Factor selection/removal:**
  - 每个交易日 \(t\)，根据过去 \(n\) 天数据重新计算每个因子的近期表现，例如 \(IC_t\)、\(ICIR_t\)、RankIC；
  - 通过阈值 \(IC'>0\)、\(ICIR'>0\) 过滤；
  - 按近期表现排序，选择 Top-N 因子：

    \[
    Z_t^{(N)}=TopN(Z_t).
    \]

- **Combination model:** linear regression。对当天选中的 Top-N 因子拟合线性模型：

  \[
  \hat y_t = \sum_{k\in Z_t^{(N)}} w_t^{(k)} f_t^{(k)}.
  \]

- **Static or dynamic weights:** dynamic weights。\(w_t^{(k)}\) 每个交易日都可以变化，同一个因子在不同日期权重甚至可以改变符号。

- **Deployment realism:**
  - 优点：动态组合更接近实际投资中 factor timing 的思想；
  - 优点：线性模型保留一定解释性，方便投资经理进行归因；
  - 局限：组合模型仍然主要依赖历史 IC/ICIR 和线性回归，没有显式处理交易成本、容量、行业/市值暴露中性化等真实约束。

### 与 AlphaGen 的关键区别

AlphaGen 倾向于：

\[
F = \sum_{k=1}^K w^{(k)}f^{(k)},
\]

其中 \(w^{(k)}\) 更偏固定权重或训练后固定组合。

AlphaForge 改成：

\[
F_t = \sum_{k=1}^K w_t^{(k)}f_t^{(k)},
\]

其中 \(K\) 和 \(w_t^{(k)}\) 都可以随时间变化，核心是 dynamic factor selection + dynamic weight adjustment。

---

## 8. Experiments and Evaluation

- **Dataset:** 中国 A 股市场公开数据，使用 Qlib 下载。
- **Universe:** CSI300 和 CSI500。
- **Time split:**
  - 测试期为 2018 到 2022；
  - 每年 retrain 一次；
  - 以第一个训练周期为例：
    - train: 2010-01-01 到 2016-12-31；
    - validation: 2017-01-01 到 2017-12-31；
    - test: 2018-01-01 到 2018-12-31。
- **Prediction target:**

  ```text
  Ref(VWAP, -21)/Ref(VWAP, -1) - 1
  ```

  可以理解为未来一段时间基于 VWAP 的收益标签，论文认为比简单 close-to-close return 更贴近真实投资。

- **Metrics:**
  - IC；
  - RankIC；
  - ICIR / RankICIR 在方法设计中使用；
  - 模拟交易中观察 cumulative return / net account value；
  - real money investment 中报告相对 CSI500 的 excess return。

- **Baselines:**
  - Formulaic alpha mining methods: GP, DSO, RL / AlphaGen-like method；
  - Machine learning methods: XGBoost, LightGBM, MLP；
  - Ablation baseline: Static，即使用 AlphaForge 的 mining model，但用 RL 风格的静态组合方式。

- **Transaction costs:**
  - 论文模拟交易中设置“每天最多换 5 只股票”以避免 excessive trading costs；
  - 但正文没有给出非常完整的手续费、滑点、冲击成本建模细节。

- **Turnover:**
  - 有通过每日最多换 5 只股票进行 turnover control；
  - 但没有像专业回测那样系统报告 turnover、capacity、implementation shortfall。

- **Random seeds:** 每个模型重复运行 5 次，以减少 random seed 影响。

- **Robustness tests:**
  1. pool size = [1, 10, 20, 50, 100]；
  2. dynamic vs static ablation；
  3. CSI300 / CSI500 双市场指数池；
  4. simulated trading 和 real money investment case。

- **Code availability:** 论文给出 GitHub：`https://github.com/DulyHao/AlphaForge`。

### 主要实验结果理解

1. 在 CSI300 和 CSI500 上，AlphaForge 的 IC 和 RankIC 高于 GP、DSO、RL、XGB、LGBM、MLP。
2. pool size 不是越大越好；CSI300 上 pool size = 10 表现最好。作者解释为：动态组合模型每个时点只需要少量最有效因子，过大 factor zoo 可能带来冗余或噪声。
3. Dynamic 版本优于 Static 版本，说明性能提升不只来自生成器挖因子，也来自动态组合。
4. case study 显示同一个因子在不同交易日权重会变化甚至改变方向，支持 factor timing 的观点。

---

## 9. Main Contributions

1. **The paper contributes a generative-predictive formula mining mechanism by addressing sparse and non-differentiable symbolic alpha search.**  
   它用 Predictor 学习公式 fitness 的 surrogate landscape，再用 Generator 通过梯度优化生成高分公式，避免完全依赖离散随机搜索或遗传变异。

2. **The paper contributes a low-correlation factor zoo construction rule by addressing redundancy among mined alpha factors.**  
   新公式不仅要有高 \(|IC|\)，还要与已有 factor zoo 的收益相关性低于阈值，从而控制 factor zoo 的重叠信息。

3. **The paper contributes dynamic factor combination by addressing the fixed-weight limitation of previous formulaic alpha pool methods such as AlphaGen.**  
   它每天根据近期 \(IC/ICIR\) 重新选择 Top-N factors，并用 linear regression 产生当天 Mega-Alpha 的动态权重。

4. **The paper contributes a deployment-oriented evaluation design by including rolling annual retraining, simulated trading, and real money investment evidence.**  
   虽然仍有真实交易成本和容量分析不足的问题，但它比只报告 IC 的论文更接近实盘应用。

---

## 10. Limitations

### 10.1 Algorithmic limitations

1. **不是严格 RL 方法，不能直接放在 PPO/MDP 框架中理解。**  
   AlphaForge 更像 neural symbolic generation + surrogate optimization。若在 review 中讨论 RL alpha mining，需要把它归为 “RL-adjacent / symbolic generation / surrogate-guided search”，而不是纯 policy-based RL。

2. **Predictor surrogate 可能产生 model bias。**  
   Generator 优化的是 \(P(x)\)，而不是真实 \(fitness(x)\)。如果 Predictor 在稀疏公式空间中外推错误，Generator 可能 exploit Predictor 的漏洞，生成高预测分但真实无效的公式。

3. **Generator 可能 mode collapse，diversity loss 只能部分缓解。**  
   论文加入 one-hot similarity 和 hidden similarity 惩罚，但这不保证真实收益来源的多样性，也不保证因子在不同市场环境中稳定。

4. **公式合法性依赖 handcrafted grammar。**  
   sequence rule mask、operator set、window set 和 constants 都是人工设计，方法表现可能高度依赖这些先验设定。

5. **挖掘阶段主要优化 standalone \(|IC|\)，不是直接优化组合后的 Mega-Alpha。**  
   这与 AlphaGen 的 pool-level reward 相比有所不同。AlphaForge 在 Discussion 中也承认，未来可以考虑把整个 batch factors 的 combined IC 纳入 mining objective。

### 10.2 Evaluation / finance limitations

1. **交易成本、滑点、容量分析不够完整。**  
   虽然模拟交易限制每日最多换 5 只股票，但没有充分报告 fee、slippage、market impact、capacity 等细节。

2. **风险暴露控制不足。**  
   论文讨论了 `-1*ts_mean(volume,20)` 可能带来小市值暴露，但方法本身没有系统加入行业、市值、风格因子中性化。

3. **real money investment 证据样本较短。**  
   约 9 个月、300 万人民币账户的结果有参考价值，但不足以证明长期稳健性。

4. **动态组合可能引入 timing overfitting。**  
   每天根据近期 IC/ICIR 选择因子，本质上是假设 factor momentum 存在；如果市场 regime 快速切换，近期表现可能反而导致追涨杀跌式的 factor timing error。

5. **公式解释性不等于经济逻辑可靠。**  
   公式是 symbolic 的，但复杂嵌套表达式可能很难解释，也可能只是历史数据中的统计模式。

---

## 11. Relation to AlphaGen and Existing Literature

- **Relation to AlphaGen:**
  - AlphaGen / Yu et al. 2023 使用 reinforcement learning 生成协同 alpha collections，并通过组合性能评价 alpha pool；
  - AlphaForge 继承 RPN formula representation，但替换了 PPO-style sequential generation，改用 Generator + Predictor；
  - AlphaForge 重点解决 AlphaGen 固定因子组合权重不适应市场变化的问题。

- **Main bottleneck addressed:**
  1. 公式空间稀疏，随机/遗传/纯 RL search 效率低；
  2. 已挖出的 alpha factors 可能存在周期性失效；
  3. fixed Mega-Alpha weights 难以适应市场风格切换；
  4. alpha pool 中大量因子没有被充分动态利用。

- **Paradigm classification:**
  - Dynamic factor combination or hierarchical decision making；
  - Model-based / surrogate evaluator methods；
  - Diversity-first symbolic generation；
  - RL-adjacent neural symbolic search。

- **Difference from prior work:**
  - 相比 GP：不是基于 tree mutation，而是 neural generator；
  - 相比 DSO：更面向金融公式 alpha，并加入 factor zoo 低相关约束和动态组合；
  - 相比 AlphaGen：不再用 PPO 式 policy 生成公式，也不固定最终组合权重，而是用 factor timing model 动态组合。

- **Whether it is more RL-like or search-like:**
  更 search-like / symbolic-generation-like。它不是典型 RL，但它解决的是 RL alpha mining 文献中非常核心的两个问题：搜索效率和因子池部署。

---

## 12. How to Incorporate This Paper into My Review

### Suggested section

- **Section:** 从 “Policy-based formula generation” 之后单独开一节：`Surrogate-guided symbolic generation and dynamic factor allocation`。
- **Reason:** AlphaForge 不适合直接放进 PPO / MDP 类方法中；它更适合作为 AlphaGen 之后的扩展方向：从 “怎么生成 factor pool” 进一步走向 “怎么动态使用 factor pool”。

### Suggested one-line description

AlphaForge extends formulaic alpha mining from static alpha-pool construction to dynamic factor deployment by combining surrogate-guided formula generation with time-varying linear Mega-Alpha allocation.

### Suggested paragraph

AlphaForge represents an important shift from pure formula discovery toward deployment-aware alpha mining. Unlike AlphaGen, which formulates alpha generation as a reinforcement learning process and learns a fixed synergistic alpha pool, AlphaForge separates the pipeline into two stages. First, it trains a Predictor to approximate the fitness landscape of formulaic alpha factors and optimizes a Generator to produce high-IC, low-correlation factors under differentiable masking and Gumbel-softmax relaxation. Second, it constructs a dynamic Mega-Alpha by reassessing each factor’s recent IC/ICIR at every trading day, selecting a small subset of effective factors, and estimating time-varying linear weights. This design addresses the practical issue that alpha factors may decay, reverse, or become temporarily ineffective under changing market regimes. Therefore, AlphaForge is best viewed as a surrogate-guided symbolic generation and dynamic factor allocation framework rather than a standard MDP-based RL method.

### Suggested comparison table row

| Paper | Venue | Core formulation | Representation | RL/Search mechanism | Reward type | Pool-aware? | Dynamic allocation? | Key contribution | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| AlphaForge | AAAI 2025 / arXiv 2024 | Two-stage formula mining + dynamic Mega-Alpha combination | RPN + one-hot matrix | Generator + Predictor surrogate optimization, not PPO | \(|IC|\) with validity and low-correlation constraint | Partly: factor zoo correlation constraint and dynamic pool selection | Yes: daily factor selection and linear regression weights | Moves from static factor pool to dynamic factor timing and deployment-aware combination | Not a strict RL/MDP method; surrogate bias, transaction cost/capacity/risk exposure analysis still limited |

---

## 13. My Overall Assessment

AlphaForge 对你的 review 最有价值的地方不是 “它提出了一个新的 RL 算法”，而是它把公式因子挖掘从 **静态发现问题** 推到了 **动态使用问题**。

如果按照研究价值分类：

- **Methodology value:** 高。Generator + Predictor 的设计提供了不同于 PPO / GP / DSO 的公式搜索路线。
- **Implementation value:** 中高。RPN one-hot、mask、gumbel-softmax、factor zoo、dynamic linear combination 都有较强实现参考价值。
- **Evaluation value:** 中高。它使用 CSI300/CSI500、rolling retrain、multiple seeds、ablation、simulated trading 和 real money investment，比只报告 IC 的论文更贴近实战。
- **Future research inspiration:** 很高。它提示你的 review 可以加入一个新维度：RL alpha mining 不仅要问 “如何生成因子”，还要问 “生成后的因子如何随市场状态动态选择和加权”。

### 对你当前 RL alpha mining review 的启发

你可以把 AlphaForge 放在 AlphaGen 之后，作为一个自然问题延伸：

\[
\text{AlphaGen: learn a synergistic alpha pool}
\quad\Longrightarrow\quad
\text{AlphaForge: dynamically time and combine a mined factor zoo}.
\]

也就是说，AlphaForge 的核心价值在于回答：

1. 已经挖出很多 symbolic alphas 后，是否应该每天全部使用？
2. 每个因子的有效性是否随时间变化？
3. 固定权重 Mega-Alpha 是否会在市场 regime 改变时失效？
4. 能否把 factor mining 和 factor timing 结合起来？

它的答案是：先尽可能挖出一批高质量低相关 factors，然后每天基于近期表现动态选择和线性组合。这一点非常适合写进你的 review 中关于 **factor pool deployment / dynamic allocation** 的部分。
