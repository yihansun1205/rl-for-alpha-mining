# Paper Reading Notes: Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning

> Venue: KDD 2023  
> Common short name in later discussions: **AlphaGen**  
> Authors: Shuo Yu, Hongyan Xue, Xiang Ao, Feiyang Pan, Jia He, Dandan Tu, Qing He

---

## 1. One-Sentence Summary

This paper proposes **AlphaGen, a PPO-based formulaic-alpha generator**, to solve **synergistic alpha pool construction** by **generating RPN-form formula expressions under grammar constraints and using downstream linear-combination performance as the RL reward**.

更直接地说：这篇论文不是只让 RL 找一个 IC 高的单因子，而是让 RL 生成一组能在组合模型中共同提高预测能力的公式化因子。

---

## 2. Core Problem

- **Task type:**  
  Formulaic alpha **pool mining / alpha collection generation**.  
  不是单因子挖掘，而是面向后续组合模型的因子集合挖掘。

- **Input:**  
  中国 A 股历史价量数据。论文实验中使用 6 个原始特征：
  \[
  \{\text{open}, \text{close}, \text{high}, \text{low}, \text{volume}, \text{vwap}\}.
  \]
  每个股票在每个交易日的输入可以理解为最近 \(\tau\) 天的价量特征展开：
  \[
  x_{t,i}\in \mathbb{R}^{m\tau}.
  \]

- **Output:**  
  一组公式化 alpha：
  \[
  \mathcal F=\{f_1,f_2,\ldots,f_k\},
  \]
  以及对应的线性组合权重：
  \[
  w=(w_1,w_2,\ldots,w_k).
  \]
  最终形成一个 combined mega-alpha：
  \[
  c(X;\mathcal F,w)=\sum_{j=1}^k w_j f_j(X).
  \]

- **Main objective:**  
  最大化组合后 alpha 对未来收益的预测能力，而不是最大化每个单因子的独立 IC。论文将 alpha set mining 表述为：
  \[
  \arg\max_{\mathcal F} c^*(\cdot;\mathcal F),
  \]
  其中 \(c^*\) 是给定因子集合后最优的组合模型。

- **Financial target:**  
  20 日未来收益：
  \[
  \frac{\operatorname{Ref}(\text{close},-20)}{\text{close}}-1.
  \]

- **Single-factor or pool-aware:**  
  明确是 **pool-aware**。核心创新就是 reward 依赖于 alpha pool 的组合效果。

- **Discovery-only or discovery-plus-deployment:**  
  主要是 **discovery-plus-combination**，不是完整交易执行系统。论文有投资模拟，但主算法没有直接优化收益、换手、交易成本、容量等真实部署指标。

---

## 3. MDP Design

AlphaGen 把公式生成过程定义为一个非平稳 MDP。非平稳性来自于：reward 依赖当前 alpha pool，而 alpha pool 会随着训练不断变化。

### State

状态是当前已经生成的 token 序列，即公式的 partial RPN prefix：

\[
s_t = [\text{BEG}, a_1,a_2,\ldots,a_{t-1}].
\]

其中 BEG 是开始 token。每个状态对应当前尚未完成的公式片段。论文为了保证可解释性，将公式长度上限设为 20 个 token。

直观理解：

- 状态不是市场状态；
- 状态不是 portfolio holding；
- 状态是“当前公式已经写到哪里了”。

### Action

动作是从 token vocabulary 中选择下一个 token：

\[
a_t \in \mathcal A(s_t).
\]

Token 包括：

- 原始特征 token，例如 open、close、volume、vwap；
- 常数 token，例如 \(-30,-10,-5,-2,-1,-0.5,-0.01,0.01,0.5,1,2,5,10,30\)；
- 时间窗口 token，例如 \(10d,20d,30d,40d,50d\)；
- 运算符 token，例如 Add、Sub、Log、Mean、Corr 等；
- 特殊 token：BEG、SEP。

### Transition

状态转移是确定性的，直接把动作 token 拼到当前序列后面：

\[
s_{t+1} = [s_t,a_t].
\]

因此这个环境没有学习到的 dynamics model，也不是 model-based RL。环境的主要作用是：

1. 维护当前 RPN 序列；
2. 判断动作是否合法；
3. 在公式结束后解析公式并调用 evaluator / combination model 计算 reward。

### Reward

中间步骤没有即时 reward：

\[
r_t=0,\quad t<T.
\]

当 episode 结束时，如果公式合法，会将生成序列解析成公式 \(f\)，插入当前 alpha pool，重新优化组合权重，然后计算新组合 alpha 的 IC：

\[
R_T = \bar\sigma_y\left(\sum_{i=1}^k w_i f_i\right).
\]

论文 Algorithm 2 中对应为：

\[
IC_{\text{new}} \leftarrow \bar\sigma_y\left(\sum_{i=1}^k w_i f_i\right),\qquad r_t \leftarrow IC_{\text{new}}.
\]

注意：论文表述中有时说 newly added alpha 的 performance boost / contribution，但 Algorithm 2 实际写的是新组合后的 IC，而不是显式差分：
\[
\bar\sigma_y(\mathcal F\cup\{f_{\text{new}}\})-\bar\sigma_y(\mathcal F).
\]
从实现/伪代码角度看，它更接近 **pool-level absolute performance reward**，而非严格 marginal contribution reward。

### Termination

episode 结束条件：

1. 生成 SEP token；
2. 或者 token 序列长度达到最大长度阈值。

结束后如果表达式有效，则解析为公式并评估；如果表达式语义非法，则给予惩罚。

### Action Legality / Grammar Constraints

动作合法性分两类。

**1. Formal legality**

通过 RPN stack 规则检查：

- 时间序列 operator 的最后一个参数必须是 time delta；
- operator 必须有足够 operands；
- 多 token 表达式不能退化为常数；
- SEP 只有在当前序列已经构成合法 RPN 时才允许出现。

**2. Semantic legality**

有些表达式形式合法但无法数值计算，例如：

\[
\log(x)
\]

如果 \(x\le 0\)，就可能非法。论文没有在生成时完全消除这类 semantic invalidity，而是将其 reward 设为 \(-1\)，即 Pearson correlation 的最小值。


### 图中 action logits / action masks / masked distribution 的作用

这张图展示的是 **PPO policy network 在每一步如何从所有 token 中选择下一个合法 token**。在 AlphaGen 中，动作不是买卖股票，而是生成公式时的下一个 token，例如 `Add`、`Log`、`SEP`、`Cov`、`$volume`、常数、时间窗口等。

#### 1. Action logits：策略网络对所有候选 token 的原始打分

Policy network 输入当前状态：

\[
s_t=[\text{BEG},0.1,\$vol,\text{Sub},\ldots]
\]

然后输出一个长度等于 token vocabulary 大小的向量：

\[
z_\theta(s_t)=(z_1,z_2,\ldots,z_{|\mathcal A|}).
\]

这个向量就是图中的 **action logits**。每个 logit 对应一个候选动作 token 的未归一化偏好分数，例如：

\[
z_{\text{Add}},\quad z_{\text{Log}},\quad z_{\text{SEP}},\quad z_{\text{Cov}},\ldots
\]

logit 越大，说明当前策略网络越倾向于选择该 token。但是 logits 本身还不是概率，也还没有考虑公式语法是否合法。

#### 2. Action masks：根据当前 RPN stack 规则判断哪些动作合法

由于 AlphaGen 生成的是 RPN 公式，不是任意 token 序列，所以并不是所有 token 都能在当前状态下被选择。**Action mask** 是一个由环境根据语法规则生成的合法性向量：

\[
m(s_t)\in\{0,1\}^{|\mathcal A|}.
\]

其中 \(m_i(s_t)=1\) 表示第 \(i\) 个 token 当前合法，\(m_i(s_t)=0\) 表示第 \(i\) 个 token 当前非法，不能被采样。

例如，在某个状态下：

- `Log` 可能合法，因为当前 stack 顶部已经有一个可作为输入的表达式；
- `Add` 可能非法，因为当前 stack 中可用表达式数量不足以支持二元运算；
- `SEP` 只有在当前序列已经构成完整合法 RPN 表达式时才合法；
- `Cov` 可能非法，因为它需要两个表达式和一个 time delta；
- time-series operator 后面如果缺少 time delta，也会被语法规则约束。

所以 action masks 的作用是：**把公式生成任务中的语法约束显式注入 RL 采样过程，避免策略网络生成大量形式非法的公式。**

#### 3. Masked distribution：屏蔽非法动作后的真实采样分布

在采样动作之前，算法会把非法动作的 logit 设为 \(-\infty\)：

\[
\tilde z_i(s_t)=
\begin{cases}
z_i(s_t), & m_i(s_t)=1,\\
-\infty, & m_i(s_t)=0.
\end{cases}
\]

然后再做 softmax：

\[
\pi_\theta(a_i\mid s_t)
=\frac{\exp(\tilde z_i(s_t))}
{\sum_j \exp(\tilde z_j(s_t))}.
\]

这个结果就是图中的 **masked distribution**。它才是真正用于 sample 下一个 token 的概率分布。由于非法动作的 logit 被设为 \(-\infty\)，softmax 之后它们的概率就是 0：

\[
\pi_\theta(a_i\mid s_t)=0,
\quad \text{if } m_i(s_t)=0.
\]

因此，最终采样过程是：

\[
a_t\sim \pi_\theta(\cdot\mid s_t, m(s_t)).
\]

图中如果从 masked distribution 里采样到 `Log`，那么下一个状态就是：

\[
s_{t+1}=[\text{BEG},0.1,\$vol,\text{Sub},\text{Log}].
\]

如果采样到 `SEP`，则当前公式结束，系统会把 RPN 序列解析成 expression tree，例如：

\[
\text{Sub}(0.1,\$vol),
\]

然后送入 combination model，计算加入当前 alpha pool 后的组合 IC，作为 episode return。

#### 4. 三者在训练中的关系

可以把三者关系写成：

\[
\text{state}
\xrightarrow{\text{policy network}}
\text{action logits}
\xrightarrow{\text{grammar mask}}
\text{masked distribution}
\xrightarrow{\text{sample}}
\text{next token}.
\]

也就是说：

- **action logits**：神经网络学到的 token 偏好；
- **action masks**：环境根据 RPN 语法和当前 stack 状态给出的合法动作集合；
- **masked distribution**：把非法动作概率归零后的最终策略分布。

#### 5. 为什么这个机制重要

如果没有 action mask，PPO 会在巨大 token 空间中频繁采样非法公式，例如 operand 数量不够、time-series operator 缺少 time delta、SEP 提前结束等。这样会导致：

1. 大量 episode 无效；
2. reward 大量浪费在语法错误上；
3. policy gradient 方差变大；
4. 学到的不是“如何生成好 alpha”，而是先花很长时间学“如何不犯语法错误”。

Invalid action masking 相当于把已知的符号语法结构作为 inductive bias 加入策略生成过程。它没有直接告诉模型哪个公式收益高，但它把搜索空间限制在形式合法的公式附近，从而让 RL 更专注于学习：**在合法公式空间中，哪些 token 组合更可能产生能提升 alpha pool 表现的因子。**

### Compact MDP Form

\[
\begin{aligned}
s_t &= [\text{BEG}, a_1,\ldots,a_{t-1}],\\
a_t &\sim \pi_\theta(a_t\mid s_t),\quad a_t\in \mathcal A_{\text{valid}}(s_t),\\
s_{t+1} &= [s_t,a_t],\\
r_t &= 0,\quad t<T,\\
R_T &= \bar\sigma_y\left(\sum_{i=1}^k w_i f_i\right),\\
\gamma &= 1.
\end{aligned}
\]

---

## 4. Representation

- **Representation type:**  
  Reverse Polish Notation, RPN，逆波兰表达式。  
  本质上公式是 expression tree，但模型为了使用自回归序列生成器，将树的后序遍历 flatten 成 token sequence。

- **Token/operator set:**  
  论文使用四类 operator：

  1. cross-sectional unary operators:  
     \[
     \operatorname{Abs}(x),\operatorname{Log}(x)
     \]

  2. cross-sectional binary operators:  
     \[
     x+y,\quad x-y,\quad x\cdot y,\quad x/y,\quad
     \operatorname{Greater}(x,y),\quad \operatorname{Less}(x,y)
     \]

  3. time-series unary operators:
     \[
     \operatorname{Ref}(x,t),\operatorname{Mean}(x,t),
     \operatorname{Med}(x,t),\operatorname{Sum}(x,t),
     \operatorname{Std}(x,t),\operatorname{Var}(x,t),
     \operatorname{Max}(x,t),\operatorname{Min}(x,t),
     \operatorname{Mad}(x,t),\operatorname{Delta}(x,t),
     \operatorname{WMA}(x,t),\operatorname{EMA}(x,t)
     \]

  4. time-series binary operators:
     \[
     \operatorname{Cov}(x,y,t),\quad \operatorname{Corr}(x,y,t).
     \]

- **Validity mechanism:**  
  使用 invalid action masking 保证形式合法；对语义非法表达式给 \(-1\) reward 惩罚。

- **Advantages:**  
  RPN 的好处是：

  1. 可以用 LSTM / Transformer 等标准序列模型生成；
  2. operator arity 固定，因此 RPN 可以无歧义地还原为 expression tree；
  3. grammar mask 容易接入 PPO；
  4. 符号公式比黑箱深度模型更容易解释。

- **Limitations:**  
  RPN flatten 会弱化树结构 inductive bias。公式语义天然是树结构，但 LSTM 看到的是线性序列；当表达式变长时，局部 token 依赖和真实语义依赖可能不一致。这也是后续 tree-based search、AST encoding、MCTS 或 GFlowNet 方法试图改进的方向。

---

## 5. Algorithm Mechanism

- **RL/search method:**  
  PPO-based policy gradient。  
  没有 MCTS、beam search、genetic programming 作为主算法。

- **On-policy or off-policy:**  
  理论上 PPO 是 on-policy 算法。但论文 Algorithm 2 写了 replay buffer \(D\)，这在严格 PPO 叙述中略微容易引起混淆。更准确地说：论文采用 PPO clipped objective 训练生成策略，整体应归类为 on-policy / near-on-policy policy optimization。

- **Learned components:**

  1. policy network:
     \[
     \pi_\theta(a_t\mid s_t)
     \]
     用于生成下一个 token；

  2. value network:
     \[
     V_\phi(s_t)
     \]
     用于估计 advantage；

  3. shared LSTM feature extractor:
     将 token sequence 编码为 dense representation；

  4. separate policy head and value head。

- **Search component:**  
  搜索主要来自 policy sampling，而不是显式搜索树。探索空间由：

  - PPO stochastic policy；
  - invalid action masking；
  - reward from combination model；

  共同控制。

- **Training loop:**

  1. 初始化 alpha pool \(\mathcal F\) 和权重 \(w\)；
  2. policy 从 BEG 开始逐 token 生成 RPN 序列；
  3. 每一步使用 action mask 保证动作合法；
  4. 当遇到 SEP 或达到长度上限后，解析公式 \(f\)；
  5. 将 \(f\) 加入 alpha pool；
  6. 重新优化组合权重 \(w\)；
  7. 如果 pool 超过容量限制，删除绝对权重最小的 alpha；
  8. 计算组合后 mega-alpha 的 IC 作为 reward；
  9. 用 PPO clipped objective 更新 policy；
  10. 重复上述过程，使 policy 越来越倾向于生成能提升当前 pool 的 alpha。

- **Key update objective:**

生成器总体目标可写为：

\[
J(\theta)
=
\mathbb E_{f\sim \pi_\theta}
\left[
\bar\sigma_y\left(c^*(X;\mathcal F\cup\{f\})\right)
\right].
\]


PPO surrogate objective 为：

\[
L^{\operatorname{CLIP}}(\theta)
=
\hat{\mathbb E}_t
\left[
\min
\left\{
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right\}
\right],
\]

其中

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\text{old}}(a_t\mid s_t)}.
\]

---

## 6. Reward Design

- **Reward signal:**  
  组合后 mega-alpha 与未来收益之间的 average IC：

  \[
  R_T = \bar\sigma_y\left(\sum_{i=1}^k w_i f_i\right).
  \]

  如果强调新增因子与已有 pool 的关系，也可以概念化为：

  \[
  \operatorname{Eval}(\mathcal F\cup\{f_{\text{new}}\}),
  \]

  但不是传统单因子：

  \[
  \bar\sigma_y(f_{\text{new}}).
  \]

- **Terminal or intermediate:**  
  主要是 terminal reward。部分公式片段没有 reward：

  \[
  r_t=0,\quad t<T.
  \]

  这意味着 credit assignment 仍然困难。

- **Pool-aware or standalone:**  
  明确是 pool-aware。Reward 依赖当前 pool 和组合权重。

- **Risk-aware or uncertainty-aware:**  
  不是 risk-aware，也不是 uncertainty-aware。没有显式建模 reward 分布、估计不确定性、尾部风险、回撤、换手、容量或交易成本。

- **Shaping mechanism:**  
  有两类弱 shaping：

  1. invalid semantic expression 给 \(-1\) reward；
  2. \(\gamma=1\)，不惩罚长公式，因为作者认为较长但有效的公式更难找到，不希望天然偏好短公式。

- **Possible reward bias:**

  1. **Training IC overfitting:** reward 来自训练集上的组合 IC，可能使生成器过拟合训练 period。
  2. **Evaluator non-stationarity:** alpha pool 一直在变，同一个公式在不同 pool 下 reward 不同。
  3. **Sparse reward:** 只有公式结束后才知道 reward，token-level credit assignment 很弱。
  4. **Absolute pool IC vs marginal contribution ambiguity:** 伪代码使用 \(IC_{\text{new}}\)，如果不是显式差分 reward，可能难以精确区分“新增因子贡献”与“已有 pool 已经强”的影响。
  5. **No deployment-aware reward:** 没有把收益、换手、交易成本、容量、行业/风格暴露作为 reward。

---

## 7. Factor Pool and Combination

- **Pool maintained:**  
  是。论文维护一个 alpha pool \(\mathcal F\)，训练过程中不在每个 episode 后重置。

- **Pool update rule:**  
  新公式生成后：

  \[
  \mathcal F \leftarrow \mathcal F\cup\{f_{\text{new}}\},
  \qquad
  w \leftarrow w \parallel \operatorname{rand}().
  \]

  然后用梯度下降优化组合权重。

- **Factor selection/removal:**  
  如果 pool 超过容量上限，则删除绝对权重最小的因子：

  \[
  p=\arg\min_i |w_i|,
  \qquad
  \mathcal F\leftarrow \mathcal F\setminus\{f_p\}.
  \]

  这个设计隐含假设是：线性组合中绝对权重小的 alpha 对 mega-alpha 贡献较小。

- **Combination model:**  
  线性组合：

  \[
  c(X;\mathcal F,w)=\sum_{j=1}^k w_j f_j(X).
  \]

  论文选择线性模型是为了保持组合后 mega-alpha 的可解释性。

- **Weight optimization:**  
  使用 MSE loss：

  \[
  L(w)=
  \frac{1}{nT}
  \sum_{t=1}^T
  \|z_t-y_t\|^2.
  \]

  由于 alpha values 和 target 都做了 normalization，MSE 可以写成 IC 和 mutual IC 的函数：

  \[
  L(w)
  =
  \frac{1}{n}
  \left(
  1
  -
  2\sum_{i=1}^k w_i\bar\sigma_y(f_i)
  +
  \sum_{i=1}^k\sum_{j=1}^k
  w_iw_j\bar\sigma(f_i(X),f_j(X))
  \right).
  \]

  这个公式很重要，因为它把组合权重优化转化为只依赖：

  1. 每个因子和 target 的 IC；
  2. 因子之间的 pairwise mutual IC。

- **Static or dynamic weights:**  
  静态权重。权重 \(w\) 对训练期优化后，在评价/回测中作为固定组合方式使用。没有 market regime dependent dynamic allocation。

- **Deployment realism:**  
  中等。论文有 top-\(k\)/drop-\(n\) 投资模拟，但主模型不是直接面向交易收益优化，也没有系统考虑交易成本、换手率、容量、行业中性、市值暴露等实盘因素。

---

## 8. Experiments and Evaluation

- **Dataset:**  
  中国 A 股 raw price-volume data，来自 Baostock。价格/成交量数据做了前复权处理。

- **Universe:**  
  CSI300 和 CSI500 成分股。

- **Time split:**  

  | Split | Period |
  |---|---|
  | Training | 2009/01/01 -- 2018/12/31 |
  | Validation | 2019/01/01 -- 2019/12/31 |
  | Test | 2020/01/01 -- 2021/12/31 |

- **Prediction target:**  
  20-day return:

  \[
  \frac{\operatorname{Ref}(\text{close},-20)}{\text{close}}-1.
  \]

- **Metrics:**  

  1. IC:
     \[
     \sigma(u,v)
     \]
     即 cross-sectional Pearson correlation；

  2. Rank IC:
     \[
     \sigma_{\text{rank}}(u,v)
     =
     \sigma(r(u),r(v)).
     \]

- **Baselines:**

  Formulaic alpha mining baselines:

  1. GP_top；
  2. GP_filter；
  3. PPO_top；
  4. PPO_filter。

  End-to-end stock trend forecasting baselines:

  1. MLP；
  2. XGBoost；
  3. LightGBM。

- **Transaction costs:**  
  主 IC / RankIC 实验不涉及交易成本。投资模拟中用 top-\(k\)/drop-\(n\) 限制每日最多换出/买入 \(n=5\) 只股票以降低交易成本，但论文没有清晰给出显式手续费率模型。

- **Turnover:**  
  没有直接报告 turnover 指标。只是通过 drop-\(n\) 机制控制交易频率。

- **Random seeds:**  
  对有随机性的实验组合使用 10 个 random seeds，并报告均值和标准差。

- **Robustness tests:**  

  1. CSI300 / CSI500 双 universe；
  2. pool size \(k\in\{1,10,20,50,100\}\)；
  3. case study 说明 mutual IC 不能有效判断 synergy；
  4. investment simulation 检查简单交易策略表现。

- **Code availability:**  
  论文给出公开代码仓库：
  \[
  \text{https://github.com/RL-MLDM/alphagen/}
  \]

- **Main reported result:**  
  在 Table 2 中，Ours 在 CSI300 上达到：

  \[
  IC=0.0725,\qquad RankIC=0.0806.
  \]

  在 CSI500 上达到：

  \[
  IC=0.0438,\qquad RankIC=0.0727.
  \]

  均高于 PPO、GP、MLP、XGBoost、LightGBM 等对照方法。

---

## 9. Main Contributions

1. **The paper contributes pool-aware reward design by addressing the mismatch between standalone alpha IC and downstream combined-factor performance.**  
   它不是直接筛选高 IC 单因子，而是让 generator 学会生成对当前 alpha pool 有用的公式。

2. **The paper contributes a PPO-based sequential formula generation framework by addressing the large discrete symbolic search space of formulaic alphas.**  
   通过 RPN token generation + action masking，把公式挖掘转化为可训练的 MDP。

3. **The paper contributes an incremental linear alpha pool optimization mechanism by addressing the computational difficulty of repeatedly combining many generated formulas.**  
   通过 IC / mutual IC cache 形式重写 MSE loss，并用绝对权重最小原则淘汰因子。

4. **The paper contributes empirical evidence against mutual-IC filtering by showing that high mutual IC alphas can still be useful in linear combinations.**  
   这点对量化因子研究很重要：相似性不等于冗余，高相关因子的差分方向可能仍然带来新的预测信息。

---

## 10. Limitations

### Algorithmic limitation 1: Sparse terminal reward

公式只有结束后才获得 reward。对于一个 20-token RPN 序列，PPO 只能从最终组合 IC 反推前面每个 token 的贡献，credit assignment 很弱。

### Algorithmic limitation 2: Sequence representation weakens tree structure

公式本质是 expression tree，但模型使用 RPN 序列和 LSTM 表示。虽然 RPN 可无歧义还原树，但 LSTM 的 inductive bias 仍是序列型，不一定适合捕捉公式的层级组合语义。

### Algorithmic limitation 3: Non-stationary reward makes policy learning unstable

由于 alpha pool 持续变化，同一个公式在不同训练阶段的 reward 可能不同。这种非平稳性让 value function 和 advantage estimation 更难。

### Algorithmic limitation 4: PPO may not be the most sample-efficient method

每个公式评估都需要调用 evaluator，计算成本高。PPO 没有显式利用树搜索、surrogate model、uncertainty 或 replay-style reuse，因此后续方法可以在探索效率上改进。

### Evaluation / finance limitation 1: Reward is not transaction-cost-aware

主 reward 是 IC，不是净收益、Sharpe、IR、换手惩罚或成本调整收益。IC 高不一定等价于实盘收益高。

### Evaluation / finance limitation 2: No capacity and liquidity analysis

没有系统分析成交容量、冲击成本、行业暴露、市值暴露、风格暴露、停牌/涨跌停等 A 股实盘约束。

### Evaluation / finance limitation 3: Possible train-period overfitting

训练期是 2009--2018，测试期是 2020--2021。虽然时间切分合理，但公式搜索空间极大，反复 evaluator feedback 仍可能导致对训练环境的结构性过拟合。

### Evaluation / finance limitation 4: Static combination weights

组合权重不是动态的。市场 regime 变化时，某些因子有效性可能发生变化，但模型没有学习 \(w_t\)：

\[
F_t=\sum_{k=1}^K w_t^{(k)} f_t^{(k)}.
\]

AlphaGen 使用的是更接近固定 \(w^{(k)}\) 的组合方式。

---

## 11. Relation to AlphaGen and Existing Literature

- **Relation to AlphaGen:**  
  这篇论文本身就是通常所说的 **AlphaGen, KDD 2023**。它是 RL-based alpha mining 中非常核心的一篇，代表了“policy-based sequential expression generation + pool-aware reward”的范式。

- **Main bottleneck addressed:**  
  解决传统 formulaic alpha mining 的两个问题：

  1. GP 或单因子方法主要优化 standalone IC；
  2. mutual IC filtering 不能准确衡量多个 alpha 的组合协同作用。

- **Paradigm classification:**  

  1. Policy-based sequential expression generation；
  2. Reward design / pool-aware evaluation；
  3. Symbolic alpha generation；
  4. Discovery-plus-combination framework。

- **Difference from prior work:**  

  | Prior work | AlphaGen difference |
  |---|---|
  | Human-designed 101 alphas | 自动生成公式 |
  | GP alpha mining | 用 PPO policy generator 替代遗传变异主流程 |
  | Single-alpha PPO | reward 改为组合模型表现，而不是单因子 IC |
  | Mutual-IC filtering | 直接用下游组合效果判断 synergy |
  | Black-box ML alpha | 保持公式 alpha 和线性组合的解释性 |

- **Whether it is more RL-like or search-like:**  
  更 RL-like。它没有显式 MCTS 或 beam search，主要依赖 PPO policy sampling 探索公式空间。  
  但从问题本质看，它仍然是在做 symbolic search，只是用 RL 训练搜索策略。

---

## 12. How to Incorporate This Paper into My Review

### Suggested section

- **Section:**  
  Policy-based Sequential Expression Generation / Pool-aware RL Alpha Mining

- **Reason:**  
  它是 RL 挖掘公式化 alpha 集合的代表性起点。后续 RiskMiner、AlphaQCM、AlphaForge 等方法都可以被解释为对它的不同瓶颈进行改进：结构搜索、reward 稳定性、不确定性探索、动态组合等。

### Suggested one-line description

AlphaGen formulates formulaic alpha generation as a PPO-based token-level MDP and uses downstream linear-combination IC as a pool-aware terminal reward to mine synergistic alpha collections.

### Suggested paragraph

AlphaGen is a representative policy-based framework for RL-driven formulaic alpha mining. Instead of optimizing the standalone IC of each generated formula, it maintains an alpha pool and evaluates newly generated formulas through their contribution to a downstream linear combination model. Candidate formulas are represented as RPN token sequences and generated by a PPO policy under grammar-based invalid action masking. This design directly addresses the mismatch between individual alpha quality and combined-factor utility, showing that mutual-IC filtering is not a reliable proxy for alpha synergy. Its main limitation is that the reward remains sparse, terminal, and evaluator-dependent, while the sequence representation only weakly preserves the tree structure of symbolic formulas.

### Suggested comparison table row

| Paper | Venue | Core formulation | Representation | RL/Search mechanism | Reward type | Pool-aware? | Dynamic allocation? | Key contribution | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| AlphaGen / Generating Synergistic Formulaic Alpha Collections via RL | KDD 2023 | Sequential generation for synergistic formulaic alpha pool construction | RPN token sequence, equivalent to expression tree postorder traversal | PPO with invalid action masking; LSTM policy/value network | Terminal pool-level combination IC | Yes | No | Optimizes downstream combined-factor performance instead of standalone alpha IC | Sparse delayed reward; weak tree-structure inductive bias; no transaction-cost-aware reward |

---

## 13. My Overall Assessment

这篇论文主要有 **methodology value** 和 **review anchor value**。

它最重要的价值不是“PPO 生成公式”本身，而是把 RL alpha mining 的目标从：

\[
\max_f \bar\sigma_y(f)
\]

推进到：

\[
\max_{\mathcal F}
\bar\sigma_y\left(\sum_i w_i f_i\right).
\]

也就是从“找一个好因子”变成“找一组能协同工作的因子”。这和真实量化研究更接近，因为实盘中很少只依赖一个单因子，而是依赖因子池和组合模型。

从你的 review 文档角度，它应该作为 2023 年核心基准论文来写。后续论文可以围绕它的不足展开：

1. reward 稀疏；
2. RPN 序列弱化树结构；
3. PPO 探索效率有限；
4. evaluator 可能过拟合；
5. 没有动态组合权重；
6. 没有真实交易成本和风险约束。

一句话评价：

> AlphaGen establishes the pool-aware RL formulation for symbolic alpha mining, but leaves open the harder problems of structured exploration, reward densification, uncertainty-aware search, and deployment-aware dynamic allocation.


---

## Appendix: Action Masking and Logits Calculation Examples

这一节补充解释 AlphaGen 图中 **action logits / action masks / masked distribution** 的具体计算过程。核心问题是：PPO actor 会对所有 token 输出 logits，但当前 RPN 语法状态下并不是所有 token 都合法。因此，实际采样不是直接对 logits 做 softmax，而是先用 action mask 屏蔽非法动作，再对 masked logits 做 softmax。

### 1. Policy network 输出的是 actor logits

在 PPO 中，图中的 policy network 更准确地说是 **actor / policy head**，它负责输出每个候选 token 的原始打分：

\[
z_\theta(s_t)=\left(z_{\$open},z_{0.5},z_{Add},z_{Log},z_{SEP},\ldots\right).
\]

这些 logits 只是未归一化偏好分数，不是最终概率。critic / value head 不负责选择 token，而是估计状态价值：

\[
V_\phi(s_t),
\]

用于计算 PPO 的 advantage：

\[
\hat A_t=\hat R_t-V_\phi(s_t).
\]

因此可以把网络结构理解为：

```text
shared LSTM backbone
        |
        |---- actor / policy head  -> action logits -> mask -> action distribution -> sample token
        |
        |---- critic / value head   -> V(s_t) -> compute advantage for PPO update
```

### 2. Masked logits 的通用计算

设当前所有候选 token 组成动作空间：

\[
\mathcal A=\{\$open,\ 0.5,\ Add,\ Log,\ SEP\}.
\]

Policy actor 输出 logits：

\[
z_\theta(s_t)=
\begin{bmatrix}
z_{\$open}\\
z_{0.5}\\
z_{Add}\\
z_{Log}\\
z_{SEP}
\end{bmatrix}.
\]

环境根据当前 RPN stack 生成 action mask：

\[
m(s_t)=
\begin{bmatrix}
m_{\$open}\\
m_{0.5}\\
m_{Add}\\
m_{Log}\\
m_{SEP}
\end{bmatrix},
\qquad
m_i(s_t)\in\{0,1\}.
\]

其中：

\[
m_i(s_t)=1
\]

表示 token \(i\) 当前合法；

\[
m_i(s_t)=0
\]

表示 token \(i\) 当前非法。

然后把非法 token 的 logit 设为 \(-\infty\)：

\[
\tilde z_i(s_t)=
\begin{cases}
z_i(s_t), & m_i(s_t)=1,\\
-\infty, & m_i(s_t)=0.
\end{cases}
\]

最后对 masked logits 做 softmax：

\[
\pi_\theta(a_i\mid s_t)=
\frac{\exp(\tilde z_i(s_t))}
{\sum_j \exp(\tilde z_j(s_t))}.
\]

因为：

\[
\exp(-\infty)=0,
\]

所以非法 token 的最终采样概率为 0。

代码中通常不会真的写 \(-\infty\)，而是用一个极小值，例如：

```python
masked_logits = logits.masked_fill(~action_mask, -1e9)
probs = torch.softmax(masked_logits, dim=-1)
action = torch.distributions.Categorical(probs).sample()
```

---

### 3. 例子一：刚开始生成，算子和 SEP 都非法

假设当前状态是：

\[
s_t=[BEG].
\]

当前 RPN stack 为空：

\[
\text{stack}=[].
\]

这时还没有任何表达式，所以不能选择 `Add`、`Log` 或 `SEP`。合法动作只有 feature 或 constant，例如：

\[
\$open,\quad 0.5.
\]

假设 actor 输出 logits：

\[
z_\theta(s_t)=
\begin{bmatrix}
z_{\$open}\\
z_{0.5}\\
z_{Add}\\
z_{Log}\\
z_{SEP}
\end{bmatrix}
=
\begin{bmatrix}
1.0\\
0.5\\
2.0\\
3.0\\
1.5
\end{bmatrix}.
\]

注意这里 `Log` 的 logit 最大：

\[
z_{Log}=3.0.
\]

但是当前 stack 为空，`Log` 没有输入对象，所以它形式非法。

对应 action mask 是：

\[
m(s_t)=
\begin{bmatrix}
1\\
1\\
0\\
0\\
0
\end{bmatrix}.
\]

masked logits 变成：

\[
\tilde z_\theta(s_t)=
\begin{bmatrix}
1.0\\
0.5\\
-\infty\\
-\infty\\
-\infty
\end{bmatrix}.
\]

softmax 只在合法动作上归一化：

\[
e^{1.0}=2.718,\qquad e^{0.5}=1.649.
\]

分母为：

\[
2.718+1.649=4.367.
\]

所以：

\[
\pi(\$open\mid s_t)=\frac{2.718}{4.367}=0.622,
\]

\[
\pi(0.5\mid s_t)=\frac{1.649}{4.367}=0.378.
\]

而非法动作概率为：

\[
\pi(Add\mid s_t)=\pi(Log\mid s_t)=\pi(SEP\mid s_t)=0.
\]

结论是：即使 actor 原始上最想选 `Log`，mask 后 `Log` 的概率仍然是 0，最终只能从 `$open` 和 `0.5` 中采样。

---

### 4. 例子二：已有一个表达式，Log 合法但 Add 非法

假设当前状态是：

\[
s_t=[BEG,\ \$open].
\]

当前 stack 是：

\[
\text{stack}=[\$open].
\]

这时：

- `Log` 合法，因为它是一元算子，只需要一个表达式；
- `Add` 非法，因为它是二元算子，需要两个表达式；
- `SEP` 合法，因为当前 stack 中已经有一个完整表达式，可以结束；
- `$open` 和 `0.5` 也可以继续加入，使后续表达式更复杂。

假设 actor 输出 logits：

\[
z_\theta(s_t)=
\begin{bmatrix}
0.2\\
0.0\\
4.0\\
1.5\\
1.0
\end{bmatrix}.
\]

这里 actor 非常想选 `Add`，因为：

\[
z_{Add}=4.0.
\]

但是当前只有一个表达式，`Add` 形式非法。mask 为：

\[
m(s_t)=
\begin{bmatrix}
1\\
1\\
0\\
1\\
1
\end{bmatrix}.
\]

masked logits 为：

\[
\tilde z_\theta(s_t)=
\begin{bmatrix}
0.2\\
0.0\\
-\infty\\
1.5\\
1.0
\end{bmatrix}.
\]

计算 softmax：

\[
e^{0.2}=1.221,\quad e^0=1,\quad e^{1.5}=4.482,\quad e^1=2.718.
\]

分母为：

\[
1.221+1+4.482+2.718=9.421.
\]

因此：

\[
\pi(\$open\mid s_t)=\frac{1.221}{9.421}=0.130,
\]

\[
\pi(0.5\mid s_t)=\frac{1}{9.421}=0.106,
\]

\[
\pi(Add\mid s_t)=0,
\]

\[
\pi(Log\mid s_t)=\frac{4.482}{9.421}=0.476,
\]

\[
\pi(SEP\mid s_t)=\frac{2.718}{9.421}=0.288.
\]

如果采样到 `Log`，新 RPN 为：

\[
[BEG,\ \$open,\ Log],
\]

对应公式：

\[
Log(\$open).
\]

如果采样到 `SEP`，episode 结束，公式为：

\[
\$open.
\]

---

### 5. 例子三：已有两个表达式，可以选 Add，但不能选 SEP

假设当前状态是：

\[
s_t=[BEG,\ \$open,\ 0.5].
\]

当前 stack 是：

\[
\text{stack}=[\$open,\ 0.5].
\]

这时：

- `Add` 合法，因为它是二元算子，需要两个表达式；
- `SEP` 非法，因为 stack 里有两个表达式，还没有合成为一个完整表达式；
- `Log` 是否允许要看规则。论文中要求 multi-token expression 不能退化成常数，因此如果 `Log` 作用在最近的 `0.5` 上，会得到 \(Log(0.5)\)，这是常数表达式，所以这里 `Log` 不允许。

一个合理 mask 是：

\[
m(s_t)=
\begin{bmatrix}
1\\
1\\
1\\
0\\
0
\end{bmatrix}.
\]

假设 actor 输出 logits：

\[
z_\theta(s_t)=
\begin{bmatrix}
0.0\\
0.5\\
1.2\\
2.5\\
3.0
\end{bmatrix}.
\]

actor 原始最想选 `SEP`，其次想选 `Log`，但这两个动作当前都非法。

masked logits 为：

\[
\tilde z_\theta(s_t)=
\begin{bmatrix}
0.0\\
0.5\\
1.2\\
-\infty\\
-\infty
\end{bmatrix}.
\]

softmax：

\[
e^0=1,\quad e^{0.5}=1.649,\quad e^{1.2}=3.320.
\]

分母为：

\[
1+1.649+3.320=5.969.
\]

所以：

\[
\pi(\$open\mid s_t)=\frac{1}{5.969}=0.168,
\]

\[
\pi(0.5\mid s_t)=\frac{1.649}{5.969}=0.276,
\]

\[
\pi(Add\mid s_t)=\frac{3.320}{5.969}=0.556,
\]

\[
\pi(Log\mid s_t)=0,
\]

\[
\pi(SEP\mid s_t)=0.
\]

如果采样到 `Add`，新状态是：

\[
[BEG,\ \$open,\ 0.5,\ Add].
\]

对应公式：

\[
Add(\$open,0.5).
\]

此时 stack 会从：

\[
[\$open,\ 0.5]
\]

变成：

\[
[Add(\$open,0.5)].
\]

下一步就可以选择 `SEP` 结束。

---

### 6. 例子四：当前公式已经完整，可以选 SEP

假设当前状态是：

\[
s_t=[BEG,\ \$open,\ 0.5,\ Add].
\]

它对应公式：

\[
Add(\$open,0.5).
\]

当前 stack 是：

\[
[Add(\$open,0.5)].
\]

这时公式已经完整，所以 `SEP` 合法。假设 actor 输出 logits：

\[
z_\theta(s_t)=
\begin{bmatrix}
0.1\\
0.0\\
0.5\\
1.0\\
2.0
\end{bmatrix}.
\]

如果当前只有一个表达式，那么：

- `SEP` 合法，可以结束公式；
- `Log` 合法，可以继续得到 \(Log(Add(\$open,0.5))\)；
- `Add` 非法，因为它需要两个表达式；
- feature / constant 可以继续加入，等待后面组合。

mask 可以写为：

\[
m(s_t)=
\begin{bmatrix}
1\\
1\\
0\\
1\\
1
\end{bmatrix}.
\]

masked logits 为：

\[
\tilde z_\theta(s_t)=
\begin{bmatrix}
0.1\\
0.0\\
-\infty\\
1.0\\
2.0
\end{bmatrix}.
\]

softmax 后，`Add` 的概率是 0，而 `SEP` 因为 logit 最大，被采样概率最高。如果采样到 `SEP`，episode 结束，最终表达式是：

\[
Add(\$open,0.5).
\]

然后系统把该公式送入 evaluator / combination model，计算加入当前 alpha pool 后的组合 IC 作为 reward。

---

### 7. 形式非法与语义非法的处理区别

这里需要特别区分 **形式非法** 和 **语义非法**。

| 类型 | 例子 | 能否提前阻止 | 处理方式 |
|---|---|---:|---|
| 形式非法 | 操作数不够时选 `Add`；公式未完成时选 `SEP`；`Mean(x,0.5)` | 可以 | action mask，采样概率直接变成 0 |
| 语义非法 | `Log(x)` 中某些样本 \(x\le 0\)；除法分母为 0 | 不一定 | 生成后 evaluate，失败则给 \(R=-1\) |

形式非法动作通过 mask 在采样前直接屏蔽；语义非法表达式如果已经生成出来，就通过最低 reward 惩罚，使 PPO 在后续更新中降低类似轨迹的概率。

---

### 8. 总结

整个过程可以概括为：

```text
当前 RPN state
→ LSTM / actor policy head
→ 所有 token 的 logits
→ RPN 语法规则生成 action mask
→ 非法 token 的 logits 设为 -∞
→ softmax 得到 masked distribution
→ 只从合法 token 中采样
→ 拼接到 RPN 序列，进入下一个 state
```

因此，logits 反映的是神经网络学到的 token 偏好；mask 反映的是当前 RPN 语法约束；masked distribution 才是真正用于采样下一个 token 的概率分布。AlphaGen 的 PPO 不是在无限制地生成公式，而是在 action mask 限定下，在合法公式空间内进行策略搜索。
