# RiskMiner 笔记：RPN、MCTS 搜索树与 \(Q(s,a)\) 的理解

## 1. 核心结论

这一部分主要回答三个问题：

\[
\boxed{
\text{RPN 本身仍然可以还原成树，但作为序列输入时，它没有显式维护树搜索结构。}
}
\]

\[
\boxed{
\text{直接 append 是一次性采样序列；MCTS 是在前缀树上反复搜索、记录、复用统计信息。}
}
\]

\[
\boxed{
\text{RiskMiner 的 }Q\text{ 通常定义在 }(s,a)\text{ 边上，而不是只定义在 state 上。}
}
\]

---

## 2. RPN 明明可以还原成树，为什么还说 flatten 后丢失显式树结构？

例如图中的公式：

\[
Div(5, Cov(open, volume, 15d))
\]

对应表达式树是：

```text
        Div
       /   \
      5    Cov
          / | \
       open vol 15d
```

它的 RPN 序列是：

\[
[BEG,\ 5,\ open,\ volume,\ 15d,\ Cov,\ Div,\ SEP]
\]

从数学表达能力上讲，RPN 当然可以唯一还原成表达式树。因此，这里的“丢失树结构”不是说信息上无法恢复表达式树，而是说：

\[
\boxed{
\text{模型生成和学习时看到的是一维 token 序列，而不是显式的 tree search object。}
}
\]

也就是说，RPN 是 tree 的一种线性编码；但普通 policy network 生成时通常只是：

\[
s_t = [token_1,\ldots,token_t]
\]

然后：

\[
a_t \sim \pi_\theta(\cdot|s_t)
\]

这种 sequential generation 没有显式存储：

- 这个 prefix 对应树上的哪个节点；
- 哪些 subtree 已经搜索过；
- 当前 operator 的 children 是否已经探索充分；
- 某个局部结构，例如 `Cov(open, volume, 15d)`，在不同公式中是否反复出现且表现好；
- 每个前缀分支的访问次数和值估计。

所以更准确的表述是：

\[
\boxed{
\text{RPN 保留了可还原的表达式树信息，但普通 sequential generation 没有显式利用搜索树统计结构。}
}
\]

---

## 3. RPN append 和 MCTS 到底哪里不一样？

假设要生成：

\[
Div(5, Cov(open, volume, 15d))
\]

RPN 生成过程是：

\[
BEG \rightarrow 5 \rightarrow open \rightarrow volume \rightarrow 15d \rightarrow Cov \rightarrow Div \rightarrow SEP
\]

### 3.1 直接 append / policy sampling

直接 append 是：

\[
a_t \sim \pi_\theta(a_t|s_t)
\]

每一步根据当前 policy 采样一个 token，直到结束。例如：

```text
s0 = [BEG]
sample 5

s1 = [BEG, 5]
sample open

s2 = [BEG, 5, open]
sample volume
...
```

它通常是一条轨迹走到底：

\[
\text{sample one complete formula}
\]

然后得到 reward，再更新 policy。

这种方式的问题是：如果某个中间 prefix 后面有很多可能，直接 append 通常只采其中一条，不能系统比较多个分支。

例如在：

\[
s=[BEG,5,open,volume,15d]
\]

下一步可以选：

\[
Cov,\ Corr,\ Add,\ Sub,\ Div,\ldots
\]

直接 append 只采一个，比如采到 `Cov`。另一个可能更好的 `Corr` 分支可能很久都不会被充分尝试。

---

### 3.2 MCTS 搜索

MCTS 不是简单生成一条序列，而是把所有 prefix 看成一棵搜索树。

例如：

```text
[BEG]
 ├── 5
 │    ├── open
 │    │    ├── volume
 │    │    │    ├── 15d
 │    │    │    │    ├── Cov
 │    │    │    │    ├── Corr
 │    │    │    │    ├── Add
 │    │    │    │    └── ...
 │    │    └── close
 │    └── volume
 └── open
```

每个 prefix 都是一个 state：

\[
s=[BEG,5,open,volume,15d]
\]

每条边是选择下一个 token：

\[
(s,Cov),\quad (s,Corr),\quad (s,Add)
\]

MCTS 会反复从 root 走到 leaf，进行：

\[
Selection \rightarrow Expansion \rightarrow Rollout \rightarrow Backpropagation
\]

并且记录：

\[
N(s,a),\quad Q(s,a),\quad P(s,a),\quad R(s,a)
\]

所以它不是一次性 append，而是：

\[
\boxed{
\text{在 token-prefix tree 上反复试验，并把经验保存在树上。}
}
\]

---

## 4. 直接 append 和 MCTS 的具体区别

假设当前前缀是：

\[
s=[BEG,5,open,volume,15d]
\]

候选动作：

\[
Cov,\ Corr,\ Add
\]

### 4.1 直接 append

```text
第 1 次采样：选 Cov → 完整公式 reward = 0.04
第 2 次采样：从头重新采样，可能走到完全不同前缀
第 3 次采样：也许很久都不回到这个 s
```

它没有显式记录“在这个 prefix 下，Cov、Corr、Add 哪个更好”。

### 4.2 MCTS

MCTS 会记录：

| Edge | \(N(s,a)\) | \(Q(s,a)\) |
|---|---:|---:|
| \((s,Cov)\) | 20 | 0.045 |
| \((s,Corr)\) | 5 | 0.060 |
| \((s,Add)\) | 2 | 0.010 |

下一次再到这个 prefix 时，MCTS 可以知道：

- `Cov` 试过很多次，表现还可以；
- `Corr` 试得少但 value 更高，值得继续扩展；
- `Add` 表现差，可以少试。

这就是 MCTS 相比直接 append 的关键优势：

\[
\boxed{
\text{MCTS 复用 prefix-level search statistics，而直接 append 不复用显式搜索树统计。}
}
\]

---

## 5. RPN 和 MCTS 的关系

RPN 是**表达式表示方式**。

MCTS 是**搜索算法**。

二者不是同一层东西。

\[
\text{RPN: formula} \leftrightarrow \text{token sequence}
\]

\[
\text{MCTS: how to search over token sequences}
\]

所以可以同时使用：

\[
\boxed{
\text{RiskMiner 用 RPN 表示公式，用 MCTS 搜索 RPN prefix tree。}
}
\]

如果只把 RPN 当成普通序列给 policy network 逐步 append，那么树结构没有被显式搜索和记录；MCTS 则把这些 prefix 组织成一棵真实的搜索树并维护统计量。

---

## 6. MCTS 中的 value \(Q\) 是针对 node 还是 \((state, action)\) pair？

在 RiskMiner 中，写法是针对边，也就是：

\[
\boxed{
Q(s,a)
}
\]

不是单独的：

\[
V(s)
\]

论文说每条 edge \((s,a)\) 存：

\[
\{N(s,a),P(s,a),Q(s,a),R(s,a)\}
\]

这和 AlphaZero / MuZero 一类 MCTS 很像：节点是 state，边是 action，统计量放在边上。

---

## 7. 为什么定义成 \(Q(s,a)\)，而不是 \(V(s)\)？

因为在公式生成里，真正需要比较的是：

\[
\text{当前 prefix 下，下一个 token 选哪个？}
\]

也就是：

\[
a^* = \arg\max_a \text{score}(s,a)
\]

如果只定义：

\[
V(s)
\]

它只能告诉你：

\[
\text{当前 prefix }s\text{ 总体好不好}
\]

但无法直接告诉你：

\[
\text{从 }s\text{ 出发，选 Cov 好，还是 Corr 好？}
\]

例如当前 state：

\[
s=[BEG,5,open,volume,15d]
\]

如果只存：

\[
V(s)=0.05
\]

你不知道这个 0.05 是因为：

- `Cov` 分支好；
- `Corr` 分支好；
- 还是两者平均出来的。

但是如果存边值：

\[
Q(s,Cov)=0.045
\]

\[
Q(s,Corr)=0.060
\]

\[
Q(s,Add)=0.010
\]

你就可以直接比较动作。

所以 MCTS selection 需要的是：

\[
\boxed{
\text{action-level value}
}
\]

这就是为什么定义成 \(Q(s,a)\)。

---

## 8. \(V(s)\) 和 \(Q(s,a)\) 的关系

它们不是矛盾的。理论上可以互相转换。

如果有 policy \(\pi(a|s)\)，可以定义：

\[
V(s)=\sum_a \pi(a|s) Q(s,a)
\]

或者在 MCTS 里粗略取：

\[
V(s)=\max_a Q(s,a)
\]

但在 selection 时，MCTS 实际要做的是：

\[
a_t=
\arg\max_a
\left[
Q(s,a)
+
P(s,a)
\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
\right]
\]

这个公式天然就是 action-level 的。因此存 \(Q(s,a)\) 比存 \(V(s)\) 更直接。

---

## 9. 用树的角度解释为什么 edge-value 更自然

你之前学的简单 MCTS 可能把 value 放在子节点上：

```text
        S0
       /  \
     S1    S2
   V(S1)  V(S2)
```

从 root \(S0\) 选择子节点，其实等价于选择动作：

\[
a_1: S0\rightarrow S1
\]

\[
a_2: S0\rightarrow S2
\]

所以子节点 value：

\[
V(S1)
\]

也可以看成边 value：

\[
Q(S0,a_1)
\]

在 deterministic transition 里：

\[
s' = T(s,a)
\]

那么：

\[
Q(s,a) \approx R(s,a)+V(s')
\]

所以你之前学的 node-value 版本和 RiskMiner 的 edge-value 版本，本质上是两种记账方式。

---

## 10. 为什么 RiskMiner 更适合 edge-based \(Q(s,a)\)？

RiskMiner 还有即时 reward：

\[
R(s,a)
\]

也就是选择某个 token 后，如果形成合法表达式，就马上得到：

\[
IC-\lambda mutIC
\]

所以边上天然有一个 action-specific 的局部反馈。

例如同一个 state：

\[
s=[BEG,5,open,volume,15d]
\]

选择不同 action：

### 选 `Cov`

\[
s'=[BEG,5,open,volume,15d,Cov]
\]

形成：

\[
Cov(open,volume,15d)
\]

可以计算：

\[
R(s,Cov)=IC(Cov(open,volume,15d))-\lambda mutIC
\]

### 选 `Corr`

\[
s'=[BEG,5,open,volume,15d,Corr]
\]

形成：

\[
Corr(open,volume,15d)
\]

可以计算：

\[
R(s,Corr)=IC(Corr(open,volume,15d))-\lambda mutIC
\]

这两个即时 reward 不一样。

所以把 reward 和 value 放在边上更自然：

\[
R(s,Cov),\quad Q(s,Cov)
\]

\[
R(s,Corr),\quad Q(s,Corr)
\]

---

## 11. 直接 append 和 MCTS 的训练差别

### 11.1 直接 append 生成

如果是普通 policy network：

\[
a_t \sim \pi_\theta(a_t|s_t)
\]

得到完整公式后，用 reward 更新：

\[
\nabla_\theta \log \pi_\theta(\tau) R(\tau)
\]

它学到的是一个全局参数化策略，但每次采样路径的局部搜索经验不会以树的形式保存。

### 11.2 MCTS 搜索

MCTS 在一次 mining iteration 中会多次从 root 出发搜索，并不断更新树：

\[
N(s,a),Q(s,a),P(s,a),R(s,a)
\]

它在同一棵树里反复比较局部分支：

\[
Cov \text{ vs } Corr \text{ vs } Add
\]

然后更系统地把搜索预算分配给高潜力分支。

| 维度 | 直接 append | MCTS |
|---|---|---|
| 生成方式 | 从 policy 一步步采样 token | 在 prefix tree 上 selection / expansion / rollout |
| 是否显式维护搜索树 | 否 | 是 |
| 是否记录每个分支访问次数 | 通常否 | 是，\(N(s,a)\) |
| 是否记录每个分支长期价值 | 通常靠网络隐式学习 | 是，\(Q(s,a)\) |
| 是否利用先验 policy | 是，直接采样 | 是，作为 \(P(s,a)\) 指导搜索 |
| 是否复用同一 prefix 的历史搜索 | 弱 | 强 |
| 适合场景 | 较简单序列生成 | 大离散搜索空间、需要规划 |

---

## 12. 压缩理解：RPN 是表示，MCTS 是搜索

不要把 RPN 和 MCTS 混在一起。

RPN 只是告诉你一个公式怎么写：

\[
[5, open, volume, 15d, Cov, Div]
\]

MCTS 是告诉你如何探索这些可能的写法：

\[
\text{哪些 prefix 试过？}
\]

\[
\text{从这个 prefix 选哪个 token 更好？}
\]

\[
\text{哪些分支应该继续扩展？}
\]

所以：

\[
\boxed{
\text{RPN 是 representation；MCTS 是 search/control mechanism。}
}
\]

---

## 13. 三个问题的最终回答

### 问题 1：RPN 也能还原成树，为什么说 flatten 后丢失显式树结构？

因为 RPN 在信息上可以还原树，但普通序列生成模型只把它作为一维 token 序列处理，没有显式维护树节点、分支、访问次数和值统计。所谓“丢失”指的是**模型和搜索过程没有显式利用 tree structure**，不是指数学上无法恢复 tree。

### 问题 2：MCTS 相比直接 append 产生，到底哪里不一样？

直接 append 是：

\[
\text{根据 policy 一步步采样一条完整序列}
\]

MCTS 是：

\[
\text{在所有 token prefix 组成的搜索树上反复 selection、expansion、rollout、backpropagation}
\]

并且为每个分支保存：

\[
N(s,a),P(s,a),Q(s,a),R(s,a)
\]

所以 MCTS 能复用局部搜索经验，更系统地探索高潜力公式分支。

### 问题 3：MCTS 中的 value \(Q\) 是针对 node 还是 \((state, action)\) pair？为什么不用 node？

RiskMiner 中是针对边：

\[
Q(s,a)
\]

因为公式生成时最重要的是在当前前缀 \(s\) 下比较下一个 token \(a\) 的好坏。

如果只用：

\[
V(s)
\]

只能知道当前 prefix 总体价值，不能区分从这里选 `Cov`、`Corr`、`Add` 分别怎么样。

而 MCTS selection 需要：

\[
a^* = \arg\max_a \text{score}(s,a)
\]

所以使用：

\[
Q(s,a)
\]

更自然、更直接。
