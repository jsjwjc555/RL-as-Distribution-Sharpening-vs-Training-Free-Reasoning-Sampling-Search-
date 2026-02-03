# Survey: RL as Distribution Sharpening vs Training-Free Reasoning (Sampling/Search)

> 主题：研究 **强化学习后训练（RL/RLVR/GRPO 类）到底改变了 base model 什么**，以及 **training-free 的推理期方法（采样/搜索/MCTS）** 是否能与 RL 相当或互补。  
> 同时收录一类重要方法：**树搜索/MCTS + 进度/过程奖励建模**（如 Think&Cite），它们常被视为“推理期可控计算 / test-time scaling”的另一条路线。

---

## 1. 关键问题（Research Questions）

### RQ1: RLVR 是否真的引入了“超出 base 的新推理能力”？
- 还是主要把 base 已有的“正确轨迹”概率变大（distribution sharpening / sampling efficiency）？
- 在 pass@1 提升的同时，pass@k（大 k）是否出现覆盖面变窄、模式塌缩？

### RQ2: Training-free 能否在推理期逼近/超过 RL 后训练？
- 如：MCMC / power sampling、rejection sampling、tree search、self-consistency、best-of-N 等。
- 代价：token 开销、延迟、可并行性、可控性。

### RQ3: 为什么 MCTS/树搜索最近又回来了？
- 任务被显式建模为“多步决策/规划”，节点是中间推理状态，动作是生成下一步；
- 需要 **过程/进度奖励** 来引导搜索（比单纯 outcome reward 更稳定）。

---

## 2. 快速结论（TL;DR）

- **RLVR 可能更像“让模型更容易抽到它本来就会的正确推理路径”**，而不一定扩展“可解题集合”。（典型证据：大 k 视角下 base 可能追平甚至超过 RLVR；RL 输出在 base 下困惑度/似然偏高）
- **Training-free 的采样/搜索** 可以在某些任务上达到接近 RL 的单次表现，并在多样性/大 k 方面更稳，但通常更耗计算。
- **MCTS + 过程/进度奖励** 适合“多步结构化生成/检索/引用/证据对齐”的任务（如 attributed generation），可解释性与可控性更强。

---

## 3. 必读核心论文（Anchor Papers）

### (A) RL 对 base 能力边界的诊断
1. **Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?** (2025)  
   - 贡献：提出用大 k 的 pass@k 评估 reasoning boundary；指出 RLVR 更像提高采样效率而非引入新推理模式，并可能缩窄覆盖面。  
   - Link: https://arxiv.org/abs/2504.13837  
   - Project: https://limit-of-RLVR.github.io

### (B) Training-free：用采样逼近 RL（MCMC / power distribution）
2. **Reasoning with Sampling: Your Base Model is Smarter Than You Think** (2025)  
   - 核心：把目标分布设为 `p_base(y)^α`（序列级 power distribution），用 Metropolis–Hastings 做 MCMC（反复重采样后缀 + 接受率）在推理期“锐化分布”。  
   - 结论：在多个基准上单次表现可接近/超过 GRPO；并强调它与 token-level temperature 不等价。  
   - Link: https://arxiv.org/abs/2510.14901  
   - Code: https://github.com/aakaran/reasoning-with-sampling

### (C) Tree Search / MCTS + Progress Reward：面向“可归因生成”
3. **Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling** (ACL 2025 / arXiv 2024)  
   - 核心：提出 SG-MCTS（Self-Guided MCTS，用 LLM 自反思指导扩展）+ Progress Reward Model（同时度量 generation progress 与 attribution progress）。  
   - 强项：对“逐句生成+证据引用”的任务，树搜索比一次性生成更稳、更可控。  
   - Link: https://arxiv.org/abs/2412.14860  
   - ACL: https://aclanthology.org/2025.acl-long.490/  
   - Code: https://github.com/nusnlp/Think-Cite

---

## 4. 相似/相关工作清单（按主题分类）

### 4.1 RL 改变了什么：熵塌缩、多样性与“分布锐化”机制
> 关键词：entropy collapse / diversity collapse / divergence choice / RLVR dynamics
- **The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models** (2025)  
  - 关注 RL 训练中的熵变化与探索-利用权衡，解释为什么“更会答对”可能伴随多样性下降。
  - https://arxiv.org/abs/2505.22617

- **Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective** (2025)  
  - 讨论熵干预（如正则、调参）如何影响 RLVR 的稳定性与泛化。
  - https://arxiv.org/abs/2510.10150

- **The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in RLVR** (2025)  
  - 从 reverse-KL 的 mode-seeking 视角解释塌缩，并提出替代散度/训练技巧缓解。
  - https://arxiv.org/abs/2509.07430


### 4.2 Training-free / Test-time scaling（与 RL 对照）
> 关键词：sampling, rejection, MCMC, inference-time optimization, test-time compute
- **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** (2024)  
  - 用重复采样展示“推理期算力扩展”的一般现象，适合作为背景动机。
  - https://arxiv.org/abs/2407.21787

- **Training-Free Group Relative Policy Optimization** (2025)  
  - 把“相对优势/组内比较”的思想搬到推理期（不更新参数），作为 training-free 的类 GRPO 路线。
  - https://arxiv.org/abs/2510.08191

- **A Minimalist Approach to LLM Reasoning: from Rejection Sampling ...** (2025)  
  - 讨论“筛样/重加权”在多大程度上能替代 RL 的收益（非常适合写 ‘RL gains may be data selection’ 的讨论段）。
  - https://arxiv.org/html/2504.11343v1


### 4.3 MCTS / 树搜索：与 Think&Cite 类似的“搜索 + 过程信号”
> 关键词：MCTS, tree search, process reward model, reflection, planning, agent
以下这些论文与你要加的 Think&Cite 在“树搜索步骤”上高度相似（selection/expansion/simulation/backprop），差异主要在 **节点/动作定义** 与 **reward/value 的来源**：

- **Teaching AI Agents to Explore with Reflective-MCTS and Self-Learning** (R-MCTS / ExACT) (2024)  
  - test-time 用反思+多智能体辩论评估节点价值，提升探索能力。
  - https://arxiv.org/abs/2410.02052  
  - Project: https://agent-e3.github.io/ExACT/

- **Ensembling Large Language Models with Process Reward Models (LE-MCTS)** (2024/2025)  
  - 用 MCTS 做“过程级集成”，动作是选择哪个 LLM 来生成下一步推理，并用 PRM 评估过程质量。
  - https://arxiv.org/abs/2412.15797  
  - PDF(NAACL): https://aclanthology.org/2025.naacl-long.515.pdf

- **MC-NEST: Enhancing Mathematical Reasoning in LLMs with Monte Carlo Self-Refine Tree** (2024)  
  - 把 self-refine/self-eval 融入 MCTS，适合数学推理。
  - https://arxiv.org/abs/2411.15645

- **REKG-MCTS: Reinforcing LLM Reasoning on Knowledge Graphs** (2025)  
  - training-free：用 MCTS 在知识图谱路径空间做动态探索。
  - https://aclanthology.org/2025.findings-acl.484.pdf

- **GroundedPRM: Tree-Guided and Fidelity-Aware Process Reward Modeling** (2025)  
  - 用 MCTS 构造结构化 reasoning trees，并训练/改进 PRM（跟 Think&Cite 的“进度/过程信号”精神接近）。
  - https://yaoz720.github.io/GroundedPRM/

- **Enhancing Reasoning through Process Supervision with Monte Carlo Tree Search** (2025)  
  - 用 MCTS 生成过程监督数据（process supervision），再用于训练模型（介于 training-free 与 training-based 之间）。
  - https://openreview.net/forum?id=OupEEi1341

- **Boosting Policy and Process Reward Models with Monte Carlo Tree Search** (2025)  
  - 用 MCTS 同时改进 policy 与 PRM/ORM，属于“搜索辅助训练”路线。
  - https://openreview.net/forum?id=4KfUpBoqvW

- **MONTE CARLO PLANNING WITH LARGE LANGUAGE MODELS** (ICLR 2025)  
  - 从 planning 视角系统化讨论 LLM action space 上的 Monte Carlo planning（含树搜索/价值估计）。
  - PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/837ff662214b04e7ea8c43f095fe0dd7-Paper-Conference.pdf

- **Monte Carlo Tree Search for Graph Reasoning in Large Language Model Agents** (ACM 2025)  
  - 面向图推理/agent 的 MCTS 框架（偏应用）。
  - https://dl.acm.org/doi/10.1145/3746252.3760854


### 4.4 与树搜索“近邻但不是 MCTS”的重要工作（可作为 related work）
> 虽然不一定是 MCTS，但常与 MCTS 并列讨论：ToT/GoT/搜索式推理、ReAct、Reflexion、自一致性、best-of-N
- Tree of Thoughts (ToT) / Graph of Thoughts (GoT) 等（建议自行补充你最认可的版本与实现）
- Self-Consistency（多采样投票）
- Best-of-N / reranking（用 reward model 或 self-eval 做重排）

---

## 5. 方法谱系（Taxonomy）——建议你在 README 顶层用这张“分类地图”
1) **Training-based (post-training)**
   - RLVR / GRPO / PPO：优化可验证奖励，提高单次成功率，但可能 mode-seeking
   - Distillation：从更强 teacher 注入新模式（常被认为更可能“扩边界”）
2) **Training-free (test-time compute)**
   - Pure sampling：best-of-N, self-consistency
   - MCMC / power sampling：对序列分布做 sharpened sampling
   - Search / planning：MCTS、tree search、heuristic reranking（可插 PRM）
3) **Search-assisted training**
   - 用 MCTS 生成过程监督数据训练 PRM / policy（如 process supervision 方向）

---

## 6. 实用复现与对照实验建议（Repo-friendly）

### 6.1 最小对照三角形
在同一个 base checkpoint 上做：
- Base + temperature / best-of-N
- Base + power sampling（MH/MCMC）
- RLVR checkpoint（如 GRPO 后模型）

比较：
- pass@1 / pass@k 曲线
- diversity（unique solutions / n-gram / embedding distance）
- token-cost（总生成 token 数 vs 性能）

### 6.2 对 Think&Cite / MCTS 系列
对 attributed generation（ASQA/ELI5/QAMPARI 等）做：
- 直接生成 vs tree search
- 不同 reward：outcome-only vs process vs progress（Think&Cite 的亮点）
- 搜索预算 ablation：branching factor、depth、rollouts、UCB 常数

---

## 7. 待补充（TODO）
- [ ] 将每篇论文的 BibTeX 补齐
- [ ] 为每一类方法收集 1–2 个公开实现（HF space / github）
- [ ] 加入你自己复现实验的脚本与结果表（建议用 `results/` + `notebooks/`）

---

## 8. 引用建议（你可以复制到论文/报告中用的表述模板）
- “Recent evidence suggests RLVR improvements often reflect *sampling efficiency gains* rather than *new reasoning patterns*, motivating training-free test-time methods that elicit latent base capabilities via sampling/search.”
- “Tree-search-based attributed generation frameworks (e.g., SG-MCTS + progress reward) reframe generation as multi-step planning, enabling controllable exploration of evidence-grounded trajectories.”

