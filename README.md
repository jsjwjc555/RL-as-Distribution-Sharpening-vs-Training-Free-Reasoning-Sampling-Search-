# Survey: RL as Distribution Sharpening vs Training-Free Reasoning (Sampling/Search)

> 主题：研究 **强化学习后训练（RL/RLVR/GRPO 类）到底改变了 base model 什么**，以及 **training-free 的推理期方法（采样/搜索/MCTS）** 是否能与 RL 相当或互补。  
- **RLVR 可能更像“让模型更容易抽到它本来就会的正确推理路径”**，而不一定扩展“可解题集合”。（典型证据：大 k 视角下 base 可能追平甚至超过 RLVR；RL 输出在 base 下困惑度/似然偏高）
- **Training-free 的采样/搜索** 可以在某些任务上达到接近 RL 的单次表现，并在多样性/大 k 方面更稳，但通常更耗计算。
- **MCTS + 过程/进度奖励** 适合“多步结构化生成/检索/引用/证据对齐”的任务（如 attributed generation），可解释性与可控性更强。
---

## 1. （Anchor Papers）2_3组会汇报

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

## 2. 相似/相关工作清单（gpt自动调研,欢迎补充）
> 关键词：entropy collapse / diversity collapse / divergence choice / RLVR dynamics / sampling, rejection, MCMC, inference-time optimization, test-time compute
- **The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models** (2025)  
  - 关注 RL 训练中的熵变化与探索-利用权衡，解释为什么“更会答对”可能伴随多样性下降。
  - https://arxiv.org/abs/2505.22617
 
- **Training-Free Group Relative Policy Optimization** (2025)  
  - 把“相对优势/组内比较”的思想搬到推理期（不更新参数），作为 training-free 的类 GRPO 路线。
  - https://arxiv.org/abs/2510.08191

- **A Minimalist Approach to LLM Reasoning: from Rejection Sampling ...** (2025)  
  - 讨论“筛样/重加权”在多大程度上能替代 RL 的收益（非常适合写 ‘RL gains may be data selection’ 的讨论段）。
  - https://arxiv.org/html/2504.11343v1

- **MC-NEST: Enhancing Mathematical Reasoning in LLMs with Monte Carlo Self-Refine Tree** (2024)  
  - 把 self-refine/self-eval 融入 MCTS，适合数学推理。
  - https://arxiv.org/abs/2411.15645

> 关键词： iclr 2026

- **The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models** (ICLR 2026 submission)
   - 直接聚焦“RLVR 可能收缩 reasoning boundary”的动态机制分析（贴合：RL 改变了 base 的什么）。
   - https://openreview.net/forum?id=sau3jEEyvR
 
- **DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search** (ICLR 2026 submission)
   - 用 MCTS/搜索提升可验证奖励场景下的探索与解空间覆盖（可理解为“RL 可能偏锐化，search 负责扩覆盖”）。
   - https://openreview.net/forum?id=Kx0G6v2c2S

- **Efficient Test-time Scaling via Iterative Deepening (ID-sampling)** (ICLR 2026 submission)
   - 通过 iterative deepening 提升 test-time scaling 的采样效率，更快“抽到正确轨迹”。
   - https://openreview.net/forum?id=72b05slV0s

- **Recursive Self-Aggregation Unlocks Deep Thinking in LLMs** (ICLR 2026 submission)
   - 融合并行多样本 + 顺序自我修订/聚合，作为 training-free 的深度推理路线。
   - https://openreview.net/forum?id=J7upvGcP9h

## 3.相反观点：Prolonged RL 可能真的“扩边界”，关键在训练阶段/动态

- **ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models**
   - 强调 prolonged RL + KL 控制 + reference reset + 多任务 等机制，声称能得到“base 大量采样也到不了”的新策略/新可解集。
   - https://arxiv.org/pdf/2505.24864

- **The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both? A Two-Stage Dynamic View**
   - 给了一个很有用的“调和框架”：把 RLVR 过程分成 exploitation stage（早期更像锐化、甚至导致收缩）和 exploration stage（训练足够久才可能真正扩展）。
   - https://arxiv.org/pdf/2510.04028
 
- **FROM f (x) AND g(x) TO f (g(x)): LLMS LEARN NEWSKILLS IN RL BY COMPOSING OLD ONES**
   - https://openreview.net/pdf?id=jt7oCtYqHE
