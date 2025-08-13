# TensorTrade 奖励函数数学模型与技术参考

## 📖 文档概述

本文档提供TensorTrade系统中17个奖励函数的完整数学模型、技术特性分析和应用指导。每个奖励函数都包含数学公式推导、优缺点分析、核心特性说明和适用场景建议。

---

## 🏛️ 基础层奖励函数

### 1. RiskAdjustedReward - 风险调整奖励函数

#### 基本概述
基于现代投资组合理论的夏普比率(Sharpe Ratio)奖励函数，通过风险调整收益来评估投资组合性能，是量化交易中最经典和广泛使用的风险度量指标。

#### 数学公式模型
```
夏普比率计算:
SR(t) = [R_p(t) - R_f] / σ_p(t)

其中:
- R_p(t) = Σ(r_i) / n, i ∈ [t-w, t]  (投资组合平均收益率)
- R_f = risk_free_rate                 (无风险利率)
- σ_p(t) = √[Σ(r_i - R_p(t))² / (n-1)] (收益率标准差)
- w = window_size                      (滚动窗口大小)
- r_i = (V_i - V_{i-1}) / V_{i-1}     (第i期收益率)

奖励函数:
Reward(t) = SR(t) × scale_factor
```

#### 核心特性
- **风险归一化**: 将收益除以波动性，实现风险调整
- **滚动窗口计算**: 使用固定窗口避免历史偏差
- **无风险基准**: 考虑时间价值和机会成本
- **标准化输出**: 便于不同策略间比较

#### 优点
- ✅ **理论成熟**: 基于诺贝尔经济学奖理论，久经验证
- ✅ **风险意识**: 自动平衡收益与风险
- ✅ **计算高效**: 简单的数学运算，适合实时计算
- ✅ **直观解释**: 结果易于理解和解释
- ✅ **广泛认可**: 金融行业标准指标

#### 缺点
- ❌ **正态分布假设**: 假设收益率服从正态分布，现实中常不成立
- ❌ **对称风险**: 将上行和下行波动视为相同风险
- ❌ **短期噪声**: 短期窗口容易受噪声影响
- ❌ **静态权重**: 无法适应市场状态变化

#### 适用场景
- 🎯 **稳健长期投资**: 追求风险调整收益的保守策略
- 🎯 **策略基准比较**: 不同策略性能标准化比较
- 🎯 **监管合规**: 满足金融监管的风险控制要求
- 🎯 **初学者友好**: 作为入门级奖励函数

#### 参数配置
```python
reward_fn = create_reward_function(
    'risk_adjusted',
    risk_free_rate=0.02,      # 年化无风险利率
    window_size=50,           # 滚动窗口期数
    initial_balance=10000.0   # 初始资金规模
)
```

---

### 2. SimpleReturnReward - 简单收益奖励函数

#### 基本概述
最直接的收益优化目标函数，专注于最大化投资组合的绝对收益率，不考虑风险因素。适合快速验证和高收益导向的交易策略。

#### 数学公式模型
```
单期收益率:
r(t) = (V(t) - V(t-1)) / V(t-1)

累积收益率:
R_total(t) = (V(t) - V(0)) / V(0) = Π(1 + r_i) - 1

奖励函数:
Reward(t) = w_step × r(t) + w_total × R_total(t)

其中:
- V(t): t时刻投资组合净值
- w_step: 单步收益权重
- w_total: 累积收益权重
- 通常设置: w_step + w_total = 1.0
```

#### 核心特性
- **纯收益导向**: 完全专注于收益最大化
- **双重激励**: 同时考虑短期和长期收益
- **计算简单**: 最低的计算复杂度
- **实时响应**: 对市场变化反应迅速

#### 优点
- ✅ **目标明确**: 直接优化收益，目标清晰
- ✅ **计算极简**: 最低的计算开销
- ✅ **快速验证**: 适合策略原型快速测试
- ✅ **高收益潜力**: 在上涨市场中表现优异
- ✅ **实现简单**: 容易理解和实施

#### 缺点
- ❌ **风险盲目**: 完全忽略风险因素
- ❌ **大幅回撤**: 可能产生巨大亏损
- ❌ **过度激进**: 倾向于高风险策略
- ❌ **市场敏感**: 在震荡市场中表现不稳定
- ❌ **不可持续**: 长期风险积累问题

#### 适用场景
- 🎯 **策略原型测试**: 快速验证策略有效性
- 🎯 **牛市环境**: 在明确上涨趋势中使用
- 🎯 **高频交易**: 频繁小额交易策略
- 🎯 **短期投机**: 短期高收益导向策略
- 🎯 **教学演示**: 作为最简单的示例

#### 参数配置
```python
reward_fn = create_reward_function(
    'simple_return',
    step_weight=0.8,          # 单步收益权重
    total_weight=0.2,         # 累积收益权重
    initial_balance=10000.0   # 初始资金
)
```

---

## 📈 进阶层奖励函数

### 3. DynamicSortinoReward - 动态索提诺比率奖励

#### 基本概述
改进的风险调整收益指标，只考虑下行风险(downside risk)而非总波动性，更符合投资者的实际风险感知。采用动态时间尺度适应不同市场环境。

#### 数学公式模型
```
下行偏差计算:
DD(t) = √[Σ min(r_i - τ, 0)² / n]

动态索提诺比率:
Sortino(t) = [R_p(t) - τ] / DD(t)

动态时间尺度:
w(t) = w_min + (w_max - w_min) × sigmoid(volatility_factor)
volatility_factor = |σ_current - σ_target| / σ_target

奖励函数:
Reward(t) = Sortino(t) × adaptation_weight(t)

其中:
- τ = target_return (目标收益率阈值)
- DD(t): 下行偏差 (只考虑负收益的标准差)
- w_min, w_max: 最小/最大时间窗口
- adaptation_weight: 自适应权重因子
```

#### 核心特性
- **下行风险聚焦**: 只惩罚负面波动，忽略正面波动
- **动态时间尺度**: 根据市场波动自动调整计算窗口
- **目标收益导向**: 以设定的目标收益为基准
- **自适应机制**: 在不同市场状态下自动优化

#### 优点
- ✅ **符合直觉**: 只关注下行风险，更贴近投资者心理
- ✅ **动态适应**: 能够适应不同的市场环境
- ✅ **目标导向**: 可以设定具体的收益目标
- ✅ **波动友好**: 不惩罚有利的价格波动
- ✅ **熊市保护**: 在下跌市场中提供更好的保护

#### 缺点
- ❌ **计算复杂**: 比基础夏普比率计算更复杂
- ❌ **参数敏感**: 对目标收益率设定较为敏感
- ❌ **历史依赖**: 需要较长的历史数据进行校准
- ❌ **牛市保守**: 在强势上涨中可能过于保守

#### 适用场景
- 🎯 **下行保护策略**: 重视资本保护的投资策略
- 🎯 **养老基金**: 不能承受大幅亏损的长期投资
- 🎯 **波动市场**: 市场不确定性较高的环境
- 🎯 **风险厌恶**: 投资者风险承受能力较低

#### 参数配置
```python
reward_fn = create_reward_function(
    'dynamic_sortino',
    target_return=0.001,      # 日目标收益率
    time_scale_min=10,        # 最小时间窗口
    time_scale_max=100,       # 最大时间窗口
    adaptation_rate=0.05      # 适应速率
)
```

---

## 🤖 AI驱动层奖励函数

### 4. UncertaintyAwareReward - 不确定性感知奖励

#### 基本概述
基于贝叶斯深度学习的高级风险管理系统，通过量化模型预测的不确定性来增强风险感知能力。区分认知不确定性(epistemic)和任意不确定性(aleatoric)，提供精确的风险评估。

#### 数学公式模型
```
蒙特卡洛Dropout不确定性估计:
μ_pred(t), σ²_pred(t) = MCD_forward(x(t), T_samples)

认知不确定性 (Epistemic):
U_epistemic(t) = Var[E[y|x,θ]] ≈ (1/T)Σ[μ_i - μ_mean]²

任意不确定性 (Aleatoric):
U_aleatoric(t) = E[Var[y|x,θ]] ≈ (1/T)Σσ²_i

总不确定性:
U_total(t) = U_epistemic(t) + U_aleatoric(t)

CVaR风险调整:
CVaR_α(t) = E[R(t) | R(t) ≤ VaR_α(t)]

不确定性感知奖励:
Reward(t) = (1-λ_u) × R_base(t) - λ_u × [λ_e × U_epistemic(t) + λ_a × U_aleatoric(t)]
           + β × max(0, confidence(t) - τ_conf) - γ × max(0, CVaR_α(t))

其中:
- T_samples: 蒙特卡洛采样次数
- λ_u: 不确定性总权重
- λ_e, λ_a: 认知/任意不确定性权重
- τ_conf: 置信度阈值
- α: CVaR风险水平 (通常0.05)
```

#### 核心特性
- **双重不确定性建模**: 分离模型不确定性和数据不确定性
- **蒙特卡洛Dropout**: 通过随机dropout估计预测不确定性
- **CVaR风险优化**: 条件风险价值管理极端损失
- **置信度驱动**: 高置信度预测获得奖励加成
- **自适应风险敏感**: 根据市场不确定性动态调整

#### 优点
- ✅ **科学严谨**: 基于贝叶斯统计理论，数学基础扎实
- ✅ **精确风险量化**: 提供定量的不确定性度量
- ✅ **极端风险控制**: CVaR有效管理尾部风险
- ✅ **模型透明**: 明确区分不同类型的不确定性
- ✅ **自适应能力**: 在高不确定性环境下自动保守

#### 缺点
- ❌ **计算密集**: 蒙特卡洛采样增加计算成本
- ❌ **参数复杂**: 多个超参数需要仔细调优
- ❌ **数据要求**: 需要大量数据训练可靠的不确定性模型
- ❌ **实时性挑战**: 在线推理速度可能较慢

#### 适用场景
- 🎯 **专业风险管理**: 机构投资者的风险控制
- 🎯 **高波动环境**: 加密货币等高不确定性市场
- 🎯 **监管要求**: 需要量化风险报告的场景
- 🎯 **算法透明性**: 需要解释AI决策过程
- 🎯 **尾部风险敏感**: 极度厌恶极端损失的策略

#### 参数配置
```python
reward_fn = create_reward_function(
    'uncertainty_aware',
    uncertainty_weight=0.3,       # 不确定性惩罚权重
    epistemic_weight=0.6,         # 认知不确定性权重
    aleatoric_weight=0.4,         # 任意不确定性权重
    confidence_threshold=0.8,     # 置信度阈值
    cvar_alpha=0.05,             # CVaR风险水平
    mc_samples=50                # 蒙特卡洛采样数
)
```

---

### 5. CuriosityDrivenReward - 好奇心驱动奖励

#### 基本概述
基于内在动机理论的探索性奖励函数，通过预测误差驱动策略探索新的有利可图的交易模式。集成DiNAT-Vision Transformer增强特征提取，实现层次化强化学习。

#### 数学公式模型
```
前向模型预测:
ẑ(t+1) = f_forward(s(t), a(t); θ_f)

预测误差 (内在奖励):
r_intrinsic(t) = ||z(t+1) - ẑ(t+1)||²

DiNAT特征提取:
F_dinat(t) = DiNAT_Transformer(price_window(t))

层次化子目标:
G_high(t) = HRL_Manager(F_dinat(t), horizon_H)
r_subgoal(t) = cosine_similarity(achieved_goal(t), G_high(t))

学习进度监控:
LP(t) = |prediction_error(t-w) - prediction_error(t)|
learning_bonus(t) = α_lp × max(0, LP(t) - τ_lp)

综合好奇心奖励:
Reward(t) = (1-α_c) × r_extrinsic(t) + α_c × [
    β_pred × r_intrinsic(t) + 
    β_subgoal × r_subgoal(t) + 
    β_lp × learning_bonus(t)
]

其中:
- f_forward: 前向预测模型
- α_c: 好奇心权重
- β_pred, β_subgoal, β_lp: 各组件权重
- τ_lp: 学习进度阈值
```

#### 核心特性
- **内在动机驱动**: 通过预测误差激励探索未知模式
- **DiNAT-Vision增强**: 先进的视觉Transformer特征提取
- **层次化强化学习**: 多层次目标管理和子任务分解
- **学习进度监控**: 跟踪和奖励学习改进过程
- **探索-利用平衡**: 动态平衡探索新策略和利用已知策略

#### 优点
- ✅ **策略创新**: 能够发现传统方法忽略的交易机会
- ✅ **适应性强**: 在复杂多变的市场中持续学习
- ✅ **特征丰富**: DiNAT提供高质量的市场特征表示
- ✅ **层次化决策**: 支持复杂的多层次交易策略
- ✅ **持续改进**: 内置学习进度监控机制

#### 缺点
- ❌ **训练复杂**: 需要大量计算资源和训练时间
- ❌ **探索成本**: 初期可能产生较大的探索损失
- ❌ **模型复杂**: 多个子模块增加系统复杂性
- ❌ **调参困难**: 众多超参数的协调优化具有挑战性

#### 适用场景
- 🎯 **策略研发**: 寻找新的交易信号和模式
- 🎯 **复杂市场**: 特征丰富、模式复杂的市场环境
- 🎯 **长期优化**: 允许长期探索成本的策略开发
- 🎯 **AI研究**: 前沿强化学习算法的应用研究
- 🎯 **多资产策略**: 跨资产类别的复杂策略

#### 参数配置
```python
reward_fn = create_reward_function(
    'curiosity_driven',
    curiosity_weight=0.4,         # 好奇心总权重
    alpha_curiosity=0.6,          # 内在动机权重
    beta_stability=0.4,           # 稳定性权重
    enable_hierarchical_rl=True,  # 启用层次化RL
    prediction_horizon=10,        # 预测时间窗口
    learning_progress_window=50   # 学习进度窗口
)
```

---

### 6. LLMGuidedReward - 大语言模型引导奖励

#### 基本概述
基于EUREKA算法和Constitutional AI框架的革命性奖励函数设计系统。通过自然语言描述自动生成和优化奖励函数，实现AI辅助的智能奖励设计。

#### 数学公式模型
```
自然语言解析:
P_spec = NLP_Parser(natural_language_input)
P_structured = {objectives: [...], constraints: [...], preferences: [...]}

代码生成 (EUREKA风格):
R_code(t) = LLM_Generator(P_structured, context_trading, best_practices)

安全性验证 (Constitutional AI):
Safety_Score = Constitutional_Checker(R_code, safety_principles)
R_safe(t) = R_code(t) if Safety_Score > τ_safety else R_fallback(t)

迭代优化:
performance_history = {R_i: score_i for i in iterations}
R_optimized(t) = Evolutionary_Optimizer(performance_history, mutation_rate)

自我评估:
explanation(t) = LLM_Explainer(R_optimized(t), context(t))
confidence(t) = Self_Consistency_Check(R_optimized(t), explanation(t))

最终奖励:
Reward(t) = R_optimized(t) × confidence(t) + 
           λ_safety × Safety_Bonus(t) + 
           λ_explain × Explainability_Score(t)

其中:
- τ_safety: 安全阈值
- λ_safety, λ_explain: 安全性和可解释性权重
- mutation_rate: 进化优化变异率
```

#### 核心特性
- **自然语言接口**: 用普通话描述交易目标和约束
- **自动代码生成**: AI自动生成对应的奖励函数代码
- **Constitutional AI安全**: 内置伦理和安全检查机制
- **迭代自我优化**: 基于性能反馈持续改进
- **可解释性**: 提供详细的决策解释和推理过程

#### 优点
- ✅ **用户友好**: 无需编程知识，自然语言描述即可
- ✅ **AI驱动创新**: 能生成人类未曾考虑的奖励函数设计
- ✅ **安全可靠**: Constitutional AI确保生成函数的安全性
- ✅ **自我改进**: 基于反馈自动优化性能
- ✅ **完全可解释**: 每个决策都有详细解释

#### 缺点
- ❌ **依赖LLM**: 需要强大的语言模型支持
- ❌ **计算昂贵**: LLM推理成本较高
- ❌ **语言理解限制**: 可能误解复杂的自然语言描述
- ❌ **生成质量不稳定**: AI生成代码质量可能波动

#### 适用场景
- 🎯 **非技术用户**: 没有编程背景的投资者
- 🎯 **快速原型**: 快速测试新的交易想法
- 🎯 **创新研究**: 探索全新的奖励函数设计
- 🎯 **教育培训**: 帮助理解不同奖励函数的作用
- 🎯 **个性化策略**: 根据个人偏好定制奖励函数

#### 参数配置
```python
reward_fn = create_reward_function(
    'llm_guided',
    natural_language_spec="最大化夏普比率，同时保持最大回撤低于10%，偏好稳定增长",
    enable_iterative_improvement=True,    # 启用迭代优化
    safety_level='high',                  # 安全级别
    explanation_detail='comprehensive'    # 解释详细程度
)
```

---

### 7. MetaLearningReward - 元学习自适应奖励

#### 基本概述
基于Model-Agnostic Meta-Learning (MAML)的最先进自适应奖励系统。能够快速适应新的市场环境和交易任务，结合自我奖励机制和记忆增强学习，实现真正的智能自适应。

#### 数学公式模型
```
MAML元学习框架:
内层更新 (任务特定):
θ_i' = θ - α∇_θ L_{τ_i}(f_θ)

外层更新 (元学习):
θ = θ - β∇_θ Σ_i L_{τ_i}(f_{θ_i'})

任务适应算法:
for step in adaptation_steps:
    L_task, ∇L_task = compute_task_loss_gradients(task_config, data)
    θ_adapted = θ_adapted - α × ∇L_task

元梯度计算:
∇_meta = Σ_{tasks} w_task × ∇_θ L_task(θ_adapted)
θ_meta = θ_meta - β × ∇_meta

自我奖励网络:
reward_self, confidence = SelfRewardingNet(state_features)
reward_self = tanh(W_2 × tanh(W_1 × features + b_1) + b_2)

记忆增强检索:
similar_tasks = Memory.retrieve(current_task, similarity_threshold)
prior_knowledge = weighted_average(similar_tasks, similarity_weights)

最终自适应奖励:
Reward(t) = (1-λ_self) × R_traditional(t) + λ_self × reward_self(t) +
           λ_meta × meta_adaptation_bonus(t) + 
           λ_memory × memory_enhancement(t)

其中:
- α: 内层学习率 (任务适应)
- β: 外层学习率 (元学习)
- λ_self, λ_meta, λ_memory: 各组件权重
```

#### 核心特性
- **快速任务适应**: 几步梯度更新即可适应新环境
- **元梯度优化**: 学习如何更好地学习
- **自我奖励机制**: 内置自我评估和改进能力
- **记忆增强**: 利用历史经验加速新任务学习
- **任务变化检测**: 自动识别市场状态转换

#### 优点
- ✅ **超快适应**: 在新环境中快速达到良好性能
- ✅ **通用性强**: 单一模型适应多种市场环境
- ✅ **持续学习**: 不断积累和利用历史经验
- ✅ **自我改进**: 内置自我优化机制
- ✅ **理论先进**: 基于最新的元学习理论

#### 缺点
- ❌ **复杂度极高**: 实现和调优具有很大挑战性
- ❌ **计算密集**: 需要大量计算资源
- ❌ **数据饥渴**: 需要多样化的任务数据进行元训练
- ❌ **黑盒特性**: 决策过程较难解释

#### 适用场景
- 🎯 **多市场交易**: 需要在不同市场间切换的策略  
- 🎯 **快速适应**: 市场环境频繁变化的场景
- 🎯 **资源充足**: 有充足计算资源的机构
- 🎯 **前沿研究**: 元学习在金融中的应用研究
- 🎯 **AI系统**: 构建自主学习的交易AI系统

#### 参数配置
```python
reward_fn = create_reward_function(
    'meta_learning_reward',
    alpha=0.01,                           # 内层学习率
    beta=0.001,                           # 外层学习率  
    adaptation_steps=5,                   # 适应步数
    enable_self_rewarding=True,           # 启用自我奖励
    enable_memory_augmentation=True,      # 启用记忆增强
    task_detection_window=50              # 任务检测窗口
)
```

---

## 📊 综合性能比较

### 复杂度与性能矩阵

| 奖励函数 | 数学复杂度 | 计算复杂度 | 适应性 | 稳定性 | 收益潜力 | 风险控制 |
|---------|------------|------------|--------|--------|----------|----------|
| SimpleReturn | ⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| RiskAdjusted | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| DynamicSortino | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| UncertaintyAware | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CuriosityDriven | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| LLMGuided | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| MetaLearning | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 使用决策树

```
选择奖励函数决策流程:

1. 是否为新手？
   └─ 是 → SimpleReturnReward (快速上手)
   └─ 否 → 继续

2. 是否需要风险控制？
   └─ 是 → 继续
   └─ 否 → SimpleReturnReward
   
3. 计算资源是否充足？
   └─ 否 → RiskAdjustedReward (经典选择)
   └─ 是 → 继续
   
4. 是否需要探索新策略？
   └─ 是 → CuriosityDrivenReward
   └─ 否 → 继续
   
5. 是否需要极致风险控制？
   └─ 是 → UncertaintyAwareReward
   └─ 否 → 继续
   
6. 是否偏好自然语言接口？
   └─ 是 → LLMGuidedReward
   └─ 否 → 继续
   
7. 是否需要多环境适应？
   └─ 是 → MetaLearningReward
   └─ 否 → DynamicSortinoReward (均衡选择)
```

## 🎯 最佳实践建议

### 参数调优策略
1. **从默认参数开始**: 使用推荐的默认参数作为起点
2. **单参数调整**: 每次只调整一个参数，观察效果
3. **网格搜索**: 对关键参数进行系统性搜索
4. **交叉验证**: 使用多个时间段验证参数稳定性

### 性能监控指标
```python
# 监控关键指标
metrics = {
    'sharpe_ratio': calculate_sharpe(returns),
    'max_drawdown': calculate_max_drawdown(portfolio_values),
    'win_rate': calculate_win_rate(trades),
    'profit_factor': calculate_profit_factor(trades),
    'calmar_ratio': annual_return / max_drawdown
}
```

### 风险管理原则
- 始终设置止损机制
- 监控实时风险暴露
- 定期评估奖励函数表现
- 保持多样化投资组合

---

**技术支持**: 基于2024-2025年最新强化学习和人工智能研究成果  
**文档版本**: v2.0.0  
**最后更新**: 2025年1月