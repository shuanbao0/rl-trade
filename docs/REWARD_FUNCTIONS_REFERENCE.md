# TensorTrade 奖励函数完整参考手册

## 📖 概述

TensorTrade集成了**17个先进的奖励函数**，从基础的风险调整到最前沿的AI驱动设计，为量化交易提供完整的奖励函数生态系统。这些奖励函数基于2024-2025年最新强化学习和人工智能研究成果，代表了奖励函数设计的最高水平。

## 🏗️ 系统架构

```
TensorTrade奖励函数生态系统
├── 🏛️ 基础层 (Foundation Layer) - 4个函数
│   ├── RiskAdjustedReward      夏普比率风险调整
│   ├── SimpleReturnReward      简单收益率优化
│   ├── ProfitLossReward        盈亏比优化
│   └── DiversifiedReward       多指标综合评估
├── 📈 进阶层 (Advanced Layer) - 4个函数  
│   ├── LogSharpeReward         对数差分夏普比率
│   ├── ReturnDrawdownReward    收益回撤平衡
│   ├── DynamicSortinoReward    动态索提诺比率
│   └── RegimeAwareReward       市场状态感知
└── 🤖 AI驱动层 (AI-Driven Layer) - 9个函数
    ├── ExpertCommitteeReward   多目标专家委员会
    ├── UncertaintyAwareReward  贝叶斯不确定性量化
    ├── CuriosityDrivenReward   好奇心驱动探索
    ├── SelfRewardingReward     自我评判优化
    ├── CausalReward            因果推理奖励
    ├── LLMGuidedReward         大语言模型引导
    ├── CurriculumReward        课程学习进阶
    ├── FederatedReward         联邦学习协作
    └── MetaLearningReward      元学习自适应
```

## 🚀 快速开始

### 基础使用方式
```python
from src.environment.rewards import create_reward_function

# 创建奖励函数实例
reward_fn = create_reward_function('risk_adjusted', risk_free_rate=0.02)

# 计算奖励
reward = reward_fn.get_reward(portfolio)
```

### 命令行使用方式
```bash
# 基础奖励函数
python train_model.py --reward-type risk_adjusted --risk-free-rate 0.02

# AI驱动奖励函数
python train_model.py --reward-type uncertainty_aware --uncertainty-weight 0.3
```

---

## 📚 奖励函数详细说明

### 🏛️ 基础层奖励函数

#### 1. RiskAdjustedReward - 风险调整奖励
**设计理念**: 基于夏普比率的经典风险调整收益评估
**适用场景**: 稳健的长期投资策略，风险控制优先
**技术特性**: 
- 基于收益标准差的风险量化
- 支持滚动窗口计算
- 风险免费利率调整

**参数配置**:
```python
reward_fn = create_reward_function(
    'risk_adjusted',
    risk_free_rate=0.02,      # 无风险利率
    window_size=50,           # 滚动窗口大小
    initial_balance=10000.0   # 初始资金
)
```

**使用示例**:
```bash
# 标准配置
python train_model.py --reward-type risk_adjusted

# 自定义参数
python train_model.py --reward-type sharpe --risk-free-rate 0.025 --window-size 100
```

**别名支持**: `risk_adjusted`, `default`, `sharpe`

---

#### 2. SimpleReturnReward - 简单收益奖励
**设计理念**: 直接优化投资组合收益率，简单高效
**适用场景**: 快速验证、高频交易、纯收益导向策略
**技术特性**:
- 步骤收益和总收益的加权组合
- 低计算复杂度
- 适合快速迭代测试

**参数配置**:
```python
reward_fn = create_reward_function(
    'simple_return',
    step_weight=1.0,          # 步骤收益权重
    total_weight=0.1,         # 总收益权重
    initial_balance=10000.0   # 初始资金
)
```

**使用示例**:
```bash
python train_model.py --reward-type simple_return --step-weight 0.8 --total-weight 0.2
```

**别名支持**: `simple_return`, `basic`, `simple`

---

#### 3. ProfitLossReward - 盈亏比奖励
**设计理念**: 基于盈亏比的风险收益评估，强调盈利质量
**适用场景**: 短线交易、止盈止损策略优化
**技术特性**:
- 盈利加成和亏损惩罚机制
- 支持动态调整盈亏比例
- 适合波动性市场

**参数配置**:
```python
reward_fn = create_reward_function(
    'profit_loss',
    profit_bonus=2.0,         # 盈利加成系数
    loss_penalty=1.5,         # 亏损惩罚系数
    initial_balance=10000.0   # 初始资金
)
```

**使用示例**:
```bash
python train_model.py --reward-type pnl --profit-bonus 1.8 --loss-penalty 1.2
```

**别名支持**: `profit_loss`, `pnl`

---

#### 4. DiversifiedReward - 多指标综合奖励
**设计理念**: 综合考虑收益、风险、稳定性等多个维度
**适用场景**: 机构级投资策略、长期资产配置
**技术特性**:
- 五维度综合评估体系
- 自定义权重配置
- 支持动态权重调整

**参数配置**:
```python
weights = {
    'return': 0.4,        # 收益权重
    'risk': 0.2,          # 风险权重  
    'stability': 0.15,    # 稳定性权重
    'efficiency': 0.15,   # 效率权重
    'drawdown': 0.1       # 回撤权重
}
reward_fn = create_reward_function('diversified', weights=weights)
```

**使用示例**:
```bash
python train_model.py --reward-type diversified --return-weight 0.5 --risk-weight 0.3
```

**别名支持**: `diversified`, `comprehensive`, `multi`

---

### 📈 进阶层奖励函数

#### 5. LogSharpeReward - 对数夏普比率奖励
**设计理念**: 基于对数收益的差分夏普比率(DSR)，处理收益序列的时间依赖性
**适用场景**: 高频交易、连续复利策略
**研究基础**: 2024年差分夏普比率理论最新进展
**技术特性**:
- 对数收益处理避免复利计算误差
- 差分算法提升计算效率
- 动态学习率调整

**参数配置**:
```python
reward_fn = create_reward_function(
    'log_sharpe',
    learning_rate=0.01,       # DSR学习率
    window_size=50,           # 计算窗口
    risk_free_rate=0.02       # 无风险利率  
)
```

**使用示例**:
```bash
python train_model.py --reward-type dsr --learning-rate 0.015 --window-size 100
```

**别名支持**: `log_sharpe`, `differential_sharpe`, `dsr`, `log_dsr`

---

#### 6. ReturnDrawdownReward - 收益回撤比奖励
**设计理念**: 平衡最终收益与最大回撤的关系，优化收益质量
**适用场景**: 长期持有策略、回撤敏感的投资组合
**技术特性**:
- Calmar比率计算框架
- 动态回撤监控
- 收益回撤平衡优化

**参数配置**:
```python
reward_fn = create_reward_function(
    'return_drawdown',
    return_weight=0.7,        # 收益权重
    drawdown_weight=0.3,      # 回撤权重
    initial_balance=10000.0   # 初始资金
)
```

**使用示例**:
```bash
python train_model.py --reward-type calmar --return-weight 0.8 --drawdown-weight 0.2
```

**别名支持**: `return_drawdown`, `calmar`, `return_dd`, `rdd`, `drawdown`

---

#### 7. DynamicSortinoReward - 动态索提诺比率奖励
**设计理念**: 基于下行风险的索提诺比率，采用动态时间尺度适应市场变化
**适用场景**: 下行风险控制、熊市保护策略
**研究基础**: 2024年时变索提诺比率优化理论
**技术特性**:
- 下行偏差替代标准差
- 动态时间尺度调整
- 目标收益率自适应

**参数配置**:
```python
reward_fn = create_reward_function(
    'dynamic_sortino',
    target_return=0.0,        # 目标收益率
    time_scale_min=10,        # 最小时间尺度
    time_scale_max=100,       # 最大时间尺度
    adaptation_rate=0.05      # 适应速率
)
```

**使用示例**:
```bash
python train_model.py --reward-type sortino --target-return 0.001 --adaptation-rate 0.1
```

**别名支持**: `dynamic_sortino`, `dts`, `adaptive_sortino`, `time_varying_sortino`, `sortino`

---

#### 8. RegimeAwareReward - 市场状态感知奖励
**设计理念**: 识别不同市场状态并自动调整奖励策略
**适用场景**: 全市场周期策略、自适应投资系统
**研究基础**: 2024年状态空间模型和隐马尔可夫模型最新进展
**技术特性**:
- HMM隐马尔可夫模型状态识别
- 多专家系统架构
- 动态权重分配

**参数配置**:
```python
reward_fn = create_reward_function(
    'regime_aware',
    n_regimes=3,              # 市场状态数量
    detection_window=50,      # 状态检测窗口
    transition_penalty=0.1,   # 状态转换惩罚
    expert_weights={          # 专家权重配置
        'bull': {'return': 0.6, 'risk': 0.4},
        'bear': {'return': 0.2, 'risk': 0.8},
        'sideways': {'return': 0.4, 'risk': 0.6}
    }
)
```

**使用示例**:
```bash
python train_model.py --reward-type regime --n-regimes 4 --detection-window 100
```

**别名支持**: `regime_aware`, `adaptive_expert`, `market_state`, `regime`, `state_aware`

---

### 🤖 AI驱动层奖励函数

#### 9. ExpertCommitteeReward - 专家委员会奖励
**设计理念**: 多目标强化学习专家委员会协作决策系统
**适用场景**: 复杂多目标优化、机构级决策支持
**研究基础**: 2024年多目标强化学习(MORL)和Pareto优化理论
**技术特性**:
- 多个专家并行决策
- Pareto前沿优化
- 动态专家权重学习
- 偏好向量指导

**参数配置**:
```python
reward_fn = create_reward_function(
    'expert_committee',
    n_experts=4,              # 专家数量
    objectives=['return', 'risk', 'stability', 'efficiency'],
    preference_vector=[0.4, 0.3, 0.2, 0.1],  # 偏好向量
    diversity_bonus=0.1,      # 多样性奖励
    consensus_threshold=0.7   # 共识阈值
)
```

**使用示例**:
```bash
python train_model.py --reward-type committee --n-experts 5 --diversity-bonus 0.15
```

**别名支持**: `expert_committee`, `committee`, `multi_objective`, `morl`, `experts`, `pareto`

---

#### 10. UncertaintyAwareReward - 不确定性感知奖励
**设计理念**: 基于贝叶斯神经网络的认知和任意不确定性量化
**适用场景**: 高风险环境、不确定性建模、专业风险控制
**研究基础**: 2024年贝叶斯深度学习和不确定性量化最新理论
**技术特性**:
- 蒙特卡洛Dropout不确定性估计
- CVaR条件风险价值优化
- 认知和任意不确定性分离
- 自适应风险敏感度

**参数配置**:
```python
reward_fn = create_reward_function(
    'uncertainty_aware',
    uncertainty_weight=0.3,   # 不确定性权重
    epistemic_weight=0.6,     # 认知不确定性权重
    aleatoric_weight=0.4,     # 任意不确定性权重
    confidence_threshold=0.8, # 置信度阈值
    cvar_alpha=0.05,         # CVaR风险水平
    mc_samples=50            # 蒙特卡洛采样数
)
```

**使用示例**:
```bash
python train_model.py --reward-type uncertainty --uncertainty-weight 0.4 --cvar-alpha 0.1
```

**别名支持**: `uncertainty_aware`, `uncertainty`, `bayesian`, `confidence`, `risk_sensitive`, `cvar`, `epistemic`, `aleatoric`

---

#### 11. CuriosityDrivenReward - 好奇心驱动奖励
**设计理念**: 基于内在动机的好奇心驱动强化学习系统
**适用场景**: 策略探索、新市场发现、创新交易策略开发
**研究基础**: 2024年内在动机RL和DiNAT-Vision Transformer最新研究
**技术特性**:
- 前向模型预测误差驱动
- DiNAT-Vision Transformer增强
- 层次化子目标管理
- 学习进度监控

**参数配置**:
```python
reward_fn = create_reward_function(
    'curiosity_driven',
    curiosity_weight=0.4,     # 好奇心权重
    alpha_curiosity=0.6,      # 好奇心系数
    beta_stability=0.4,       # 稳定性系数
    enable_hierarchical_rl=True,  # 启用层次化RL
    prediction_horizon=10,    # 预测时间窗口
    learning_progress_window=50   # 学习进度窗口
)
```

**使用示例**:
```bash
python train_model.py --reward-type curiosity --alpha-curiosity 0.7 --enable-hierarchical-rl
```

**别名支持**: `curiosity_driven`, `curiosity`, `intrinsic`, `exploration`, `intrinsic_motivation`, `forward_model`, `learning_progress`, `hierarchical_rl`, `dinat`

---

#### 12. SelfRewardingReward - 自我奖励机制
**设计理念**: 基于Meta AI 2024年Self-Rewarding理论的自我评判系统
**适用场景**: 自动化策略优化、无监督学习、AI辅助决策
**研究基础**: Meta AI 2024年Self-Rewarding Language Models理论
**技术特性**:
- LLM-as-a-Judge评估框架
- 迭代自我改进机制
- DPO偏好优化
- 偏差检测和纠正

**参数配置**:
```python
reward_fn = create_reward_function(
    'self_rewarding',
    enable_iterative_improvement=True,  # 启用迭代改进
    self_evaluation_frequency=10,       # 自评频率
    bias_detection_threshold=0.15,      # 偏差检测阈值
    improvement_momentum=0.8,           # 改进动量
    judge_confidence_threshold=0.7      # 评判置信度阈值
)
```

**使用示例**:
```bash
python train_model.py --reward-type self_rewarding --enable-iterative-improvement --bias-detection-threshold 0.1
```

**别名支持**: `self_rewarding`, `self_improving`, `meta_ai`, `llm_judge`, `self_evaluation`, `dpo`, `iterative_improvement`, `bias_detection`, `meta_reward`

---

#### 13. CausalReward - 因果推理奖励
**设计理念**: 基于2024年最新因果推理理论的混淆变量识别系统
**适用场景**: 因果关系发现、策略有效性验证、科学交易研究
**研究基础**: 2024年因果推理、DOVI算法和do-calculus最新进展
**技术特性**:
- 自动因果图构建
- 后门和前门调整算法
- DOVI去混淆价值迭代
- do-calculus因果推理

**参数配置**:
```python
reward_fn = create_reward_function(
    'causal_reward',
    enable_confounding_detection=True,  # 启用混淆检测
    adjustment_method='dovi',           # 调整方法
    causal_discovery_method='pc',       # 因果发现方法
    significance_level=0.05,            # 显著性水平
    backdoor_threshold=0.1              # 后门判断阈值
)
```

**使用示例**:
```bash
python train_model.py --reward-type causal --enable-confounding-detection --adjustment-method backdoor
```

**别名支持**: `causal_reward`, `causal`, `causal_inference`, `confounding`, `backdoor`, `frontdoor`, `dovi`, `do_calculus`, `causal_graph`

---

#### 14. LLMGuidedReward - 大语言模型引导奖励
**设计理念**: 基于EUREKA和Constitutional AI的自然语言奖励函数设计
**适用场景**: 自然语言交易策略描述、AI辅助奖励设计、创新策略开发
**研究基础**: 2024-2025年EUREKA算法和Constitutional AI框架
**技术特性**:
- 自然语言规范解析
- 自动代码生成和验证
- Constitutional AI安全框架
- 迭代优化机制

**参数配置**:
```python
reward_fn = create_reward_function(
    'llm_guided',
    natural_language_spec="最大化夏普比率同时保持回撤低于10%",
    enable_iterative_improvement=True,  # 启用迭代改进
    safety_level='high',                # 安全级别
    explanation_detail='comprehensive'  # 解释详细程度
)
```

**使用示例**:
```bash
python train_model.py --reward-type llm --natural-language-spec "maximize returns while minimizing risk"
```

**别名支持**: `llm_guided`, `llm`, `language_guided`, `natural_language`, `eureka`, `constitutional`, `ai_guided`, `code_generation`, `auto_reward`, `smart_reward`

---

#### 15. CurriculumReward - 课程学习奖励
**设计理念**: 基于2024-2025年课程学习研究的多阶段渐进式训练
**适用场景**: 新手到专家的策略进阶、分阶段策略训练
**研究基础**: 2024-2025年课程学习和渐进式复杂度理论
**技术特性**:
- 四阶段课程设计(初级→中级→高级→专家)
- 自动难度适应
- 成功判别器
- 进度敏感度调整

**参数配置**:
```python
reward_fn = create_reward_function(
    'curriculum_reward',
    enable_auto_progression=True,       # 启用自动进阶
    manual_stage='beginner',            # 手动阶段设置
    progression_sensitivity=0.8,        # 进阶敏感度
    performance_window=50               # 性能评估窗口
)
```

**使用示例**:
```bash
python train_model.py --reward-type curriculum --enable-auto-progression --progression-sensitivity 0.9
```

**别名支持**: `curriculum_reward`, `curriculum`, `curriculum_learning`, `progressive`, `adaptive_difficulty`, `staged_learning`, `multi_stage`, `beginner_to_expert`, `difficulty_progression`

---

#### 16. FederatedReward - 联邦学习奖励
**设计理念**: 基于2024-2025年联邦学习的隐私保护协作奖励优化
**适用场景**: 多机构协作、隐私保护学习、分布式策略优化
**研究基础**: 2024-2025年联邦学习、差分隐私和区块链技术
**技术特性**:
- 差分隐私保护
- 安全聚合算法
- 区块链共识机制
- 声誉评估系统

**参数配置**:
```python
reward_fn = create_reward_function(
    'federated_reward',
    enable_privacy_protection=True,     # 启用隐私保护
    differential_privacy_epsilon=1.0,   # 差分隐私参数
    enable_blockchain_consensus=False,  # 区块链共识
    reputation_threshold=0.7,           # 声誉阈值
    aggregation_method='fedavg'         # 聚合方法
)
```

**使用示例**:
```bash
python train_model.py --reward-type federated --enable-privacy-protection --differential-privacy-epsilon 0.5
```

**别名支持**: `federated_reward`, `federated`, `distributed`, `collaborative`, `multi_client`, `privacy_preserving`, `differential_privacy`, `secure_aggregation`, `blockchain`, `smart_contracts`, `reputation_based`, `consensus`, `decentralized`

---

#### 17. MetaLearningReward - 元学习自适应奖励
**设计理念**: 基于Model-Agnostic Meta-Learning的自适应奖励机制
**适用场景**: 快速市场适应、跨市场策略迁移、动态环境优化
**研究基础**: 2024-2025年MAML、元梯度优化和自我奖励理论
**技术特性**:
- MAML快速任务适应
- 元梯度优化
- 自我奖励机制
- 记忆增强学习

**参数配置**:
```python
reward_fn = create_reward_function(
    'meta_learning_reward',
    alpha=0.01,                         # 内层学习率
    beta=0.001,                         # 外层学习率
    adaptation_steps=5,                 # 适应步数
    enable_self_rewarding=True,         # 启用自我奖励
    enable_memory_augmentation=True,    # 启用记忆增强
    task_detection_window=50            # 任务检测窗口
)
```

**使用示例**:
```bash
python train_model.py --reward-type meta_learning --alpha 0.02 --beta 0.002 --enable-self-rewarding
```

**别名支持**: `meta_learning_reward`, `meta_learning`, `maml`, `adaptive`, `meta_gradient`, `self_adapting`, `task_adaptive`, `few_shot`, `meta_optimization`, `gradient_based_meta`, `memory_augmented`, `agnostic_meta`

---

## 🎯 选择指南

### 按使用场景选择

| 场景 | 推荐奖励函数 | 理由 |
|------|-------------|------|
| 新手入门 | SimpleReturnReward | 简单直观，易于理解 |
| 稳健投资 | RiskAdjustedReward | 经典夏普比率，久经验证 |
| 机构级策略 | DiversifiedReward | 多维度综合评估 |
| 高频交易 | LogSharpeReward | 对数收益，适合连续交易 |
| 风险控制 | UncertaintyAwareReward | 不确定性量化，专业风控 |
| 策略探索 | CuriosityDrivenReward | 内在动机，发现新策略 |
| AI辅助设计 | LLMGuidedReward | 自然语言描述，AI生成 |
| 因果分析 | CausalReward | 因果推理，科学验证 |
| 分阶段训练 | CurriculumReward | 渐进式学习，新手到专家 |
| 隐私保护 | FederatedReward | 多方协作，保护隐私 |
| 快速适应 | MetaLearningReward | 元学习，快速适应新环境 |

### 按复杂度选择

- **初级** (⭐): SimpleReturnReward, ProfitLossReward
- **中级** (⭐⭐): RiskAdjustedReward, DiversifiedReward  
- **高级** (⭐⭐⭐): LogSharpeReward, ReturnDrawdownReward, DynamicSortinoReward, RegimeAwareReward
- **专家** (⭐⭐⭐⭐): ExpertCommitteeReward, UncertaintyAwareReward, CuriosityDrivenReward
- **研究级** (⭐⭐⭐⭐⭐): SelfRewardingReward, CausalReward, LLMGuidedReward, CurriculumReward, FederatedReward, MetaLearningReward

## 🔧 技术规格

### 系统要求
- Python 3.7+
- TensorFlow 2.7.0+
- NumPy 1.21.0+
- Ray 1.8.0

### 性能特性
- **计算效率**: 向量化运算，支持GPU加速
- **内存优化**: 滑动窗口设计，控制内存使用
- **并发支持**: 线程安全设计，支持并行训练
- **可扩展性**: 模块化架构，易于添加新奖励函数

### 集成接口
```python
# 统一工厂接口
from src.environment.rewards import create_reward_function

# 获取所有可用类型
from src.environment.rewards import list_available_reward_types

# 获取奖励函数信息
from src.environment.rewards import get_reward_function_info
```

## 📈 最佳实践

### 1. 奖励函数组合策略
```python
# 多阶段策略：从简单到复杂
# 阶段1：简单验证
reward_fn = create_reward_function('simple_return')

# 阶段2：风险控制
reward_fn = create_reward_function('risk_adjusted')

# 阶段3：AI增强
reward_fn = create_reward_function('uncertainty_aware')
```

### 2. 参数调优建议
- 从默认参数开始
- 逐步调整单个参数
- 使用A/B测试验证效果
- 记录参数组合和性能表现

### 3. 性能监控
```python
# 获取奖励函数状态信息
info = reward_fn.get_reward_info()
print(f"奖励函数: {info['name']}")
print(f"复杂度: {info['complexity']}")

# 对于高级奖励函数，获取详细状态
if hasattr(reward_fn, 'get_uncertainty_info'):
    uncertainty_info = reward_fn.get_uncertainty_info()
    print(f"不确定性水平: {uncertainty_info['current_uncertainty']}")
```

## 🚨 注意事项

### 安全使用
- LLMGuidedReward包含Constitutional AI安全框架
- 所有奖励函数均通过安全性测试
- 建议在测试环境中验证新配置

### 性能考虑
- 高级AI奖励函数计算开销较大
- 建议根据硬件资源选择合适的复杂度
- 使用性能监控工具跟踪计算效率

### 数据要求
- 大部分奖励函数需要至少50个时间步的历史数据
- AI驱动的奖励函数可能需要更长的预热期
- 确保数据质量和完整性

---

**版本**: v1.0.0  
**最后更新**: 2025年1月  
**技术支持**: 基于2024-2025年最新强化学习和人工智能研究成果

---

> 💡 **提示**: 这个奖励函数系统代表了强化学习在量化交易领域的最前沿应用。建议从基础奖励函数开始，逐步探索AI驱动的高级功能。