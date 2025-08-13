"""
奖励函数模块

模块化的奖励函数系统，支持多种奖励策略：

奖励函数类型：
- RiskAdjustedReward: 基于夏普比率的风险调整奖励
- SimpleReturnReward: 基于简单收益率的奖励
- ProfitLossReward: 基于盈亏比的奖励  
- DiversifiedReward: 多指标综合奖励
- LogSharpeReward: 基于对数收益和差分夏普比率的奖励
- ReturnDrawdownReward: 基于最终收益与最大回撤结合的奖励
- DynamicSortinoReward: 基于动态时间尺度索提诺比率的奖励
- RegimeAwareReward: 基于市场状态感知的自适应专家奖励系统
- ExpertCommitteeReward: 基于多目标强化学习的专家委员会协作决策奖励
- UncertaintyAwareReward: 基于认知和任意不确定性量化的风险敏感奖励
- CuriosityDrivenReward: 基于好奇心驱动的内在动机强化学习奖励
- SelfRewardingReward: 基于Meta AI 2024年Self-Rewarding理论的自我评判奖励
- CausalReward: 基于2024年最新因果推理理论的因果图构建和混淆变量识别奖励
- LLMGuidedReward: 基于2024-2025年EUREKA和Constitutional AI的LLM引导奖励函数设计
- CurriculumReward: 基于2024-2025年课程学习研究的多阶段渐进式复杂度奖励系统
- FederatedReward: 基于2024-2025年联邦学习研究的分布式协作奖励优化系统
- MetaLearningReward: 基于2024-2025年Model-Agnostic Meta-Learning的自适应奖励机制
- OptimizedForexReward: Experiment #005优化外汇奖励函数，解决奖励-回报不一致问题

使用方式：
```python
from src.environment.rewards import create_reward_function

# 通过配置创建奖励函数
reward_fn = create_reward_function(
    reward_type="risk_adjusted",
    risk_free_rate=0.02,
    window_size=50
)

# 或直接实例化
from src.environment.rewards.risk_adjusted import RiskAdjustedReward
reward_fn = RiskAdjustedReward(risk_free_rate=0.02)
```
"""

from .base_reward import BaseRewardScheme
from .reward_factory import RewardFactory, create_reward_function, get_reward_function_info, list_available_reward_types
from .risk_adjusted import RiskAdjustedReward
from .simple_return import SimpleReturnReward
from .profit_loss import ProfitLossReward
from .diversified import DiversifiedReward
from .log_sharpe import LogSharpeReward
from .return_drawdown import ReturnDrawdownReward
from .dynamic_sortino import DynamicSortinoReward
from .regime_aware import RegimeAwareReward
from .expert_committee import ExpertCommitteeReward
from .uncertainty_aware import UncertaintyAwareReward
from .curiosity_driven import CuriosityDrivenReward
from .self_rewarding import SelfRewardingReward
from .causal_reward import CausalReward
from .llm_guided import LLMGuidedReward
from .curriculum_reward import CurriculumReward
from .federated_reward import FederatedReward
from .meta_learning_reward import MetaLearningReward
from .optimized_forex_reward import OptimizedForexReward

__all__ = [
    'BaseRewardScheme',
    'RewardFactory', 
    'create_reward_function',
    'get_reward_function_info',
    'list_available_reward_types',
    'RiskAdjustedReward',
    'SimpleReturnReward', 
    'ProfitLossReward',
    'DiversifiedReward',
    'LogSharpeReward',
    'ReturnDrawdownReward',
    'DynamicSortinoReward',
    'RegimeAwareReward',
    'ExpertCommitteeReward',
    'UncertaintyAwareReward',
    'CuriosityDrivenReward',
    'SelfRewardingReward',
    'CausalReward',
    'LLMGuidedReward',
    'CurriculumReward',
    'FederatedReward',
    'MetaLearningReward',
    'OptimizedForexReward'
]

# 版本信息
__version__ = "1.0.0"