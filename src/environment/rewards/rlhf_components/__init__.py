"""
RLHF奖励函数组件模块

包含人类反馈强化学习所需的所有核心组件：
- 专家反馈收集接口
- 偏好学习模型
- 奖励模型训练
- PPO人类对齐算法
"""

from .expert_feedback_interface import (
    ExpertFeedbackInterface,
    FeedbackType,
    ExpertiseLevel,
    TradingScenario,
    ExpertProfile,
    FeedbackData
)

# 偏好学习组件 (已实现)
from .preference_learning import (
    BradleyTerryPreferenceModel,
    PreferenceLearningTrainer,
    MultiExpertPreferenceFusion,
    PreferenceLearningMethod,
    PreferenceTrainingConfig,
    PreferenceData,
    create_preference_learner
)

# 奖励模型组件 (已实现)
from .reward_model import (
    CritiqueGuidedRewardModel,
    HierarchicalRewardModel,
    AdaptiveRewardModel,
    RewardModelTrainer,
    RewardModelType,
    RewardModelConfig,
    RewardTrainingData,
    create_reward_model_trainer
)

# PPO人类对齐组件 (已实现)
from .ppo_human_alignment import (
    PPOWithHumanAlignment,
    OnlineAdaptiveLearning,
    ConstitutionalAI,
    PPOHumanConfig,
    AlignmentObjective,
    PolicyNetwork
)

__all__ = [
    # 反馈收集组件 (已实现)
    'ExpertFeedbackInterface',
    'FeedbackType',
    'ExpertiseLevel', 
    'TradingScenario',
    'ExpertProfile',
    'FeedbackData',
    
    # 偏好学习组件 (已实现)
    'BradleyTerryPreferenceModel',
    'PreferenceLearningTrainer', 
    'MultiExpertPreferenceFusion',
    'PreferenceLearningMethod',
    'PreferenceTrainingConfig',
    'PreferenceData',
    'create_preference_learner',
    
    # 奖励模型组件 (已实现)
    'CritiqueGuidedRewardModel',
    'HierarchicalRewardModel',
    'AdaptiveRewardModel',
    'RewardModelTrainer',
    'RewardModelType',
    'RewardModelConfig',
    'RewardTrainingData',
    'create_reward_model_trainer',
    
    # PPO人类对齐组件 (已实现)
    'PPOWithHumanAlignment',
    'OnlineAdaptiveLearning',
    'ConstitutionalAI',
    'PPOHumanConfig',
    'AlignmentObjective',
    'PolicyNetwork'
]