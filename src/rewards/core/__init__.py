"""
Rewards Core Module - 奖励函数核心组件
"""

from .reward_context import RewardContext, RewardResult, RewardContextBuilder
from .base_reward import BaseReward, RewardPlugin, SimpleRewardMixin, HistoryAwareRewardMixin
from .reward_registry import RewardRegistry, register_reward, get_global_registry
from .reward_factory import SmartRewardFactory, CompositeReward

__all__ = [
    'RewardContext',
    'RewardResult', 
    'RewardContextBuilder',
    'BaseReward',
    'RewardPlugin',
    'SimpleRewardMixin',
    'HistoryAwareRewardMixin',
    'RewardRegistry',
    'register_reward',
    'get_global_registry',
    'SmartRewardFactory',
    'CompositeReward'
]