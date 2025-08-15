"""
基础奖励函数模块
"""

from .simple_return_reward import SimpleReturnReward
from .profit_loss_reward import ProfitLossReward

__all__ = [
    'SimpleReturnReward',
    'ProfitLossReward'
]