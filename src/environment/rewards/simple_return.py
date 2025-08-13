"""
基于简单收益率的奖励函数

简单直接的奖励函数，主要基于资产组合的收益率变化计算奖励。
适合初学者理解和简单的交易策略训练。
"""

import numpy as np
import logging
from typing import Dict, Any
from .base_reward import BaseRewardScheme


class SimpleReturnReward(BaseRewardScheme):
    """
    基于简单收益率的奖励函数
    
    这是一个简单直接的奖励函数，主要根据资产组合价值的变化来计算奖励。
    奖励 = 当前步骤的收益率百分比，简单易懂。
    """
    
    def __init__(self,
                 initial_balance: float = 10000.0,
                 step_weight: float = 1.0,
                 total_weight: float = 0.1,
                 **kwargs):
        """
        初始化简单收益率奖励函数
        
        Args:
            initial_balance: 初始资金
            step_weight: 步骤收益率权重，默认1.0
            total_weight: 总收益率权重，默认0.1
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        self.step_weight = step_weight
        self.total_weight = total_weight
        
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 计算简单收益率奖励
        
        Args:
            portfolio_value: 当前投资组合价值
            action: 执行的动作
            price: 当前价格
            portfolio_info: 投资组合详细信息
            trade_info: 交易执行信息
            step: 当前步数
            **kwargs: 其他参数
            
        Returns:
            float: 奖励值
        """
        # 更新历史记录
        self.update_history(portfolio_value)
        
        # 计算步骤收益率
        if self.previous_value is None or len(self.portfolio_history) < 2:
            step_return_pct = 0.0
        else:
            prev_value = self.portfolio_history[-2] if len(self.portfolio_history) > 1 else self.initial_balance
            if prev_value > 0:
                step_return_pct = (portfolio_value - prev_value) / prev_value * 100
            else:
                step_return_pct = 0.0
        
        # 计算总收益率
        if self.initial_balance > 0:
            total_return_pct = (portfolio_value - self.initial_balance) / self.initial_balance * 100
        else:
            total_return_pct = 0.0
        
        # 组合奖励
        reward = (step_return_pct * self.step_weight + 
                 total_return_pct * self.total_weight)
        
        # 记录奖励历史
        self.reward_history.append(reward)
        
        return reward
    
    def reward(self, env) -> float:
        """
        向后兼容的奖励计算方法
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 奖励值
        """
        # 使用基类的兼容方法
        return super().reward(env)
