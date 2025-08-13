"""
汇率简化奖励函数

基于实验分析创建的简化版汇率奖励函数，专注解决核心问题：
1. 合理的奖励规模 (避免极端值)
2. 点数导向的奖励计算
3. 简单但有效的趋势跟随
4. 控制过度交易
"""

import numpy as np
import logging
from typing import Dict, Any
from collections import deque
from .base_reward import BaseRewardScheme


class ForexSimpleReward(BaseRewardScheme):
    """
    汇率简化奖励函数
    
    简化版本，专注核心功能：
    - 基于点数的直观奖励
    - 趋势方向奖励
    - 交易频率控制  
    - 稳定的奖励范围
    """
    
    def __init__(self,
                 initial_balance: float = 10000.0,
                 pip_size: float = 0.0001,
                 target_daily_pips: float = 10.0,
                 trend_window: int = 10,
                 overtrading_penalty: float = 0.1,
                 **kwargs):
        """
        初始化汇率简化奖励函数
        
        Args:
            initial_balance: 初始资金
            pip_size: 点大小
            target_daily_pips: 目标日点数
            trend_window: 趋势窗口
            overtrading_penalty: 过度交易惩罚
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        self.pip_size = pip_size
        self.target_daily_pips = target_daily_pips
        self.trend_window = trend_window
        self.overtrading_penalty = overtrading_penalty
        
        # 历史数据
        self.price_history = deque(maxlen=trend_window + 5)
        self.action_history = deque(maxlen=10)
        
    def calculate_reward(self, portfolio_value: float, action: float, price: float,
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        简化奖励计算
        
        Returns:
            float: 奖励值 (范围约[-1, 1])
        """
        # 更新历史
        self.update_history(portfolio_value)
        self.price_history.append(price)
        self.action_history.append(action)
        
        if len(self.portfolio_history) < 2 or len(self.price_history) < 2:
            return 0.0
        
        # 1. 基础收益奖励 (主要组件)
        step_return = (portfolio_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
        base_reward = step_return * 100.0  # 转换为百分比
        
        # 2. 点数方向奖励
        price_change = self.price_history[-1] - self.price_history[-2] 
        pip_change = price_change / self.pip_size
        direction_reward = pip_change * action * 0.1  # 方向一致性奖励
        
        # 3. 趋势跟随奖励
        trend_reward = 0.0
        if len(self.price_history) >= self.trend_window:
            recent_prices = list(self.price_history)[-self.trend_window:]
            trend_direction = recent_prices[-1] - recent_prices[0]
            if abs(trend_direction) > self.pip_size * 2:  # 明显趋势
                trend_reward = np.sign(trend_direction) * action * 0.05
        
        # 4. 过度交易惩罚
        trading_penalty = 0.0
        if len(self.action_history) >= 5:
            recent_actions = list(self.action_history)[-5:]
            action_volatility = np.std(recent_actions)
            if action_volatility > 0.5:  # 动作变化过大
                trading_penalty = -self.overtrading_penalty
        
        # 5. 综合奖励
        total_reward = base_reward + direction_reward + trend_reward + trading_penalty
        
        # 6. 限制奖励范围
        normalized_reward = np.clip(total_reward, -1.0, 1.0)
        
        self.reward_history.append(normalized_reward)
        return normalized_reward
    
    def get_reward_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            "name": "Forex Simple Reward",
            "description": "简化版汇率奖励函数，专注核心功能",
            "category": "forex_basic",
            "features": [
                "点数导向奖励",
                "趋势跟随",
                "过度交易控制",
                "稳定奖励范围"
            ],
            "parameters": {
                "pip_size": self.pip_size,
                "target_daily_pips": self.target_daily_pips,
                "trend_window": self.trend_window
            },
            "expected_reward_range": [-1.0, 1.0],
            "suitable_for": ["EURUSD", "主要货币对", "汇率交易"]
        }
    
    def reset(self):
        """重置状态"""
        super().reset()
        self.price_history.clear()
        self.action_history.clear()