"""
基于盈亏比的奖励函数

专注于盈利和亏损的比例关系，鼓励高盈亏比的交易策略。
适合专注于交易质量而非频率的策略训练。
"""

import numpy as np
import logging
from typing import Dict, Any, List
from .base_reward import BaseRewardScheme


class ProfitLossReward(BaseRewardScheme):
    """
    基于盈亏比的奖励函数
    
    这个奖励函数专注于盈利交易和亏损交易的比例关系。
    奖励策略产生高盈亏比的交易，惩罚连续亏损。
    """
    
    def __init__(self,
                 initial_balance: float = 10000.0,
                 min_trade_threshold: float = 0.001,  # 最小交易阈值（0.1%）
                 profit_bonus: float = 2.0,           # 盈利奖励倍数
                 loss_penalty: float = 1.5,           # 亏损惩罚倍数
                 consecutive_loss_penalty: float = 0.5, # 连续亏损额外惩罚
                 win_rate_bonus: float = 0.1,         # 胜率奖励
                 **kwargs):
        """
        初始化盈亏比奖励函数
        
        Args:
            initial_balance: 初始资金
            min_trade_threshold: 最小交易阈值
            profit_bonus: 盈利奖励倍数
            loss_penalty: 亏损惩罚倍数
            consecutive_loss_penalty: 连续亏损额外惩罚
            win_rate_bonus: 胜率奖励
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        self.min_trade_threshold = min_trade_threshold
        self.profit_bonus = profit_bonus
        self.loss_penalty = loss_penalty
        self.consecutive_loss_penalty = consecutive_loss_penalty
        self.win_rate_bonus = win_rate_bonus
        
        # 交易统计
        self.trades = []
        self.consecutive_losses = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 计算盈亏比奖励
        
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
        
        # 计算步骤收益
        if len(self.portfolio_history) < 2:
            step_return = 0.0
        else:
            prev_value = self.portfolio_history[-2]
            step_return = portfolio_value - prev_value
        
        # 基础奖励：步骤收益的百分比
        base_reward = (step_return / self.initial_balance) * 100
        
        # 检查是否有有效交易
        if abs(action) < self.min_trade_threshold:
            # 无交易时的小幅惩罚，鼓励行动
            return base_reward - 0.1
        
        # 记录交易
        self.trades.append(step_return)
        
        # 计算盈亏比奖励
        if step_return > 0:
            # 盈利交易
            self.total_profit += step_return
            self.winning_trades += 1
            self.consecutive_losses = 0  # 重置连续亏损
            
            # 盈利奖励
            profit_reward = base_reward * self.profit_bonus
            
        else:
            # 亏损交易
            self.total_loss += abs(step_return)
            self.losing_trades += 1
            self.consecutive_losses += 1
            
            # 亏损惩罚
            loss_reward = base_reward * self.loss_penalty
            
            # 连续亏损额外惩罚
            consecutive_penalty = -self.consecutive_losses * self.consecutive_loss_penalty
            
            profit_reward = loss_reward + consecutive_penalty
        
        # 胜率奖励（每10个交易计算一次）
        win_rate_reward = 0.0
        if len(self.trades) % 10 == 0 and len(self.trades) > 0:
            win_rate = self.winning_trades / len(self.trades)
            win_rate_reward = win_rate * self.win_rate_bonus * 100
        
        # 总奖励
        total_reward = profit_reward + win_rate_reward
        
        # 记录奖励历史
        self.reward_history.append(total_reward)
        
        return total_reward
    
    def _is_profitable_trade(self, return_value: float) -> bool:
        """判断是否为盈利交易"""
        return return_value > 0
    
    def _is_significant_trade(self, return_value: float) -> bool:
        """判断是否为有意义的交易（超过最小交易阈值）"""
        return abs(return_value) >= self.min_trade_threshold
    
    def _count_consecutive_losses(self, trade_sequence: list) -> int:
        """计算连续亏损次数"""
        consecutive = 0
        for trade in reversed(trade_sequence):
            if trade <= 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def _calculate_win_rate(self, trade_results: list) -> float:
        """计算胜率"""
        if not trade_results:
            return 0.0
        
        winning_trades = sum(1 for trade in trade_results if trade > 0)
        return winning_trades / len(trade_results)
