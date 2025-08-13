"""
基于夏普比率的风险调整奖励函数

这是原有RiskAdjustedReward的重构版本，保持了原有的功能和特性，
同时使其符合新的模块化架构。
"""

import numpy as np
import logging
from typing import Dict, Any
from .base_reward import BaseRewardScheme


class RiskAdjustedReward(BaseRewardScheme):
    """
    基于夏普比率的风险调整奖励函数
    
    计算考虑风险的收益奖励，鼓励稳定盈利，惩罚过度风险。
    采用渐进式训练策略，从基础阶段逐步提升到高级阶段。
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02, 
                 window_size: int = 50, 
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化风险调整奖励函数
        
        Args:
            risk_free_rate: 无风险收益率(年化)，默认2%
            window_size: 计算夏普比率的窗口大小
            initial_balance: 初始资金，用于计算总收益率
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        self.risk_free_rate = risk_free_rate / 252  # 转换为日收益率
        self.window_size = window_size
        
        # 渐进式训练阶段管理
        self.current_stage = "basic"  # basic -> intermediate -> advanced
        self.total_rewards = []
        
        # 阶段切换条件 - 延长学习期，更宽松的阈值
        self.stage_thresholds = {
            "basic_to_intermediate": {"min_episodes": 100, "avg_reward_threshold": 5.0},
            "intermediate_to_advanced": {"min_episodes": 300, "avg_reward_threshold": 10.0}
        }
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 计算风险调整奖励
        
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
        if len(self.portfolio_history) < 2:
            step_return_pct = 0.0
        else:
            prev_value = self.portfolio_history[-2]
            if prev_value > 0:
                step_return_pct = (portfolio_value - prev_value) / prev_value * 100
            else:
                step_return_pct = 0.0
        
        # 计算总收益率
        total_return_pct = (portfolio_value - self.initial_balance) / self.initial_balance * 100
        
        # 根据当前阶段计算奖励
        reward = self._calculate_stage_reward(step_return_pct, action, portfolio_value, total_return_pct)
        
        # 记录奖励历史
        self.reward_history.append(reward)
        
        return reward
    
    def _calculate_stage_reward(self, step_return_pct: float, action: float, current_value: float, total_return_pct: float) -> float:
        """根据当前阶段计算奖励 - 修复资金缩放问题"""
        # 计算资金缩放因子，基于初始资金调整奖励幅度
        balance_scale_factor = self.initial_balance / 10000.0  # 基准：10,000
        reward_scale = 1.0 / balance_scale_factor  # 资金越大，缩放越小
        
        if self.current_stage == "basic":
            # 基础阶段：主要基于总收益率，鼓励正收益
            total_return_reward = total_return_pct * 1.0 * reward_scale  # 从10.0降到1.0
            
            # 基础风险惩罚：避免极端动作
            risk_penalty = 0.0
            if abs(action) > 0.8:
                risk_penalty = -abs(action) * 5.0 * reward_scale  # 从50.0降到5.0
            
            return total_return_reward + risk_penalty
            
        elif self.current_stage == "intermediate":
            # 中级阶段：结合步骤收益和风险调整
            step_reward = step_return_pct * 2.0 * reward_scale  # 从20.0降到2.0
            total_reward = total_return_pct * 0.5 * reward_scale  # 从5.0降到0.5
            
            # 中级风险调整：基于收益波动
            risk_adjustment = self._calculate_basic_risk_adjustment() * reward_scale
            
            return step_reward + total_reward + risk_adjustment
            
        else:  # advanced
            # 高级阶段：完整的夏普比率计算
            return self._calculate_sharpe_reward(step_return_pct, current_value, total_return_pct, reward_scale)
    
    def _calculate_basic_risk_adjustment(self) -> float:
        """计算基础风险调整"""
        if len(self.reward_history) < 5:
            return 0.0
        
        recent_rewards = self.reward_history[-5:]
        volatility = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
        
        # 惩罚高波动率
        return -volatility * 2.0
    
    def _calculate_sharpe_reward(self, step_return_pct: float, current_value: float, total_return_pct: float, reward_scale: float) -> float:
        """计算完整的夏普比率奖励 - 修复缩放问题"""
        if len(self.portfolio_history) < self.window_size:
            return step_return_pct * 15.0 * reward_scale  # 降级处理，应用缩放
        
        # 计算窗口内的收益序列
        window_values = self.portfolio_history[-self.window_size:]
        returns = []
        
        for i in range(1, len(window_values)):
            if window_values[i-1] > 0:
                ret = (window_values[i] - window_values[i-1]) / window_values[i-1]
                returns.append(ret)
        
        if len(returns) < 2:
            return step_return_pct * 15.0 * reward_scale
        
        # 计算夏普比率
        excess_returns = [r - self.risk_free_rate for r in returns]
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess > 0:
            sharpe_ratio = mean_excess / std_excess
            # 大幅减少夏普比率权重，避免奖励爆炸
            sharpe_reward = sharpe_ratio * 100.0 * reward_scale  # 从1000.0降到100.0
        else:
            sharpe_reward = mean_excess * 100.0 * reward_scale  # 从1000.0降到100.0
        
        # 结合多个因子，都应用缩放
        step_reward = step_return_pct * 10.0 * reward_scale
        total_reward = total_return_pct * 2.0 * reward_scale
        
        return step_reward + total_reward + sharpe_reward
