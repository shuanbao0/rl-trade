"""
自动迁移的奖励函数: ProfitLossReward
原始文件: src/environment/rewards/profit_loss.py
市场兼容性: forex, stock, crypto
时间粒度兼容性: 1min, 5min, 1h, 1d, 1w
复杂度: 4/10
别名: profit, pnl, profit_loss
"""

import numpy as np
from src.rewards.core.base_reward import BaseReward, SimpleRewardMixin
from src.rewards.core.reward_context import RewardContext
from typing import Optional, Dict, Any, List


class ProfitLossReward(BaseReward, SimpleRewardMixin):
    """
    基于盈亏比的奖励函数
    
    这个奖励函数专注于盈利交易和亏损交易的比例关系。
    奖励策略产生高盈亏比的交易，惩罚连续亏损。
    
    迁移自原始ProfitLossReward类，适配新的RewardContext架构。
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        
        # 交易参数
        self.min_trade_threshold = config.get('min_trade_threshold', 0.001)  # 最小交易阈值
        self.profit_bonus = config.get('profit_bonus', 2.0)  # 盈利奖励倍数
        self.loss_penalty = config.get('loss_penalty', 1.5)  # 亏损惩罚倍数
        self.consecutive_loss_penalty = config.get('consecutive_loss_penalty', 0.5)  # 连续亏损惩罚
        self.win_rate_bonus = config.get('win_rate_bonus', 0.1)  # 胜率奖励
        
        # 交易统计
        self.trades = []
        self.consecutive_losses = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # 设置名称和描述
        self.name = config.get('name', 'profit_loss')
        self.description = "基于盈亏比的奖励函数，专注于交易质量而非频率"
    
    def calculate(self, context: RewardContext) -> float:
        """
        计算盈亏比奖励
        
        Args:
            context: 奖励上下文对象
            
        Returns:
            float: 计算的奖励值
        """
        # 计算步骤收益
        step_return = self.get_step_return(context)
        
        # 获取初始余额用于百分比计算
        initial_balance = context.portfolio_info.get('initial_balance', 10000.0)
        
        # 基础奖励：步骤收益的百分比
        base_reward = (step_return * context.portfolio_value) / initial_balance * 100
        
        # 检查是否有有效交易
        if abs(context.action) < self.min_trade_threshold:
            # 无交易时的小幅惩罚，鼓励行动
            return base_reward - 0.1
        
        # 记录交易
        trade_return = step_return * context.portfolio_value
        self.trades.append(trade_return)
        
        # 计算盈亏比奖励
        if trade_return > 0:
            # 盈利交易
            profit_reward = self._calculate_profit_reward(base_reward, trade_return)
        else:
            # 亏损交易
            profit_reward = self._calculate_loss_penalty(base_reward, trade_return)
        
        # 胜率奖励（每10个交易计算一次）
        win_rate_reward = self._calculate_win_rate_reward()
        
        # 总奖励
        total_reward = profit_reward + win_rate_reward
        
        return total_reward
    
    def _calculate_profit_reward(self, base_reward: float, trade_return: float) -> float:
        """计算盈利奖励"""
        self.total_profit += trade_return
        self.winning_trades += 1
        self.consecutive_losses = 0  # 重置连续亏损
        
        # 盈利奖励
        return base_reward * self.profit_bonus
    
    def _calculate_loss_penalty(self, base_reward: float, trade_return: float) -> float:
        """计算亏损惩罚"""
        self.total_loss += abs(trade_return)
        self.losing_trades += 1
        self.consecutive_losses += 1
        
        # 亏损惩罚
        loss_reward = base_reward * self.loss_penalty
        
        # 连续亏损额外惩罚
        consecutive_penalty = -self.consecutive_losses * self.consecutive_loss_penalty
        
        return loss_reward + consecutive_penalty
    
    def _calculate_win_rate_reward(self) -> float:
        """计算胜率奖励"""
        if len(self.trades) % 10 == 0 and len(self.trades) > 0:
            win_rate = self.winning_trades / len(self.trades)
            return win_rate * self.win_rate_bonus * 100
        return 0.0
    
    def reset(self):
        """重置奖励函数状态"""
        super().reset()
        self.trades = []
        self.consecutive_losses = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def get_trade_statistics(self) -> Dict[str, float]:
        """获取交易统计信息"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'consecutive_losses': 0
            }
        
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        avg_profit = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0.0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0.0
        
        return {
            'total_trades': len(self.trades),
            'win_rate': self.winning_trades / len(self.trades),
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'consecutive_losses': self.consecutive_losses
        }
    
    def get_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        trade_stats = self.get_trade_statistics()
        
        return {
            'name': self.name,
            'type': 'profit_loss',
            'description': self.description,
            'market_compatibility': ['forex', 'stock', 'crypto'],
            'granularity_compatibility': ['1min', '5min', '1h', '1d', '1w'],
            'parameters': {
                'min_trade_threshold': self.min_trade_threshold,
                'profit_bonus': self.profit_bonus,
                'loss_penalty': self.loss_penalty,
                'consecutive_loss_penalty': self.consecutive_loss_penalty,
                'win_rate_bonus': self.win_rate_bonus
            },
            'complexity_score': 4,
            'category': 'basic',
            'features': ['profit_loss_ratio', 'win_rate_tracking', 'consecutive_loss_penalty', 'trade_statistics'],
            'trade_statistics': trade_stats,
            'migrated_from': 'src.environment.rewards.profit_loss.ProfitLossReward'
        }
    
    # 向后兼容方法
    def compute_reward(self, old_context):
        """兼容旧的compute_reward方法"""
        from src.rewards.migration.compatibility_mapper import CompatibilityMapper
        mapper = CompatibilityMapper()
        new_context = mapper.map_context(old_context)
        return self.calculate(new_context)
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """兼容旧的calculate_reward方法"""
        context = RewardContext(
            portfolio_value=portfolio_value,
            action=action,
            current_price=price,
            step=step,
            portfolio_info=portfolio_info,
            **kwargs
        )
        return self.calculate(context)
    
    def get_reward_info(self):
        """兼容旧的get_reward_info方法"""
        return self.get_info()