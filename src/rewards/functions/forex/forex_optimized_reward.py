"""
自动迁移的奖励函数: ForexOptimizedReward
原始文件: src/environment/rewards/forex_optimized.py
市场兼容性: forex
时间粒度兼容性: 1min, 5min, 1h
复杂度: 8/10
别名: forex, forex_optimized, pip_based
"""

import numpy as np
from collections import deque
from src.rewards.core.base_reward import BaseReward, HistoryAwareRewardMixin
from src.rewards.core.reward_context import RewardContext
from typing import Optional, Dict, Any, List


class ForexOptimizedReward(BaseReward, HistoryAwareRewardMixin):
    """
    汇率专用优化奖励函数
    
    专门为EURUSD等主要货币对设计的奖励函数，结合：
    - 点数收益计算 (forex核心指标)
    - 趋势跟随奖励 (主要盈利模式)
    - 交易质量控制 (避免噪声交易)
    - 风险管理机制 (控制回撤)
    - 时间序列一致性 (适应汇率持续性特征)
    
    迁移自原始ForexOptimizedReward类，适配新的RewardContext架构。
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        
        # 外汇专用参数
        self.pip_size = config.get('pip_size', 0.0001)  # 标准点大小
        self.currency_pair = config.get('base_currency_pair', 'EURUSD')
        self.trend_window = config.get('trend_window', 20)
        self.quality_window = config.get('quality_window', 10)
        self.daily_target_pips = config.get('daily_target_pips', 20.0)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.consistency_weight = config.get('consistency_weight', 0.3)
        
        # 历史数据配置
        max_window = max(self.trend_window, self.quality_window) + 10
        self.min_history_steps = max_window
        
        # 交易状态追踪
        self.price_history_buffer = deque(maxlen=max_window)
        self.action_history_buffer = deque(maxlen=self.quality_window)
        self.pip_profits = deque(maxlen=100)
        self.trade_decisions = deque(maxlen=50)
        
        # 性能指标
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.daily_pip_count = 0.0
        
        # 奖励权重配置
        self.weights = {
            'pip_reward': config.get('pip_weight', 0.40),
            'trend_reward': config.get('trend_weight', 0.25),
            'quality_reward': config.get('quality_weight', 0.20),
            'risk_reward': config.get('risk_weight', 0.15)
        }
        
        # 设置名称和描述
        self.name = config.get('name', 'forex_optimized')
        self.description = f"外汇优化奖励函数，专为{self.currency_pair}等主要货币对设计"
    
    def calculate(self, context: RewardContext) -> float:
        """
        计算外汇优化奖励
        
        Args:
            context: 奖励上下文对象
            
        Returns:
            float: 标准化奖励值 (范围约[-2, 2])
        """
        # 更新历史数据
        self.price_history_buffer.append(context.current_price)
        self.action_history_buffer.append(context.action)
        
        # 如果数据不足，返回零奖励
        if len(self.price_history_buffer) < 2:
            return 0.0
        
        # 1. 计算点数奖励 (核心组件)
        pip_reward = self._calculate_pip_reward(context)
        
        # 2. 计算趋势跟随奖励
        trend_reward = self._calculate_trend_reward(context)
        
        # 3. 计算交易质量奖励
        quality_reward = self._calculate_quality_reward(context)
        
        # 4. 计算风险管理奖励
        risk_reward = self._calculate_risk_reward(context)
        
        # 5. 综合奖励计算
        total_reward = (
            pip_reward * self.weights['pip_reward'] +
            trend_reward * self.weights['trend_reward'] + 
            quality_reward * self.weights['quality_reward'] +
            risk_reward * self.weights['risk_reward']
        )
        
        # 6. 奖励标准化到合理范围
        normalized_reward = np.tanh(total_reward) * 2.0  # 范围约[-2, 2]
        
        return normalized_reward
    
    def _calculate_pip_reward(self, context: RewardContext) -> float:
        """计算点数收益奖励"""
        if len(self.price_history_buffer) < 2:
            return 0.0
            
        # 计算价格变化 (pips)
        prev_price = self.price_history_buffer[-2]
        current_price = context.current_price
        price_change = current_price - prev_price
        pip_change = price_change / self.pip_size
        
        # 计算点数收益 (考虑交易方向)
        pip_profit = pip_change * context.action
        self.pip_profits.append(pip_profit)
        self.daily_pip_count += pip_profit
        
        # 奖励计算：基础点数收益
        base_pip_reward = pip_profit / self.daily_target_pips  # 标准化到日目标
        
        # 连续盈利/亏损调整
        if pip_profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
            # 连续盈利奖励递减 (避免过度冒险)
            win_bonus = min(1.2, 1.0 + 0.05 * min(self.consecutive_wins, 4))
            base_pip_reward *= win_bonus
        elif pip_profit < 0:
            self.consecutive_losses += 1  
            self.consecutive_wins = 0
            # 连续亏损惩罚递增
            loss_penalty = max(0.8, 1.0 - 0.1 * min(self.consecutive_losses, 3))
            base_pip_reward *= loss_penalty
        
        return base_pip_reward
    
    def _calculate_trend_reward(self, context: RewardContext) -> float:
        """计算趋势跟随奖励"""
        if len(self.price_history_buffer) < self.trend_window:
            return 0.0
            
        # 计算多周期趋势
        recent_prices = list(self.price_history_buffer)[-self.trend_window:]
        
        # 短期趋势 (5周期)
        short_trend = (recent_prices[-1] - recent_prices[-5]) if len(recent_prices) >= 5 else 0
        
        # 中期趋势 (20周期)  
        medium_trend = recent_prices[-1] - recent_prices[0]
        
        # 趋势强度计算
        price_range = max(recent_prices) - min(recent_prices)
        if price_range > 0:
            trend_strength = abs(medium_trend) / price_range
        else:
            trend_strength = 0.0
        
        # 趋势方向一致性检查
        short_direction = np.sign(short_trend)
        medium_direction = np.sign(medium_trend)
        trend_consistency = short_direction * medium_direction  # 1: 一致, -1: 相反, 0: 无趋势
        
        # 奖励计算
        if trend_consistency > 0 and trend_strength > 0.3:  # 强趋势且方向一致
            # 奖励顺势交易
            trend_alignment = context.action * medium_direction
            trend_reward = trend_alignment * trend_strength * 2.0
        elif trend_strength < 0.1:  # 无明显趋势
            # 轻微惩罚大幅动作 (避免在盘整中过度交易)
            trend_reward = -abs(context.action) * 0.5
        else:  # 趋势不明确
            trend_reward = 0.0
            
        return trend_reward
    
    def _calculate_quality_reward(self, context: RewardContext) -> float:
        """计算交易质量奖励"""
        if len(self.action_history_buffer) < 3:
            return 0.0
        
        # 1. 动作一致性奖励 (避免频繁变向)
        recent_actions = list(self.action_history_buffer)[-self.quality_window:]
        action_changes = sum([abs(recent_actions[i] - recent_actions[i-1]) 
                            for i in range(1, len(recent_actions))])
        
        # 动作稳定性 (变化越少越好)
        max_possible_change = 2.0 * (len(recent_actions) - 1)  # 最大可能变化
        if max_possible_change > 0:
            stability_score = 1.0 - (action_changes / max_possible_change)
            consistency_reward = stability_score * self.consistency_weight * 2.0
        else:
            consistency_reward = 0.0
        
        # 2. 仓位大小合理性
        position_size = abs(context.action)
        if position_size < 0.1:  # 过小仓位
            size_reward = -0.2  # 轻微惩罚过于保守
        elif position_size > 0.8:  # 过大仓位
            size_reward = -0.5  # 惩罚过度冒险
        else:  # 合理仓位
            size_reward = 0.1
        
        # 3. 交易频率控制
        non_zero_actions = sum([1 for a in recent_actions if abs(a) > 0.05])
        trade_frequency = non_zero_actions / len(recent_actions)
        
        if trade_frequency > 0.8:  # 过度交易
            frequency_reward = -1.0
        elif trade_frequency < 0.2:  # 交易过少
            frequency_reward = -0.3
        else:  # 适度交易
            frequency_reward = 0.2
        
        return consistency_reward + size_reward + frequency_reward
    
    def _calculate_risk_reward(self, context: RewardContext) -> float:
        """计算风险管理奖励"""
        if not self.has_sufficient_history(context, min_steps=10):
            return 0.0
        
        portfolio_history = context.portfolio_history
        
        # 1. 回撤控制
        recent_values = portfolio_history[-20:] if len(portfolio_history) >= 20 else portfolio_history
        peak_value = max(recent_values)
        current_drawdown = (peak_value - context.portfolio_value) / peak_value if peak_value > 0 else 0
        
        if current_drawdown <= 0.02:  # 2%以内回撤
            drawdown_reward = 1.0
        elif current_drawdown <= 0.05:  # 2-5%回撤  
            drawdown_reward = 0.5
        elif current_drawdown <= 0.10:  # 5-10%回撤
            drawdown_reward = -0.5
        else:  # 超过10%回撤
            drawdown_reward = -2.0
        
        # 2. 收益稳定性
        if len(portfolio_history) >= 20:
            returns_series = self.get_returns_series(context, window=19)
            if len(returns_series) > 1:
                return_volatility = np.std(returns_series)
                
                # 低波动奖励
                if return_volatility < 0.01:  # 1%以下波动
                    stability_reward = 0.5
                elif return_volatility < 0.02:  # 1-2%波动
                    stability_reward = 0.2
                else:  # 高波动惩罚
                    stability_reward = -0.3
            else:
                stability_reward = 0.0
        else:
            stability_reward = 0.0
        
        # 3. 资金使用效率
        initial_balance = context.portfolio_info.get('initial_balance', 10000.0)
        total_return = (context.portfolio_value - initial_balance) / initial_balance
        days_passed = len(portfolio_history) / (24 * 12)  # 假设5分钟数据
        
        if days_passed > 0:
            daily_return = total_return / days_passed
            if daily_return > 0.005:  # 日收益>0.5%
                efficiency_reward = 1.0
            elif daily_return > 0.002:  # 日收益>0.2%
                efficiency_reward = 0.5
            elif daily_return > 0:  # 正收益
                efficiency_reward = 0.1
            else:  # 负收益
                efficiency_reward = -0.5
        else:
            efficiency_reward = 0.0
        
        return drawdown_reward + stability_reward + efficiency_reward
    
    def reset(self):
        """重置奖励函数状态"""
        super().reset()
        self.price_history_buffer.clear()
        self.action_history_buffer.clear()
        self.pip_profits.clear()
        self.trade_decisions.clear()
        
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.daily_pip_count = 0.0
    
    def get_forex_statistics(self) -> Dict[str, Any]:
        """获取外汇特定统计信息"""
        return {
            'daily_pip_count': self.daily_pip_count,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_wins': self.max_consecutive_wins,
            'total_trades': len(self.pip_profits),
            'avg_pip_profit': np.mean(self.pip_profits) if self.pip_profits else 0.0,
            'pip_win_rate': sum(1 for p in self.pip_profits if p > 0) / len(self.pip_profits) if self.pip_profits else 0.0
        }
    
    def get_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        forex_stats = self.get_forex_statistics()
        
        return {
            'name': self.name,
            'type': 'forex_optimized',
            'description': self.description,
            'market_compatibility': ['forex'],
            'granularity_compatibility': ['1min', '5min', '1h'],
            'parameters': {
                'pip_size': self.pip_size,
                'currency_pair': self.currency_pair,
                'daily_target_pips': self.daily_target_pips,
                'trend_window': self.trend_window,
                'quality_window': self.quality_window,
                'max_risk_per_trade': self.max_risk_per_trade,
                'consistency_weight': self.consistency_weight,
                'weights': self.weights
            },
            'complexity_score': 8,
            'category': 'forex_specialized',
            'features': [
                'pip_calculation',
                'trend_following', 
                'quality_control',
                'risk_management',
                'forex_specific_features'
            ],
            'requires_history': True,
            'min_history_steps': self.min_history_steps,
            'optimization_focus': '汇率小波动环境下的稳定盈利',
            'expected_reward_range': [-2.0, 2.0],
            'suitable_for': ['EURUSD', 'GBPUSD', 'USDJPY', '主要货币对'],
            'forex_statistics': forex_stats,
            'migrated_from': 'src.environment.rewards.forex_optimized.ForexOptimizedReward'
        }
    
    # 向后兼容方法
    def compute_reward(self, old_context):
        """兼容旧的compute_reward方法"""
        from src.rewards.migration.compatibility_mapper import CompatibilityMapper
        mapper = CompatibilityMapper()
        new_context = mapper.map_context(old_context)
        
        # 添加外汇特定元数据
        if not hasattr(new_context, 'metadata') or new_context.metadata is None:
            new_context.metadata = {}
        new_context.metadata.update({
            'pip_size': self.pip_size,
            'currency_pair': self.currency_pair
        })
        
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
            market_type='forex',
            metadata={
                'pip_size': self.pip_size,
                'currency_pair': self.currency_pair
            },
            **kwargs
        )
        return self.calculate(context)
    
    def get_reward_info(self):
        """兼容旧的get_reward_info方法"""
        return self.get_info()