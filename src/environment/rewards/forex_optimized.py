"""
汇率专用优化奖励函数

基于EURUSD实际交易特征设计的专门奖励函数，解决传统奖励函数在汇率交易中的问题：
1. 奖励规模适配汇率小波动特性
2. 平衡趋势跟随和风险控制
3. 鼓励高质量交易，避免过度频繁交易
4. 考虑汇率特有的时间序列特征
"""

import numpy as np
import logging
from typing import Dict, Any, List
from collections import deque
from .base_reward import BaseRewardScheme


class ForexOptimizedReward(BaseRewardScheme):
    """
    汇率优化奖励函数
    
    专门为EURUSD等主要货币对设计的奖励函数，结合：
    - 点数收益计算 (forex核心指标)
    - 趋势跟随奖励 (主要盈利模式)
    - 交易质量控制 (避免噪声交易)
    - 风险管理机制 (控制回撤)
    - 时间序列一致性 (适应汇率持续性特征)
    """
    
    def __init__(self,
                 initial_balance: float = 10000.0,
                 pip_size: float = 0.0001,           # 标准点大小
                 base_currency_pair: str = "EURUSD",  # 货币对
                 trend_window: int = 20,              # 趋势判断窗口
                 quality_window: int = 10,            # 交易质量评估窗口
                 daily_target_pips: float = 20.0,    # 日目标点数
                 max_risk_per_trade: float = 0.02,   # 单笔最大风险2%
                 consistency_weight: float = 0.3,    # 一致性权重
                 **kwargs):
        """
        初始化汇率优化奖励函数
        
        Args:
            initial_balance: 初始资金
            pip_size: 点大小 (EURUSD: 0.0001)
            base_currency_pair: 基础货币对
            trend_window: 趋势判断的回望窗口
            quality_window: 交易质量评估窗口
            daily_target_pips: 每日目标点数
            max_risk_per_trade: 单笔交易最大风险比例
            consistency_weight: 一致性奖励权重
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 汇率专用参数
        self.pip_size = pip_size
        self.currency_pair = base_currency_pair
        self.trend_window = trend_window
        self.quality_window = quality_window  
        self.daily_target_pips = daily_target_pips
        self.max_risk_per_trade = max_risk_per_trade
        self.consistency_weight = consistency_weight
        
        # 交易状态追踪
        self.price_history = deque(maxlen=max(trend_window, quality_window) + 10)
        self.action_history = deque(maxlen=quality_window)
        self.pip_profits = deque(maxlen=100)  # 点数收益历史
        self.trade_decisions = deque(maxlen=50)  # 交易决策历史
        
        # 性能指标
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.daily_pip_count = 0
        self.session_start_balance = initial_balance
        
        # 奖励权重配置 (总和=1.0)
        self.weights = {
            'pip_reward': 0.40,      # 点数收益 (核心)
            'trend_reward': 0.25,    # 趋势跟随
            'quality_reward': 0.20,  # 交易质量  
            'risk_reward': 0.15      # 风险管理
        }
        
        self.logger = logging.getLogger(f"ForexOptimizedReward_{self.currency_pair}")
        
    def calculate_reward(self, portfolio_value: float, action: float, price: float,
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        汇率优化奖励计算
        
        Args:
            portfolio_value: 当前投资组合价值
            action: 交易动作 (-1到1)
            price: 当前汇率价格
            portfolio_info: 投资组合信息
            trade_info: 交易执行信息  
            step: 当前步数
            
        Returns:
            float: 标准化奖励值 (范围约[-1, 1])
        """
        # 更新历史数据
        self.update_history(portfolio_value)
        self.price_history.append(price)
        self.action_history.append(action)
        
        # 如果数据不足，返回零奖励
        if len(self.portfolio_history) < 2 or len(self.price_history) < 2:
            return 0.0
        
        # 1. 计算点数奖励 (核心组件)
        pip_reward = self._calculate_pip_reward(price, action)
        
        # 2. 计算趋势跟随奖励
        trend_reward = self._calculate_trend_reward(price, action)
        
        # 3. 计算交易质量奖励
        quality_reward = self._calculate_quality_reward(action)
        
        # 4. 计算风险管理奖励
        risk_reward = self._calculate_risk_reward(portfolio_value)
        
        # 5. 综合奖励计算
        total_reward = (
            pip_reward * self.weights['pip_reward'] +
            trend_reward * self.weights['trend_reward'] + 
            quality_reward * self.weights['quality_reward'] +
            risk_reward * self.weights['risk_reward']
        )
        
        # 6. 奖励标准化到合理范围
        normalized_reward = np.tanh(total_reward) * 2.0  # 范围约[-2, 2]
        
        # 记录奖励历史
        self.reward_history.append(normalized_reward)
        
        return normalized_reward
    
    def _calculate_pip_reward(self, current_price: float, action: float) -> float:
        """计算点数收益奖励"""
        if len(self.price_history) < 2:
            return 0.0
            
        # 计算价格变化 (pips)
        prev_price = self.price_history[-2]
        price_change = current_price - prev_price
        pip_change = price_change / self.pip_size
        
        # 计算点数收益 (考虑交易方向)
        pip_profit = pip_change * action
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
    
    def _calculate_trend_reward(self, current_price: float, action: float) -> float:
        """计算趋势跟随奖励"""
        if len(self.price_history) < self.trend_window:
            return 0.0
            
        # 计算多周期趋势
        recent_prices = list(self.price_history)[-self.trend_window:]
        
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
            trend_alignment = action * medium_direction
            trend_reward = trend_alignment * trend_strength * 2.0
        elif trend_strength < 0.1:  # 无明显趋势
            # 轻微惩罚大幅动作 (避免在盘整中过度交易)
            trend_reward = -abs(action) * 0.5
        else:  # 趋势不明确
            trend_reward = 0.0
            
        return trend_reward
    
    def _calculate_quality_reward(self, action: float) -> float:
        """计算交易质量奖励"""
        if len(self.action_history) < 3:
            return 0.0
        
        # 1. 动作一致性奖励 (避免频繁变向)
        recent_actions = list(self.action_history)[-self.quality_window:]
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
        position_size = abs(action)
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
    
    def _calculate_risk_reward(self, portfolio_value: float) -> float:
        """计算风险管理奖励"""
        if len(self.portfolio_history) < 10:
            return 0.0
        
        # 1. 回撤控制
        recent_values = list(self.portfolio_history)[-20:]  # 最近20步
        peak_value = max(recent_values)
        current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        
        if current_drawdown <= 0.02:  # 2%以内回撤
            drawdown_reward = 1.0
        elif current_drawdown <= 0.05:  # 2-5%回撤  
            drawdown_reward = 0.5
        elif current_drawdown <= 0.10:  # 5-10%回撤
            drawdown_reward = -0.5
        else:  # 超过10%回撤
            drawdown_reward = -2.0
        
        # 2. 收益稳定性
        if len(self.portfolio_history) >= 20:
            recent_returns = [(self.portfolio_history[i] - self.portfolio_history[i-1]) / self.portfolio_history[i-1]
                            for i in range(-19, 0)]  # 最近19个收益率
            return_volatility = np.std(recent_returns) if recent_returns else 0
            
            # 低波动奖励
            if return_volatility < 0.01:  # 1%以下波动
                stability_reward = 0.5
            elif return_volatility < 0.02:  # 1-2%波动
                stability_reward = 0.2
            else:  # 高波动惩罚
                stability_reward = -0.3
        else:
            stability_reward = 0.0
        
        # 3. 资金使用效率
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        days_passed = len(self.portfolio_history) / (24 * 12)  # 假设5分钟数据
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
    
    def get_reward_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            "name": "Forex Optimized Reward",
            "description": "专为汇率交易优化的综合奖励函数",
            "category": "forex_specialized",
            "features": [
                "点数收益计算",
                "多周期趋势跟随", 
                "交易质量控制",
                "风险管理机制",
                "汇率特征适配"
            ],
            "parameters": {
                "pip_size": self.pip_size,
                "currency_pair": self.currency_pair,
                "daily_target_pips": self.daily_target_pips,
                "trend_window": self.trend_window,
                "weights": self.weights
            },
            "optimization_focus": "汇率小波动环境下的稳定盈利",
            "expected_reward_range": [-2.0, 2.0],
            "suitable_for": ["EURUSD", "GBPUSD", "USDJPY", "主要货币对"]
        }
    
    def reset(self):
        """重置奖励函数状态"""
        super().reset()
        self.price_history.clear()
        self.action_history.clear()
        self.pip_profits.clear()
        self.trade_decisions.clear()
        
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.daily_pip_count = 0
        self.session_start_balance = self.initial_balance