"""
自动迁移的奖励函数: RiskAdjustedReward
原始文件: src/environment/rewards/risk_adjusted.py
市场兼容性: forex, stock, crypto
时间粒度兼容性: 5min, 1h, 1d, 1w
复杂度: 7/10
别名: risk_adj, risk_adjusted, sharpe
"""

import numpy as np
from src.rewards.core.base_reward import BaseReward, HistoryAwareRewardMixin
from src.rewards.core.reward_context import RewardContext
from typing import Optional, Dict, Any


class RiskAdjustedReward(BaseReward, HistoryAwareRewardMixin):
    """
    基于夏普比率的风险调整奖励函数
    
    计算考虑风险的收益奖励，鼓励稳定盈利，惩罚过度风险。
    采用渐进式训练策略，从基础阶段逐步提升到高级阶段。
    
    迁移自原始RiskAdjustedReward类，适配新的RewardContext架构。
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        
        # 风险参数
        self.risk_free_rate = config.get('risk_free_rate', 0.02) / 252  # 日收益率
        self.window_size = config.get('window_size', 50)
        
        # 渐进式训练配置
        self.current_stage = config.get('initial_stage', 'basic')  # basic -> intermediate -> advanced
        self.balance_scale_factor = config.get('balance_scale_factor', 1.0)
        
        # 阶段切换条件
        self.stage_thresholds = {
            "basic_to_intermediate": {"min_episodes": 100, "avg_reward_threshold": 5.0},
            "intermediate_to_advanced": {"min_episodes": 300, "avg_reward_threshold": 10.0}
        }
        
        # 历史数据配置
        self.min_history_steps = max(self.window_size, 10)
        
        # 设置名称和描述
        self.name = config.get('name', 'risk_adjusted')
        self.description = "基于夏普比率的风险调整奖励，鼓励稳定盈利，采用渐进式训练"
    
    def calculate(self, context: RewardContext) -> float:
        """
        计算风险调整奖励
        
        Args:
            context: 奖励上下文对象
            
        Returns:
            float: 计算的奖励值
        """
        # 计算基本收益率
        step_return = context.get_step_return()
        total_return = context.get_return_pct()
        
        # 根据当前阶段计算奖励
        reward = self._calculate_stage_reward(
            step_return=step_return,
            action=context.action,
            total_return=total_return,
            context=context
        )
        
        return reward
    
    def _calculate_stage_reward(self, step_return: float, action: float, 
                               total_return: float, context: RewardContext) -> float:
        """根据当前阶段计算奖励"""
        # 动态计算资金缩放因子
        initial_balance = context.portfolio_info.get('initial_balance', 10000.0)
        balance_scale = initial_balance / 10000.0
        reward_scale = 1.0 / balance_scale
        
        if self.current_stage == "basic":
            return self._calculate_basic_reward(step_return, total_return, action, reward_scale)
        elif self.current_stage == "intermediate":
            return self._calculate_intermediate_reward(step_return, total_return, action, reward_scale, context)
        else:  # advanced
            return self._calculate_advanced_reward(step_return, total_return, context, reward_scale)
    
    def _calculate_basic_reward(self, step_return: float, total_return: float, 
                               action: float, reward_scale: float) -> float:
        """基础阶段：主要基于总收益率，鼓励正收益"""
        total_return_reward = total_return * 1.0 * reward_scale
        
        # 基础风险惩罚：避免极端动作
        risk_penalty = 0.0
        if abs(action) > 0.8:
            risk_penalty = -abs(action) * 5.0 * reward_scale
        
        return total_return_reward + risk_penalty
    
    def _calculate_intermediate_reward(self, step_return: float, total_return: float,
                                     action: float, reward_scale: float, context: RewardContext) -> float:
        """中级阶段：结合步骤收益和风险调整"""
        step_reward = step_return * 100.0 * 2.0 * reward_scale  # 步骤收益转百分比
        total_reward = total_return * 0.5 * reward_scale
        
        # 中级风险调整：基于收益波动
        risk_adjustment = self._calculate_basic_risk_adjustment(context) * reward_scale
        
        return step_reward + total_reward + risk_adjustment
    
    def _calculate_advanced_reward(self, step_return: float, total_return: float,
                                  context: RewardContext, reward_scale: float) -> float:
        """高级阶段：完整的夏普比率计算"""
        if not self.has_sufficient_history(context):
            return step_return * 100.0 * 15.0 * reward_scale
        
        # 获取收益序列
        returns_series = self.get_returns_series(context)
        
        if len(returns_series) < 2:
            return step_return * 100.0 * 15.0 * reward_scale
        
        # 计算夏普比率
        sharpe_reward = self._calculate_sharpe_ratio_reward(returns_series, reward_scale)
        
        # 结合多个因子
        step_reward = step_return * 100.0 * 10.0 * reward_scale
        total_reward = total_return * 2.0 * reward_scale
        
        return step_reward + total_reward + sharpe_reward
    
    def _calculate_basic_risk_adjustment(self, context: RewardContext) -> float:
        """计算基础风险调整"""
        if not hasattr(context, 'portfolio_history') or context.portfolio_history is None:
            return 0.0
            
        if len(context.portfolio_history) < 5:
            return 0.0
        
        # 计算最近的收益率波动
        recent_values = context.portfolio_history[-5:]
        returns = []
        for i in range(1, len(recent_values)):
            if recent_values[i-1] > 0:
                ret = (recent_values[i] - recent_values[i-1]) / recent_values[i-1]
                returns.append(ret)
        
        if len(returns) > 1:
            volatility = np.std(returns)
            return -volatility * 200.0  # 惩罚高波动率
        
        return 0.0
    
    def _calculate_sharpe_ratio_reward(self, returns_series: np.ndarray, reward_scale: float) -> float:
        """计算夏普比率奖励"""
        # 计算超额收益
        excess_returns = returns_series - self.risk_free_rate
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess > 0:
            sharpe_ratio = mean_excess / std_excess
            sharpe_reward = sharpe_ratio * 100.0 * reward_scale
        else:
            sharpe_reward = mean_excess * 100.0 * reward_scale
        
        return sharpe_reward
    
    def update_stage(self, episode_count: int, avg_reward: float):
        """更新训练阶段"""
        if self.current_stage == "basic":
            threshold = self.stage_thresholds["basic_to_intermediate"]
            if (episode_count >= threshold["min_episodes"] and 
                avg_reward >= threshold["avg_reward_threshold"]):
                self.current_stage = "intermediate"
                
        elif self.current_stage == "intermediate":
            threshold = self.stage_thresholds["intermediate_to_advanced"]
            if (episode_count >= threshold["min_episodes"] and 
                avg_reward >= threshold["avg_reward_threshold"]):
                self.current_stage = "advanced"
    
    def get_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            'name': self.name,
            'type': 'risk_adjusted',
            'description': self.description,
            'market_compatibility': ['forex', 'stock', 'crypto'],
            'granularity_compatibility': ['5min', '1h', '1d', '1w'],
            'parameters': {
                'risk_free_rate': self.risk_free_rate * 252,  # 转回年化
                'window_size': self.window_size,
                'current_stage': self.current_stage,
                'balance_scale_factor': self.balance_scale_factor
            },
            'complexity_score': 7,
            'category': 'advanced',
            'features': ['sharpe_ratio', 'risk_adjustment', 'progressive_training', 'volatility_penalty'],
            'requires_history': True,
            'min_history_steps': self.min_history_steps,
            'migrated_from': 'src.environment.rewards.risk_adjusted.RiskAdjustedReward'
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