"""
自动迁移的奖励函数: SimpleReturnReward
原始文件: src/environment/rewards/simple_return.py
市场兼容性: forex, stock, crypto
时间粒度兼容性: 1min, 5min, 1h, 1d, 1w
复杂度: 2/10
别名: simple, return, basic_return
"""

from src.rewards.core.base_reward import BaseReward, SimpleRewardMixin
from src.rewards.core.reward_context import RewardContext
from typing import Optional, Dict, Any


class SimpleReturnReward(BaseReward, SimpleRewardMixin):
    """
    基于简单收益率的奖励函数
    
    这是一个简单直接的奖励函数，主要根据资产组合价值的变化来计算奖励。
    奖励 = 当前步骤的收益率百分比，简单易懂。
    
    迁移自原始SimpleReturnReward类，适配新的RewardContext架构。
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        
        # 参数配置
        self.step_weight = config.get('step_weight', 1.0)
        self.total_weight = config.get('total_weight', 0.1)
        self.return_scale = config.get('return_scale', 100.0)  # 收益率缩放
        
        # 设置名称和描述
        self.name = config.get('name', 'simple_return')
        self.description = "基于简单收益率的奖励函数，适合初学者理解和简单交易策略"
    
    def calculate(self, context: RewardContext) -> float:
        """
        计算简单收益率奖励
        
        Args:
            context: 奖励上下文对象
            
        Returns:
            float: 计算的奖励值
        """
        # 计算步骤收益率
        step_return = self.get_step_return(context)
        
        # 计算总收益率
        total_return = self.get_total_return(context)
        
        # 组合奖励
        reward = (step_return * self.step_weight * self.return_scale + 
                 total_return * self.total_weight)
        
        return reward
    
    def get_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            'name': self.name,
            'type': 'basic_return',
            'description': self.description,
            'market_compatibility': ['forex', 'stock', 'crypto'],
            'granularity_compatibility': ['1min', '5min', '1h', '1d', '1w'],
            'parameters': {
                'step_weight': self.step_weight,
                'total_weight': self.total_weight,
                'return_scale': self.return_scale
            },
            'complexity_score': 2,
            'category': 'basic',
            'features': ['step_return', 'total_return', 'percentage_based'],
            'migrated_from': 'src.environment.rewards.simple_return.SimpleReturnReward'
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