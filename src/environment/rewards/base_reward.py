"""
Advanced Reward Functions Base Class
奖励函数基类定义 - 已移除TensorTrade依赖

定义了所有奖励函数必须实现的接口标准，确保统一性和可扩展性。
完全兼容Gymnasium环境。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import numpy as np


class BaseRewardScheme(ABC):
    """
    奖励函数抽象基类
    
    所有自定义奖励函数都应继承此类并实现必要的方法。
    这确保了所有奖励函数具有统一的接口和行为标准。
    """
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化奖励函数基类
        
        Args:
            initial_balance: 初始资金
            **kwargs: 其他参数
        """
        self.initial_balance = initial_balance
        self.initial_value = initial_balance
        self.previous_value = None
        self.step_count = 0
        self.episode_count = 0
        
        # 性能跟踪
        self.portfolio_history = []
        self.reward_history = []
        self.episode_rewards = []
        
        # 配置参数
        self.config = kwargs
        
    @abstractmethod
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 替代TensorTrade接口
        
        Args:
            portfolio_value: 当前投资组合价值
            action: 执行的动作
            price: 当前价格
            portfolio_info: 投资组合详细信息
            trade_info: 交易执行信息
            step: 当前步数
            **kwargs: 其他参数
            
        Returns:
            float: 计算得到的奖励值
        """
        pass
    
    # 向后兼容方法 - 保持原有17个奖励函数能正常工作
    def reward(self, env) -> float:
        """向后兼容的奖励计算方法"""
        # 将TensorTrade环境信息转换为标准格式
        try:
            portfolio_value = getattr(env, 'portfolio', {}).get('net_worth', self.initial_balance)
            action = getattr(env, 'action', 0.0)
            price = getattr(env, 'price', 100.0)
            
            # 构建兼容的信息字典
            portfolio_info = {
                'total_value': portfolio_value,
                'cash': portfolio_value * 0.5,  # 估算
                'shares': portfolio_value * 0.5,  # 估算
                'return_pct': (portfolio_value - self.initial_balance) / self.initial_balance * 100
            }
            
            trade_info = {'executed': True, 'amount': 0, 'cost': 0}
            
            return self.calculate_reward(
                portfolio_value=portfolio_value,
                action=action,
                price=price,
                portfolio_info=portfolio_info,
                trade_info=trade_info,
                step=self.step_count
            )
        except Exception:
            return 0.0
    
    def get_reward(self, portfolio) -> float:
        """TensorTrade框架兼容的get_reward方法"""
        # 简化的兼容实现
        try:
            portfolio_value = getattr(portfolio, 'net_worth', self.initial_balance)
            self.update_history(portfolio_value)
            return self.reward(portfolio)
        except Exception:
            return 0.0
    
    def update_history(self, portfolio_value: float):
        """更新历史记录"""
        self.portfolio_history.append(portfolio_value)
        self.previous_value = portfolio_value
        self.step_count += 1
    
    def get_reward_info(self) -> Dict[str, Any]:
        """获取奖励函数信息 - 用于17个奖励函数的元数据"""
        return {
            "name": self.__class__.__name__,
            "description": getattr(self, '_description', f"Advanced {self.__class__.__name__} reward function"),
            "category": getattr(self, '_category', "basic"),
            "features": getattr(self, '_features', ["reward_calculation", "performance_tracking"]),
            "type": "rl_reward",
            "initial_balance": self.initial_balance,
            "step_count": self.step_count,
            "episode_count": self.episode_count
        }
