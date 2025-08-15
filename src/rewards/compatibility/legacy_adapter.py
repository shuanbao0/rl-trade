"""
旧版奖励函数适配器 - 使旧环境能够使用新的奖励函数
"""

from typing import Any, Dict, Optional, Union
import numpy as np

from ..core.reward_context import RewardContext
from ..core.base_reward import BaseReward
from .context_converter import ContextConverter


class LegacyRewardAdapter:
    """
    旧版奖励函数适配器
    
    这个适配器允许旧版的TensorTrade环境和其他系统
    使用新的RewardContext架构下的奖励函数。
    """
    
    def __init__(self, reward_function: BaseReward):
        """
        初始化适配器
        
        Args:
            reward_function: 新版本的奖励函数实例
        """
        self.reward_function = reward_function
        self.converter = ContextConverter()
        
        # 为了完全兼容，我们需要模拟旧版本的接口
        self.initial_balance = getattr(reward_function, 'initial_balance', 10000.0)
        self.step_count = 0
        self.episode_count = 0
        
        # 历史数据存储（兼容旧系统）
        self.portfolio_history = []
        self.reward_history = []
    
    def reward(self, env) -> float:
        """
        兼容TensorTrade的reward方法
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 奖励值
        """
        try:
            # 从环境中提取信息
            portfolio_value = self._extract_portfolio_value(env)
            action = self._extract_action(env)
            price = self._extract_price(env)
            
            # 构建上下文
            context = self.converter.from_tensortrade_env(
                env=env,
                portfolio_value=portfolio_value,
                action=action,
                price=price,
                step=self.step_count
            )
            
            # 计算奖励
            reward_value = self.reward_function.calculate(context)
            
            # 更新历史记录
            self.portfolio_history.append(portfolio_value)
            self.reward_history.append(reward_value)
            self.step_count += 1
            
            return float(reward_value)
            
        except Exception as e:
            # 如果出错，返回0奖励，避免训练中断
            print(f"LegacyRewardAdapter error: {e}")
            return 0.0
    
    def get_reward(self, portfolio) -> float:
        """
        兼容TensorTrade的get_reward方法
        
        Args:
            portfolio: TensorTrade的Portfolio对象
            
        Returns:
            float: 奖励值
        """
        try:
            portfolio_value = self._extract_portfolio_value_from_portfolio(portfolio)
            
            # 构建简化的上下文
            context = RewardContext(
                portfolio_value=portfolio_value,
                action=0.0,  # 如果没有动作信息，使用默认值
                current_price=1.0,  # 如果没有价格信息，使用默认值
                step=self.step_count,
                portfolio_history=np.array(self.portfolio_history) if self.portfolio_history else None
            )
            
            reward_value = self.reward_function.calculate(context)
            
            # 更新历史记录
            self.portfolio_history.append(portfolio_value)
            self.reward_history.append(reward_value)
            self.step_count += 1
            
            return float(reward_value)
            
        except Exception as e:
            print(f"LegacyRewardAdapter.get_reward error: {e}")
            return 0.0
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        兼容旧版本的calculate_reward接口
        
        Args:
            portfolio_value: 投资组合价值
            action: 动作
            price: 价格
            portfolio_info: 投资组合信息
            trade_info: 交易信息
            step: 步数
            **kwargs: 其他参数
            
        Returns:
            float: 奖励值
        """
        context = self.converter.from_legacy_params(
            portfolio_value=portfolio_value,
            action=action,
            price=price,
            portfolio_info=portfolio_info,
            trade_info=trade_info,
            step=step,
            **kwargs
        )
        
        return self.reward_function.calculate(context)
    
    def get_reward_info(self) -> Dict[str, Any]:
        """
        兼容旧版本的get_reward_info方法
        
        Returns:
            Dict: 奖励函数信息
        """
        info = self.reward_function.get_info()
        
        # 添加兼容旧格式的字段
        legacy_info = {
            "name": info.get('name', 'unknown'),
            "description": info.get('description', ''),
            "category": info.get('category', 'basic'),
            "features": info.get('features', []),
            "type": "rl_reward",
            "initial_balance": self.initial_balance,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            # 保留新版本的完整信息
            "full_info": info
        }
        
        return legacy_info
    
    def reset(self):
        """重置适配器状态"""
        self.step_count = 0
        self.episode_count += 1
        self.portfolio_history = []
        self.reward_history = []
        
        # 重置底层奖励函数
        if hasattr(self.reward_function, 'reset'):
            self.reward_function.reset()
    
    def _extract_portfolio_value(self, env) -> float:
        """从环境中提取投资组合价值"""
        try:
            # 尝试多种可能的属性名
            if hasattr(env, 'portfolio') and hasattr(env.portfolio, 'net_worth'):
                return float(env.portfolio.net_worth)
            elif hasattr(env, 'net_worth'):
                return float(env.net_worth)
            elif hasattr(env, 'balance'):
                return float(env.balance)
            elif hasattr(env, 'portfolio_value'):
                return float(env.portfolio_value)
            else:
                return self.initial_balance
        except:
            return self.initial_balance
    
    def _extract_action(self, env) -> float:
        """从环境中提取动作"""
        try:
            if hasattr(env, 'action'):
                action = env.action
                if isinstance(action, (list, np.ndarray)) and len(action) > 0:
                    return float(action[0])
                else:
                    return float(action)
            elif hasattr(env, 'last_action'):
                return float(env.last_action)
            else:
                return 0.0
        except:
            return 0.0
    
    def _extract_price(self, env) -> float:
        """从环境中提取价格"""
        try:
            if hasattr(env, 'price'):
                return float(env.price)
            elif hasattr(env, 'current_price'):
                return float(env.current_price)
            elif hasattr(env, 'observation') and hasattr(env.observation, 'price'):
                return float(env.observation.price)
            else:
                return 1.0
        except:
            return 1.0
    
    def _extract_portfolio_value_from_portfolio(self, portfolio) -> float:
        """从Portfolio对象中提取价值"""
        try:
            if hasattr(portfolio, 'net_worth'):
                return float(portfolio.net_worth)
            elif hasattr(portfolio, 'total_value'):
                return float(portfolio.total_value)
            elif hasattr(portfolio, 'balance'):
                return float(portfolio.balance)
            else:
                return self.initial_balance
        except:
            return self.initial_balance
    
    # 以下方法提供完整的旧版本接口兼容性
    
    def __call__(self, *args, **kwargs) -> float:
        """使适配器可以直接调用"""
        if len(args) == 1:
            # 单参数调用，假设是环境
            return self.reward(args[0])
        else:
            # 多参数调用，假设是calculate_reward格式
            return self.calculate_reward(*args, **kwargs)
    
    @property
    def previous_value(self):
        """兼容属性"""
        return self.portfolio_history[-1] if self.portfolio_history else self.initial_balance
    
    def update_history(self, portfolio_value: float):
        """兼容方法"""
        self.portfolio_history.append(portfolio_value)
        self.step_count += 1