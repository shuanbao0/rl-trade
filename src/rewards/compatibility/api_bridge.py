"""
API桥接器 - 提供统一的API接口用于新旧系统集成
"""

from typing import Any, Dict, List, Optional, Union, Type
import importlib
import warnings

from ..core.base_reward import BaseReward
from ..core.reward_context import RewardContext
from .legacy_adapter import LegacyRewardAdapter
from .context_converter import ContextConverter


class APIBridge:
    """
    API桥接器
    
    提供统一的接口来创建和使用奖励函数，自动处理新旧版本之间的兼容性。
    允许用户使用旧的API方式来访问新的奖励函数。
    """
    
    def __init__(self):
        self.converter = ContextConverter()
        self._reward_registry = {}
        self._legacy_aliases = {}
        
        # 注册默认的奖励函数映射
        self._register_default_mappings()
    
    def create_reward(self, reward_type: str, legacy_mode: bool = False, **config) -> Union[BaseReward, LegacyRewardAdapter]:
        """
        创建奖励函数实例
        
        Args:
            reward_type: 奖励函数类型名称或别名
            legacy_mode: 是否返回旧版本兼容的适配器
            **config: 奖励函数配置参数
            
        Returns:
            BaseReward或LegacyRewardAdapter: 奖励函数实例
        """
        # 解析奖励类型
        reward_class = self._resolve_reward_type(reward_type)
        
        if reward_class is None:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        # 创建奖励函数实例
        reward_instance = reward_class(**config)
        
        if legacy_mode:
            # 返回兼容旧版本的适配器
            return LegacyRewardAdapter(reward_instance)
        else:
            # 返回新版本的奖励函数
            return reward_instance
    
    def create_legacy_reward(self, reward_type: str, **config) -> LegacyRewardAdapter:
        """
        创建旧版本兼容的奖励函数
        
        Args:
            reward_type: 奖励函数类型
            **config: 配置参数
            
        Returns:
            LegacyRewardAdapter: 兼容旧版本的适配器
        """
        return self.create_reward(reward_type, legacy_mode=True, **config)
    
    def get_available_rewards(self, category: str = None) -> List[str]:
        """
        获取可用的奖励函数列表
        
        Args:
            category: 可选的类别过滤
            
        Returns:
            List[str]: 可用奖励函数名称列表
        """
        available = []
        
        # 从新版本奖励函数模块获取
        try:
            from ..functions import get_available_rewards
            available.extend(get_available_rewards(category=category))
        except ImportError:
            pass
        
        # 添加注册的自定义奖励函数
        for name in self._reward_registry.keys():
            if name not in available:
                available.append(name)
        
        return sorted(available)
    
    def get_reward_info(self, reward_type: str) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Args:
            reward_type: 奖励函数类型
            
        Returns:
            Dict: 奖励函数详细信息
        """
        reward_class = self._resolve_reward_type(reward_type)
        
        if reward_class is None:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        # 创建临时实例获取信息
        temp_instance = reward_class()
        return temp_instance.get_info()
    
    def register_reward(self, name: str, reward_class: Type[BaseReward], aliases: List[str] = None):
        """
        注册自定义奖励函数
        
        Args:
            name: 奖励函数名称
            reward_class: 奖励函数类
            aliases: 别名列表
        """
        self._reward_registry[name] = reward_class
        
        if aliases:
            for alias in aliases:
                self._legacy_aliases[alias] = name
    
    def wrap_legacy_function(self, legacy_reward_func) -> BaseReward:
        """
        包装旧版本的奖励函数为新版本兼容
        
        Args:
            legacy_reward_func: 旧版本的奖励函数
            
        Returns:
            BaseReward: 包装后的奖励函数
        """
        return LegacyFunctionWrapper(legacy_reward_func, self.converter)
    
    def create_context_from_env(self, env, portfolio_value: float = None, 
                               action: float = None, price: float = None) -> RewardContext:
        """
        从环境创建RewardContext
        
        Args:
            env: 环境对象
            portfolio_value: 可选的投资组合价值
            action: 可选的动作
            price: 可选的价格
            
        Returns:
            RewardContext: 创建的上下文对象
        """
        # 如果没有提供参数，尝试从环境中提取
        if portfolio_value is None:
            portfolio_value = getattr(env, 'portfolio_value', 10000.0)
        if action is None:
            action = getattr(env, 'action', 0.0)
        if price is None:
            price = getattr(env, 'price', 1.0)
        
        step = getattr(env, 'step_count', 0)
        
        return self.converter.from_tensortrade_env(env, portfolio_value, action, price, step)
    
    def _register_default_mappings(self):
        """注册默认的奖励函数映射"""
        # 这里定义新旧奖励函数之间的映射关系
        default_mappings = {
            # 基础奖励函数别名
            'simple': 'simple_return',
            'return': 'simple_return',
            'basic_return': 'simple_return',
            
            # 盈亏相关
            'profit': 'profit_loss',
            'pnl': 'profit_loss',
            'profit_loss': 'profit_loss',
            
            # 风险调整相关
            'risk_adj': 'risk_adjusted',
            'risk_adjusted': 'risk_adjusted',
            'sharpe': 'risk_adjusted',
            
            # 外汇相关
            'forex': 'forex_optimized',
            'forex_optimized': 'forex_optimized',
            'pip_based': 'forex_optimized',
            
            # 其他常见别名
            'log_return': 'log_return',
            'volatility_adjusted': 'volatility_adjusted',
            'sortino': 'sortino_ratio',
        }
        
        self._legacy_aliases.update(default_mappings)
    
    def _resolve_reward_type(self, reward_type: str) -> Optional[Type[BaseReward]]:
        """解析奖励函数类型"""
        # 标准化输入
        reward_type = reward_type.lower().strip()
        
        # 1. 检查别名映射
        if reward_type in self._legacy_aliases:
            reward_type = self._legacy_aliases[reward_type]
        
        # 2. 检查注册的自定义奖励函数
        if reward_type in self._reward_registry:
            return self._reward_registry[reward_type]
        
        # 3. 尝试从新版本奖励函数模块导入
        return self._import_reward_class(reward_type)
    
    def _import_reward_class(self, reward_type: str) -> Optional[Type[BaseReward]]:
        """动态导入奖励函数类"""
        # 构建类名（驼峰命名）
        class_name = self._to_camel_case(reward_type) + 'Reward'
        
        # 尝试从不同模块导入
        import_paths = [
            f'..functions.basic.{reward_type}_reward',
            f'..functions.advanced.{reward_type}_reward', 
            f'..functions.forex.{reward_type}_reward',
            f'..functions.{reward_type}_reward',
        ]
        
        for import_path in import_paths:
            try:
                module = importlib.import_module(import_path, package=__name__)
                if hasattr(module, class_name):
                    return getattr(module, class_name)
            except ImportError:
                continue
        
        # 如果都失败了，尝试直接从functions模块导入
        try:
            import sys
            from .. import functions
            
            # 获取所有可能的类名
            possible_names = [
                class_name,
                reward_type.title() + 'Reward',
                reward_type.upper() + 'Reward',
                ''.join(word.title() for word in reward_type.split('_')) + 'Reward'
            ]
            
            for name in possible_names:
                if hasattr(functions, name):
                    return getattr(functions, name)
        except ImportError:
            pass
        
        return None
    
    def _to_camel_case(self, snake_str: str) -> str:
        """将下划线命名转换为驼峰命名"""
        components = snake_str.split('_')
        return ''.join(word.title() for word in components)


class LegacyFunctionWrapper(BaseReward):
    """
    旧版本奖励函数包装器
    
    将不符合新版本接口的旧奖励函数包装为新版本兼容。
    """
    
    def __init__(self, legacy_function, converter: ContextConverter, **config):
        super().__init__(**config)
        self.legacy_function = legacy_function
        self.converter = converter
        self.name = getattr(legacy_function, '__name__', 'legacy_reward')
    
    def calculate(self, context: RewardContext) -> float:
        """计算奖励值"""
        try:
            # 尝试直接调用（如果支持新格式）
            if hasattr(self.legacy_function, 'calculate'):
                return self.legacy_function.calculate(context)
            
            # 转换为旧格式并调用
            legacy_params = self.converter.to_legacy_format(context)
            
            if hasattr(self.legacy_function, 'calculate_reward'):
                return self.legacy_function.calculate_reward(*legacy_params)
            elif hasattr(self.legacy_function, 'reward'):
                # 模拟环境对象
                mock_env = type('MockEnv', (), {
                    'portfolio': type('Portfolio', (), {'net_worth': context.portfolio_value})(),
                    'action': context.action,
                    'price': context.current_price
                })()
                return self.legacy_function.reward(mock_env)
            elif callable(self.legacy_function):
                return self.legacy_function(context.portfolio_value, context.action, context.current_price)
            else:
                raise ValueError("Cannot determine how to call legacy function")
        
        except Exception as e:
            warnings.warn(f"Error calling legacy function {self.name}: {e}")
            return 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        if hasattr(self.legacy_function, 'get_reward_info'):
            return self.legacy_function.get_reward_info()
        elif hasattr(self.legacy_function, 'get_info'):
            return self.legacy_function.get_info()
        else:
            return {
                'name': self.name,
                'type': 'legacy_wrapped',
                'description': f'Wrapped legacy function: {self.legacy_function.__class__.__name__}',
                'wrapped_function': str(self.legacy_function)
            }