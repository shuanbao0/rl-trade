"""
上下文转换器 - 在新旧格式之间转换RewardContext
"""

from typing import Any, Dict, Optional, Union, List
import numpy as np

from ..core.reward_context import RewardContext


class ContextConverter:
    """
    上下文转换器
    
    负责在新的RewardContext格式和旧的各种格式之间进行转换，
    确保向后兼容性。
    """
    
    def __init__(self):
        # 默认值配置
        self.defaults = {
            'portfolio_value': 10000.0,
            'action': 0.0,
            'current_price': 1.0,
            'step': 0
        }
    
    def from_dict(self, context_dict: Dict[str, Any]) -> RewardContext:
        """
        从字典格式转换为RewardContext
        
        Args:
            context_dict: 包含上下文信息的字典
            
        Returns:
            RewardContext: 新格式的上下文对象
        """
        # 标准化字段名
        normalized = self._normalize_field_names(context_dict)
        
        # 构建RewardContext参数
        context_kwargs = {}
        
        # 必需字段
        context_kwargs['portfolio_value'] = normalized.get('portfolio_value', self.defaults['portfolio_value'])
        context_kwargs['action'] = normalized.get('action', self.defaults['action'])
        context_kwargs['current_price'] = normalized.get('current_price', self.defaults['current_price'])
        context_kwargs['step'] = normalized.get('step', self.defaults['step'])
        
        # 可选字段
        if 'portfolio_history' in normalized:
            context_kwargs['portfolio_history'] = self._convert_to_numpy_array(normalized['portfolio_history'])
        
        if 'price_history' in normalized:
            context_kwargs['price_history'] = self._convert_to_numpy_array(normalized['price_history'])
            
        if 'action_history' in normalized:
            context_kwargs['action_history'] = self._convert_to_numpy_array(normalized['action_history'])
        
        if 'portfolio_info' in normalized:
            context_kwargs['portfolio_info'] = normalized['portfolio_info']
        
        if 'market_type' in normalized:
            context_kwargs['market_type'] = normalized['market_type']
            
        if 'granularity' in normalized:
            context_kwargs['granularity'] = normalized['granularity']
        
        if 'metadata' in normalized:
            context_kwargs['metadata'] = normalized['metadata']
        
        # 其他字段作为额外参数
        for key, value in normalized.items():
            if key not in context_kwargs:
                context_kwargs[key] = value
        
        return RewardContext(**context_kwargs)
    
    def from_legacy_params(self, portfolio_value: float, action: float, price: float,
                          portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> RewardContext:
        """
        从旧版本的参数格式转换为RewardContext
        
        Args:
            portfolio_value: 投资组合价值
            action: 动作
            price: 价格
            portfolio_info: 投资组合信息
            trade_info: 交易信息
            step: 步数
            **kwargs: 其他参数
            
        Returns:
            RewardContext: 新格式的上下文对象
        """
        context_kwargs = {
            'portfolio_value': portfolio_value,
            'action': action,
            'current_price': price,
            'step': step,
            'portfolio_info': portfolio_info or {}
        }
        
        # 从kwargs中提取已知字段
        known_fields = ['portfolio_history', 'price_history', 'action_history', 
                       'market_type', 'granularity', 'metadata']
        
        for field in known_fields:
            if field in kwargs:
                if field.endswith('_history'):
                    context_kwargs[field] = self._convert_to_numpy_array(kwargs[field])
                else:
                    context_kwargs[field] = kwargs[field]
        
        # 其余kwargs作为metadata
        metadata = context_kwargs.get('metadata', {})
        for key, value in kwargs.items():
            if key not in known_fields and key not in context_kwargs:
                metadata[key] = value
        
        if metadata:
            context_kwargs['metadata'] = metadata
        
        return RewardContext(**context_kwargs)
    
    def from_tensortrade_env(self, env, portfolio_value: float, action: float, 
                           price: float, step: int) -> RewardContext:
        """
        从TensorTrade环境转换为RewardContext
        
        Args:
            env: TensorTrade环境实例
            portfolio_value: 投资组合价值
            action: 动作
            price: 价格  
            step: 步数
            
        Returns:
            RewardContext: 新格式的上下文对象
        """
        context_kwargs = {
            'portfolio_value': portfolio_value,
            'action': action,
            'current_price': price,
            'step': step
        }
        
        # 尝试从环境中提取更多信息
        portfolio_info = {}
        metadata = {}
        
        # 提取投资组合信息
        if hasattr(env, 'portfolio'):
            portfolio = env.portfolio
            if hasattr(portfolio, 'initial_balance'):
                portfolio_info['initial_balance'] = portfolio.initial_balance
            if hasattr(portfolio, 'balance'):
                portfolio_info['balance'] = portfolio.balance
            if hasattr(portfolio, 'net_worth'):
                portfolio_info['net_worth'] = portfolio.net_worth
        
        # 提取历史数据
        if hasattr(env, 'observation') and hasattr(env.observation, 'features'):
            # 尝试从观察中提取价格历史
            features = env.observation.features
            if isinstance(features, np.ndarray) and len(features.shape) >= 2:
                # 假设最后一列是价格
                context_kwargs['price_history'] = features[:, -1]
        
        # 提取市场信息
        if hasattr(env, 'trading_pairs') and env.trading_pairs:
            pair = env.trading_pairs[0] if isinstance(env.trading_pairs, list) else env.trading_pairs
            if hasattr(pair, 'base') and hasattr(pair, 'quote'):
                context_kwargs['market_type'] = 'forex' if 'USD' in str(pair) else 'stock'
                metadata['trading_pair'] = f"{pair.base}/{pair.quote}"
        
        # 提取时间信息
        if hasattr(env, 'clock') and hasattr(env.clock, 'step_size'):
            granularity_map = {
                '1T': '1min',
                '5T': '5min', 
                '1H': '1h',
                '1D': '1d',
                '1W': '1w'
            }
            step_size = str(env.clock.step_size)
            context_kwargs['granularity'] = granularity_map.get(step_size, step_size)
        
        if portfolio_info:
            context_kwargs['portfolio_info'] = portfolio_info
        if metadata:
            context_kwargs['metadata'] = metadata
        
        return RewardContext(**context_kwargs)
    
    def to_dict(self, context: RewardContext) -> Dict[str, Any]:
        """
        将RewardContext转换为字典格式
        
        Args:
            context: RewardContext对象
            
        Returns:
            Dict: 字典格式的上下文数据
        """
        return context.to_dict()
    
    def to_legacy_format(self, context: RewardContext) -> tuple:
        """
        将RewardContext转换为旧版本的参数格式
        
        Args:
            context: RewardContext对象
            
        Returns:
            tuple: (portfolio_value, action, price, portfolio_info, trade_info, step)
        """
        portfolio_info = context.portfolio_info or {}
        trade_info = {}  # 新版本中没有trade_info概念，使用空字典
        
        return (
            context.portfolio_value,
            context.action,
            context.current_price,
            portfolio_info,
            trade_info,
            context.step
        )
    
    def _normalize_field_names(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化字段名称"""
        field_mappings = {
            # 投资组合相关
            'portfolio_val': 'portfolio_value',
            'portfolio_worth': 'portfolio_value',
            'net_worth': 'portfolio_value',
            'balance': 'portfolio_value',
            
            # 价格相关
            'price': 'current_price',
            'close': 'current_price',
            'close_price': 'current_price',
            
            # 步数相关
            'current_step': 'step',
            'time_step': 'step',
            'episode_step': 'step',
            
            # 其他
            'obs': 'observation',
            'observation': 'observation'
        }
        
        normalized = {}
        for key, value in data.items():
            # 使用映射或原键名
            new_key = field_mappings.get(key, key)
            normalized[new_key] = value
        
        return normalized
    
    def _convert_to_numpy_array(self, data: Any) -> Optional[np.ndarray]:
        """将数据转换为numpy数组"""
        if data is None:
            return None
        
        if isinstance(data, np.ndarray):
            return data
        
        if isinstance(data, (list, tuple)):
            try:
                return np.array(data, dtype=np.float64)
            except:
                return np.array(data)
        
        # 单个值转换为数组
        try:
            return np.array([float(data)])
        except:
            return np.array([data])
    
    def validate_context(self, context: RewardContext) -> tuple[bool, List[str]]:
        """
        验证RewardContext的有效性
        
        Args:
            context: 要验证的上下文
            
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        
        # 检查必需字段
        if context.portfolio_value is None:
            errors.append("portfolio_value is required")
        elif not isinstance(context.portfolio_value, (int, float)):
            errors.append("portfolio_value must be numeric")
        elif context.portfolio_value < 0:
            errors.append("portfolio_value cannot be negative")
        
        if context.action is None:
            errors.append("action is required")
        elif not isinstance(context.action, (int, float, np.ndarray)):
            errors.append("action must be numeric or array")
        
        if context.current_price is None:
            errors.append("current_price is required")
        elif not isinstance(context.current_price, (int, float)):
            errors.append("current_price must be numeric")
        elif context.current_price <= 0:
            errors.append("current_price must be positive")
        
        if context.step is None:
            errors.append("step is required")
        elif not isinstance(context.step, int):
            errors.append("step must be integer")
        elif context.step < 0:
            errors.append("step cannot be negative")
        
        # 检查历史数据格式
        if context.portfolio_history is not None:
            if not isinstance(context.portfolio_history, np.ndarray):
                errors.append("portfolio_history must be numpy array")
            elif len(context.portfolio_history) == 0:
                errors.append("portfolio_history cannot be empty")
        
        is_valid = len(errors) == 0
        return is_valid, errors