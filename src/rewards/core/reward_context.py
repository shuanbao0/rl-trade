"""
Reward Context - 标准化奖励计算上下文
为所有奖励函数提供统一的数据接口，解耦环境与奖励计算
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import numpy as np


@dataclass
class RewardContext:
    """
    标准化的奖励计算上下文
    
    提供奖励函数计算所需的所有数据，实现环境与奖励函数的解耦
    """
    # === 核心交易数据 ===
    portfolio_value: float                              # 当前投资组合价值
    action: Union[float, np.ndarray, Dict]              # 执行的交易动作
    current_price: float                                # 当前市场价格
    step: int                                          # 当前时间步
    timestamp: Optional[datetime] = None               # 时间戳
    
    # === 投资组合详细信息 ===
    portfolio_info: Dict[str, Any] = field(default_factory=dict)  # 投资组合详情
    trade_info: Dict[str, Any] = field(default_factory=dict)      # 交易执行信息
    
    # === 市场上下文信息 ===
    market_type: Optional[str] = None                  # 市场类型 (forex/stock/crypto)
    granularity: Optional[str] = None                  # 时间粒度 (1min/5min/1h/1d)
    market_state: Optional[str] = None                 # 市场状态 (trending/ranging/volatile)
    
    # === 历史数据 (按需提供) ===
    price_history: Optional[np.ndarray] = None         # 价格历史
    portfolio_history: Optional[np.ndarray] = None     # 投资组合价值历史
    action_history: Optional[np.ndarray] = None        # 动作历史
    
    # === 技术指标数据 ===
    technical_indicators: Dict[str, float] = field(default_factory=dict)  # 技术指标
    
    # === 扩展数据和元信息 ===
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def get_return_pct(self, initial_value: Optional[float] = None) -> float:
        """计算收益率百分比"""
        if initial_value is None:
            initial_value = self.portfolio_info.get('initial_balance', 10000.0)
        
        if initial_value <= 0:
            return 0.0
        
        return (self.portfolio_value - initial_value) / initial_value * 100
    
    def get_step_return(self) -> float:
        """计算步骤收益率"""
        if (self.portfolio_history is not None and 
            len(self.portfolio_history) >= 2):
            prev_value = self.portfolio_history[-2]
            if prev_value > 0:
                return (self.portfolio_value - prev_value) / prev_value
        return 0.0
    
    def get_price_change(self) -> float:
        """计算价格变化"""
        if (self.price_history is not None and 
            len(self.price_history) >= 2):
            return self.price_history[-1] - self.price_history[-2]
        return 0.0
    
    def get_volatility(self, window: int = 20) -> float:
        """计算历史波动率"""
        if (self.price_history is not None and 
            len(self.price_history) >= window):
            prices = self.price_history[-window:]
            returns = np.diff(prices) / prices[:-1]
            return np.std(returns)
        return 0.0
    
    def has_sufficient_history(self, min_steps: int) -> bool:
        """检查是否有足够的历史数据"""
        return (self.portfolio_history is not None and 
                len(self.portfolio_history) >= min_steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'portfolio_value': self.portfolio_value,
            'action': self.action,
            'current_price': self.current_price,
            'step': self.step,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'portfolio_info': self.portfolio_info,
            'trade_info': self.trade_info,
            'market_type': self.market_type,
            'granularity': self.granularity,
            'market_state': self.market_state,
            'technical_indicators': self.technical_indicators,
            'metadata': self.metadata,
            'return_pct': self.get_return_pct(),
            'step_return': self.get_step_return(),
            'price_change': self.get_price_change()
        }


@dataclass
class RewardResult:
    """
    奖励计算结果
    """
    reward_value: float                                # 奖励值
    components: Dict[str, float] = field(default_factory=dict)  # 奖励组件分解
    metadata: Dict[str, Any] = field(default_factory=dict)      # 计算元数据
    computation_time: float = 0.0                      # 计算耗时
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'reward_value': self.reward_value,
            'components': self.components,
            'metadata': self.metadata,
            'computation_time': self.computation_time
        }


class RewardContextBuilder:
    """
    奖励上下文构建器 - 帮助构建复杂的RewardContext
    """
    
    def __init__(self):
        self._context_data = {}
        
    def with_portfolio_value(self, value: float) -> 'RewardContextBuilder':
        """设置投资组合价值"""
        self._context_data['portfolio_value'] = value
        return self
    
    def with_action(self, action: Union[float, np.ndarray, Dict]) -> 'RewardContextBuilder':
        """设置交易动作"""
        self._context_data['action'] = action
        return self
    
    def with_price(self, price: float) -> 'RewardContextBuilder':
        """设置当前价格"""
        self._context_data['current_price'] = price
        return self
    
    def with_step(self, step: int) -> 'RewardContextBuilder':
        """设置时间步"""
        self._context_data['step'] = step
        return self
    
    def with_market_info(self, market_type: str, granularity: str) -> 'RewardContextBuilder':
        """设置市场信息"""
        self._context_data['market_type'] = market_type
        self._context_data['granularity'] = granularity
        return self
    
    def with_portfolio_info(self, info: Dict[str, Any]) -> 'RewardContextBuilder':
        """设置投资组合详细信息"""
        self._context_data['portfolio_info'] = info
        return self
    
    def with_trade_info(self, info: Dict[str, Any]) -> 'RewardContextBuilder':
        """设置交易信息"""
        self._context_data['trade_info'] = info
        return self
    
    def with_history(self, price_history: np.ndarray, 
                    portfolio_history: np.ndarray = None,
                    action_history: np.ndarray = None) -> 'RewardContextBuilder':
        """设置历史数据"""
        self._context_data['price_history'] = price_history
        if portfolio_history is not None:
            self._context_data['portfolio_history'] = portfolio_history
        if action_history is not None:
            self._context_data['action_history'] = action_history
        return self
    
    def with_technical_indicators(self, indicators: Dict[str, float]) -> 'RewardContextBuilder':
        """设置技术指标"""
        self._context_data['technical_indicators'] = indicators
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'RewardContextBuilder':
        """设置元数据"""
        self._context_data['metadata'] = metadata
        return self
    
    def build(self) -> RewardContext:
        """构建RewardContext"""
        return RewardContext(**self._context_data)


# 便捷函数
def create_simple_context(portfolio_value: float, action: float, price: float, step: int) -> RewardContext:
    """创建简单的奖励上下文"""
    return RewardContext(
        portfolio_value=portfolio_value,
        action=action,
        current_price=price,
        step=step
    )


def create_forex_context(portfolio_value: float, action: float, price: float, 
                        step: int, pip_size: float = 0.0001, **kwargs) -> RewardContext:
    """创建外汇专用上下文"""
    context = RewardContext(
        portfolio_value=portfolio_value,
        action=action,
        current_price=price,
        step=step,
        market_type='forex',
        granularity=kwargs.get('granularity', '5min')
    )
    
    # 添加外汇专用元数据
    context.metadata['pip_size'] = pip_size
    context.metadata['spread'] = kwargs.get('spread', 2)
    context.metadata['leverage'] = kwargs.get('leverage', 100)
    
    return context