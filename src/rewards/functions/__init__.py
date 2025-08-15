"""
奖励函数集合 - 新架构下的奖励函数模块
"""

# 基础奖励函数
from .basic.simple_return_reward import SimpleReturnReward
from .basic.profit_loss_reward import ProfitLossReward

# 高级奖励函数
from .advanced.risk_adjusted_reward import RiskAdjustedReward

# 外汇专用奖励函数
from .forex.forex_optimized_reward import ForexOptimizedReward

__all__ = [
    # 基础函数
    'SimpleReturnReward',
    'ProfitLossReward',
    
    # 高级函数
    'RiskAdjustedReward',
    
    # 外汇专用
    'ForexOptimizedReward',
]

# 函数分类映射
REWARD_CATEGORIES = {
    'basic': [
        'SimpleReturnReward',
        'ProfitLossReward',
    ],
    'advanced': [
        'RiskAdjustedReward',
    ],
    'forex': [
        'ForexOptimizedReward',
    ]
}

# 市场兼容性映射
MARKET_COMPATIBILITY = {
    'forex': [
        'SimpleReturnReward',
        'ProfitLossReward', 
        'RiskAdjustedReward',
        'ForexOptimizedReward',
    ],
    'stock': [
        'SimpleReturnReward',
        'ProfitLossReward',
        'RiskAdjustedReward',
    ],
    'crypto': [
        'SimpleReturnReward',
        'ProfitLossReward',
        'RiskAdjustedReward',
    ]
}

# 时间粒度兼容性映射
GRANULARITY_COMPATIBILITY = {
    '1min': [
        'SimpleReturnReward',
        'ProfitLossReward',
        'ForexOptimizedReward',
    ],
    '5min': [
        'SimpleReturnReward',
        'ProfitLossReward',
        'RiskAdjustedReward',
        'ForexOptimizedReward',
    ],
    '1h': [
        'SimpleReturnReward',
        'ProfitLossReward',
        'RiskAdjustedReward',
        'ForexOptimizedReward',
    ],
    '1d': [
        'SimpleReturnReward',
        'ProfitLossReward',
        'RiskAdjustedReward',
    ],
    '1w': [
        'SimpleReturnReward',
        'ProfitLossReward',
        'RiskAdjustedReward',
    ]
}

def get_available_rewards(market_type: str = None, granularity: str = None, category: str = None):
    """
    获取可用的奖励函数
    
    Args:
        market_type: 市场类型 ('forex', 'stock', 'crypto')
        granularity: 时间粒度 ('1min', '5min', '1h', '1d', '1w')
        category: 类别 ('basic', 'advanced', 'forex')
        
    Returns:
        List[str]: 符合条件的奖励函数名称列表
    """
    available = set(__all__)
    
    if market_type and market_type in MARKET_COMPATIBILITY:
        available &= set(MARKET_COMPATIBILITY[market_type])
    
    if granularity and granularity in GRANULARITY_COMPATIBILITY:
        available &= set(GRANULARITY_COMPATIBILITY[granularity])
        
    if category and category in REWARD_CATEGORIES:
        available &= set(REWARD_CATEGORIES[category])
    
    return sorted(list(available))

def get_reward_info(reward_name: str):
    """
    获取奖励函数详细信息
    
    Args:
        reward_name: 奖励函数名称
        
    Returns:
        Dict: 奖励函数信息
    """
    if reward_name not in __all__:
        raise ValueError(f"Unknown reward function: {reward_name}")
    
    # 动态导入并获取信息
    reward_class = globals()[reward_name]
    instance = reward_class()
    return instance.get_info()

# 版本信息
__version__ = "2.0.0"