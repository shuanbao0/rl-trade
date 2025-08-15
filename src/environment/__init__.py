"""
交易环境模块
负责交易环境创建、动作空间定义、投资组合管理等功能

注意：奖励函数已迁移到 src.rewards 模块
"""

from .trading_environment import TradingEnvironment

# 环境适配器（新增）
from .adapters import (
    BaseEnvironmentAdapter,
    AutoEnvironmentAdapter,
    create_environment_adapter,
    analyze_environment_requirements
)

# 环境特征枚举（新增）
from .enums import EnvironmentFeature

__all__ = [
    'TradingEnvironment',
    'BaseEnvironmentAdapter',
    'AutoEnvironmentAdapter', 
    'create_environment_adapter',
    'analyze_environment_requirements',
    'EnvironmentFeature'
] 