"""
奖励工厂模块 - Reward Factories Module

提供基于市场类型和时间粒度的智能工厂选择和创建系统。

主要组件：
- AbstractFactory: 抽象工厂基类和接口定义
- StockFactory: 股票市场专用工厂
- ForexFactory: 外汇市场专用工厂  
- FactoryManager: 工厂管理器，提供统一的工厂选择和管理

使用示例：
    ```python
    from src.rewards.factories import get_best_factory, create_trading_system
    
    # 获取股票日线工厂
    factory = get_best_factory("stock", "1d")
    
    # 创建完整交易系统
    system = create_trading_system("forex", "15min", risk_profile="moderate")
    ```
"""

from .abstract_factory import (
    # 抽象基类
    AbstractRewardFactory,
    AbstractEnvironmentFactory, 
    AbstractComponentFactory,
    MasterAbstractFactory,
    
    # 配置和结果类
    FactoryConfiguration,
    CreationResult,
    
    # 注册表
    FactoryRegistry,
    factory_registry,
    
    # 便利函数
    create_factory_configuration,
    get_factory
)

from .stock_factory import (
    StockRewardFactory,
    StockEnvironmentFactory,
    StockComponentFactory,
    StockMasterFactory
)

from .forex_factory import (
    ForexRewardFactory,
    ForexEnvironmentFactory,
    ForexComponentFactory,
    ForexMasterFactory
)

from .factory_manager import (
    FactoryManager,
    FactoryPerformanceMetrics,
    factory_manager,
    
    # 便利函数
    get_best_factory,
    create_trading_system,
    get_factory_recommendations
)


# 导出主要接口
__all__ = [
    # 抽象基类
    "AbstractRewardFactory",
    "AbstractEnvironmentFactory", 
    "AbstractComponentFactory",
    "MasterAbstractFactory",
    
    # 配置和结果
    "FactoryConfiguration",
    "CreationResult",
    "FactoryPerformanceMetrics",
    
    # 具体工厂实现
    "StockMasterFactory",
    "ForexMasterFactory",
    
    # 管理器
    "FactoryManager",
    "factory_manager",
    "factory_registry",
    
    # 便利函数
    "get_best_factory",
    "create_trading_system", 
    "create_factory_configuration",
    "get_factory_recommendations",
    "get_factory"
]


def get_available_markets():
    """获取所有支持的市场类型"""
    return factory_manager.get_statistics()["available_markets"]


def get_available_granularities():
    """获取所有支持的时间粒度"""
    return factory_manager.get_statistics()["available_granularities"]


def get_factory_info():
    """获取所有工厂的详细信息"""
    return factory_manager.list_available_factories()


# 模块级配置
VERSION = "1.0.0"
AUTHOR = "Reward Factories Team"
DESCRIPTION = "智能交易系统工厂模块"