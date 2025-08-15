"""
工厂测试模块 - Factory Tests Module

包含对奖励工厂系统的全面测试。

测试覆盖：
- 抽象工厂基类和接口
- 具体工厂实现（股票、外汇）
- 工厂管理器功能
- 配置验证和错误处理
- 性能评估和推荐系统
"""

from .test_abstract_factory import *
from .test_factory_manager import *

__all__ = [
    "TestFactoryConfiguration",
    "TestAbstractFactories", 
    "TestFactoryRegistry",
    "TestFactoryConfigurationUtils",
    "TestFactoryManager",
    "TestFactoryManagerConvenienceFunctions",
    "TestFactoryManagerErrorHandling",
    "TestFactoryManagerCompatibility"
]