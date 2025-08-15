"""
环境适配器模块 - Environment Adapters Module

提供环境的自动适配、配置和优化功能。

主要组件：
- BaseEnvironmentAdapter: 环境适配器基类
- AutoEnvironmentAdapter: 自动环境适配器
- EnvironmentAdapterFactory: 环境适配器工厂

使用示例：
    ```python
    from src.environment.adapters import create_environment_adapter
    
    # 创建环境适配器
    adapter = create_environment_adapter(
        adaptation_level="intermediate",
        optimization_goal="performance"
    )
    
    # 适配环境
    if adapter.should_adapt(env, data, performance_history, current_step):
        result = adapter.adapt_environment(env, data, performance_history, current_step)
    ```
"""

from .environment_adapter import (
    # 核心类
    BaseEnvironmentAdapter,
    AutoEnvironmentAdapter,
    EnvironmentAdapterFactory,
    
    # 枚举
    AdaptationLevel,
    EnvironmentOptimizationGoal,
    
    # 数据类
    DataCharacteristics,
    EnvironmentAdaptationConfig,
    EnvironmentAdaptationResult,
    
    # 便利函数
    create_environment_adapter,
    analyze_environment_requirements
)

__all__ = [
    # 核心类
    "BaseEnvironmentAdapter",
    "AutoEnvironmentAdapter",
    "EnvironmentAdapterFactory",
    
    # 枚举
    "AdaptationLevel",
    "EnvironmentOptimizationGoal",
    
    # 数据类
    "DataCharacteristics",
    "EnvironmentAdaptationConfig", 
    "EnvironmentAdaptationResult",
    
    # 便利函数
    "create_environment_adapter",
    "analyze_environment_requirements"
]