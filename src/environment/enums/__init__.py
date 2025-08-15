"""
环境枚举模块 - Environment Enumerations Module

定义环境相关的枚举类型和常量。

主要组件：
- EnvironmentFeature: 环境特征枚举

使用示例：
    ```python
    from src.environment.enums import EnvironmentFeature
    
    # 定义环境特征
    features = EnvironmentFeature.HIGH_LIQUIDITY | EnvironmentFeature.LOW_SPREAD
    
    # 获取兼容的奖励函数
    compatible_rewards = features.get_compatible_rewards()
    ```
"""

from .environment_features import EnvironmentFeature, FeatureConfiguration

__all__ = [
    "EnvironmentFeature",
    "FeatureConfiguration"
]