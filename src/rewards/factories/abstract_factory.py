"""
抽象工厂基类 - Abstract Factory Base Classes

定义创建奖励函数、环境和相关组件的抽象工厂接口，
使用市场类型和时间粒度作为决策因子。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Union, Tuple
from dataclasses import dataclass
import numpy as np

from ..enums.market_types import MarketType
from ..enums.time_granularities import TimeGranularity
from ..enums.reward_categories import RewardCategory
# 从环境模块导入 EnvironmentFeature
from ...environment.enums.environment_features import EnvironmentFeature
from ..enums.risk_profiles import RiskProfile
from ..core.base_reward import BaseReward
from ..core.reward_context import RewardContext


@dataclass
class FactoryConfiguration:
    """工厂配置"""
    market_type: MarketType
    time_granularity: TimeGranularity
    risk_profile: RiskProfile
    environment_features: EnvironmentFeature
    preferred_categories: List[RewardCategory]
    custom_parameters: Dict[str, Any]
    performance_mode: str = "balanced"  # "speed", "balanced", "accuracy"
    logging_enabled: bool = True
    cache_enabled: bool = True


@dataclass
class CreationResult:
    """创建结果"""
    instance: Any
    metadata: Dict[str, Any]
    warnings: List[str]
    configuration_used: Dict[str, Any]
    performance_metrics: Dict[str, float]


class AbstractRewardFactory(ABC):
    """
    抽象奖励函数工厂
    
    定义创建奖励函数的标准接口，子类需要实现具体的创建逻辑。
    """
    
    def __init__(self, config: FactoryConfiguration):
        self.config = config
        self._cache: Dict[str, Any] = {}
        self._creation_history: List[Dict] = []
    
    @abstractmethod
    def create_reward(self, reward_type: str, **kwargs) -> CreationResult:
        """创建奖励函数实例"""
        pass
    
    @abstractmethod
    def get_available_rewards(self) -> List[str]:
        """获取可用的奖励函数类型"""
        pass
    
    @abstractmethod
    def get_recommended_rewards(self) -> List[Tuple[str, float]]:
        """获取推荐的奖励函数及其评分"""
        pass
    
    @abstractmethod
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """验证当前配置的有效性"""
        pass
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
    
    def get_creation_history(self) -> List[Dict]:
        """获取创建历史"""
        return self._creation_history.copy()


class AbstractEnvironmentFactory(ABC):
    """
    抽象环境工厂
    
    定义创建交易环境的标准接口。
    """
    
    def __init__(self, config: FactoryConfiguration):
        self.config = config
        self._cache: Dict[str, Any] = {}
    
    @abstractmethod
    def create_environment(self, env_type: str = "trading", **kwargs) -> CreationResult:
        """创建环境实例"""
        pass
    
    @abstractmethod
    def create_observation_space(self, **kwargs) -> Any:
        """创建观察空间"""
        pass
    
    @abstractmethod
    def create_action_space(self, **kwargs) -> Any:
        """创建动作空间"""
        pass
    
    @abstractmethod
    def get_environment_configuration(self) -> Dict[str, Any]:
        """获取环境配置"""
        pass


class AbstractComponentFactory(ABC):
    """
    抽象组件工厂
    
    定义创建其他组件（如特征工程器、风险管理器）的接口。
    """
    
    def __init__(self, config: FactoryConfiguration):
        self.config = config
    
    @abstractmethod
    def create_feature_engineer(self, **kwargs) -> CreationResult:
        """创建特征工程器"""
        pass
    
    @abstractmethod
    def create_risk_manager(self, **kwargs) -> CreationResult:
        """创建风险管理器"""
        pass
    
    @abstractmethod
    def create_data_processor(self, **kwargs) -> CreationResult:
        """创建数据处理器"""
        pass


class MasterAbstractFactory(ABC):
    """
    主抽象工厂
    
    协调所有子工厂，提供统一的创建接口。
    """
    
    def __init__(self, config: FactoryConfiguration):
        self.config = config
        self._reward_factory: Optional[AbstractRewardFactory] = None
        self._environment_factory: Optional[AbstractEnvironmentFactory] = None
        self._component_factory: Optional[AbstractComponentFactory] = None
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """初始化所有子工厂"""
        pass
    
    @property
    def reward_factory(self) -> AbstractRewardFactory:
        """获取奖励工厂"""
        if not self._initialized:
            self.initialize()
        return self._reward_factory
    
    @property
    def environment_factory(self) -> AbstractEnvironmentFactory:
        """获取环境工厂"""
        if not self._initialized:
            self.initialize()
        return self._environment_factory
    
    @property
    def component_factory(self) -> AbstractComponentFactory:
        """获取组件工厂"""
        if not self._initialized:
            self.initialize()
        return self._component_factory
    
    @abstractmethod
    def create_complete_system(self, **kwargs) -> Dict[str, CreationResult]:
        """创建完整的交易系统"""
        pass
    
    @abstractmethod
    def get_system_compatibility_score(self) -> float:
        """获取系统兼容性评分"""
        pass
    
    def validate_system_configuration(self) -> Tuple[bool, List[str]]:
        """验证系统配置"""
        errors = []
        
        # 验证基础配置
        if not isinstance(self.config.market_type, MarketType):
            errors.append("Invalid market type")
        
        if not isinstance(self.config.time_granularity, TimeGranularity):
            errors.append("Invalid time granularity")
        
        if not isinstance(self.config.risk_profile, RiskProfile):
            errors.append("Invalid risk profile")
        
        # 验证兼容性
        if not self.config.time_granularity.is_compatible_with_market(
            self.config.market_type.value
        ):
            errors.append(
                f"Time granularity {self.config.time_granularity.value} "
                f"is not compatible with market {self.config.market_type.value}"
            )
        
        if not self.config.risk_profile.is_compatible_with_market(
            self.config.market_type.value
        ):
            errors.append(
                f"Risk profile {self.config.risk_profile.value} "
                f"is not compatible with market {self.config.market_type.value}"
            )
        
        # 验证子工厂
        if self._initialized:
            if self._reward_factory:
                valid, factory_errors = self._reward_factory.validate_configuration()
                if not valid:
                    errors.extend([f"Reward factory: {e}" for e in factory_errors])
        
        return len(errors) == 0, errors


class FactoryRegistry:
    """
    工厂注册表
    
    管理不同市场类型和配置组合的工厂实现。
    """
    
    def __init__(self):
        self._factories: Dict[str, Type[MasterAbstractFactory]] = {}
        self._configurations: Dict[str, FactoryConfiguration] = {}
        self._instances: Dict[str, MasterAbstractFactory] = {}
    
    def register_factory(
        self, 
        key: str, 
        factory_class: Type[MasterAbstractFactory],
        default_config: Optional[FactoryConfiguration] = None
    ) -> None:
        """注册工厂类"""
        self._factories[key] = factory_class
        if default_config:
            self._configurations[key] = default_config
    
    def create_factory(
        self, 
        key: str, 
        config: Optional[FactoryConfiguration] = None
    ) -> MasterAbstractFactory:
        """创建工厂实例"""
        if key not in self._factories:
            raise ValueError(f"Factory not registered: {key}")
        
        # 使用提供的配置或默认配置
        factory_config = config or self._configurations.get(key)
        if not factory_config:
            raise ValueError(f"No configuration available for factory: {key}")
        
        # 创建实例
        factory_class = self._factories[key]
        instance = factory_class(factory_config)
        
        # 缓存实例（可选）
        cache_key = f"{key}_{id(factory_config)}"
        self._instances[cache_key] = instance
        
        return instance
    
    def get_factory_for_market(
        self, 
        market_type: MarketType,
        time_granularity: TimeGranularity,
        **kwargs
    ) -> str:
        """根据市场类型和时间粒度获取最适合的工厂key"""
        
        # 构建查找key的优先级列表
        candidates = [
            f"{market_type.value}_{time_granularity.value}",  # 精确匹配
            f"{market_type.value}_*",  # 市场类型匹配
            f"*_{time_granularity.value}",  # 时间粒度匹配
            "default"  # 默认
        ]
        
        for candidate in candidates:
            if candidate in self._factories:
                return candidate
        
        raise ValueError(
            f"No suitable factory found for market {market_type.value} "
            f"and granularity {time_granularity.value}"
        )
    
    def list_available_factories(self) -> List[str]:
        """列出所有可用的工厂"""
        return list(self._factories.keys())
    
    def get_factory_info(self, key: str) -> Dict[str, Any]:
        """获取工厂信息"""
        if key not in self._factories:
            raise ValueError(f"Factory not found: {key}")
        
        factory_class = self._factories[key]
        config = self._configurations.get(key)
        
        return {
            "key": key,
            "class": factory_class.__name__,
            "module": factory_class.__module__,
            "has_default_config": config is not None,
            "config_summary": {
                "market_type": config.market_type.value if config else None,
                "time_granularity": config.time_granularity.value if config else None,
                "risk_profile": config.risk_profile.value if config else None
            } if config else None
        }


# 全局工厂注册表实例
factory_registry = FactoryRegistry()


def create_factory_configuration(
    market_type: Union[str, MarketType],
    time_granularity: Union[str, TimeGranularity],
    risk_profile: Union[str, RiskProfile] = RiskProfile.BALANCED,
    environment_features: Optional[EnvironmentFeature] = None,
    **kwargs
) -> FactoryConfiguration:
    """
    便利函数：创建工厂配置
    """
    
    # 转换参数类型
    if isinstance(market_type, str):
        market_type = MarketType.from_string(market_type)
    
    if isinstance(time_granularity, str):
        time_granularity = TimeGranularity.from_string(time_granularity)
    
    if isinstance(risk_profile, str):
        risk_profile = RiskProfile.from_string(risk_profile)
    
    # 设置默认环境特征
    if environment_features is None:
        # 根据市场类型和时间粒度设置默认特征
        features = []
        
        if time_granularity.is_high_frequency():
            features.append(EnvironmentFeature.HIGH_FREQUENCY)
            features.append(EnvironmentFeature.REAL_TIME_DATA)
        
        if market_type == MarketType.FOREX:
            features.extend([
                EnvironmentFeature.HIGH_LIQUIDITY,
                EnvironmentFeature.LOW_SPREAD,
                EnvironmentFeature.LEVERAGE_AVAILABLE
            ])
        elif market_type == MarketType.CRYPTO:
            features.extend([
                EnvironmentFeature.HIGH_VOLATILITY,
                EnvironmentFeature.MULTI_SESSION
            ])
        elif market_type == MarketType.STOCK:
            features.extend([
                EnvironmentFeature.NEWS_DRIVEN,
                EnvironmentFeature.EARNINGS_SENSITIVE
            ])
        
        # 组合所有特征
        environment_features = EnvironmentFeature(0)  # 空特征集
        for feature in features:
            environment_features |= feature
    
    # 设置默认奖励类别偏好
    preferred_categories = kwargs.pop('preferred_categories', [])
    if not preferred_categories:
        if risk_profile in [RiskProfile.ULTRA_CONSERVATIVE, RiskProfile.CONSERVATIVE]:
            preferred_categories = [RewardCategory.RISK_ADJUSTED, RewardCategory.BASIC]
        elif risk_profile in [RiskProfile.AGGRESSIVE, RiskProfile.ULTRA_AGGRESSIVE]:
            preferred_categories = [RewardCategory.MOMENTUM, RewardCategory.VOLATILITY_AWARE]
        else:
            preferred_categories = [RewardCategory.RISK_ADJUSTED, RewardCategory.TREND_FOLLOWING]
    
    return FactoryConfiguration(
        market_type=market_type,
        time_granularity=time_granularity,
        risk_profile=risk_profile,
        environment_features=environment_features,
        preferred_categories=preferred_categories,
        custom_parameters=kwargs.pop('custom_parameters', {}),
        performance_mode=kwargs.pop('performance_mode', 'balanced'),
        logging_enabled=kwargs.pop('logging_enabled', True),
        cache_enabled=kwargs.pop('cache_enabled', True)
    )


def get_factory(
    market_type: Union[str, MarketType],
    time_granularity: Union[str, TimeGranularity],
    **kwargs
) -> MasterAbstractFactory:
    """
    便利函数：获取工厂实例
    """
    
    # 创建配置
    config = create_factory_configuration(
        market_type=market_type,
        time_granularity=time_granularity,
        **kwargs
    )
    
    # 转换参数以查找工厂
    if isinstance(market_type, str):
        market_type = MarketType.from_string(market_type)
    if isinstance(time_granularity, str):
        time_granularity = TimeGranularity.from_string(time_granularity)
    
    # 获取合适的工厂key
    factory_key = factory_registry.get_factory_for_market(
        market_type, time_granularity
    )
    
    # 创建并返回工厂实例
    return factory_registry.create_factory(factory_key, config)