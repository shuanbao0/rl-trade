"""
工厂管理器 - Factory Manager

统一管理所有市场类型的工厂，提供自动化的工厂选择和创建服务。
"""

import logging
from typing import Dict, List, Optional, Any, Type, Union, Tuple
from dataclasses import dataclass
import importlib
import numpy as np

from .abstract_factory import (
    MasterAbstractFactory, FactoryConfiguration, CreationResult,
    factory_registry, create_factory_configuration, get_factory
)
from .stock_factory import StockMasterFactory
from .forex_factory import ForexMasterFactory
from ..enums.market_types import MarketType
from ..enums.time_granularities import TimeGranularity
from ..enums.risk_profiles import RiskProfile
# 从环境模块导入 EnvironmentFeature
from ...environment.enums.environment_features import EnvironmentFeature


# 使用统一日志系统
from ...utils.logger import get_logger
logger = get_logger(__name__)


@dataclass
class FactoryPerformanceMetrics:
    """工厂性能指标"""
    compatibility_score: float
    creation_time: float
    memory_usage: float
    success_rate: float
    recommendation_accuracy: float


class FactoryManager:
    """
    工厂管理器
    
    负责注册、选择和管理所有市场类型的工厂实现。
    """
    
    def __init__(self):
        self._registered_factories: Dict[str, Type[MasterAbstractFactory]] = {}
        self._factory_configs: Dict[str, FactoryConfiguration] = {}
        self._factory_instances: Dict[str, MasterAbstractFactory] = {}
        self._performance_metrics: Dict[str, FactoryPerformanceMetrics] = {}
        self._initialization_history: List[Dict[str, Any]] = []
        
        # 自动注册内置工厂
        self._register_builtin_factories()
    
    def _register_builtin_factories(self) -> None:
        """注册内置的工厂实现"""
        
        logger.info("Registering builtin factories...")
        
        # 注册股票工厂
        self.register_factory(
            "stock_default",
            StockMasterFactory,
            create_factory_configuration(
                market_type=MarketType.STOCK,
                time_granularity=TimeGranularity.DAY_1,
                risk_profile=RiskProfile.BALANCED
            )
        )
        
        # 注册不同时间粒度的股票工厂
        for granularity in ["5min", "15min", "1h", "1d", "1w"]:
            key = f"stock_{granularity}"
            self.register_factory(
                key,
                StockMasterFactory,
                create_factory_configuration(
                    market_type=MarketType.STOCK,
                    time_granularity=granularity,
                    risk_profile=RiskProfile.BALANCED
                )
            )
        
        # 注册外汇工厂
        self.register_factory(
            "forex_default",
            ForexMasterFactory,
            create_factory_configuration(
                market_type=MarketType.FOREX,
                time_granularity=TimeGranularity.MINUTE_15,
                risk_profile=RiskProfile.MODERATE
            )
        )
        
        # 注册不同时间粒度的外汇工厂
        for granularity in ["1min", "5min", "15min", "1h", "4h"]:
            key = f"forex_{granularity}"
            self.register_factory(
                key,
                ForexMasterFactory,
                create_factory_configuration(
                    market_type=MarketType.FOREX,
                    time_granularity=granularity,
                    risk_profile=RiskProfile.MODERATE
                )
            )
        
        # 注册通用工厂（占位符）
        self.register_factory(
            "default",
            StockMasterFactory,  # 默认使用股票工厂
            create_factory_configuration(
                market_type=MarketType.STOCK,
                time_granularity=TimeGranularity.DAY_1,
                risk_profile=RiskProfile.BALANCED
            )
        )
        
        logger.info(f"Registered {len(self._registered_factories)} builtin factories")
    
    def register_factory(
        self,
        key: str,
        factory_class: Type[MasterAbstractFactory],
        default_config: Optional[FactoryConfiguration] = None
    ) -> None:
        """注册工厂"""
        
        self._registered_factories[key] = factory_class
        
        if default_config:
            self._factory_configs[key] = default_config
            
        # 同时注册到全局注册表
        factory_registry.register_factory(key, factory_class, default_config)
        
        logger.debug(f"Registered factory: {key} -> {factory_class.__name__}")
    
    def get_factory(
        self,
        market_type: Union[str, MarketType],
        time_granularity: Union[str, TimeGranularity],
        risk_profile: Union[str, RiskProfile] = RiskProfile.BALANCED,
        **kwargs
    ) -> MasterAbstractFactory:
        """获取最适合的工厂实例"""
        
        # 标准化参数
        if isinstance(market_type, str):
            market_type = MarketType.from_string(market_type)
        if isinstance(time_granularity, str):
            time_granularity = TimeGranularity.from_string(time_granularity)
        if isinstance(risk_profile, str):
            risk_profile = RiskProfile.from_string(risk_profile)
        
        # 查找最佳工厂
        factory_key = self._find_best_factory_key(market_type, time_granularity, risk_profile)
        
        # 创建配置
        config = create_factory_configuration(
            market_type=market_type,
            time_granularity=time_granularity,
            risk_profile=risk_profile,
            **kwargs
        )
        
        # 获取或创建工厂实例
        instance_key = f"{factory_key}_{id(config)}"
        
        if instance_key not in self._factory_instances:
            factory_class = self._registered_factories[factory_key]
            instance = factory_class(config)
            self._factory_instances[instance_key] = instance
            
            # 记录创建历史
            self._initialization_history.append({
                "factory_key": factory_key,
                "instance_key": instance_key,
                "config": config,
                "timestamp": str(np.datetime64('now'))
            })
            
            logger.info(f"Created factory instance: {factory_key} for {market_type.value}")
        
        return self._factory_instances[instance_key]
    
    def _find_best_factory_key(
        self,
        market_type: MarketType,
        time_granularity: TimeGranularity,
        risk_profile: RiskProfile
    ) -> str:
        """查找最适合的工厂key"""
        
        # 构建候选列表（按优先级排序）
        candidates = []
        
        # 1. 精确匹配：市场类型 + 时间粒度
        exact_key = f"{market_type.value}_{time_granularity.value}"
        if exact_key in self._registered_factories:
            candidates.append((exact_key, 10.0))
        
        # 2. 市场类型匹配
        market_key = f"{market_type.value}_default"
        if market_key in self._registered_factories:
            candidates.append((market_key, 8.0))
        
        # 3. 市场类型 + 类似时间粒度
        similar_granularities = self._get_similar_granularities(time_granularity)
        for gran in similar_granularities:
            similar_key = f"{market_type.value}_{gran}"
            if similar_key in self._registered_factories:
                candidates.append((similar_key, 6.0))
        
        # 4. 通用默认
        if "default" in self._registered_factories:
            candidates.append(("default", 2.0))
        
        if not candidates:
            raise ValueError(f"No suitable factory found for {market_type.value}")
        
        # 返回评分最高的工厂
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_key = candidates[0][0]
        
        logger.debug(f"Selected factory {selected_key} for {market_type.value}_{time_granularity.value}")
        
        return selected_key
    
    def _get_similar_granularities(self, target: TimeGranularity) -> List[str]:
        """获取相似的时间粒度"""
        
        # 根据时间粒度类别分组
        granularity_groups = {
            "ultra_high_frequency": ["1s", "5s", "15s", "30s", "1min"],
            "high_frequency": ["1min", "5min"],
            "short_term": ["5min", "15min", "30min"],
            "medium_term": ["1h", "4h", "8h", "12h"],
            "long_term": ["1d", "3d"],
            "very_long_term": ["1w", "2w"],
            "strategic": ["1M", "3M"]
        }
        
        target_category = target.category
        similar = granularity_groups.get(target_category, [])
        
        # 移除目标值本身
        if target.value in similar:
            similar.remove(target.value)
        
        return similar
    
    def create_complete_system(
        self,
        market_type: Union[str, MarketType],
        time_granularity: Union[str, TimeGranularity],
        **kwargs
    ) -> Dict[str, CreationResult]:
        """创建完整的交易系统"""
        
        factory = self.get_factory(market_type, time_granularity, **kwargs)
        return factory.create_complete_system(**kwargs)
    
    def evaluate_factory_performance(
        self,
        factory_key: str,
        test_config: Optional[FactoryConfiguration] = None
    ) -> FactoryPerformanceMetrics:
        """评估工厂性能"""
        
        if factory_key not in self._registered_factories:
            raise ValueError(f"Factory not found: {factory_key}")
        
        # 使用测试配置或默认配置
        config = test_config or self._factory_configs.get(factory_key)
        if not config:
            raise ValueError(f"No configuration available for {factory_key}")
        
        # 创建工厂实例并测试
        factory_class = self._registered_factories[factory_key]
        
        import time
        start_time = time.time()
        
        try:
            # 创建实例
            factory = factory_class(config)
            
            # 测试兼容性
            compatibility_score = factory.get_system_compatibility_score()
            
            # 测试组件创建
            system = factory.create_complete_system()
            success_rate = len(system) / 5.0  # 期望5个组件
            
            creation_time = time.time() - start_time
            
            # 简单的内存使用估算（实际应该使用更精确的方法）
            memory_usage = len(str(system)) / 1024.0  # KB
            
            # 评估推荐准确性（占位符）
            recommendation_accuracy = 0.85  # 应该基于实际测试
            
            metrics = FactoryPerformanceMetrics(
                compatibility_score=compatibility_score,
                creation_time=creation_time,
                memory_usage=memory_usage,
                success_rate=success_rate,
                recommendation_accuracy=recommendation_accuracy
            )
            
            self._performance_metrics[factory_key] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate factory {factory_key}: {e}")
            return FactoryPerformanceMetrics(
                compatibility_score=0.0,
                creation_time=float('inf'),
                memory_usage=float('inf'),
                success_rate=0.0,
                recommendation_accuracy=0.0
            )
    
    def get_factory_recommendations(
        self,
        market_type: Union[str, MarketType],
        time_granularity: Union[str, TimeGranularity],
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """获取工厂推荐列表"""
        
        if isinstance(market_type, str):
            market_type = MarketType.from_string(market_type)
        if isinstance(time_granularity, str):
            time_granularity = TimeGranularity.from_string(time_granularity)
        
        recommendations = []
        
        for factory_key in self._registered_factories:
            try:
                # 创建测试配置
                test_config = create_factory_configuration(
                    market_type=market_type,
                    time_granularity=time_granularity
                )
                
                # 评估性能
                if factory_key not in self._performance_metrics:
                    self.evaluate_factory_performance(factory_key, test_config)
                
                metrics = self._performance_metrics[factory_key]
                score = (
                    metrics.compatibility_score * 0.4 +
                    metrics.success_rate * 30 +  # 转换为0-30分
                    (1 / max(metrics.creation_time, 0.001)) * 0.1 +
                    metrics.recommendation_accuracy * 10  # 转换为0-10分
                )
                
                recommendations.append((factory_key, score))
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {factory_key}: {e}")
                continue
        
        # 排序并返回top_n
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def list_available_factories(self) -> Dict[str, Dict[str, Any]]:
        """列出所有可用的工厂及其信息"""
        
        result = {}
        
        for factory_key, factory_class in self._registered_factories.items():
            config = self._factory_configs.get(factory_key)
            metrics = self._performance_metrics.get(factory_key)
            
            result[factory_key] = {
                "class_name": factory_class.__name__,
                "module": factory_class.__module__,
                "has_config": config is not None,
                "config_summary": {
                    "market_type": config.market_type.value if config else None,
                    "time_granularity": config.time_granularity.value if config else None,
                    "risk_profile": config.risk_profile.value if config else None
                } if config else None,
                "performance_evaluated": metrics is not None,
                "compatibility_score": metrics.compatibility_score if metrics else None
            }
        
        return result
    
    def clear_cache(self) -> None:
        """清空所有缓存"""
        self._factory_instances.clear()
        self._performance_metrics.clear()
        logger.info("Factory manager cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取工厂管理器统计信息"""
        
        return {
            "total_factories": len(self._registered_factories),
            "active_instances": len(self._factory_instances),
            "evaluated_factories": len(self._performance_metrics),
            "initialization_count": len(self._initialization_history),
            "available_markets": list(set(
                config.market_type.value for config in self._factory_configs.values()
            )),
            "available_granularities": list(set(
                config.time_granularity.value for config in self._factory_configs.values()
            ))
        }


# 全局工厂管理器实例
factory_manager = FactoryManager()


def get_best_factory(
    market_type: Union[str, MarketType],
    time_granularity: Union[str, TimeGranularity],
    **kwargs
) -> MasterAbstractFactory:
    """便利函数：获取最佳工厂实例"""
    return factory_manager.get_factory(market_type, time_granularity, **kwargs)


def create_trading_system(
    market_type: Union[str, MarketType],
    time_granularity: Union[str, TimeGranularity],
    **kwargs
) -> Dict[str, CreationResult]:
    """便利函数：创建完整的交易系统"""
    return factory_manager.create_complete_system(market_type, time_granularity, **kwargs)


def get_factory_recommendations(
    market_type: Union[str, MarketType],
    time_granularity: Union[str, TimeGranularity],
    top_n: int = 3
) -> List[Tuple[str, float]]:
    """便利函数：获取工厂推荐"""
    return factory_manager.get_factory_recommendations(market_type, time_granularity, top_n)