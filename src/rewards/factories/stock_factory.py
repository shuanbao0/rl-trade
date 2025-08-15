"""
股票市场工厂 - Stock Market Factory

专门为股票市场优化的工厂实现，提供股票交易特有的奖励函数和环境配置。
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Type
from dataclasses import dataclass
import numpy as np
import time

from .abstract_factory import (
    AbstractRewardFactory, AbstractEnvironmentFactory, 
    AbstractComponentFactory, MasterAbstractFactory,
    FactoryConfiguration, CreationResult
)
from ..enums.market_types import MarketType
from ..enums.time_granularities import TimeGranularity
from ..enums.reward_categories import RewardCategory
# 从环境模块导入 EnvironmentFeature
from ...environment.enums.environment_features import EnvironmentFeature
from ..enums.risk_profiles import RiskProfile
from ..core.base_reward import BaseReward
from ..core.reward_context import RewardContext


# 使用统一日志系统
from ...utils.logger import get_logger
logger = get_logger(__name__)


class StockRewardFactory(AbstractRewardFactory):
    """股票奖励函数工厂"""
    
    def __init__(self, config: FactoryConfiguration):
        super().__init__(config)
        self._available_rewards = self._initialize_stock_rewards()
        self._performance_cache: Dict[str, float] = {}
    
    def _initialize_stock_rewards(self) -> Dict[str, Dict[str, Any]]:
        """初始化股票市场可用的奖励函数"""
        return {
            "simple_return": {
                "class": "SimpleReturnReward",
                "category": RewardCategory.BASIC,
                "complexity": 1,
                "best_granularities": ["5min", "15min", "1h", "1d"],
                "description": "基础收益率计算，适合股票长期投资"
            },
            "risk_adjusted": {
                "class": "RiskAdjustedReward", 
                "category": RewardCategory.RISK_ADJUSTED,
                "complexity": 3,
                "best_granularities": ["1h", "1d", "1w"],
                "description": "风险调整收益，考虑股票波动率"
            },
            "dividend_adjusted": {
                "class": "DividendAdjustedReward",
                "category": RewardCategory.BASIC,
                "complexity": 2,
                "best_granularities": ["1d", "1w"],
                "description": "包含股息收益的总回报计算"
            },
            "sector_relative": {
                "class": "SectorRelativeReward",
                "category": RewardCategory.MULTI_FACTOR,
                "complexity": 4,
                "best_granularities": ["1d", "1w"],
                "description": "相对板块表现的奖励函数"
            },
            "earnings_momentum": {
                "class": "EarningsMomentumReward",
                "category": RewardCategory.MOMENTUM,
                "complexity": 5,
                "best_granularities": ["1d", "1w"],
                "description": "基于财报动量的奖励函数"
            },
            "value_growth": {
                "class": "ValueGrowthReward",
                "category": RewardCategory.MULTI_FACTOR,
                "complexity": 6,
                "best_granularities": ["1d", "1w", "1M"],
                "description": "价值成长混合策略奖励"
            },
            "technical_pattern": {
                "class": "TechnicalPatternReward",
                "category": RewardCategory.TREND_FOLLOWING,
                "complexity": 4,
                "best_granularities": ["5min", "15min", "1h"],
                "description": "技术形态识别奖励"
            },
            "volume_price": {
                "class": "VolumePriceReward",
                "category": RewardCategory.MOMENTUM,
                "complexity": 3,
                "best_granularities": ["5min", "15min", "1h", "1d"],
                "description": "量价配合分析奖励"
            }
        }
    
    def create_reward(self, reward_type: str, **kwargs) -> CreationResult:
        """创建股票奖励函数实例"""
        if reward_type not in self._available_rewards:
            available = ", ".join(self._available_rewards.keys())
            raise ValueError(
                f"Unknown stock reward type: {reward_type}. "
                f"Available: {available}"
            )
        
        reward_config = self._available_rewards[reward_type]
        warnings = []
        
        # 检查时间粒度兼容性
        current_granularity = self.config.time_granularity.value
        best_granularities = reward_config["best_granularities"]
        
        if current_granularity not in best_granularities:
            warnings.append(
                f"Time granularity {current_granularity} may not be optimal "
                f"for {reward_type}. Best: {best_granularities}"
            )
        
        # 检查风险配置兼容性
        complexity = reward_config["complexity"]
        if self.config.risk_profile == RiskProfile.ULTRA_CONSERVATIVE and complexity > 2:
            warnings.append(
                f"High complexity reward {reward_type} may not suit "
                f"ultra conservative risk profile"
            )
        
        # 创建奖励函数（这里使用占位符实现）
        reward_instance = self._create_stock_reward_instance(reward_type, **kwargs)
        
        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(reward_type)
        
        # 记录创建历史
        creation_record = {
            "type": reward_type,
            "timestamp": np.datetime64('now'),
            "config": self.config,
            "warnings": warnings,
            "performance": performance_metrics
        }
        self._creation_history.append(creation_record)
        
        return CreationResult(
            instance=reward_instance,
            metadata={
                "reward_type": reward_type,
                "category": reward_config["category"].value,
                "complexity": complexity,
                "optimized_for": "stock_market"
            },
            warnings=warnings,
            configuration_used={
                "market_type": self.config.market_type.value,
                "time_granularity": self.config.time_granularity.value,
                "risk_profile": self.config.risk_profile.value,
                **kwargs
            },
            performance_metrics=performance_metrics
        )
    
    def _create_stock_reward_instance(self, reward_type: str, **kwargs) -> BaseReward:
        """创建具体的奖励函数实例（占位符实现）"""
        
        # 这里应该根据reward_type创建实际的奖励函数实例
        # 现在使用占位符类
        class PlaceholderStockReward(BaseReward):
            def __init__(self, reward_type: str, config: FactoryConfiguration, **kwargs):
                self.reward_type = reward_type
                self.config = config
                self.kwargs = kwargs
            
            def calculate(self, context: RewardContext) -> float:
                # 占位符实现 - 实际应该根据reward_type实现具体逻辑
                if self.reward_type == "simple_return":
                    return (context.portfolio_value - 10000) / 10000  # 假设初始资金10000
                elif self.reward_type == "risk_adjusted":
                    base_return = (context.portfolio_value - 10000) / 10000
                    # 简单的风险调整（实际应该更复杂）
                    return base_return * 0.8  # 降低风险权重
                else:
                    return 0.0
            
            def reset(self):
                pass
            
            def get_info(self) -> Dict[str, Any]:
                return {
                    "type": self.reward_type,
                    "config": self.config,
                    "kwargs": self.kwargs
                }
        
        return PlaceholderStockReward(reward_type, self.config, **kwargs)
    
    def _calculate_performance_metrics(self, reward_type: str) -> Dict[str, float]:
        """计算性能指标"""
        
        # 缓存检查
        cache_key = f"{reward_type}_{self.config.time_granularity.value}"
        if cache_key in self._performance_cache:
            return {"cached_score": self._performance_cache[cache_key]}
        
        # 基于奖励类型和配置计算评分
        reward_config = self._available_rewards[reward_type]
        base_score = 5.0  # 基础评分
        
        # 时间粒度匹配度
        if self.config.time_granularity.value in reward_config["best_granularities"]:
            granularity_bonus = 2.0
        else:
            granularity_bonus = -1.0
        
        # 复杂度适配度
        complexity = reward_config["complexity"]
        if self.config.risk_profile == RiskProfile.ULTRA_CONSERVATIVE:
            complexity_factor = max(0.5, (10 - complexity) / 10)
        elif self.config.risk_profile == RiskProfile.ULTRA_AGGRESSIVE:
            complexity_factor = min(1.5, complexity / 10 + 0.5)
        else:
            complexity_factor = 1.0
        
        # 市场特征匹配度
        feature_bonus = 0.0
        if self.config.environment_features & EnvironmentFeature.NEWS_DRIVEN:
            if reward_type in ["earnings_momentum", "sector_relative"]:
                feature_bonus += 1.0
        
        if self.config.environment_features & EnvironmentFeature.HIGH_FREQUENCY:
            if reward_type in ["technical_pattern", "volume_price"]:
                feature_bonus += 1.0
        
        # 计算最终评分
        final_score = (base_score + granularity_bonus + feature_bonus) * complexity_factor
        final_score = max(0.0, min(10.0, final_score))  # 限制在0-10范围
        
        performance_metrics = {
            "overall_score": final_score,
            "granularity_match": granularity_bonus,
            "complexity_fit": complexity_factor,
            "feature_alignment": feature_bonus,
            "estimated_sharpe": final_score / 5.0,  # 粗略估算
            "computational_cost": complexity / 10.0
        }
        
        # 缓存结果
        self._performance_cache[cache_key] = final_score
        
        return performance_metrics
    
    def get_available_rewards(self) -> List[str]:
        """获取可用的奖励函数类型"""
        return list(self._available_rewards.keys())
    
    def get_recommended_rewards(self) -> List[Tuple[str, float]]:
        """获取推荐的奖励函数及其评分"""
        recommendations = []
        
        for reward_type in self._available_rewards:
            performance = self._calculate_performance_metrics(reward_type)
            score = performance["overall_score"]
            recommendations.append((reward_type, score))
        
        # 按评分排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """验证当前配置的有效性"""
        errors = []
        
        # 检查市场类型
        if self.config.market_type != MarketType.STOCK:
            errors.append(f"Expected STOCK market, got {self.config.market_type.value}")
        
        # 检查时间粒度兼容性
        compatible_granularities = self.config.market_type.compatible_granularities
        if self.config.time_granularity.value not in compatible_granularities:
            errors.append(
                f"Time granularity {self.config.time_granularity.value} "
                f"not compatible with stock market"
            )
        
        # 检查奖励类别偏好
        stock_suitable_categories = {
            RewardCategory.BASIC, RewardCategory.RISK_ADJUSTED,
            RewardCategory.MOMENTUM, RewardCategory.MULTI_FACTOR,
            RewardCategory.TREND_FOLLOWING
        }
        
        for category in self.config.preferred_categories:
            if category not in stock_suitable_categories:
                errors.append(f"Category {category.value} may not be optimal for stocks")
        
        return len(errors) == 0, errors


class StockEnvironmentFactory(AbstractEnvironmentFactory):
    """股票环境工厂"""
    
    def create_environment(self, env_type: str = "trading", **kwargs) -> CreationResult:
        """创建股票交易环境"""
        
        # 占位符实现
        class PlaceholderStockEnvironment:
            def __init__(self, config: FactoryConfiguration, **kwargs):
                self.config = config
                self.kwargs = kwargs
                self.observation_space = None
                self.action_space = None
            
            def reset(self):
                return np.zeros(10)  # 占位符观察
            
            def step(self, action):
                return np.zeros(10), 0.0, False, {}
        
        env_instance = PlaceholderStockEnvironment(self.config, **kwargs)
        
        return CreationResult(
            instance=env_instance,
            metadata={
                "env_type": env_type,
                "market_optimized": "stock",
                "features_enabled": self.config.environment_features.name
            },
            warnings=[],
            configuration_used=self.config.__dict__,
            performance_metrics={"initialization_time": 0.001}
        )
    
    def create_observation_space(self, **kwargs):
        """创建观察空间（占位符）"""
        return None  # 应该返回gym.Space对象
    
    def create_action_space(self, **kwargs):
        """创建动作空间（占位符）"""
        return None  # 应该返回gym.Space对象
    
    def get_environment_configuration(self) -> Dict[str, Any]:
        """获取环境配置"""
        return {
            "market_type": "stock",
            "trading_hours": "market_hours",
            "commission_rate": 0.001,
            "slippage_model": "linear",
            "position_limits": True,
            "short_selling": True,
            "corporate_actions": True
        }


class StockComponentFactory(AbstractComponentFactory):
    """股票组件工厂"""
    
    def create_feature_engineer(self, **kwargs) -> CreationResult:
        """创建股票特征工程器"""
        
        class PlaceholderStockFeatureEngineer:
            def __init__(self, config: FactoryConfiguration):
                self.config = config
                self.features = [
                    "price_return", "volume_ratio", "rsi", "macd", 
                    "bollinger_bands", "pe_ratio", "sector_momentum"
                ]
            
            def engineer_features(self, data):
                return data  # 占位符
        
        instance = PlaceholderStockFeatureEngineer(self.config)
        
        return CreationResult(
            instance=instance,
            metadata={"feature_count": len(instance.features)},
            warnings=[],
            configuration_used=self.config.__dict__,
            performance_metrics={"processing_speed": 1000}  # 每秒处理行数
        )
    
    def create_risk_manager(self, **kwargs) -> CreationResult:
        """创建股票风险管理器"""
        
        class PlaceholderStockRiskManager:
            def __init__(self, config: FactoryConfiguration):
                self.config = config
                self.max_position_size = 0.1  # 10%最大仓位
                self.stop_loss = 0.05  # 5%止损
            
            def check_risk(self, portfolio, action):
                return True  # 占位符
        
        instance = PlaceholderStockRiskManager(self.config)
        
        return CreationResult(
            instance=instance,
            metadata={"risk_rules": ["position_limit", "stop_loss", "sector_limit"]},
            warnings=[],
            configuration_used=self.config.__dict__,
            performance_metrics={"check_latency": 0.001}
        )
    
    def create_data_processor(self, **kwargs) -> CreationResult:
        """创建股票数据处理器"""
        
        class PlaceholderStockDataProcessor:
            def __init__(self, config: FactoryConfiguration):
                self.config = config
            
            def process_data(self, raw_data):
                return raw_data  # 占位符
        
        instance = PlaceholderStockDataProcessor(self.config)
        
        return CreationResult(
            instance=instance,
            metadata={"processor_type": "stock_optimized"},
            warnings=[],
            configuration_used=self.config.__dict__,
            performance_metrics={"throughput": 10000}
        )


class StockMasterFactory(MasterAbstractFactory):
    """股票主工厂"""
    
    def initialize(self) -> None:
        """初始化所有子工厂"""
        if self._initialized:
            return
        
        logger.info(f"Initializing stock factory for {self.config.market_type.value}")
        
        self._reward_factory = StockRewardFactory(self.config)
        self._environment_factory = StockEnvironmentFactory(self.config)
        self._component_factory = StockComponentFactory(self.config)
        
        self._initialized = True
        logger.info("Stock factory initialization completed")
    
    def create_complete_system(self, **kwargs) -> Dict[str, CreationResult]:
        """创建完整的股票交易系统"""
        self.initialize()
        
        # 获取推荐的奖励函数
        recommendations = self._reward_factory.get_recommended_rewards()
        best_reward_type = recommendations[0][0] if recommendations else "simple_return"
        
        # 创建所有组件
        results = {}
        
        results["reward"] = self._reward_factory.create_reward(
            best_reward_type, **kwargs.get("reward_kwargs", {})
        )
        
        results["environment"] = self._environment_factory.create_environment(
            **kwargs.get("env_kwargs", {})
        )
        
        results["feature_engineer"] = self._component_factory.create_feature_engineer(
            **kwargs.get("feature_kwargs", {})
        )
        
        results["risk_manager"] = self._component_factory.create_risk_manager(
            **kwargs.get("risk_kwargs", {})
        )
        
        results["data_processor"] = self._component_factory.create_data_processor(
            **kwargs.get("data_kwargs", {})
        )
        
        logger.info(f"Created complete stock trading system with reward: {best_reward_type}")
        
        return results
    
    def get_system_compatibility_score(self) -> float:
        """获取系统兼容性评分"""
        
        # 基础兼容性检查
        base_score = 8.0  # 股票市场基础评分
        
        # 时间粒度适配性
        if self.config.time_granularity.category in ["short_term", "medium_term", "long_term"]:
            granularity_bonus = 1.0
        else:
            granularity_bonus = -1.0
        
        # 风险配置适配性
        if self.config.risk_profile in [
            RiskProfile.CONSERVATIVE, RiskProfile.MODERATE, 
            RiskProfile.BALANCED, RiskProfile.GROWTH
        ]:
            risk_bonus = 1.0
        else:
            risk_bonus = 0.0
        
        # 环境特征适配性
        feature_score = 0.0
        if self.config.environment_features & EnvironmentFeature.NEWS_DRIVEN:
            feature_score += 0.5
        if self.config.environment_features & EnvironmentFeature.EARNINGS_SENSITIVE:
            feature_score += 0.5
        if self.config.environment_features & EnvironmentFeature.CROSS_ASSET_CORRELATION:
            feature_score += 0.5
        
        final_score = base_score + granularity_bonus + risk_bonus + feature_score
        return max(0.0, min(10.0, final_score))