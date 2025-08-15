"""
外汇市场工厂 - Forex Market Factory

专门为外汇市场优化的工厂实现，提供外汇交易特有的奖励函数和环境配置。
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


class ForexRewardFactory(AbstractRewardFactory):
    """外汇奖励函数工厂"""
    
    def __init__(self, config: FactoryConfiguration):
        super().__init__(config)
        self._available_rewards = self._initialize_forex_rewards()
        self._pip_values = self._initialize_pip_values()
    
    def _initialize_forex_rewards(self) -> Dict[str, Dict[str, Any]]:
        """初始化外汇市场可用的奖励函数"""
        return {
            "pip_based": {
                "class": "PipBasedReward",
                "category": RewardCategory.BASIC,
                "complexity": 2,
                "best_granularities": ["1min", "5min", "15min", "1h"],
                "description": "基于点差(pip)的收益计算，外汇市场标准"
            },
            "forex_optimized": {
                "class": "ForexOptimizedReward",
                "category": RewardCategory.MARKET_SPECIFIC,
                "complexity": 4,
                "best_granularities": ["5min", "15min", "1h"],
                "description": "外汇优化奖励，考虑利差和隔夜费用"
            },
            "spread_aware": {
                "class": "SpreadAwareReward",
                "category": RewardCategory.RISK_ADJUSTED,
                "complexity": 3,
                "best_granularities": ["1min", "5min", "15min"],
                "description": "点差感知奖励，适合高频外汇交易"
            },
            "carry_trade": {
                "class": "CarryTradeReward",
                "category": RewardCategory.MULTI_FACTOR,
                "complexity": 5,
                "best_granularities": ["1h", "4h", "1d"],
                "description": "利差交易奖励，考虑隔夜利息"
            },
            "momentum_breakout": {
                "class": "MomentumBreakoutReward",
                "category": RewardCategory.MOMENTUM,
                "complexity": 4,
                "best_granularities": ["5min", "15min", "1h"],
                "description": "动量突破奖励，适合外汇趋势交易"
            },
            "range_trading": {
                "class": "RangeTradingReward",
                "category": RewardCategory.MEAN_REVERSION,
                "complexity": 3,
                "best_granularities": ["1min", "5min", "15min"],
                "description": "区间交易奖励，适合外汇震荡行情"
            },
            "correlation_pair": {
                "class": "CorrelationPairReward",
                "category": RewardCategory.MULTI_FACTOR,
                "complexity": 6,
                "best_granularities": ["15min", "1h", "4h"],
                "description": "货币对相关性奖励"
            },
            "news_sentiment": {
                "class": "NewsSentimentReward",
                "category": RewardCategory.ML_ENHANCED,
                "complexity": 7,
                "best_granularities": ["1min", "5min", "15min"],
                "description": "新闻情绪驱动的外汇奖励"
            }
        }
    
    def _initialize_pip_values(self) -> Dict[str, float]:
        """初始化主要货币对的点值"""
        return {
            "EURUSD": 0.0001,
            "GBPUSD": 0.0001,
            "USDJPY": 0.01,
            "USDCHF": 0.0001,
            "AUDUSD": 0.0001,
            "USDCAD": 0.0001,
            "NZDUSD": 0.0001,
            "EURJPY": 0.01,
            "GBPJPY": 0.01,
            "CHFJPY": 0.01
        }
    
    def create_reward(self, reward_type: str, **kwargs) -> CreationResult:
        """创建外汇奖励函数实例"""
        if reward_type not in self._available_rewards:
            available = ", ".join(self._available_rewards.keys())
            raise ValueError(
                f"Unknown forex reward type: {reward_type}. "
                f"Available: {available}"
            )
        
        reward_config = self._available_rewards[reward_type]
        warnings = []
        
        # 检查时间粒度适配性
        current_granularity = self.config.time_granularity.value
        best_granularities = reward_config["best_granularities"]
        
        if current_granularity not in best_granularities:
            warnings.append(
                f"Time granularity {current_granularity} may not be optimal "
                f"for {reward_type}. Best: {best_granularities}"
            )
        
        # 检查高频交易适配性
        if (self.config.environment_features & EnvironmentFeature.HIGH_FREQUENCY and
            reward_type in ["carry_trade", "correlation_pair"]):
            warnings.append(
                f"Reward {reward_type} may not be suitable for high frequency trading"
            )
        
        # 检查杠杆配置
        if (self.config.environment_features & EnvironmentFeature.LEVERAGE_AVAILABLE and
            self.config.risk_profile in [RiskProfile.ULTRA_CONSERVATIVE, RiskProfile.CONSERVATIVE]):
            warnings.append(
                "High leverage may conflict with conservative risk profile"
            )
        
        # 创建奖励函数实例
        reward_instance = self._create_forex_reward_instance(reward_type, **kwargs)
        
        # 计算性能指标
        performance_metrics = self._calculate_forex_performance_metrics(reward_type)
        
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
                "complexity": reward_config["complexity"],
                "optimized_for": "forex_market",
                "pip_calculation": True
            },
            warnings=warnings,
            configuration_used={
                "market_type": self.config.market_type.value,
                "time_granularity": self.config.time_granularity.value,
                "risk_profile": self.config.risk_profile.value,
                "leverage_aware": bool(self.config.environment_features & EnvironmentFeature.LEVERAGE_AVAILABLE),
                **kwargs
            },
            performance_metrics=performance_metrics
        )
    
    def _create_forex_reward_instance(self, reward_type: str, **kwargs) -> BaseReward:
        """创建具体的外汇奖励函数实例"""
        
        class PlaceholderForexReward(BaseReward):
            def __init__(self, reward_type: str, config: FactoryConfiguration, pip_values: Dict[str, float], **kwargs):
                self.reward_type = reward_type
                self.config = config
                self.pip_values = pip_values
                self.kwargs = kwargs
                self.symbol = kwargs.get('symbol', 'EURUSD')
                self.position_size = kwargs.get('position_size', 100000)  # 标准手
            
            def calculate(self, context: RewardContext) -> float:
                """计算外汇奖励"""
                pip_value = self.pip_values.get(self.symbol, 0.0001)
                
                if self.reward_type == "pip_based":
                    # 基于点差的收益计算
                    price_change = context.current_price - context.previous_price if hasattr(context, 'previous_price') else 0
                    pip_change = price_change / pip_value
                    return pip_change * 0.1  # 标准化
                
                elif self.reward_type == "forex_optimized":
                    # 外汇优化奖励，考虑利差
                    base_reward = (context.portfolio_value - 100000) / 100000
                    
                    # 模拟利差影响（实际应该从外部数据获取）
                    if self.symbol.startswith('AUD') or self.symbol.startswith('NZD'):
                        carry_bonus = 0.001  # 高息货币奖励
                    else:
                        carry_bonus = 0.0
                    
                    return base_reward + carry_bonus
                
                elif self.reward_type == "spread_aware":
                    # 点差感知奖励
                    base_reward = (context.portfolio_value - 100000) / 100000
                    spread_penalty = 0.0001  # 模拟点差成本
                    return base_reward - spread_penalty
                
                elif self.reward_type == "momentum_breakout":
                    # 动量突破奖励
                    if hasattr(context, 'momentum_signal'):
                        momentum = context.momentum_signal
                    else:
                        momentum = np.random.normal(0, 0.1)  # 占位符
                    
                    base_reward = (context.portfolio_value - 100000) / 100000
                    return base_reward * (1 + abs(momentum))
                
                else:
                    # 默认处理
                    return (context.portfolio_value - 100000) / 100000
            
            def reset(self):
                pass
            
            def get_info(self) -> Dict[str, Any]:
                return {
                    "type": self.reward_type,
                    "symbol": self.symbol,
                    "position_size": self.position_size,
                    "pip_value": self.pip_values.get(self.symbol, 0.0001),
                    "config": self.config
                }
        
        return PlaceholderForexReward(reward_type, self.config, self._pip_values, **kwargs)
    
    def _calculate_forex_performance_metrics(self, reward_type: str) -> Dict[str, float]:
        """计算外汇奖励函数性能指标"""
        
        reward_config = self._available_rewards[reward_type]
        base_score = 6.0  # 外汇市场基础评分
        
        # 时间粒度匹配度
        granularity_match = 0.0
        if self.config.time_granularity.value in reward_config["best_granularities"]:
            granularity_match = 2.0
        elif self.config.time_granularity.is_high_frequency():
            if reward_type in ["pip_based", "spread_aware", "range_trading"]:
                granularity_match = 1.5
            else:
                granularity_match = -1.0
        
        # 24/7交易特性加分
        if self.config.time_granularity.value in ["1min", "5min", "15min"]:
            forex_bonus = 1.0
        else:
            forex_bonus = 0.5
        
        # 杠杆适配性
        leverage_factor = 1.0
        if self.config.environment_features & EnvironmentFeature.LEVERAGE_AVAILABLE:
            if reward_type in ["carry_trade", "momentum_breakout"]:
                leverage_factor = 1.2
            elif reward_type in ["spread_aware", "range_trading"]:
                leverage_factor = 1.1
        
        # 风险配置适配性
        risk_factor = 1.0
        complexity = reward_config["complexity"]
        
        if self.config.risk_profile == RiskProfile.ULTRA_CONSERVATIVE:
            risk_factor = max(0.3, (8 - complexity) / 8)
        elif self.config.risk_profile == RiskProfile.ULTRA_AGGRESSIVE:
            if reward_type in ["momentum_breakout", "news_sentiment"]:
                risk_factor = 1.3
        
        # 计算最终评分
        final_score = (base_score + granularity_match + forex_bonus) * leverage_factor * risk_factor
        final_score = max(0.0, min(10.0, final_score))
        
        return {
            "overall_score": final_score,
            "granularity_match": granularity_match,
            "forex_optimization": forex_bonus,
            "leverage_compatibility": leverage_factor,
            "risk_alignment": risk_factor,
            "estimated_sharpe": final_score / 5.0,
            "pip_accuracy": 0.95,  # 外汇特有指标
            "spread_efficiency": 0.90,
            "computational_cost": complexity / 10.0
        }
    
    def get_available_rewards(self) -> List[str]:
        """获取可用的奖励函数类型"""
        return list(self._available_rewards.keys())
    
    def get_recommended_rewards(self) -> List[Tuple[str, float]]:
        """获取推荐的奖励函数及其评分"""
        recommendations = []
        
        for reward_type in self._available_rewards:
            performance = self._calculate_forex_performance_metrics(reward_type)
            score = performance["overall_score"]
            recommendations.append((reward_type, score))
        
        # 按评分排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """验证当前配置的有效性"""
        errors = []
        
        # 检查市场类型
        if self.config.market_type != MarketType.FOREX:
            errors.append(f"Expected FOREX market, got {self.config.market_type.value}")
        
        # 检查时间粒度兼容性
        compatible_granularities = self.config.market_type.compatible_granularities
        if self.config.time_granularity.value not in compatible_granularities:
            errors.append(
                f"Time granularity {self.config.time_granularity.value} "
                f"not compatible with forex market"
            )
        
        # 检查外汇特有配置
        if not (self.config.environment_features & EnvironmentFeature.HIGH_LIQUIDITY):
            errors.append("Forex market should have HIGH_LIQUIDITY feature")
        
        # 检查风险配置合理性
        if (self.config.risk_profile == RiskProfile.ULTRA_CONSERVATIVE and
            self.config.environment_features & EnvironmentFeature.LEVERAGE_AVAILABLE):
            errors.append("Ultra conservative profile conflicts with leverage usage")
        
        return len(errors) == 0, errors


class ForexEnvironmentFactory(AbstractEnvironmentFactory):
    """外汇环境工厂"""
    
    def create_environment(self, env_type: str = "forex_trading", **kwargs) -> CreationResult:
        """创建外汇交易环境"""
        
        class PlaceholderForexEnvironment:
            def __init__(self, config: FactoryConfiguration, **kwargs):
                self.config = config
                self.kwargs = kwargs
                self.symbol = kwargs.get('symbol', 'EURUSD')
                self.leverage = kwargs.get('leverage', 100)
                self.spread = kwargs.get('spread', 1.5)  # 1.5点
                
            def reset(self):
                return np.zeros(15)  # 外汇特征更多
            
            def step(self, action):
                # 外汇特有的step逻辑
                return np.zeros(15), 0.0, False, {
                    "pip_change": 0.0,
                    "spread_cost": self.spread,
                    "swap": 0.0
                }
        
        env_instance = PlaceholderForexEnvironment(self.config, **kwargs)
        
        return CreationResult(
            instance=env_instance,
            metadata={
                "env_type": env_type,
                "market_optimized": "forex",
                "24_7_trading": True,
                "leverage_enabled": bool(self.config.environment_features & EnvironmentFeature.LEVERAGE_AVAILABLE)
            },
            warnings=[],
            configuration_used=self.config.__dict__,
            performance_metrics={"initialization_time": 0.001}
        )
    
    def create_observation_space(self, **kwargs):
        """创建外汇观察空间"""
        # 外汇观察空间应该包括：价格、成交量、技术指标、经济指标等
        return None  # 占位符
    
    def create_action_space(self, **kwargs):
        """创建外汇动作空间"""
        # 外汇动作空间：买入/卖出/平仓，仓位大小
        return None  # 占位符
    
    def get_environment_configuration(self) -> Dict[str, Any]:
        """获取外汇环境配置"""
        return {
            "market_type": "forex",
            "trading_hours": "24/5",  # 24小时，周一到周五
            "leverage_available": True,
            "short_selling": True,
            "pip_calculation": True,
            "swap_rates": True,
            "spread_model": "dynamic",
            "slippage_model": "market_impact",
            "rollover_time": "22:00_GMT"
        }


class ForexComponentFactory(AbstractComponentFactory):
    """外汇组件工厂"""
    
    def create_feature_engineer(self, **kwargs) -> CreationResult:
        """创建外汇特征工程器"""
        
        class PlaceholderForexFeatureEngineer:
            def __init__(self, config: FactoryConfiguration):
                self.config = config
                self.features = [
                    "pip_change", "spread", "bid_ask_volume", "rsi", "macd",
                    "bollinger_bands", "atr", "interest_rate_diff", 
                    "economic_calendar", "correlation_index", "volatility_smile"
                ]
            
            def engineer_features(self, data):
                # 外汇特有的特征工程
                return data
        
        instance = PlaceholderForexFeatureEngineer(self.config)
        
        return CreationResult(
            instance=instance,
            metadata={
                "feature_count": len(instance.features),
                "forex_specific": True,
                "pip_aware": True
            },
            warnings=[],
            configuration_used=self.config.__dict__,
            performance_metrics={"processing_speed": 2000}
        )
    
    def create_risk_manager(self, **kwargs) -> CreationResult:
        """创建外汇风险管理器"""
        
        class PlaceholderForexRiskManager:
            def __init__(self, config: FactoryConfiguration):
                self.config = config
                self.max_leverage = 100 if config.risk_profile == RiskProfile.AGGRESSIVE else 50
                self.max_drawdown = config.risk_profile.configuration.max_drawdown
                self.correlation_limit = 0.7  # 货币对相关性限制
            
            def check_risk(self, portfolio, action):
                # 外汇特有的风险检查
                return True
        
        instance = PlaceholderForexRiskManager(self.config)
        
        return CreationResult(
            instance=instance,
            metadata={
                "risk_rules": ["leverage_limit", "correlation_limit", "drawdown_limit", "news_filter"],
                "forex_specific": True
            },
            warnings=[],
            configuration_used=self.config.__dict__,
            performance_metrics={"check_latency": 0.0005}  # 外汇需要更快的风险检查
        )
    
    def create_data_processor(self, **kwargs) -> CreationResult:
        """创建外汇数据处理器"""
        
        class PlaceholderForexDataProcessor:
            def __init__(self, config: FactoryConfiguration):
                self.config = config
                self.pip_precision = 4  # 大多数货币对4位小数
            
            def process_data(self, raw_data):
                # 外汇数据特有处理：点差计算、隔夜费等
                return raw_data
        
        instance = PlaceholderForexDataProcessor(self.config)
        
        return CreationResult(
            instance=instance,
            metadata={"processor_type": "forex_optimized", "pip_precision": 4},
            warnings=[],
            configuration_used=self.config.__dict__,
            performance_metrics={"throughput": 50000}  # 外汇数据量更大
        )


class ForexMasterFactory(MasterAbstractFactory):
    """外汇主工厂"""
    
    def initialize(self) -> None:
        """初始化所有子工厂"""
        if self._initialized:
            return
        
        logger.info(f"Initializing forex factory for {self.config.market_type.value}")
        
        self._reward_factory = ForexRewardFactory(self.config)
        self._environment_factory = ForexEnvironmentFactory(self.config)
        self._component_factory = ForexComponentFactory(self.config)
        
        self._initialized = True
        logger.info("Forex factory initialization completed")
    
    def create_complete_system(self, **kwargs) -> Dict[str, CreationResult]:
        """创建完整的外汇交易系统"""
        self.initialize()
        
        # 获取推荐的奖励函数
        recommendations = self._reward_factory.get_recommended_rewards()
        best_reward_type = recommendations[0][0] if recommendations else "pip_based"
        
        logger.info(f"Using recommended reward: {best_reward_type}")
        
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
        
        logger.info(f"Created complete forex trading system with reward: {best_reward_type}")
        
        return results
    
    def get_system_compatibility_score(self) -> float:
        """获取系统兼容性评分"""
        
        base_score = 9.0  # 外汇市场基础评分（流动性好）
        
        # 24/7交易优势
        if self.config.time_granularity.is_high_frequency():
            trading_hours_bonus = 1.0
        else:
            trading_hours_bonus = 0.5
        
        # 杠杆使用适配性
        leverage_score = 0.0
        if self.config.environment_features & EnvironmentFeature.LEVERAGE_AVAILABLE:
            if self.config.risk_profile in [RiskProfile.GROWTH, RiskProfile.AGGRESSIVE]:
                leverage_score = 1.0
            elif self.config.risk_profile == RiskProfile.ULTRA_AGGRESSIVE:
                leverage_score = 0.5  # 过度激进可能有风险
        
        # 流动性匹配度
        liquidity_score = 0.0
        if self.config.environment_features & EnvironmentFeature.HIGH_LIQUIDITY:
            liquidity_score = 0.5
        
        final_score = base_score + trading_hours_bonus + leverage_score + liquidity_score
        return max(0.0, min(10.0, final_score))