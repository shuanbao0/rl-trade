"""
动态参数调整器测试 - Parameter Adapter Tests

测试参数动态调整和自适应功能。
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.rewards.adapters.parameter_adapter import (
    AdaptationStrategy, MarketRegime, ParameterBounds, AdaptationConfig,
    AdaptationResult, MarketContext, BaseParameterAdapter,
    PerformanceBasedAdapter, MarketRegimeAdapter, PerformanceTracker,
    MarketRegimeDetector, ParameterAdapterFactory,
    create_parameter_adapter, detect_market_regime, get_market_context
)
from src.rewards.core.reward_context import RewardContext
from src.rewards.core.base_reward import BaseReward


class MockReward(BaseReward):
    """测试用奖励函数"""
    
    def __init__(self):
        self.lookback_period = 20
        self.smoothing_factor = 0.1
        self.risk_adjustment = 1.0
        self.volatility_window = 30
        self.threshold = 0.0
        self.alpha = 0.5
        self.beta = 1.0
        self.history = []
    
    def calculate(self, context: RewardContext) -> float:
        reward = np.random.normal(0.01, 0.1)
        self.history.append(reward)
        return reward
    
    def reset(self):
        self.history = []
    
    def get_info(self):
        return {"type": "mock_reward"}


class TestParameterBounds(unittest.TestCase):
    """测试参数边界"""
    
    def test_bounds_creation(self):
        """测试边界创建"""
        bounds = ParameterBounds(
            min_value=0.0,
            max_value=1.0,
            step_size=0.1,
            constraint_type="continuous"
        )
        
        self.assertEqual(bounds.min_value, 0.0)
        self.assertEqual(bounds.max_value, 1.0)
        self.assertEqual(bounds.step_size, 0.1)
        self.assertEqual(bounds.constraint_type, "continuous")
    
    def test_discrete_bounds(self):
        """测试离散边界"""
        bounds = ParameterBounds(
            min_value=5,
            max_value=50,
            constraint_type="discrete"
        )
        
        self.assertEqual(bounds.constraint_type, "discrete")
        self.assertIsNone(bounds.allowed_values)
    
    def test_categorical_bounds(self):
        """测试分类边界"""
        bounds = ParameterBounds(
            min_value=0,
            max_value=2,
            constraint_type="categorical",
            allowed_values=["low", "medium", "high"]
        )
        
        self.assertEqual(bounds.constraint_type, "categorical")
        self.assertEqual(len(bounds.allowed_values), 3)


class TestAdaptationConfig(unittest.TestCase):
    """测试自适应配置"""
    
    def test_config_creation(self):
        """测试配置创建"""
        bounds = {
            "lookback_period": ParameterBounds(5, 100, constraint_type="discrete"),
            "smoothing_factor": ParameterBounds(0.01, 0.5, constraint_type="continuous")
        }
        
        config = AdaptationConfig(
            adaptation_strategy=AdaptationStrategy.PERFORMANCE_BASED,
            adaptation_frequency=15,
            lookback_window=60,
            performance_threshold=0.08,
            parameter_bounds=bounds
        )
        
        self.assertEqual(config.adaptation_strategy, AdaptationStrategy.PERFORMANCE_BASED)
        self.assertEqual(config.adaptation_frequency, 15)
        self.assertEqual(config.lookback_window, 60)
        self.assertEqual(config.performance_threshold, 0.08)
        self.assertEqual(len(config.parameter_bounds), 2)
    
    def test_config_defaults(self):
        """测试配置默认值"""
        config = AdaptationConfig(
            adaptation_strategy=AdaptationStrategy.MARKET_REGIME
        )
        
        self.assertEqual(config.adaptation_frequency, 10)
        self.assertEqual(config.lookback_window, 50)
        self.assertEqual(config.performance_threshold, 0.05)
        self.assertEqual(config.adaptation_rate, 0.1)


class TestMarketContext(unittest.TestCase):
    """测试市场上下文"""
    
    def test_context_creation(self):
        """测试上下文创建"""
        context = MarketContext(
            volatility=0.15,
            trend_strength=0.8,
            momentum=0.05,
            volume_profile=1.2,
            correlation_breakdown=0.3,
            regime=MarketRegime.BULL_MARKET,
            features={"rsi": 65.0, "macd": 0.02}
        )
        
        self.assertEqual(context.volatility, 0.15)
        self.assertEqual(context.regime, MarketRegime.BULL_MARKET)
        self.assertEqual(context.features["rsi"], 65.0)
        self.assertEqual(len(context.features), 2)


class TestPerformanceTracker(unittest.TestCase):
    """测试性能追踪器"""
    
    def setUp(self):
        """设置测试"""
        self.tracker = PerformanceTracker(window_size=20)
    
    def test_tracker_initialization(self):
        """测试追踪器初始化"""
        self.assertEqual(self.tracker.window_size, 20)
        self.assertEqual(len(self.tracker.performance_history), 0)
        self.assertEqual(len(self.tracker.timestamps), 0)
    
    def test_performance_update(self):
        """测试性能更新"""
        performance_values = [0.1, 0.05, -0.02, 0.08, 0.03]
        
        for perf in performance_values:
            self.tracker.update(perf)
        
        self.assertEqual(len(self.tracker.performance_history), 5)
        self.assertEqual(len(self.tracker.timestamps), 5)
        self.assertEqual(self.tracker.performance_history[-1], 0.03)
    
    def test_window_size_limit(self):
        """测试窗口大小限制"""
        # 添加超过窗口大小的数据
        for i in range(25):
            self.tracker.update(i * 0.01)
        
        # 应该只保留最新的20个数据点
        self.assertEqual(len(self.tracker.performance_history), 20)
        self.assertEqual(self.tracker.performance_history[0], 0.05)  # (25-20) * 0.01
        self.assertEqual(self.tracker.performance_history[-1], 0.24)  # 24 * 0.01
    
    def test_recent_performance(self):
        """测试获取最近性能"""
        for i in range(15):
            self.tracker.update(i * 0.01)
        
        recent_5 = self.tracker.get_recent_performance(5)
        self.assertEqual(len(recent_5), 5)
        self.assertEqual(recent_5[-1], 0.14)  # 14 * 0.01
    
    def test_performance_statistics(self):
        """测试性能统计"""
        performance_values = [0.1, 0.05, -0.02, 0.08, 0.03, 0.06, -0.01, 0.04]
        
        for perf in performance_values:
            self.tracker.update(perf)
        
        stats = self.tracker.get_performance_statistics()
        
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("sharpe", stats)
        self.assertIn("trend", stats)
        
        self.assertAlmostEqual(stats["mean"], np.mean(performance_values), places=6)
        self.assertAlmostEqual(stats["min"], -0.02, places=6)
        self.assertAlmostEqual(stats["max"], 0.1, places=6)
    
    def test_empty_statistics(self):
        """测试空数据统计"""
        stats = self.tracker.get_performance_statistics()
        self.assertEqual(stats, {})


class TestMarketRegimeDetector(unittest.TestCase):
    """测试市场状态检测器"""
    
    def setUp(self):
        """设置测试"""
        self.detector = MarketRegimeDetector()
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        self.assertEqual(len(self.detector.regime_history), 0)
        self.assertEqual(len(self.detector.features_history), 0)
    
    def test_bull_market_detection(self):
        """测试牛市检测"""
        # 创建上升趋势价格
        np.random.seed(42)
        prices = []
        price = 100
        for i in range(50):
            price += np.random.normal(0.5, 1.0)  # 正向漂移
            prices.append(price)
        
        regime = self.detector.detect_regime(prices)
        
        # 由于强上升趋势，应该检测为牛市或恢复市
        self.assertIn(regime, [MarketRegime.BULL_MARKET, MarketRegime.RECOVERY, MarketRegime.HIGH_VOLATILITY])
    
    def test_bear_market_detection(self):
        """测试熊市检测"""
        # 创建下降趋势价格
        np.random.seed(42)
        prices = []
        price = 100
        for i in range(50):
            price += np.random.normal(-0.5, 1.0)  # 负向漂移
            prices.append(price)
        
        regime = self.detector.detect_regime(prices)
        
        # 应该检测为熊市、危机或高波动
        self.assertIn(regime, [MarketRegime.BEAR_MARKET, MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY])
    
    def test_sideways_market_detection(self):
        """测试震荡市检测"""
        # 创建无趋势价格
        np.random.seed(42)
        prices = []
        price = 100
        for i in range(50):
            price += np.random.normal(0, 0.5)  # 无漂移，低波动
            prices.append(price)
        
        regime = self.detector.detect_regime(prices)
        
        # 应该检测为震荡市或低波动
        self.assertIn(regime, [MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY])
    
    def test_high_volatility_detection(self):
        """测试高波动检测"""
        # 创建高波动价格
        np.random.seed(42)
        prices = []
        price = 100
        for i in range(50):
            price += np.random.normal(0, 5.0)  # 高波动
            prices.append(price)
        
        regime = self.detector.detect_regime(prices)
        
        # 应该检测为高波动或危机
        self.assertIn(regime, [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS])
    
    def test_insufficient_data(self):
        """测试数据不足的情况"""
        prices = [100, 101, 99, 102]  # 少于20个数据点
        
        regime = self.detector.detect_regime(prices)
        
        # 数据不足时应该返回默认值
        self.assertEqual(regime, MarketRegime.SIDEWAYS)
    
    def test_market_context_creation(self):
        """测试市场上下文创建"""
        # 创建测试价格序列
        np.random.seed(42)
        prices = [100 + np.random.normal(0, 2) for _ in range(30)]
        volumes = [1000 + np.random.normal(0, 200) for _ in range(30)]
        
        context = self.detector.get_market_context(prices, volumes)
        
        self.assertIsInstance(context, MarketContext)
        self.assertIsInstance(context.volatility, float)
        self.assertIsInstance(context.trend_strength, float)
        self.assertIsInstance(context.momentum, float)
        self.assertIsInstance(context.volume_profile, float)
        self.assertIsInstance(context.regime, MarketRegime)
        
        # 检查合理范围
        self.assertTrue(0 <= context.trend_strength <= 1)
        self.assertTrue(context.volatility >= 0)
    
    def test_insufficient_data_context(self):
        """测试数据不足时的上下文"""
        prices = [100, 101, 99]  # 少于10个数据点
        
        context = self.detector.get_market_context(prices)
        
        self.assertEqual(context.volatility, 0.0)
        self.assertEqual(context.trend_strength, 0.0)
        self.assertEqual(context.momentum, 0.0)
        self.assertEqual(context.volume_profile, 1.0)


class TestPerformanceBasedAdapter(unittest.TestCase):
    """测试基于性能的适配器"""
    
    def setUp(self):
        """设置测试"""
        bounds = {
            "lookback_period": ParameterBounds(5, 100, constraint_type="discrete"),
            "smoothing_factor": ParameterBounds(0.01, 0.5, constraint_type="continuous"),
            "risk_adjustment": ParameterBounds(0.1, 3.0, constraint_type="continuous")
        }
        
        config = AdaptationConfig(
            adaptation_strategy=AdaptationStrategy.PERFORMANCE_BASED,
            adaptation_frequency=5,
            lookback_window=20,
            performance_threshold=0.02,
            parameter_bounds=bounds
        )
        
        self.adapter = PerformanceBasedAdapter(config)
        self.reward_function = MockReward()
    
    def test_adapter_initialization(self):
        """测试适配器初始化"""
        self.assertEqual(self.adapter.config.adaptation_strategy, AdaptationStrategy.PERFORMANCE_BASED)
        self.assertEqual(self.adapter.config.adaptation_frequency, 5)
        self.assertEqual(len(self.adapter.adaptation_history), 0)
    
    def test_should_adapt_frequency(self):
        """测试适配频率检查"""
        market_context = MarketContext(
            volatility=0.1, trend_strength=0.5, momentum=0.02,
            volume_profile=1.0, correlation_breakdown=0.1,
            regime=MarketRegime.SIDEWAYS
        )
        
        performance_history = [0.01] * 25
        
        # 步骤不是频率的倍数
        should_adapt = self.adapter.should_adapt(market_context, performance_history, 7)
        self.assertFalse(should_adapt)
        
        # 步骤是频率的倍数，并且有足够数据和性能下降
        performance_history_declining = [0.05] * 15 + [-0.01] * 10  # 创建性能下降
        should_adapt = self.adapter.should_adapt(market_context, performance_history_declining, 10)
        self.assertTrue(should_adapt)
    
    def test_should_adapt_insufficient_data(self):
        """测试数据不足时不适配"""
        market_context = MarketContext(
            volatility=0.1, trend_strength=0.5, momentum=0.02,
            volume_profile=1.0, correlation_breakdown=0.1,
            regime=MarketRegime.SIDEWAYS
        )
        
        performance_history = [0.01] * 10  # 少于lookback_window
        
        should_adapt = self.adapter.should_adapt(market_context, performance_history, 10)
        self.assertFalse(should_adapt)
    
    def test_should_adapt_performance_decline(self):
        """测试性能下降时适配"""
        market_context = MarketContext(
            volatility=0.1, trend_strength=0.5, momentum=0.02,
            volume_profile=1.0, correlation_breakdown=0.1,
            regime=MarketRegime.SIDEWAYS
        )
        
        # 创建性能下降的历史
        performance_history = [0.05] * 10 + [-0.01] * 15  # 前期好，后期差
        
        should_adapt = self.adapter.should_adapt(market_context, performance_history, 25)
        self.assertTrue(should_adapt)
    
    def test_parameter_extraction(self):
        """测试参数提取"""
        parameters = self.adapter._extract_current_parameters(self.reward_function)
        
        self.assertIn("lookback_period", parameters)
        self.assertIn("smoothing_factor", parameters)
        self.assertIn("risk_adjustment", parameters)
        self.assertEqual(parameters["lookback_period"], 20)
        self.assertEqual(parameters["smoothing_factor"], 0.1)
    
    def test_performance_pattern_analysis(self):
        """测试性能模式分析"""
        # 创建下降趋势的性能
        performance_history = [0.1 - i * 0.005 for i in range(20)]
        
        analysis = self.adapter._analyze_performance_pattern(performance_history)
        
        self.assertIn("trend", analysis)
        self.assertIn("slope", analysis)
        self.assertIn("volatility", analysis)
        self.assertIn("stability", analysis)
        
        self.assertEqual(analysis["trend"], "declining")
        self.assertTrue(analysis["slope"] < 0)
    
    def test_parameter_adjustment(self):
        """测试参数调整"""
        current_parameters = {
            "lookback_period": 20,
            "smoothing_factor": 0.1,
            "risk_adjustment": 1.0
        }
        
        performance_analysis = {
            "trend": "declining",
            "volatility": 0.15,
            "stability": 0.6
        }
        
        market_context = MarketContext(
            volatility=0.2, trend_strength=0.3, momentum=-0.02,
            volume_profile=0.8, correlation_breakdown=0.4,
            regime=MarketRegime.HIGH_VOLATILITY
        )
        
        new_parameters = self.adapter._adjust_parameters_for_performance(
            current_parameters, performance_analysis, market_context
        )
        
        # 性能下降时应该调整参数
        self.assertNotEqual(new_parameters, current_parameters)
        
        # 检查特定调整逻辑
        if "lookback_period" in new_parameters:
            # 下降趋势时应该减少回顾期
            self.assertLessEqual(new_parameters["lookback_period"], current_parameters["lookback_period"])
    
    def test_parameter_validation(self):
        """测试参数验证"""
        # 有效参数
        valid_parameters = {
            "lookback_period": 25,
            "smoothing_factor": 0.15,
            "risk_adjustment": 1.5
        }
        
        is_valid, errors = self.adapter.validate_parameters(valid_parameters, self.reward_function)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # 无效参数
        invalid_parameters = {
            "lookback_period": 150,  # 超出边界
            "smoothing_factor": 0.8,  # 超出边界
            "risk_adjustment": 1.5
        }
        
        is_valid, errors = self.adapter.validate_parameters(invalid_parameters, self.reward_function)
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)
    
    def test_adaptation_execution(self):
        """测试适配执行"""
        np.random.seed(42)
        
        market_context = MarketContext(
            volatility=0.15, trend_strength=0.6, momentum=0.03,
            volume_profile=1.2, correlation_breakdown=0.2,
            regime=MarketRegime.BULL_MARKET
        )
        
        # 创建性能下降历史
        performance_history = [0.08 - i * 0.003 for i in range(25)]
        
        result = self.adapter.adapt_parameters(
            self.reward_function, market_context, performance_history, 25
        )
        
        self.assertIsInstance(result, AdaptationResult)
        self.assertIsInstance(result.old_parameters, dict)
        self.assertIsInstance(result.new_parameters, dict)
        self.assertIsInstance(result.performance_improvement, float)
        self.assertIsInstance(result.confidence_score, float)
        
        # 检查记录
        self.assertEqual(len(self.adapter.adaptation_history), 1)
        
        # 检查参数确实应用到了奖励函数
        for param_name, param_value in result.new_parameters.items():
            if hasattr(self.reward_function, param_name):
                self.assertEqual(getattr(self.reward_function, param_name), param_value)


class TestMarketRegimeAdapter(unittest.TestCase):
    """测试市场状态适配器"""
    
    def setUp(self):
        """设置测试"""
        config = AdaptationConfig(
            adaptation_strategy=AdaptationStrategy.MARKET_REGIME,
            adaptation_frequency=8
        )
        
        self.adapter = MarketRegimeAdapter(config)
        self.reward_function = MockReward()
    
    def test_adapter_initialization(self):
        """测试适配器初始化"""
        self.assertEqual(self.adapter.config.adaptation_strategy, AdaptationStrategy.MARKET_REGIME)
        self.assertIsNotNone(self.adapter.regime_parameters)
        
        # 检查是否有预定义的状态参数
        self.assertIn(MarketRegime.BULL_MARKET, self.adapter.regime_parameters)
        self.assertIn(MarketRegime.BEAR_MARKET, self.adapter.regime_parameters)
        self.assertIn(MarketRegime.HIGH_VOLATILITY, self.adapter.regime_parameters)
    
    def test_should_adapt_regime_change(self):
        """测试状态变化时适配"""
        market_context = MarketContext(
            volatility=0.1, trend_strength=0.8, momentum=0.05,
            volume_profile=1.0, correlation_breakdown=0.1,
            regime=MarketRegime.BULL_MARKET
        )
        
        performance_history = [0.01] * 20
        
        # 第一次适配（没有历史）
        should_adapt = self.adapter.should_adapt(market_context, performance_history, 8)
        self.assertTrue(should_adapt)
        
        # 执行一次适配
        result = self.adapter.adapt_parameters(
            self.reward_function, market_context, performance_history, 8
        )
        
        # 相同状态，不应该再适配
        should_adapt = self.adapter.should_adapt(market_context, performance_history, 16)
        self.assertFalse(should_adapt)
        
        # 状态变化，应该适配
        new_context = MarketContext(
            volatility=0.3, trend_strength=0.2, momentum=-0.03,
            volume_profile=0.8, correlation_breakdown=0.6,
            regime=MarketRegime.HIGH_VOLATILITY
        )
        
        should_adapt = self.adapter.should_adapt(new_context, performance_history, 24)
        self.assertTrue(should_adapt)
    
    def test_regime_parameter_application(self):
        """测试状态参数应用"""
        bull_market_context = MarketContext(
            volatility=0.1, trend_strength=0.8, momentum=0.05,
            volume_profile=1.0, correlation_breakdown=0.1,
            regime=MarketRegime.BULL_MARKET
        )
        
        performance_history = [0.01] * 20
        
        result = self.adapter.adapt_parameters(
            self.reward_function, bull_market_context, performance_history, 8
        )
        
        # 检查是否应用了牛市参数（考虑微调影响）
        bull_params = self.adapter.regime_parameters[MarketRegime.BULL_MARKET]
        for param_name, expected_value in bull_params.items():
            if param_name in result.new_parameters:
                # 由于有市场强度微调，值可能有显著变化，特别是risk_adjustment参数
                actual_value = result.new_parameters[param_name]
                if isinstance(expected_value, (int, float)):
                    if param_name == "risk_adjustment":
                        # risk_adjustment参数会受到波动率微调的大幅影响，检查方向而非精确值
                        self.assertIsInstance(actual_value, (int, float))
                        self.assertGreater(actual_value, 0)  # 应该为正值
                    else:
                        # 其他参数检查合理范围
                        self.assertAlmostEqual(actual_value, expected_value, delta=abs(expected_value * 0.5))
    
    def test_market_strength_fine_tuning(self):
        """测试市场强度微调"""
        base_parameters = {
            "lookback_period": 20,
            "risk_adjustment": 1.0
        }
        
        # 强趋势市场
        strong_trend_context = MarketContext(
            volatility=0.15, trend_strength=0.9, momentum=0.08,
            volume_profile=1.2, correlation_breakdown=0.1,
            regime=MarketRegime.BULL_MARKET
        )
        
        tuned_params = self.adapter._fine_tune_for_market_strength(
            base_parameters, strong_trend_context
        )
        
        # 强趋势时回顾期应该减少
        if "lookback_period" in tuned_params:
            self.assertLess(tuned_params["lookback_period"], base_parameters["lookback_period"])
        
        # 高波动时风险调整应该增加
        if "risk_adjustment" in tuned_params:
            vol_factor = strong_trend_context.volatility / 0.02
            expected_adjustment = base_parameters["risk_adjustment"] * vol_factor
            self.assertAlmostEqual(tuned_params["risk_adjustment"], expected_adjustment, places=2)


class TestParameterAdapterFactory(unittest.TestCase):
    """测试参数适配器工厂"""
    
    def test_create_performance_based_adapter(self):
        """测试创建基于性能的适配器"""
        config = AdaptationConfig(
            adaptation_strategy=AdaptationStrategy.PERFORMANCE_BASED
        )
        
        adapter = ParameterAdapterFactory.create_adapter(
            AdaptationStrategy.PERFORMANCE_BASED, config
        )
        
        self.assertIsInstance(adapter, PerformanceBasedAdapter)
        self.assertEqual(adapter.config.adaptation_strategy, AdaptationStrategy.PERFORMANCE_BASED)
    
    def test_create_market_regime_adapter(self):
        """测试创建市场状态适配器"""
        config = AdaptationConfig(
            adaptation_strategy=AdaptationStrategy.MARKET_REGIME
        )
        
        adapter = ParameterAdapterFactory.create_adapter(
            AdaptationStrategy.MARKET_REGIME, config
        )
        
        self.assertIsInstance(adapter, MarketRegimeAdapter)
        self.assertEqual(adapter.config.adaptation_strategy, AdaptationStrategy.MARKET_REGIME)
    
    def test_unsupported_strategy(self):
        """测试不支持的策略"""
        config = AdaptationConfig(
            adaptation_strategy=AdaptationStrategy.REINFORCEMENT_LEARNING
        )
        
        with self.assertRaises(ValueError):
            ParameterAdapterFactory.create_adapter(
                AdaptationStrategy.REINFORCEMENT_LEARNING, config
            )


class TestConvenienceFunctions(unittest.TestCase):
    """测试便利函数"""
    
    def test_create_parameter_adapter_function(self):
        """测试创建参数适配器便利函数"""
        adapter = create_parameter_adapter(
            strategy="performance_based",
            adaptation_frequency=12,
            lookback_window=40,
            performance_threshold=0.03
        )
        
        self.assertIsInstance(adapter, PerformanceBasedAdapter)
        self.assertEqual(adapter.config.adaptation_frequency, 12)
        self.assertEqual(adapter.config.lookback_window, 40)
        self.assertEqual(adapter.config.performance_threshold, 0.03)
    
    def test_create_adapter_with_bounds(self):
        """测试带参数边界的适配器创建"""
        parameter_bounds = {
            "lookback_period": {
                "min_value": 10,
                "max_value": 80,
                "constraint_type": "discrete"
            },
            "smoothing_factor": {
                "min_value": 0.05,
                "max_value": 0.3,
                "constraint_type": "continuous"
            }
        }
        
        adapter = create_parameter_adapter(
            strategy="market_regime",
            parameter_bounds=parameter_bounds
        )
        
        self.assertIsInstance(adapter, MarketRegimeAdapter)
        self.assertEqual(len(adapter.config.parameter_bounds), 2)
        
        # 检查边界设置
        bounds = adapter.config.parameter_bounds["lookback_period"]
        self.assertEqual(bounds.min_value, 10)
        self.assertEqual(bounds.max_value, 80)
        self.assertEqual(bounds.constraint_type, "discrete")
    
    def test_detect_market_regime_function(self):
        """测试检测市场状态便利函数"""
        # 创建上升趋势价格
        np.random.seed(42)
        prices = [100 + i * 0.5 + np.random.normal(0, 1) for i in range(30)]
        
        regime = detect_market_regime(prices)
        
        self.assertIsInstance(regime, MarketRegime)
    
    def test_get_market_context_function(self):
        """测试获取市场上下文便利函数"""
        np.random.seed(42)
        prices = [100 + np.random.normal(0, 2) for _ in range(25)]
        volumes = [1000 + np.random.normal(0, 100) for _ in range(25)]
        
        context = get_market_context(prices, volumes)
        
        self.assertIsInstance(context, MarketContext)
        self.assertIsInstance(context.regime, MarketRegime)
        self.assertTrue(context.volatility >= 0)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_performance_history(self):
        """测试空性能历史"""
        adapter = create_parameter_adapter("performance_based")
        
        market_context = MarketContext(
            volatility=0.1, trend_strength=0.5, momentum=0.0,
            volume_profile=1.0, correlation_breakdown=0.1,
            regime=MarketRegime.SIDEWAYS
        )
        
        should_adapt = adapter.should_adapt(market_context, [], 10)
        self.assertFalse(should_adapt)
    
    def test_extreme_market_conditions(self):
        """测试极端市场条件"""
        adapter = create_parameter_adapter("market_regime")
        
        crisis_context = MarketContext(
            volatility=0.8, trend_strength=0.9, momentum=-0.3,
            volume_profile=0.3, correlation_breakdown=0.9,
            regime=MarketRegime.CRISIS
        )
        
        performance_history = [-0.1, -0.2, -0.15, -0.05] * 10
        reward_function = MockReward()
        
        # 应该能够处理极端条件
        should_adapt = adapter.should_adapt(crisis_context, performance_history, 8)
        if should_adapt:
            result = adapter.adapt_parameters(
                reward_function, crisis_context, performance_history, 8
            )
            self.assertIsInstance(result, AdaptationResult)
    
    def test_invalid_parameter_handling(self):
        """测试无效参数处理"""
        bounds = {
            "lookback_period": ParameterBounds(5, 50, constraint_type="discrete")
        }
        
        adapter = create_parameter_adapter(
            "performance_based",
            parameter_bounds={"lookback_period": {
                "min_value": 5, "max_value": 50, "constraint_type": "discrete"
            }}
        )
        
        # 创建会产生无效参数的场景
        reward_function = MockReward()
        reward_function.lookback_period = 100  # 超出边界
        
        invalid_params = {"lookback_period": 100}
        is_valid, errors = adapter.validate_parameters(invalid_params, reward_function)
        
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)
    
    def test_adaptation_statistics(self):
        """测试适配统计"""
        adapter = create_parameter_adapter("performance_based")
        
        # 执行几次适配
        market_context = MarketContext(
            volatility=0.1, trend_strength=0.5, momentum=0.0,
            volume_profile=1.0, correlation_breakdown=0.1,
            regime=MarketRegime.SIDEWAYS
        )
        
        performance_history = [0.05] * 10 + [-0.02] * 15
        reward_function = MockReward()
        
        for i in range(3):
            if adapter.should_adapt(market_context, performance_history, 10 * (i + 1)):
                adapter.adapt_parameters(
                    reward_function, market_context, performance_history, 10 * (i + 1)
                )
        
        stats = adapter.get_adaptation_statistics()
        
        self.assertIn("total_adaptations", stats)
        if stats["total_adaptations"] > 0:
            self.assertIn("average_improvement", stats)
            self.assertIn("success_rate", stats)
            self.assertIn("average_confidence", stats)


if __name__ == '__main__':
    unittest.main()