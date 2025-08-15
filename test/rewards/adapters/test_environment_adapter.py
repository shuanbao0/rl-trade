"""
环境适配系统测试 - Environment Adapter Tests

测试环境自动适配和配置优化功能。
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.rewards.adapters.environment_adapter import (
    AdaptationLevel, EnvironmentOptimizationGoal, DataCharacteristics,
    EnvironmentAdaptationConfig, EnvironmentAdaptationResult,
    BaseEnvironmentAdapter, AutoEnvironmentAdapter, EnvironmentAdapterFactory,
    create_environment_adapter, analyze_environment_requirements
)
from src.rewards.adapters.parameter_adapter import MarketContext, MarketRegime
from src.rewards.environments.base_environment import EnvironmentConfiguration
from src.rewards.enums.market_types import MarketType
from src.rewards.enums.time_granularities import TimeGranularity
from src.rewards.enums.risk_profiles import RiskProfile
from src.rewards.enums.environment_features import EnvironmentFeature


class MockTradingEnvironment:
    """测试用交易环境"""
    
    def __init__(self, config: EnvironmentConfiguration):
        self.config = config


class TestDataCharacteristics(unittest.TestCase):
    """测试数据特征"""
    
    def test_characteristics_creation(self):
        """测试特征创建"""
        characteristics = DataCharacteristics(
            sample_size=1000,
            feature_count=10,
            data_quality_score=0.85,
            missing_ratio=0.05,
            noise_level=0.2,
            stationarity_score=0.75,
            seasonality_detected=True,
            outlier_ratio=0.08,
            correlation_structure={"max_correlation": 0.6, "mean_correlation": 0.3},
            temporal_consistency=0.9
        )
        
        self.assertEqual(characteristics.sample_size, 1000)
        self.assertEqual(characteristics.feature_count, 10)
        self.assertEqual(characteristics.data_quality_score, 0.85)
        self.assertTrue(characteristics.seasonality_detected)
        self.assertEqual(len(characteristics.correlation_structure), 2)


class TestEnvironmentAdaptationConfig(unittest.TestCase):
    """测试环境适配配置"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = EnvironmentAdaptationConfig(
            adaptation_level=AdaptationLevel.ADVANCED,
            optimization_goal=EnvironmentOptimizationGoal.PERFORMANCE,
            auto_feature_detection=True,
            auto_reward_selection=True,
            adaptation_frequency=150,
            performance_threshold=0.08
        )
        
        self.assertEqual(config.adaptation_level, AdaptationLevel.ADVANCED)
        self.assertEqual(config.optimization_goal, EnvironmentOptimizationGoal.PERFORMANCE)
        self.assertTrue(config.auto_feature_detection)
        self.assertEqual(config.adaptation_frequency, 150)
        self.assertEqual(config.performance_threshold, 0.08)
    
    def test_config_defaults(self):
        """测试配置默认值"""
        config = EnvironmentAdaptationConfig(
            adaptation_level=AdaptationLevel.BASIC,
            optimization_goal=EnvironmentOptimizationGoal.STABILITY
        )
        
        self.assertTrue(config.auto_feature_detection)
        self.assertTrue(config.auto_reward_selection)
        self.assertEqual(config.adaptation_frequency, 100)
        self.assertEqual(config.performance_threshold, 0.05)


class TestAutoEnvironmentAdapter(unittest.TestCase):
    """测试自动环境适配器"""
    
    def setUp(self):
        """设置测试"""
        self.config = EnvironmentAdaptationConfig(
            adaptation_level=AdaptationLevel.INTERMEDIATE,
            optimization_goal=EnvironmentOptimizationGoal.PERFORMANCE,
            adaptation_frequency=10,
            minimum_adaptation_interval=5
        )
        
        self.adapter = AutoEnvironmentAdapter(self.config)
        
        # 创建测试环境配置
        self.env_config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN,
            window_size=20,
            max_steps=1000
        )
        
        self.mock_env = MockTradingEnvironment(self.env_config)
        
        # 创建测试数据
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 200),
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200) * 0.5
        })
    
    def test_adapter_initialization(self):
        """测试适配器初始化"""
        self.assertEqual(self.adapter.config.adaptation_level, AdaptationLevel.INTERMEDIATE)
        self.assertEqual(len(self.adapter.adaptation_history), 0)
        self.assertIsNotNone(self.adapter.regime_detector)
        self.assertIsNotNone(self.adapter.reward_selector)
    
    def test_data_characteristics_analysis(self):
        """测试数据特征分析"""
        characteristics = self.adapter.analyze_data_characteristics(self.test_data)
        
        self.assertIsInstance(characteristics, DataCharacteristics)
        self.assertEqual(characteristics.sample_size, 200)
        self.assertEqual(characteristics.feature_count, 4)
        self.assertTrue(0.0 <= characteristics.data_quality_score <= 1.0)
        self.assertTrue(0.0 <= characteristics.missing_ratio <= 1.0)
        self.assertTrue(characteristics.noise_level >= 0.0)
        self.assertTrue(0.0 <= characteristics.outlier_ratio <= 1.0)
        self.assertIsInstance(characteristics.correlation_structure, dict)
    
    def test_empty_data_analysis(self):
        """测试空数据分析"""
        empty_data = pd.DataFrame()
        characteristics = self.adapter.analyze_data_characteristics(empty_data)
        
        self.assertEqual(characteristics.sample_size, 0)
        self.assertEqual(characteristics.feature_count, 0)
        self.assertEqual(characteristics.data_quality_score, 0.0)
        self.assertEqual(characteristics.missing_ratio, 1.0)
    
    def test_stationarity_estimation(self):
        """测试平稳性估计"""
        # 创建平稳数据
        stationary_data = pd.DataFrame({
            'price': np.random.randn(100)
        })
        
        stationarity = self.adapter._estimate_stationarity(stationary_data)
        self.assertTrue(0.0 <= stationarity <= 1.0)
        
        # 创建非平稳数据
        non_stationary_data = pd.DataFrame({
            'price': np.random.randn(100).cumsum()
        })
        
        stationarity_ns = self.adapter._estimate_stationarity(non_stationary_data)
        self.assertTrue(0.0 <= stationarity_ns <= 1.0)
    
    def test_seasonality_detection(self):
        """测试季节性检测"""
        # 创建有季节性的数据
        t = np.arange(100)
        seasonal_data = pd.DataFrame({
            'price': np.sin(2 * np.pi * t / 12) + np.random.randn(100) * 0.1
        })
        
        seasonality = self.adapter._detect_seasonality(seasonal_data)
        self.assertIsInstance(seasonality, bool)
        
        # 测试数据不足的情况
        small_data = pd.DataFrame({'price': [1, 2, 3]})
        seasonality_small = self.adapter._detect_seasonality(small_data)
        self.assertFalse(seasonality_small)
    
    def test_outlier_ratio_calculation(self):
        """测试异常值比例计算"""
        # 创建包含异常值的数据
        data_with_outliers = pd.DataFrame({
            'price': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]  # 100是异常值
        })
        
        outlier_ratio = self.adapter._calculate_outlier_ratio(data_with_outliers)
        self.assertTrue(0.0 <= outlier_ratio <= 1.0)
        self.assertGreater(outlier_ratio, 0.0)  # 应该检测到异常值
    
    def test_correlation_structure_analysis(self):
        """测试相关性结构分析"""
        # 创建相关数据
        corr_data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })
        corr_data['z'] = corr_data['x'] + np.random.randn(100) * 0.1  # 与x高度相关
        
        correlation_structure = self.adapter._analyze_correlation_structure(corr_data)
        
        self.assertIn("max_correlation", correlation_structure)
        self.assertIn("mean_correlation", correlation_structure)
        self.assertIn("correlation_diversity", correlation_structure)
        self.assertTrue(0.0 <= correlation_structure["max_correlation"] <= 1.0)
    
    def test_temporal_consistency_calculation(self):
        """测试时间一致性计算"""
        # 创建时间索引数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        time_data = pd.DataFrame({
            'price': np.random.randn(50)
        }, index=dates)
        
        consistency = self.adapter._calculate_temporal_consistency(time_data)
        self.assertTrue(0.0 <= consistency <= 1.0)
        
        # 测试无时间索引的情况
        no_time_data = pd.DataFrame({'price': np.random.randn(20)})
        consistency_no_time = self.adapter._calculate_temporal_consistency(no_time_data)
        self.assertTrue(0.0 <= consistency_no_time <= 1.0)
    
    def test_price_column_identification(self):
        """测试价格列识别"""
        # 标准价格列名
        price_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        price_col = self.adapter._identify_price_column(price_data)
        self.assertEqual(price_col, 'close')
        
        # 无标准价格列名
        no_price_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        price_col_alt = self.adapter._identify_price_column(no_price_data)
        self.assertIn(price_col_alt, ['feature1', 'feature2'])
    
    def test_market_context_generation(self):
        """测试市场上下文生成"""
        context = self.adapter._get_market_context(self.test_data)
        
        self.assertIsInstance(context, MarketContext)
        self.assertIsInstance(context.regime, MarketRegime)
        self.assertTrue(context.volatility >= 0.0)
        self.assertTrue(0.0 <= context.trend_strength <= 1.0)
    
    def test_performance_issues_analysis(self):
        """测试性能问题分析"""
        # 创建下降性能历史
        declining_performance = [0.1 - i * 0.01 for i in range(30)]
        
        characteristics = DataCharacteristics(
            sample_size=200, feature_count=4, data_quality_score=0.8,
            missing_ratio=0.1, noise_level=0.3, stationarity_score=0.7,
            seasonality_detected=False, outlier_ratio=0.05,
            correlation_structure={}, temporal_consistency=0.8
        )
        
        analysis = self.adapter._analyze_performance_issues(declining_performance, characteristics)
        
        self.assertIn("trend", analysis)
        self.assertIn("issues", analysis)
        self.assertIn("volatility", analysis)
        self.assertEqual(analysis["trend"], "declining")
        self.assertIn("performance_declining", analysis["issues"])
    
    def test_configuration_optimization(self):
        """测试配置优化"""
        characteristics = DataCharacteristics(
            sample_size=500, feature_count=8, data_quality_score=0.75,
            missing_ratio=0.15, noise_level=0.4, stationarity_score=0.6,
            seasonality_detected=True, outlier_ratio=0.12,
            correlation_structure={"max_correlation": 0.8}, temporal_consistency=0.7
        )
        
        market_context = MarketContext(
            volatility=0.25, trend_strength=0.8, momentum=0.05,
            volume_profile=1.2, correlation_breakdown=0.3,
            regime=MarketRegime.HIGH_VOLATILITY
        )
        
        performance_analysis = {
            "trend": "declining",
            "issues": ["high_volatility", "poor_data_quality"]
        }
        
        new_config = self.adapter._generate_optimized_configuration(
            self.env_config, characteristics, market_context, performance_analysis
        )
        
        self.assertIsInstance(new_config, EnvironmentConfiguration)
        # 高噪声应该增加窗口大小
        self.assertGreaterEqual(new_config.window_size, self.env_config.window_size)
        # 高波动应该启用标准化
        self.assertTrue(new_config.normalize_observations)
    
    def test_should_adapt_frequency(self):
        """测试适配频率检查"""
        performance_history = [0.01] * 25
        
        # 不满足频率要求
        should_adapt = self.adapter.should_adapt(
            self.mock_env, self.test_data, performance_history, 7
        )
        self.assertFalse(should_adapt)
        
        # 满足频率但性能没有下降
        should_adapt = self.adapter.should_adapt(
            self.mock_env, self.test_data, performance_history, 10
        )
        self.assertFalse(should_adapt)  # 性能稳定，不需要适配
    
    def test_should_adapt_performance_decline(self):
        """测试性能下降时适配"""
        # 创建性能下降历史
        declining_performance = [0.05] * 20 + [-0.02] * 15
        
        should_adapt = self.adapter.should_adapt(
            self.mock_env, self.test_data, declining_performance, 10
        )
        self.assertTrue(should_adapt)
    
    def test_should_adapt_insufficient_data(self):
        """测试数据不足时不适配"""
        short_performance = [0.01] * 10  # 少于20个数据点
        
        should_adapt = self.adapter.should_adapt(
            self.mock_env, self.test_data, short_performance, 10
        )
        self.assertFalse(should_adapt)
    
    def test_environment_configuration_validation(self):
        """测试环境配置验证"""
        # 有效配置
        valid_config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN,
            window_size=10,
            max_steps=100
        )
        
        is_valid, errors = self.adapter.validate_environment_configuration(valid_config, self.test_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # 无效配置（窗口大小过大）
        invalid_config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN,
            window_size=300,  # 大于数据长度
            max_steps=100
        )
        
        is_valid, errors = self.adapter.validate_environment_configuration(invalid_config, self.test_data)
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)
    
    def test_adaptation_execution(self):
        """测试适配执行"""
        # 创建需要适配的场景
        declining_performance = [0.08 - i * 0.002 for i in range(40)]
        
        result = self.adapter.adapt_environment(
            self.mock_env, self.test_data, declining_performance, 10
        )
        
        self.assertIsInstance(result, EnvironmentAdaptationResult)
        self.assertIsInstance(result.old_configuration, EnvironmentConfiguration)
        self.assertIsInstance(result.new_configuration, EnvironmentConfiguration)
        self.assertIsInstance(result.performance_improvement, float)
        self.assertIsInstance(result.confidence_score, float)
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.warnings, list)
        
        # 检查适配历史记录
        self.assertEqual(len(self.adapter.adaptation_history), 1)
        self.assertEqual(self.adapter.last_adaptation_step, 10)
    
    def test_adaptation_improvement_estimation(self):
        """测试适配改善估计"""
        old_config = self.env_config
        new_config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN | EnvironmentFeature.HIGH_VOLATILITY,
            window_size=25,  # 变化
            max_steps=1000,
            normalize_observations=True  # 变化
        )
        
        performance_analysis = {"issues": ["performance_declining"]}
        market_context = MarketContext(
            volatility=0.3, trend_strength=0.7, momentum=0.0,
            volume_profile=1.0, correlation_breakdown=0.2,
            regime=MarketRegime.HIGH_VOLATILITY
        )
        
        improvement = self.adapter._estimate_adaptation_improvement(
            old_config, new_config, performance_analysis, market_context
        )
        
        self.assertTrue(0.0 <= improvement <= 0.1)
        self.assertGreater(improvement, 0.0)  # 应该有改善
    
    def test_adaptation_confidence_calculation(self):
        """测试适配置信度计算"""
        characteristics = DataCharacteristics(
            sample_size=1000, feature_count=5, data_quality_score=0.9,
            missing_ratio=0.05, noise_level=0.1, stationarity_score=0.8,
            seasonality_detected=True, outlier_ratio=0.03,
            correlation_structure={}, temporal_consistency=0.85
        )
        
        market_context = MarketContext(
            volatility=0.15, trend_strength=0.75, momentum=0.02,
            volume_profile=1.1, correlation_breakdown=0.1,
            regime=MarketRegime.BULL_MARKET
        )
        
        performance_analysis = {"issues": ["high_volatility"]}
        
        confidence = self.adapter._calculate_adaptation_confidence(
            characteristics, market_context, performance_analysis
        )
        
        self.assertTrue(0.1 <= confidence <= 0.95)
    
    def test_recommendations_generation(self):
        """测试建议生成"""
        characteristics = DataCharacteristics(
            sample_size=200, feature_count=3, data_quality_score=0.6,
            missing_ratio=0.3, noise_level=0.4, stationarity_score=0.5,
            seasonality_detected=False, outlier_ratio=0.15,
            correlation_structure={}, temporal_consistency=0.6
        )
        
        market_context = MarketContext(
            volatility=0.35, trend_strength=0.5, momentum=0.0,
            volume_profile=1.0, correlation_breakdown=0.4,
            regime=MarketRegime.HIGH_VOLATILITY
        )
        
        recommendations = self.adapter._generate_recommendations(
            self.env_config, characteristics, market_context
        )
        
        self.assertIsInstance(recommendations, list)
        # 应该包含数据质量和噪声相关的建议
        recommendation_text = " ".join(recommendations)
        self.assertTrue(any(keyword in recommendation_text.lower() for keyword in 
                          ["data", "noise", "volatility", "outlier"]))
    
    def test_warnings_generation(self):
        """测试警告生成"""
        characteristics = DataCharacteristics(
            sample_size=50, feature_count=2, data_quality_score=0.4,
            missing_ratio=0.4, noise_level=0.2, stationarity_score=0.3,
            seasonality_detected=False, outlier_ratio=0.08,
            correlation_structure={}, temporal_consistency=0.3
        )
        
        warnings = self.adapter._generate_adaptation_warnings(self.env_config, characteristics)
        
        self.assertIsInstance(warnings, list)
        # 应该包含数据量不足和数据质量相关的警告
        warning_text = " ".join(warnings)
        self.assertTrue(any(keyword in warning_text.lower() for keyword in 
                          ["limited", "missing", "consistency"]))
    
    def test_adaptation_statistics(self):
        """测试适配统计"""
        # 执行几次适配
        declining_performance = [0.08 - i * 0.002 for i in range(40)]
        
        for i in range(3):
            if self.adapter.should_adapt(self.mock_env, self.test_data, declining_performance, 10 * (i + 1)):
                result = self.adapter.adapt_environment(
                    self.mock_env, self.test_data, declining_performance, 10 * (i + 1)
                )
        
        stats = self.adapter.get_adaptation_statistics()
        
        self.assertIn("total_adaptations", stats)
        if stats["total_adaptations"] > 0:
            self.assertIn("average_improvement", stats)
            self.assertIn("success_rate", stats)
            self.assertIn("average_confidence", stats)


class TestEnvironmentAdapterFactory(unittest.TestCase):
    """测试环境适配器工厂"""
    
    def test_create_auto_adapter(self):
        """测试创建自动适配器"""
        adapter = EnvironmentAdapterFactory.create_adapter(
            adaptation_level=AdaptationLevel.ADVANCED,
            optimization_goal=EnvironmentOptimizationGoal.ROBUSTNESS
        )
        
        self.assertIsInstance(adapter, AutoEnvironmentAdapter)
        self.assertEqual(adapter.config.adaptation_level, AdaptationLevel.ADVANCED)
        self.assertEqual(adapter.config.optimization_goal, EnvironmentOptimizationGoal.ROBUSTNESS)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便利函数"""
    
    def setUp(self):
        """设置测试"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'close': np.random.randn(150).cumsum() + 100,
            'volume': np.random.randint(1000, 5000, 150),
            'feature1': np.random.randn(150) * 0.5
        })
    
    def test_create_environment_adapter_function(self):
        """测试创建环境适配器便利函数"""
        adapter = create_environment_adapter(
            adaptation_level="advanced",
            optimization_goal="performance",
            adaptation_frequency=80,
            performance_threshold=0.03
        )
        
        self.assertIsInstance(adapter, AutoEnvironmentAdapter)
        self.assertEqual(adapter.config.adaptation_level, AdaptationLevel.ADVANCED)
        self.assertEqual(adapter.config.optimization_goal, EnvironmentOptimizationGoal.PERFORMANCE)
        self.assertEqual(adapter.config.adaptation_frequency, 80)
        self.assertEqual(adapter.config.performance_threshold, 0.03)
    
    def test_analyze_environment_requirements_function(self):
        """测试分析环境需求便利函数"""
        analysis = analyze_environment_requirements(
            data=self.test_data,
            market_type="stock",
            time_granularity="1d"
        )
        
        self.assertIn("data_characteristics", analysis)
        self.assertIn("market_context", analysis)
        self.assertIn("recommended_configuration", analysis)
        self.assertIn("analysis_summary", analysis)
        
        # 检查数据特征
        self.assertIsInstance(analysis["data_characteristics"], DataCharacteristics)
        
        # 检查市场上下文
        self.assertIsInstance(analysis["market_context"], MarketContext)
        
        # 检查推荐配置
        recommended = analysis["recommended_configuration"]
        self.assertIn("window_size", recommended)
        self.assertIn("normalize_observations", recommended)
        self.assertIn("environment_features", recommended)
        
        # 检查分析摘要
        summary = analysis["analysis_summary"]
        self.assertIn("data_quality", summary)
        self.assertIn("market_regime", summary)
        self.assertIn("complexity_level", summary)
        self.assertIn("recommended_adaptation_level", summary)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        adapter = create_environment_adapter()
        empty_data = pd.DataFrame()
        
        # 分析空数据不应该崩溃
        characteristics = adapter.analyze_data_characteristics(empty_data)
        self.assertEqual(characteristics.sample_size, 0)
        
        # 分析环境需求不应该崩溃
        analysis = analyze_environment_requirements(empty_data)
        self.assertIsInstance(analysis, dict)
    
    def test_single_column_data(self):
        """测试单列数据"""
        adapter = create_environment_adapter()
        single_col_data = pd.DataFrame({'price': np.random.randn(50)})
        
        characteristics = adapter.analyze_data_characteristics(single_col_data)
        self.assertEqual(characteristics.feature_count, 1)
        self.assertTrue(characteristics.data_quality_score >= 0)
    
    def test_high_missing_data(self):
        """测试高缺失率数据"""
        adapter = create_environment_adapter()
        
        # 创建50%缺失的数据
        data_with_missing = pd.DataFrame({
            'price': [100, np.nan, 102, np.nan, 104] * 20,
            'volume': [1000, 1100, np.nan, np.nan, 1400] * 20
        })
        
        characteristics = adapter.analyze_data_characteristics(data_with_missing)
        self.assertGreater(characteristics.missing_ratio, 0.3)
        self.assertLess(characteristics.data_quality_score, 0.8)
    
    def test_constant_data(self):
        """测试常数数据"""
        adapter = create_environment_adapter()
        
        # 所有值相同的数据
        constant_data = pd.DataFrame({
            'price': [100] * 50,
            'volume': [1000] * 50
        })
        
        characteristics = adapter.analyze_data_characteristics(constant_data)
        self.assertEqual(characteristics.noise_level, 0.0)
        self.assertEqual(characteristics.outlier_ratio, 0.0)
    
    def test_extreme_adaptation_scenarios(self):
        """测试极端适配场景"""
        config = EnvironmentAdaptationConfig(
            adaptation_level=AdaptationLevel.EXPERT,
            optimization_goal=EnvironmentOptimizationGoal.MULTI_OBJECTIVE,
            adaptation_frequency=1,  # 每步都适配
            performance_threshold=0.001  # 非常低的阈值
        )
        
        adapter = AutoEnvironmentAdapter(config)
        
        # 这种极端配置应该能正常创建
        self.assertIsInstance(adapter, AutoEnvironmentAdapter)
        self.assertEqual(adapter.config.adaptation_frequency, 1)
    
    def test_very_small_data(self):
        """测试极小数据集"""
        adapter = create_environment_adapter()
        tiny_data = pd.DataFrame({'price': [100, 101, 99]})
        
        characteristics = adapter.analyze_data_characteristics(tiny_data)
        self.assertEqual(characteristics.sample_size, 3)
        
        # 季节性检测应该返回False（数据太少）
        seasonality = adapter._detect_seasonality(tiny_data)
        self.assertFalse(seasonality)
    
    def test_non_numeric_data_handling(self):
        """测试非数值数据处理"""
        adapter = create_environment_adapter()
        
        mixed_data = pd.DataFrame({
            'price': [100, 101, 102, 103, 104],
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        })
        
        characteristics = adapter.analyze_data_characteristics(mixed_data)
        # 应该只计算数值列
        self.assertEqual(characteristics.feature_count, 3)  # 总列数
        # 但分析应该基于数值列


if __name__ == '__main__':
    unittest.main()