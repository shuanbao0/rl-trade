"""
工厂管理器测试 - Factory Manager Tests

测试工厂管理器的功能，包括工厂注册、选择、性能评估等。
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.rewards.factories.factory_manager import (
    FactoryManager, FactoryPerformanceMetrics,
    get_best_factory, create_trading_system, get_factory_recommendations
)
from src.rewards.factories.abstract_factory import create_factory_configuration
from src.rewards.factories.stock_factory import StockMasterFactory
from src.rewards.factories.forex_factory import ForexMasterFactory
from src.rewards.enums.market_types import MarketType
from src.rewards.enums.time_granularities import TimeGranularity
from src.rewards.enums.risk_profiles import RiskProfile


class TestFactoryManager(unittest.TestCase):
    """测试工厂管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = FactoryManager()
    
    def test_builtin_factories_registered(self):
        """测试内置工厂是否正确注册"""
        factories = self.manager.list_available_factories()
        
        # 验证股票工厂
        self.assertIn("stock_default", factories)
        self.assertIn("stock_1d", factories)
        self.assertIn("stock_1h", factories)
        
        # 验证外汇工厂
        self.assertIn("forex_default", factories)
        self.assertIn("forex_15min", factories)
        self.assertIn("forex_1h", factories)
        
        # 验证默认工厂
        self.assertIn("default", factories)
    
    def test_register_custom_factory(self):
        """测试注册自定义工厂"""
        # 创建测试配置
        config = create_factory_configuration(
            market_type=MarketType.CRYPTO,
            time_granularity=TimeGranularity.MINUTE_15,
            risk_profile=RiskProfile.AGGRESSIVE
        )
        
        # 注册自定义工厂
        self.manager.register_factory("crypto_test", StockMasterFactory, config)
        
        # 验证注册成功
        factories = self.manager.list_available_factories()
        self.assertIn("crypto_test", factories)
        
        factory_info = factories["crypto_test"]
        self.assertEqual(factory_info["class_name"], "StockMasterFactory")
        self.assertTrue(factory_info["has_config"])
    
    def test_get_factory_stock(self):
        """测试获取股票工厂"""
        factory = self.manager.get_factory(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED
        )
        
        self.assertIsInstance(factory, StockMasterFactory)
        self.assertEqual(factory.config.market_type, MarketType.STOCK)
        self.assertEqual(factory.config.time_granularity, TimeGranularity.DAY_1)
    
    def test_get_factory_forex(self):
        """测试获取外汇工厂"""
        factory = self.manager.get_factory(
            market_type=MarketType.FOREX,
            time_granularity=TimeGranularity.MINUTE_15,
            risk_profile=RiskProfile.MODERATE
        )
        
        self.assertIsInstance(factory, ForexMasterFactory)
        self.assertEqual(factory.config.market_type, MarketType.FOREX)
        self.assertEqual(factory.config.time_granularity, TimeGranularity.MINUTE_15)
    
    def test_get_factory_with_strings(self):
        """测试使用字符串参数获取工厂"""
        factory = self.manager.get_factory(
            market_type="stock",
            time_granularity="1d",
            risk_profile="balanced"
        )
        
        self.assertIsInstance(factory, StockMasterFactory)
        self.assertEqual(factory.config.market_type, MarketType.STOCK)
    
    def test_find_best_factory_key(self):
        """测试查找最佳工厂key"""
        # 测试精确匹配
        key = self.manager._find_best_factory_key(
            MarketType.STOCK, TimeGranularity.DAY_1, RiskProfile.BALANCED
        )
        self.assertEqual(key, "stock_1d")
        
        # 测试相似匹配
        key = self.manager._find_best_factory_key(
            MarketType.FOREX, TimeGranularity.MINUTE_1, RiskProfile.BALANCED
        )
        # 应该选择相似的时间粒度
        self.assertTrue(key.startswith("forex_"))
    
    def test_get_similar_granularities(self):
        """测试获取相似时间粒度"""
        similar = self.manager._get_similar_granularities(TimeGranularity.MINUTE_15)
        
        # 15分钟属于短期，应该包含其他短期粒度
        self.assertIn("5min", similar)
        self.assertIn("30min", similar)
        self.assertNotIn("15min", similar)  # 不包含自己
    
    def test_create_complete_system(self):
        """测试创建完整交易系统"""
        system = self.manager.create_complete_system(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1
        )
        
        # 验证系统组件
        self.assertIn("reward", system)
        self.assertIn("environment", system)
        self.assertIn("feature_engineer", system)
        self.assertIn("risk_manager", system)
        self.assertIn("data_processor", system)
        
        # 验证组件创建成功
        for component_name, creation_result in system.items():
            self.assertIsNotNone(creation_result.instance)
    
    def test_evaluate_factory_performance(self):
        """测试工厂性能评估"""
        metrics = self.manager.evaluate_factory_performance("stock_default")
        
        self.assertIsInstance(metrics, FactoryPerformanceMetrics)
        self.assertGreaterEqual(metrics.compatibility_score, 0.0)
        self.assertLessEqual(metrics.compatibility_score, 10.0)
        self.assertGreaterEqual(metrics.success_rate, 0.0)
        self.assertLessEqual(metrics.success_rate, 1.0)
        self.assertGreater(metrics.creation_time, 0.0)
        self.assertGreater(metrics.memory_usage, 0.0)
    
    def test_get_factory_recommendations(self):
        """测试获取工厂推荐"""
        recommendations = self.manager.get_factory_recommendations(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            top_n=3
        )
        
        self.assertLessEqual(len(recommendations), 3)
        
        # 验证推荐格式
        for factory_key, score in recommendations:
            self.assertIsInstance(factory_key, str)
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)
        
        # 验证排序（评分从高到低）
        if len(recommendations) > 1:
            for i in range(len(recommendations) - 1):
                self.assertGreaterEqual(recommendations[i][1], recommendations[i + 1][1])
    
    def test_clear_cache(self):
        """测试清空缓存"""
        # 先创建一些实例
        self.manager.get_factory(MarketType.STOCK, TimeGranularity.DAY_1)
        self.manager.evaluate_factory_performance("stock_default")
        
        # 验证缓存有内容
        self.assertGreater(len(self.manager._factory_instances), 0)
        self.assertGreater(len(self.manager._performance_metrics), 0)
        
        # 清空缓存
        self.manager.clear_cache()
        
        # 验证缓存已清空
        self.assertEqual(len(self.manager._factory_instances), 0)
        self.assertEqual(len(self.manager._performance_metrics), 0)
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        stats = self.manager.get_statistics()
        
        self.assertIn("total_factories", stats)
        self.assertIn("active_instances", stats)
        self.assertIn("available_markets", stats)
        self.assertIn("available_granularities", stats)
        
        # 验证数据类型
        self.assertIsInstance(stats["total_factories"], int)
        self.assertIsInstance(stats["available_markets"], list)
        self.assertIsInstance(stats["available_granularities"], list)
        
        # 验证包含预期的市场和粒度
        self.assertIn("stock", stats["available_markets"])
        self.assertIn("forex", stats["available_markets"])


class TestFactoryManagerConvenienceFunctions(unittest.TestCase):
    """测试工厂管理器便利函数"""
    
    def test_get_best_factory(self):
        """测试获取最佳工厂便利函数"""
        factory = get_best_factory("stock", "1d")
        
        self.assertIsInstance(factory, StockMasterFactory)
        self.assertEqual(factory.config.market_type, MarketType.STOCK)
        self.assertEqual(factory.config.time_granularity, TimeGranularity.DAY_1)
    
    def test_create_trading_system(self):
        """测试创建交易系统便利函数"""
        system = create_trading_system("forex", "15min", risk_profile="moderate")
        
        # 验证系统完整性
        expected_components = ["reward", "environment", "feature_engineer", "risk_manager", "data_processor"]
        for component in expected_components:
            self.assertIn(component, system)
            self.assertIsNotNone(system[component].instance)
    
    def test_get_factory_recommendations_function(self):
        """测试获取工厂推荐便利函数"""
        recommendations = get_factory_recommendations("stock", "1d", top_n=2)
        
        self.assertLessEqual(len(recommendations), 2)
        self.assertIsInstance(recommendations, list)
        
        if recommendations:
            factory_key, score = recommendations[0]
            self.assertIsInstance(factory_key, str)
            self.assertIsInstance(score, (int, float))


class TestFactoryManagerErrorHandling(unittest.TestCase):
    """测试工厂管理器错误处理"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = FactoryManager()
    
    def test_invalid_market_type(self):
        """测试无效市场类型"""
        with self.assertRaises(ValueError):
            self.manager.get_factory("invalid_market", "1d")
    
    def test_invalid_time_granularity(self):
        """测试无效时间粒度"""
        with self.assertRaises(ValueError):
            self.manager.get_factory("stock", "invalid_granularity")
    
    def test_invalid_risk_profile(self):
        """测试无效风险配置"""
        with self.assertRaises(ValueError):
            self.manager.get_factory("stock", "1d", risk_profile="invalid_risk")
    
    def test_nonexistent_factory_performance_evaluation(self):
        """测试评估不存在的工厂性能"""
        with self.assertRaises(ValueError):
            self.manager.evaluate_factory_performance("nonexistent_factory")
    
    def test_factory_performance_evaluation_without_config(self):
        """测试评估没有配置的工厂性能"""
        # 注册一个没有默认配置的工厂
        self.manager.register_factory("no_config_factory", StockMasterFactory)
        
        with self.assertRaises(ValueError):
            self.manager.evaluate_factory_performance("no_config_factory")


class TestFactoryManagerCompatibility(unittest.TestCase):
    """测试工厂管理器兼容性"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = FactoryManager()
    
    def test_stock_factory_compatibility(self):
        """测试股票工厂兼容性"""
        # 测试各种股票配置
        valid_configs = [
            ("stock", "5min", "conservative"),
            ("stock", "1h", "balanced"), 
            ("stock", "1d", "growth"),
            ("stock", "1w", "moderate")
        ]
        
        for market, granularity, risk in valid_configs:
            factory = self.manager.get_factory(market, granularity, risk_profile=risk)
            self.assertIsInstance(factory, StockMasterFactory)
            
            # 验证配置有效性
            valid, errors = factory.validate_system_configuration()
            self.assertTrue(valid, f"Configuration invalid for {market}/{granularity}/{risk}: {errors}")
    
    def test_forex_factory_compatibility(self):
        """测试外汇工厂兼容性"""
        # 测试各种外汇配置
        valid_configs = [
            ("forex", "1min", "aggressive"),
            ("forex", "5min", "moderate"),
            ("forex", "15min", "balanced"),
            ("forex", "1h", "conservative"),
            ("forex", "4h", "growth")
        ]
        
        for market, granularity, risk in valid_configs:
            factory = self.manager.get_factory(market, granularity, risk_profile=risk)
            self.assertIsInstance(factory, ForexMasterFactory)
            
            # 验证配置有效性
            valid, errors = factory.validate_system_configuration()
            self.assertTrue(valid, f"Configuration invalid for {market}/{granularity}/{risk}: {errors}")
    
    def test_cross_market_factory_selection(self):
        """测试跨市场工厂选择"""
        # 测试不同市场类型选择不同工厂
        stock_factory = self.manager.get_factory("stock", "1d")
        forex_factory = self.manager.get_factory("forex", "1h")
        
        self.assertIsInstance(stock_factory, StockMasterFactory)
        self.assertIsInstance(forex_factory, ForexMasterFactory)
        self.assertNotEqual(type(stock_factory), type(forex_factory))


if __name__ == '__main__':
    unittest.main()